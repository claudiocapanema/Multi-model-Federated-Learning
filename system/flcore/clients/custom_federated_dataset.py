# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""FederatedDataset."""


from typing import Any, Optional, Union

import random

import datasets
from datasets import Dataset, DatasetDict
from PIL import Image
from flwr_datasets.common import EventType, event
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.preprocessor import Preprocessor
from flwr_datasets.utils import (
    # _check_if_dataset_tested,
    _instantiate_merger_if_needed,
    _instantiate_partitioners,
)


# flake8: noqa: E501
# pylint: disable=line-too-long
class CustomFederatedDataset:
    """Representation of a dataset for federated learning/evaluation/analytics.

    Download, partition data among clients (edge devices), or load full dataset.

    Partitions are created per-split-basis using Partitioners from
    `flwr_datasets.partitioner` specified in `partitioners` (see `partitioners`
    parameter for more information).

    Parameters
    ----------
    dataset : str
        The name of the dataset in the Hugging Face Hub.
    subset : str
        Secondary information regarding the dataset, most often subset or version
        (that is passed to the name in datasets.load_dataset).
    preprocessor : Optional[Union[Preprocessor, Dict[str, Tuple[str, ...]]]]
        `Callable` that transforms `DatasetDict` by resplitting, removing
        features, creating new features, performing any other preprocessing operation,
        or configuration dict for `Merger`. Applied after shuffling. If None,
        no operation is applied.
    partitioners : Dict[str, Union[Partitioner, int]]
        A dictionary mapping the Dataset split (a `str`) to a `Partitioner` or an `int`
        (representing the number of IID partitions that this split should be
        partitioned into, i.e., using the default partitioner
        `IidPartitioner <https://flower.ai/docs/datasets/ref-api/flwr_
        datasets.partitioner.IidPartitioner.html>`_). One or multiple `Partitioner`
        objects can be specified in that manner, but at most, one per split.
    shuffle : bool
        Whether to randomize the order of samples. Applied prior to preprocessing
        operations, speratelly to each of the present splits in the dataset. It uses
        the `seed` argument. Defaults to True.
    seed : Optional[int]
        Seed used for dataset shuffling. It has no effect if `shuffle` is False. The
        seed cannot be set in the later stages. If `None`, then fresh, unpredictable
        entropy will be pulled from the OS. Defaults to 42.
    load_dataset_kwargs : Any
        Additional keyword arguments passed to `datasets.load_dataset` function.
        Currently used paramters used are dataset => path (in load_dataset),
        subset => name (in load_dataset). You can pass e.g., `num_proc=4`,
        `trust_remote_code=True`. Do not pass any parameters that modify the
        return type such as another type than DatasetDict is returned.

    Examples
    --------
    Use MNIST dataset for Federated Learning with 100 clients (edge devices):

    >>> from flwr_datasets import FederatedDataset
    >>>
    >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})
    >>> # Load partition for a client with ID 10.
    >>> partition = fds.load_partition(10)
    >>> # Use test split for centralized evaluation.
    >>> centralized = fds.load_split("test")

    Use CIFAR10 dataset for Federated Laerning with 100 clients:

    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import DirichletPartitioner
    >>>
    >>> partitioner = DirichletPartitioner(num_partitions=10, partition_by="label",
    >>>                                    alpha=0.5, min_partition_size=10)
    >>> fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
    >>> partition = fds.load_partition(partition_id=0)

    Visualize the partitioned datasets:

    >>> from flwr_datasets.visualization import plot_label_distributions
    >>>
    >>> _ = plot_label_distributions(
    >>>     partitioner=fds.partitioners["train"],
    >>>     label_name="label",
    >>>     legend=True,
    >>> )
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    def __init__(
        self,
        *,
        dataset: str,
        path: Optional[str] = None,
        subset: Optional[str] = None,
        dataset_name: Optional[str] = None,
        preprocessor: Optional[Union[Preprocessor, dict[str, tuple[str, ...]]]] = None,
        partitioners: dict[str, Union[Partitioner, int]],
        shuffle: bool = True,
        seed: Optional[int] = 42,
        duplication_factors: Optional[dict[str, int]] = None,
        duplication_client_threshold: int = 100,
        augmentation_datasets: Optional[dict[str, bool]] = None,
        rotation_degrees: float = 10.0,
        translation_pixels: int = 2,
        **load_dataset_kwargs: Any,
    ) -> None:
        # _check_if_dataset_tested(dataset)
        self._dataset_id: str = dataset
        self._dataset_name: str = dataset_name or dataset
        self._dataset_path: Optional[str] = path
        self._subset: Optional[str] = subset
        self._preprocessor: Optional[Preprocessor] = _instantiate_merger_if_needed(
            preprocessor
        )
        self._partitioners: dict[str, Partitioner] = _instantiate_partitioners(
            partitioners
        )
        self._check_partitioners_correctness()
        self._shuffle = shuffle
        self._seed = seed
        self._duplication_factors = duplication_factors or {}
        self._duplication_client_threshold = duplication_client_threshold
        self._augmentation_datasets = (
            {"CIFAR10": True, "MNIST": True, "GTSRB": True, "ImageNet10": True, "F-MNIST": True}
            if augmentation_datasets is None
            else augmentation_datasets
        )
        self._rotation_degrees = float(rotation_degrees)
        self._translation_pixels = int(translation_pixels)
        #  _dataset is prepared lazily on the first call to `load_partition`
        #  or `load_split`. See _prepare_datasets for more details
        self._dataset: Optional[DatasetDict] = None
        # Indicate if the dataset is prepared for `load_partition` or `load_split`
        self._dataset_prepared: bool = False
        self._event = {
            "load_partition": {split: False for split in self._partitioners},
        }
        self._load_dataset_kwargs = load_dataset_kwargs

    def load_partition(
        self,
        partition_id: int,
        split: Optional[str] = None,
    ) -> Dataset:
        """Load the partition specified by the idx in the selected split.

        The dataset is downloaded only when the first call to `load_partition` or
        `load_split` is made.

        Parameters
        ----------
        partition_id : int
            Partition index for the selected split, idx in {0, ..., num_partitions - 1}.
        split : Optional[str]
            Name of the (partitioned) split (e.g. "train", "test"). You can skip this
            parameter if there is only one partitioner for the dataset. The name will be
            inferred automatically. For example, if `partitioners={"train": 10}`, you do
            not need to provide this argument, but if `partitioners={"train": 10,
            "test": 100}`, you need to set it to differentiate which partitioner should
            be used.
            The split names you can choose from vary from dataset to dataset. You need
            to check the dataset on the `Hugging Face Hub`<https://huggingface.co/
            datasets>_ to see which splits are available. You can resplit the dataset
            by using the `preprocessor` parameter (to rename, merge, divide, etc. the
            available splits).

        Returns
        -------
        partition : Dataset
            Single partition from the dataset split.
        """
        if not self._dataset_prepared:
            self._prepare_dataset()
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")
        if split is None:
            self._check_if_no_split_keyword_possible()
            split = list(self._partitioners.keys())[0]
        self._check_if_split_present(split)
        self._check_if_split_possible_to_federate(split)
        partitioner: Partitioner = self._partitioners[split]
        self._assign_dataset_to_partitioner(split)
        partition = partitioner.load_partition(partition_id)
        if not self._event["load_partition"][split]:
            event(
                EventType.LOAD_PARTITION_CALLED,
                {
                    "federated_dataset_id": id(self),
                    "dataset_name": self._dataset_name,
                    "split": split,
                    "partitioner": partitioner.__class__.__name__,
                    "num_partitions": partitioner.num_partitions,
                },
            )
            self._event["load_partition"][split] = True
        return partition

    def load_split(self, split: str) -> Dataset:
        """Load the full split of the dataset.

        The dataset is downloaded only when the first call to `load_partition` or
        `load_split` is made.

        Parameters
        ----------
        split : str
            Split name of the downloaded dataset (e.g. "train", "test").
            The split names you can choose from vary from dataset to dataset. You need
            to check the dataset on the `Hugging Face Hub`<https://huggingface.co/
            datasets>_ to see which splits are available. You can resplit the dataset
            by using the `preprocessor` parameter (to rename, merge, divide, etc. the
            available splits).

        Returns
        -------
        dataset_split : Dataset
            Part of the dataset identified by its split name.
        """
        if not self._dataset_prepared:
            self._prepare_dataset()
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")
        self._check_if_split_present(split)
        dataset_split = self._dataset[split]

        if not self._event["load_split"][split]:
            event(
                EventType.LOAD_SPLIT_CALLED,
                {
                    "federated_dataset_id": id(self),
                    "dataset_name": self._dataset_name,
                    "split": split,
                },
            )
            self._event["load_split"][split] = True

        return dataset_split

    @property
    def partitioners(self) -> dict[str, Partitioner]:
        """Dictionary mapping each split to its associated partitioner.

        The returned partitioners have the splits of the dataset assigned to them.
        """
        # This function triggers the dataset download (lazy download) and checks
        # the partitioner specification correctness (which can also happen lazily only
        # after the dataset download).
        if not self._dataset_prepared:
            self._prepare_dataset()
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")
        partitioners_keys = list(self._partitioners.keys())
        for split in partitioners_keys:
            self._check_if_split_present(split)
            self._assign_dataset_to_partitioner(split)
        return self._partitioners

    def _check_if_split_present(self, split: str) -> None:
        """Check if the split (for partitioning or full return) is in the dataset."""
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")
        available_splits = list(self._dataset.keys())
        if split not in available_splits:
            raise ValueError(
                f"The given split: '{split}' is not present in the dataset's splits: "
                f"'{available_splits}'."
            )

    def _check_if_split_possible_to_federate(self, split: str) -> None:
        """Check if the split has corresponding partitioner."""
        partitioners_keys = list(self._partitioners.keys())
        if split not in partitioners_keys:
            raise ValueError(
                f"The given split: '{split}' does not have a partitioner to perform "
                f"partitioning. Partitioners were specified for the following splits:"
                f"'{partitioners_keys}'."
            )

    def _assign_dataset_to_partitioner(self, split: str) -> None:
        """Assign the corresponding split of the dataset to the partitioner.

        Assign only if the dataset is not assigned yet.
        """
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")
        if not self._partitioners[split].is_dataset_assigned():
            self._partitioners[split].dataset = self._dataset[split]

    def _prepare_dataset(self) -> None:
        """Prepare the dataset (prior to partitioning) by download, shuffle, replit.

        Run only ONCE when triggered by load_* function. (In future more control whether
        this should happen lazily or not can be added). The operations done here should
        not happen more than once.

        It is controlled by a single flag, `_dataset_prepared` that is set True at the
        end of the function.

        Notes
        -----
        The shuffling should happen before the resplitting. Here is the explanation.
        If the dataset has a non-random order of samples e.g. each split has first
        only label 0, then only label 1. Then in case of resplitting e.g.
        someone creates: "train" train[:int(0.75 * len(train))], test: concat(
        train[int(0.75 * len(train)):], test). The new test took the 0.25 of e.g.
        the train that is only label 0 (assuming the equal count of labels).
        Therefore, for such edge cases (for which we have split) the split should
        happen before the resplitting.
        """
        # Load either from a local Arrow dataset directory or directly from the
        # Hugging Face dataset identifier. The latter is useful when ``path`` is
        # not supplied by the caller.
        if self._dataset_path is not None:
            self._dataset = datasets.load_from_disk(
                dataset_path=self._dataset_path
            )
        else:
            self._dataset = datasets.load_dataset(
                self._dataset_id,
                **self._load_dataset_kwargs,
            )

        if not isinstance(self._dataset, datasets.DatasetDict):
            raise ValueError(
                "Probably one of the specified parameter in `load_dataset_kwargs` "
                "change the return type of the datasets.load_dataset function. "
                "Make sure to use parameter such that the return type is DatasetDict. "
                f"The return type is currently: {type(self._dataset)}."
            )

        # Expand only the TRAIN split, and only when both conditions hold:
        #   1) the dataset name is configured for duplication; and
        #   2) the number of federated clients is greater than the threshold.
        # This happens before shuffling and before assigning the dataset to the
        # partitioner, so the partitioner sees the expanded dataset.
        num_partitions = max(
            (partitioner.num_partitions for partitioner in self._partitioners.values()),
            default=0,
        )
        dataset_key = self._dataset_name.strip().upper()
        duplication_factor = self._duplication_factors.get(dataset_key, 1)

        if (
            "train" in self._dataset
            and duplication_factor > 1
            and num_partitions >= self._duplication_client_threshold
        ):
            original_train = self._dataset["train"]
            original_train_size = len(original_train)
            train_copies = [original_train]

            use_augmentation = self._augmentation_datasets.get(dataset_key, False)

            if use_augmentation:
                # Keep the first copy unchanged. The additional copies are
                # independently augmented versions of the original samples.
                for copy_id in range(1, duplication_factor):
                    augmented_train = self._augment_train_split(
                        original_train,
                        seed=(
                            None
                            if self._seed is None
                            else self._seed + copy_id
                        ),
                    )
                    train_copies.append(augmented_train)
            else:
                # Fallback for configured datasets without augmentation.
                train_copies.extend(
                    [original_train] * (duplication_factor - 1)
                )

            self._dataset["train"] = datasets.concatenate_datasets(train_copies)

            print(
                f"Expanded dataset {self._dataset_name}: train size "
                f"{original_train_size} -> {len(self._dataset['train'])} "
                f"({duplication_factor}x) for {num_partitions} clients; "
                f"augmentation={use_augmentation}, "
                f"rotation=+/-{self._rotation_degrees}deg, "
                f"translation=+/-{self._translation_pixels}px."
            )

        if self._shuffle:
            # Note it shuffles all the splits. The self._dataset is DatasetDict
            # so e.g. {"train": train_data, "test": test_data}. All splits get shuffled.
            self._dataset = self._dataset.shuffle(seed=self._seed)
        if self._preprocessor:
            self._dataset = self._preprocessor(self._dataset)
        available_splits = list(self._dataset.keys())
        self._event["load_split"] = {split: False for split in available_splits}
        self._dataset_prepared = True

    def _augment_train_split(
        self,
        train_dataset: Dataset,
        seed: Optional[int],
    ) -> Dataset:
        """Create a mildly augmented copy of an image training split.

        The original split is not modified. Each sample receives a small
        random rotation and a small random horizontal/vertical translation.
        Labels and all non-image columns are preserved unchanged.
        """
        image_column = self._find_image_column(train_dataset)
        base_seed = 0 if seed is None else int(seed)

        def augment_example(example: dict[str, Any], idx: int) -> dict[str, Any]:
            image = example[image_column]

            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            rng = random.Random(base_seed + idx)

            angle = rng.uniform(
                -self._rotation_degrees,
                self._rotation_degrees,
            )
            tx = rng.randint(
                -self._translation_pixels,
                self._translation_pixels,
            )
            ty = rng.randint(
                -self._translation_pixels,
                self._translation_pixels,
            )

            augmented = image.rotate(
                angle,
                resample=Image.Resampling.BILINEAR,
                fillcolor=0,
            )

            # Apply a small translation after rotation.
            augmented = augmented.transform(
                augmented.size,
                Image.Transform.AFFINE,
                (1, 0, -tx, 0, 1, -ty),
                resample=Image.Resampling.BILINEAR,
                fillcolor=0,
            )

            example[image_column] = augmented
            return example

        return train_dataset.map(
            augment_example,
            with_indices=True,
            desc=f"Mild augmentation: {self._dataset_name} train",
        )

    @staticmethod
    def _find_image_column(dataset: Dataset) -> str:
        """Find the image column without assuming a dataset-specific name."""
        for column_name, feature in dataset.features.items():
            if isinstance(feature, datasets.Image):
                return column_name

        for column_name in ("image", "img", "pixel_values"):
            if column_name in dataset.column_names:
                return column_name

        raise ValueError(
            "Could not identify an image column in dataset. "
            f"Available columns: {dataset.column_names}"
        )

    def _check_if_no_split_keyword_possible(self) -> None:
        if len(self._partitioners) != 1:
            raise ValueError(
                "Please set the `split` argument. You can only omit the split keyword "
                "if there is exactly one partitioner specified."
            )

    def _check_partitioners_correctness(self) -> None:
        """Check if the partitioners are correctly specified.

        Check if each partitioner is a different Python object. Using the same
        partitioner for different splits is not allowed.
        """
        partitioners_keys = list(self._partitioners.keys())
        for i, first_split in enumerate(partitioners_keys):
            for j in range(i + 1, len(partitioners_keys)):
                second_split = partitioners_keys[j]
                if self._partitioners[first_split] is self._partitioners[second_split]:
                    raise ValueError(
                        f"The same partitioner object is used for multiple splits: "
                        f"('{first_split}', '{second_split}'). "
                        "Each partitioner should be a separate object."
                    )