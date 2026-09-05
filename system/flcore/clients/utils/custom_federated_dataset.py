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

"""Custom FederatedDataset with optional dataset duplication."""

from typing import Any, Optional, Union

import datasets
from datasets import Dataset, DatasetDict
from flwr_datasets.common import EventType, event
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.preprocessor import Preprocessor
from flwr_datasets.utils import (
    _instantiate_merger_if_needed,
    _instantiate_partitioners,
)


class CustomFederatedDataset:
    """Federated dataset supporting local/Hugging Face loading and train duplication.

    The class intentionally keeps the same partitioning flow as Flower's
    FederatedDataset. The only additional operation is optional duplication of
    the TRAIN split before shuffling and before the partitioner receives it.
    """

    # pylint: disable=too-many-instance-attributes, too-many-arguments
    def __init__(
        self,
        *,
        dataset: str,
        path: Optional[str] = None,
        subset: Optional[str] = None,
        dataset_name: Optional[str] = None,
        preprocessor: Optional[
            Union[Preprocessor, dict[str, tuple[str, ...]]]
        ] = None,
        partitioners: dict[str, Union[Partitioner, int]],
        shuffle: bool = True,
        seed: Optional[int] = 42,
        duplication_factors: Optional[dict[str, int]] = None,
        duplication_client_threshold: int = 100,
        **load_dataset_kwargs: Any,
    ) -> None:
        # ``dataset`` is the Hugging Face dataset identifier.
        self._dataset_id: str = dataset

        # ``dataset_name`` is the experiment's short/local name, e.g. CIFAR10.
        self._dataset_name: str = dataset_name or dataset

        # If path is supplied, the dataset is loaded from disk. Otherwise it is
        # loaded from the Hugging Face Hub.
        self._dataset_path: Optional[str] = path

        self._subset: Optional[str] = subset

        self._preprocessor: Optional[Preprocessor] = (
            _instantiate_merger_if_needed(preprocessor)
        )

        self._partitioners: dict[str, Partitioner] = _instantiate_partitioners(
            partitioners
        )

        self._check_partitioners_correctness()

        self._shuffle = shuffle
        self._seed = seed

        self._duplication_factors: dict[str, int] = (
            duplication_factors.copy() if duplication_factors else {}
        )

        if duplication_client_threshold < 0:
            raise ValueError(
                "duplication_client_threshold must be greater than or equal to 0."
            )

        self._duplication_client_threshold = duplication_client_threshold

        # Dataset loading is lazy, exactly as in Flower's implementation.
        self._dataset: Optional[DatasetDict] = None
        self._dataset_prepared = False

        self._event = {
            "load_partition": {split: False for split in self._partitioners}
        }

        # Additional arguments passed to datasets.load_dataset().
        self._load_dataset_kwargs = load_dataset_kwargs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_partition(
        self,
        partition_id: int,
        split: Optional[str] = None,
    ) -> Dataset:
        """Load one federated partition."""

        if not self._dataset_prepared:
            self._prepare_dataset()

        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")

        if split is None:
            self._check_if_no_split_keyword_possible()
            split = list(self._partitioners.keys())[0]

        self._check_if_split_present(split)
        self._check_if_split_possible_to_federate(split)

        partitioner = self._partitioners[split]

        # This is the same assignment order used by Flower.
        self._assign_dataset_to_partitioner(split)

        # Let the partitioner itself validate the partition id and construct
        # the requested partition. Do not catch the exception here: swallowing
        # it makes the caller incorrectly continue with an undefined variable.
        partition = partitioner.load_partition(partition_id)

        if not self._event["load_partition"][split]:
            event(
                EventType.LOAD_PARTITION_CALLED,
                {
                    "federated_dataset_id": id(self),
                    "dataset_name": self._dataset_name,
                    "split": split,
                    "partitioner": partitioner.__class__.__name__,
                    "num_partitions": getattr(partitioner, "_num_partitions", None),
                },
            )
            self._event["load_partition"][split] = True

        return partition

    def load_split(self, split: str) -> Dataset:
        """Load a complete dataset split."""

        if not self._dataset_prepared:
            self._prepare_dataset()

        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")

        self._check_if_split_present(split)

        dataset_split = self._dataset[split]

        if split not in self._event.setdefault("load_split", {}):
            self._event["load_split"][split] = False

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
        """Return the configured partitioners with their datasets assigned."""

        if not self._dataset_prepared:
            self._prepare_dataset()

        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")

        for split in self._partitioners:
            self._check_if_split_present(split)
            self._assign_dataset_to_partitioner(split)

        return self._partitioners

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def _prepare_dataset(self) -> None:
        """Load, optionally duplicate, shuffle and preprocess the dataset."""

        # --------------------------------------------------------------
        # 1. Load the dataset
        # --------------------------------------------------------------
        if self._dataset_path is not None:
            self._dataset = datasets.load_from_disk(
                dataset_path=self._dataset_path
            )
        else:
            load_kwargs = dict(self._load_dataset_kwargs)

            # Flower's original API stores ``subset`` as a separate argument.
            # Forward it as HF's ``name`` only when the caller supplied it.
            if self._subset is not None and "name" not in load_kwargs:
                load_kwargs["name"] = self._subset

            self._dataset = datasets.load_dataset(
                self._dataset_id,
                **load_kwargs,
            )

        if not isinstance(self._dataset, DatasetDict):
            raise ValueError(
                "CustomFederatedDataset requires datasets.load_dataset/load_from_disk "
                "to return a DatasetDict. "
                f"Received: {type(self._dataset)}."
            )

        # --------------------------------------------------------------
        # 2. Duplicate TRAIN before partitioning
        # --------------------------------------------------------------
        self._duplicate_train_if_required()

        # --------------------------------------------------------------
        # 3. Keep Flower's original preparation order
        # --------------------------------------------------------------
        if self._shuffle:
            self._dataset = self._dataset.shuffle(seed=self._seed)

        if self._preprocessor:
            self._dataset = self._preprocessor(self._dataset)

        available_splits = list(self._dataset.keys())

        self._event["load_split"] = {
            split: False for split in available_splits
        }

        self._dataset_prepared = True

    def _duplicate_train_if_required(self) -> None:
        if self._dataset is None or "train" not in self._dataset:
            return

        # Do not call partitioner.num_partitions here. In Flower's
        # DirichletPartitioner that property validates against
        # partitioner.dataset, which has not been assigned yet.
        configured_num_partitions = []
        for partitioner in self._partitioners.values():
            value = getattr(partitioner, "_num_partitions", None)
            if value is not None:
                configured_num_partitions.append(int(value))

        num_partitions = max(configured_num_partitions, default=0)
        factor = self._get_duplication_factor()

        if factor <= 1 or num_partitions < self._duplication_client_threshold:
            return

        original_train = self._dataset["train"]
        original_size = len(original_train)

        # Duplicate before assigning the dataset to the partitioner.
        self._dataset["train"] = datasets.concatenate_datasets(
            [original_train] * factor
        )

        print(
            f"Expanded dataset {self._dataset_name}: train size "
            f"{original_size} -> {len(self._dataset['train'])} "
            f"({factor}x) for {num_partitions} clients."
        )

    def _get_duplication_factor(self) -> int:
        """Resolve the configured factor robustly.

        The calling code may use either:
          - the experiment name (e.g. ``WISDM-W``), or
          - the complete HF repository name
            (e.g. ``claudiogsc/WISDM-W``).

        Both forms are accepted.
        """

        if not self._duplication_factors:
            return 1

        candidates = [
            self._dataset_name,
            self._dataset_name.strip(),
            self._dataset_name.upper(),
            self._dataset_id,
            self._dataset_id.strip(),
        ]

        # Also accept the repository basename:
        # ``claudiogsc/WISDM-W`` -> ``WISDM-W``.
        if "/" in self._dataset_id:
            candidates.append(self._dataset_id.rsplit("/", 1)[-1])

        # And compare case-insensitively.
        normalized = {
            str(key).strip().casefold(): value
            for key, value in self._duplication_factors.items()
        }

        for candidate in candidates:
            key = str(candidate).strip().casefold()
            if key in normalized:
                return normalized[key]

        return 1

    # ------------------------------------------------------------------
    # Flower-compatible validation/assignment
    # ------------------------------------------------------------------

    def _check_if_split_present(self, split: str) -> None:
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")

        available_splits = list(self._dataset.keys())

        if split not in available_splits:
            raise ValueError(
                f"The given split: '{split}' is not present in the dataset's "
                f"splits: '{available_splits}'."
            )

    def _check_if_split_possible_to_federate(self, split: str) -> None:
        if split not in self._partitioners:
            raise ValueError(
                f"The given split: '{split}' does not have a partitioner to "
                f"perform partitioning. Partitioners were specified for: "
                f"'{list(self._partitioners.keys())}'."
            )

    def _assign_dataset_to_partitioner(self, split: str) -> None:
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")

        partitioner = self._partitioners[split]

        if not partitioner.is_dataset_assigned():
            partitioner.dataset = self._dataset[split]

    def _check_if_no_split_keyword_possible(self) -> None:
        if len(self._partitioners) != 1:
            raise ValueError(
                "Please set the `split` argument. You can only omit the split "
                "keyword if there is exactly one partitioner specified."
            )

    def _check_partitioners_correctness(self) -> None:
        """Ensure that the same partitioner object is not reused."""

        partitioners_keys = list(self._partitioners.keys())

        for i, first_split in enumerate(partitioners_keys):
            for j in range(i + 1, len(partitioners_keys)):
                second_split = partitioners_keys[j]

                if (
                    self._partitioners[first_split]
                    is self._partitioners[second_split]
                ):
                    raise ValueError(
                        "The same partitioner object is used for multiple "
                        f"splits: ('{first_split}', '{second_split}'). "
                        "Each partition should have a separate partitioner."
                    )