import sys
import os

import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

fds = {}  # Cache FederatedDataset

def download_datasets(datasets_name: list, alphas: list, num_partitions: int):
    try:
        logger.info(
            """Loading {} {} {} data.""".format(
                datasets_name,
                num_partitions,
                alphas
            )
        )

        global fds

        dataset_paths = {
            "EMNIST": "claudiogsc/emnist_balanced",
            "CIFAR10": "uoft-cs/cifar10",
            "MNIST": "ylecun/mnist",
            "F-MNIST": "zalando-datasets/fashion_mnist",
            "GTSRB": "claudiogsc/GTSRB",
            "Gowalla": "claudiogsc/Gowalla-State-of-Texas-Window-4-overlap-0.5",
            "WISDM-W": "claudiogsc/WISDM-W",
            "ImageNet": "claudiogsc/ImageNet-15_household_objects",
            "ImageNet10": "claudiogsc/ImageNet-10_household_objects",
            "wikitext": "claudiogsc/wikitext-Window-1-Words-3743",
            "Foursquare": "claudiogsc/foursquare-us-sequences-highlevel-40000-samples-10-seq-len-8-classes"
        }

        for i in range(len(datasets_name)):

            dataset_name = datasets_name[i]
            filename = f"datasets/{dataset_name}"

            if dataset_name not in fds and not os.path.isdir(filename):

                logger.info("Downloading {}".format(dataset_name))

                # SVHN requires an explicit configuration
                if dataset_name == "SVHN":
                    dataset = load_dataset(
                        "ufldl-stanford/svhn",
                        "cropped_digits"
                    )
                else:
                    dataset = load_dataset(
                        dataset_paths[dataset_name]
                    )

                if dataset_name in ["Gowalla"]:
                    dataset["train"] = (
                        dataset["train"]
                        .shuffle(seed=42)
                        .select(range(120000))
                    )

                    dataset["test"] = (
                        dataset["test"]
                        .shuffle(seed=42)
                        .select(range(30000))
                    )

                elif dataset_name in ["wikitext"]:
                    dataset["train"] = (
                        dataset["train"]
                        .shuffle(seed=42)
                        .select(range(480000))
                    )

                    dataset["test"] = (
                        dataset["test"]
                        .shuffle(seed=42)
                        .select(range(12000))
                    )

                dataset.save_to_disk(filename)

            else:
                logger.info("Found {}".format(dataset_name))

    except Exception as e:
        print("load_data aa error")
        print(
            """Error on line {} {} {}""".format(
                sys.exc_info()[-1].tb_lineno,
                type(e).__name__,
                e
            )
        )