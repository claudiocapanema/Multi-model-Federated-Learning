import sys
import os

import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

fds = {}  # Cache FederatedDataset

def download_datasets(datasets_name: list, alphas: list, num_partitions: int):
    try:
        # Only initialize `FederatedDataset` once
        logger.info(
            """Loading {} {} {} data.""".format(datasets_name, num_partitions, alphas))
        pasta = './datasets/'
        # for arquivo in os.listdir(pasta):
        #     caminho_completo = os.path.join(pasta, arquivo)
        #     if os.path.isfile(caminho_completo):
        #         os.remove(caminho_completo)
        #         logger.info("Removed {}".format(caminho_completo))
        global fds
        for i in range(len(datasets_name)):
            dataset_name = datasets_name[i]
            filename = f"datasets/{dataset_name}"
            if dataset_name not in fds and not os.path.isdir(filename):
                logger.info("Downloading {}".format(dataset_name))
                dataset = load_dataset({"EMNIST": "claudiogsc/emnist_balanced", "CIFAR10": "uoft-cs/cifar10", "MNIST": "ylecun/mnist",
                             "GTSRB": "claudiogsc/GTSRB", "Gowalla": "claudiogsc/Gowalla-State-of-Texas",
                             "WISDM-W": "claudiogsc/WISDM-W", "ImageNet": "claudiogsc/ImageNet-15_household_objects"}[
                        dataset_name])
                dataset.save_to_disk(filename)
            else:
                logger.info("Found {}".format(dataset_name))

    except Exception as e:
        logger.error("load_data error")
        logger.error("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))