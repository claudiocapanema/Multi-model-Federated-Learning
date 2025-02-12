from collections import defaultdict
from typing import List

import random
import numpy as np
from torch.utils.data import Dataset

from dataset.partition.utils import IndexedSubset

np.random.seed(0)
random.seed(0)


class DirichletPartition:
    def __init__(
            self,
            num_clients: int,
            alpha: float,
            num_class: int = 10,
            minimum_data_size: int = 20,
            max_iter=10000
    ):
        self.num_clients = num_clients
        self.alpha = alpha
        self.num_class = num_class
        self.minimum_data_size = minimum_data_size
        self.max_iter = max_iter
        self.distributions = defaultdict(lambda: np.random.dirichlet(np.repeat(self.alpha, self.num_clients)))
        self.partitions = None

    def __call__(self, dataset):

        it = 0
        if not isinstance(dataset.targets, np.ndarray):
            dataset.targets = np.array(
                dataset.targets, dtype=np.int64
            )
        net_dataidx_map = {}
        min_size = 0
        idx_batch = [[] for _ in range(self.num_clients)]
        while min_size < self.minimum_data_size and it < self.max_iter:
            it += 1
            idx_batch = [[] for _ in range(self.num_clients)]
            # for each class in the dataset
            for k in range(self.num_class):
                # print("""\n==== iteracao {}""".format(k))
                idx_k = np.where(dataset.targets == k)[0]
                # np.random.shuffle(idx_k)
                proportions = self.distributions[k]
                # print("inicial: ", k, " proportions: ", proportions)
                ## Balance
                proportions = np.array(
                    [
                        p * (len(idx_j) < len(dataset.targets) / self.num_clients)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )

                # print("meio: ", k, " proportions: ", proportions)
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # print("meio 2: ", k, " proportions: ", proportions)
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                # print("""//// idx_k {} ////""".format(k, len(idx_k)))
                # print("proporcao: ", proportions)
                for j in range(self.num_clients):
                    # print("""=cliente {}""".format(j))
                    # np.random.shuffle(idx_batch[j])
                    net_dataidx_map[j] = idx_batch[j]
                    # if len(net_dataidx_map[j]) > 0:
                    #     print("Quantidade cliente: ", j, len(net_dataidx_map[j]),
                    #           np.unique(dataset.targets[np.array(net_dataidx_map[j])], return_counts=True))
            min_size = min([len(idx_j) for idx_j in idx_batch])

        # Redistribution loop
        it = 0
        while min_size < self.minimum_data_size and it < self.max_iter:
            # Find client with minimum and maximum samples
            min_samples_client = min(idx_batch, key=len)
            max_samples_client = max(idx_batch, key=len)
            # Get count of samples needed to reach minimum_data_size
            transfer_samples_count = self.minimum_data_size - len(min_samples_client)
            # Transfer samples from max_samples_client to min_samples_client
            min_samples_client.extend(max_samples_client[-transfer_samples_count:])
            del max_samples_client[-transfer_samples_count:]
            # Recalculate min_size
            min_size = min([len(idx_j) for idx_j in idx_batch])
            it += 1

        for j in range(self.num_clients):
            # np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
            print("Quantidade cliente: ", j, len(net_dataidx_map[j]),
                  np.unique(dataset.targets[np.array(net_dataidx_map[j])], return_counts=True))
        dataset_ref = dataset
        return [
            IndexedSubset(
                dataset_ref,
                indices=net_dataidx_map[i],
            )
            for i in range(self.num_clients)
        ]
