import random
import json
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomResizedCrop, RandomAffine, ColorJitter,  Normalize, ToTensor, RandomRotation, Lambda, Grayscale, GaussianBlur, RandomInvert, AutoAugment, AutoAugmentPolicy, RandomAutocontrast, RandomGrayscale, ElasticTransform, RandomCrop, RandAugment
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import numpy as np
import sys
from .models import CNN, CNN_3, CNNDistillation, GRU, LSTM, TinyImageNetCNN
import  datasets as dt
from .custom_federated_dataset import CustomFederatedDataset
from torch.utils.data import Dataset
from collections import Counter


import logging


logging.basicConfig(level=logging.INFO)  # Configure logging
logger = logging.getLogger(__name__)  # Create logger for the module

def fedpredict_client_weight_predictions_torch(output: torch.Tensor, t: int, current_proportion: np.array, similarity: float) -> np.array:
    """
        This function gives more weight to the predominant classes in the current dataset. This function is part of
        FedPredict-Dynamic
    Args:
        output: torch.Tensor, required
            The output of the model after applying 'softmax' activation function.
        t: int, required
            The current round.
        current_proportion:  np.array, required
            The classes proportion in the current training data.
        similarity: float, required
            The similarity between the old data (i.e., the one that the local model was previously trained on) and the new
        data. Note that s \in [0, 1].

    Returns:
        np.array containing the weighted predictions

    """

    try:
        _has_torch = True
        if similarity != 1:
            if _has_torch:
                output = torch.multiply(output, torch.from_numpy(current_proportion * (1 - similarity)).to(output.device))
            else:
                raise ValueError("Framework 'torch' not found")

        return output

    except Exception as e:
        print("FedPredict client weight prediction")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

DATASET_INPUT_MAP = {"CIFAR10": "img", "MNIST": "image", "EMNIST": "image", "GTSRB": "image", "Gowalla": "sequence",
                     "WISDM-W": "sequence", "ImageNet": "image", "ImageNet10": "image", "wikitext": "sequence"}

def load_model(model_name, dataset, strategy, device):
    try:
        num_classes = {'EMNIST': 47, 'MNIST': 10, 'CIFAR10': 10, 'GTSRB': 43, 'WISDM-W': 12, 'WISDM-P': 12, 'Tiny-ImageNet': 200,
         'ImageNet100': 15, 'ImageNet': 15, "ImageNet10": 10, "ImageNet_v2": 15, "Gowalla": 7, "wikitext": 30}[dataset]
        if model_name == 'CNN':
            if dataset in ['MNIST']:
                input_shape = 1
                mid_dim = 256*4
                logger.info("""leu mnist com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset in ['EMNIST']:
                input_shape = 1
                mid_dim = 256*4
                logger.info("""leu emnist com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset in ['GTSRB']:
                input_shape = 3
                mid_dim = 36*4
                logger.info("""leu gtsrb com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset == "ImageNet":
                input_shape=3
                mid_dim=1600
            elif dataset == "ImageNet10":
                input_shape=3
                mid_dim=1600
            elif dataset == "CIFAR10":
                input_shape = 3
                mid_dim = 400*4
                logger.info("""leu cifar com {} {} {}""".format(input_shape, mid_dim, num_classes))
            return CNN(input_shape=input_shape, num_classes=num_classes, mid_dim=mid_dim)
        elif model_name == 'CNN_3':
            if dataset in ['MNIST']:
                input_shape = 1
                mid_dim = 4
                logger.info("""leu mnist com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset in ['EMNIST']:
                input_shape = 1
                mid_dim = 4
                logger.info("""leu emnist com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset in ['GTSRB']:
                input_shape = 3
                mid_dim = 16
                logger.info("""leu gtsrb com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset == "ImageNet":
                input_shape = 3
                mid_dim = 16
                logger.info("""leu imagenet com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset == "ImageNet10":
                input_shape = 3
                mid_dim = 16
                return TinyImageNetCNN()
                logger.info("""leu imagenet10 com {} {} {}""".format(input_shape, mid_dim, num_classes))
            elif dataset == "CIFAR10":
                input_shape = 3
                mid_dim = 16
                logger.info("""leu cifar com {} {} {}""".format(input_shape, mid_dim, num_classes))

            if "FedKD" in strategy:
                return CNNDistillation(input_shape=input_shape, mid_dim=mid_dim, num_classes=num_classes, dataset=dataset)
            else:
                return CNN_3(input_shape=input_shape, num_classes=num_classes, mid_dim=mid_dim)

        elif model_name == "gru":
            if dataset in ["WISDM-W", "WISDM-P"]:
                return GRU(6, num_layers=1, hidden_size=2, sequence_length=200, num_classes=num_classes)

        elif model_name == "lstm":
            if dataset in ["Gowalla"]:
                return LSTM(4, device=device, num_layers=1, hidden_size=1, sequence_length=10, num_classes=num_classes)

        raise ValueError("""Model not found for model {} and dataset {}""".format(model_name, dataset))

    except Exception as e:
        print("""load_model error""")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


fds = None

def get_weights(net):
    try:
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    except Exception as e:
        print("get_weights error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def get_weights_fedkd(net):
    try:
        return [val.cpu().numpy() for _, val in net.student.state_dict().items()]
    except Exception as e:
        print("get_weights_fedkd error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def set_weights(net, parameters):
    try:
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        # for new_param, old_param in zip(parameters, net.parameters()):
        #     old_param.data = new_param.data.clone()
    except Exception as e:
        print("set_weights error")
        print(f"tipo {type(net)}")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def set_weights_fedkd(net, parameters):
    try:
        params_dict = zip(net.student.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.student.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print("set_weights_fedkd error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def wikitext_preprocess(dataset):
    def tokenize(text_lines):
        text = " <eos> ".join(text_lines).lower()
        tokens = text.split()
        return tokens

    full_train_tokens = tokenize(dataset['train']['text'])
    valid_tokens_raw = tokenize(dataset['validation']['text'])
    test_tokens_raw = tokenize(dataset['test']['text'])

    # 2. Construir vocabulário com as 100 palavras mais frequentes
    counter = Counter(full_train_tokens)
    most_common_words = [word for word, _ in counter.most_common(100)]
    vocab = {word: idx for idx, word in enumerate(most_common_words)}
    vocab_size = len(vocab)
    print(f"Vocabulário: {vocab_size} palavras (as 100 mais frequentes)")

    # 3. Filtrar os tokens que estão no vocabulário
    def filter_tokens(tokens, vocab):
        filtered = [vocab[t] for t in tokens if t in vocab]
        return filtered

    train_tokens = filter_tokens(full_train_tokens, vocab)
    valid_tokens = filter_tokens(valid_tokens_raw, vocab)
    test_tokens = filter_tokens(test_tokens_raw, vocab)

    # 4. Dataset para prever a próxima palavra
    class NextWordDataset(Dataset):
        def __init__(self, data, seq_len):
            self.data = data
            self.seq_len = seq_len

        def __len__(self):
            return len(self.data) - self.seq_len

        def __getitem__(self, idx):
            x = self.data[idx:idx + self.seq_len]
            y = self.data[idx + self.seq_len]
            return {"text": torch.tensor(x, dtype=torch.long), "label": torch.tensor(y, dtype=torch.long)}

    seq_len = 1
    batch_size = 256

    train_dataset = NextWordDataset(train_tokens, seq_len)
    valid_dataset = NextWordDataset(valid_tokens, seq_len)
    test_dataset = NextWordDataset(test_tokens, seq_len)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataset, test_dataset


fds = {}  # Cache FederatedDataset

def get_transform(dataset_name, train_test):
    pytorch_transforms = {"CIFAR10": {"train":
                                          Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                      "test": Compose(
                                          [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])},
                          "MNIST": Compose([ToTensor(), RandomRotation(10),
                                            Normalize([0.5], [0.5])]),
                          "EMNIST": Compose([ToTensor(), RandomRotation(10),
                                             Normalize([0.5], [0.5])]),
                          "GTSRB": Compose(
                              [

                                  Resize((32, 32)),
                                  RandomHorizontalFlip(),  # FLips the image w.r.t horizontal axis
                                  RandomRotation(10),  # Rotates the image to a specified angel
                                  RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                                  # Performs actions like zooms, change shear angles.
                                  ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                                  ToTensor(),
                                  Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
                              ]
                          ),
                          "ImageNet": Compose(
                              [

                                  Resize(32),
                                  RandomHorizontalFlip(),
                                  ToTensor(),
                                  Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
                                  # transforms.Resize((32, 32)),
                                  # transforms.ToTensor(),
                                  # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              ]
                          ),
                          "ImageNet10": {"train":
                              Compose(
                                  [

                                      Resize(32),
                                      ToTensor(),
                                      Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                                      # Resize((32, 32)),
                                      # ToTensor(),
                                      # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                  ]
                              ),
                              "test":
                                  Compose(
                                      [
                                          Resize(32),
                                          ToTensor(),
                                          Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                                          # Resize((32, 32)),
                                          # ToTensor(),
                                          # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                      ]
                                  )
                          }
                          # Compose([AutoAugment(policy=AutoAugmentPolicy.CIFAR10), Resize(32), ToTensor(),
                          #             Normalize(mean=[0.485, 0.456, 0.406],
                          #                          std=[0.229, 0.224, 0.225])])
        ,
                          "WISDM-W": {"train": Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.float32))),
                                      "test": Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.float32)))},
                          "Gowalla": {"train": Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.float32))),
                                      "test": Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.float32)))},
                          "wikitext": {"train": Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.float32))),
                                      "test": Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.float32)))}

                          }[dataset_name][train_test]

    return pytorch_transforms

def load_data(dataset_name: str, alpha: float, partition_id: int, num_partitions: int, batch_size: int,
              data_sampling_percentage: int, get_from_volume: bool = True, k_fold: int = 5, fold_id: int = 1):
    try:
        # Only initialize `FederatedDataset` once
        print(
            """Loading {} {} {} {} {} {} data.""".format(dataset_name, partition_id, num_partitions, batch_size, data_sampling_percentage, alpha))
        global fds
        if not get_from_volume:

            if dataset_name not in fds:
                partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label",

                                                   alpha=alpha, min_partition_size=10, seed=1,

                                                   self_balancing=True)
                fds[dataset_name] = CustomFederatedDataset(
                    dataset={"EMNIST": "claudiogsc/emnist_balanced", "CIFAR10": "uoft-cs/cifar10", "MNIST": "ylecun/mnist",
                         "GTSRB": "claudiogsc/GTSRB", "Gowalla": "claudiogsc/Gowalla-State-of-Texas-Window-4-overlap-0.5",
                         "WISDM-W": "claudiogsc/WISDM-W", "ImageNet": "claudiogsc/ImageNet-15_household_objects"
                         , "ImageNet10": "claudiogsc/ImageNet-10_household_objects", 'wikitext': 'claudiogsc/wikitext-Window-1-Words-3743'}[dataset_name],
                    partitioners={"train": partitioner},
                    seed=1
                )
        else:
            # dts = dt.load_from_disk(f"datasets/{dataset_name}")
            partitioner = DirichletPartitioner(num_partitions=num_partitions, partition_by="label",
                                               alpha=alpha, min_partition_size=10, seed=1,
                                               self_balancing=True)
            print("dataset from volume")
            fd = CustomFederatedDataset(
                dataset={"EMNIST": "claudiogsc/emnist_balanced", "CIFAR10": "uoft-cs/cifar10", "MNIST": "ylecun/mnist",
                         "GTSRB": "claudiogsc/GTSRB", "Gowalla": "claudiogsc/Gowalla-State-of-Texas-Window-4-overlap-0.5",
                         "WISDM-W": "claudiogsc/WISDM-W", "ImageNet": "claudiogsc/ImageNet-15_household_objects"
                         , "ImageNet10": "claudiogsc/ImageNet-10_household_objects", 'wikitext': 'claudiogsc/wikitext-Window-1-Words-3743'}[
                    dataset_name],
                partitioners={"train": partitioner},
                path=f"datasets/{dataset_name}",
                seed=1
            )
            fds[dataset_name] = fd
            print("passou dataset")
        attempts = 0
        while True:
            attempts += 1
            try:
                time.sleep(random.randint(1, 1))
                partition = fds[dataset_name].load_partition(partition_id)
                logger.info("""Loaded dataset {} in the {} attempt for client {}""".format(dataset_name, attempts, partition_id))
                break
            except Exception as e:
                logger.info("""Tried to load dataset {} for the {} time for the client {} error""".format(dataset_name, attempts, partition_id))
                logger.info("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
                time.sleep(1)
        test_size = 1 - data_sampling_percentage
        partition_train_test = partition.train_test_split(test_size=test_size, seed=1)
        # ==============================
        # K-FOLD LOCAL (intra-client)
        # ==============================
        # num_samples = len(partition)
        # indices = np.arange(num_samples)
        #
        # rng = np.random.default_rng(seed=k_fold)
        # rng.shuffle(indices)
        #
        # folds = np.array_split(indices, k_fold)
        #
        # val_idx = folds[fold_id-1]
        # train_idx = np.concatenate([f for i, f in enumerate(folds) if i != fold_id])
        #
        # partition_train = partition.select(train_idx)
        # partition_test = partition.select(val_idx)

        if dataset_name in ["CIFAR10", "MNIST", "EMNIST", "GTSRB", "ImageNet", "ImageNet10", "WISDM-W", "Gowalla", "wikitext"]:
            # Divide data on each node: 80% train, 20% test

            pytorch_transforms_train = get_transform(dataset_name, "train")
            pytorch_transforms_test = get_transform(dataset_name, "test")

        # import torchvision.datasets as datasets
        # datasets.EMNIST
        key = DATASET_INPUT_MAP[dataset_name]

        def apply_transforms_train(batch):
            """Apply transforms to the partition from FederatedDataset."""

            batch[key] = [pytorch_transforms_train(img) for img in batch[key]]
            # logger.info("""bath key: {}""".format(batch[key]))
            return batch

        def apply_transforms_test(batch):
            """Apply transforms to the partition from FederatedDataset."""

            batch[key] = [pytorch_transforms_test(img) for img in batch[key]]
            # logger.info("""bath key: {}""".format(batch[key]))
            return batch

        if dataset_name in ["CIFAR10", "MNIST", "EMNIST", "GTSRB", "ImageNet", "ImageNet10", "WISDM-W", "Gowalla", "wikitext"]:
            partition_train = partition_train_test["train"].with_transform(apply_transforms_train)
            partition_test = partition_train_test["test"].with_transform(apply_transforms_test)

            # partition_train = partition_train.with_transform(apply_transforms_train)
            # partition_test = partition_test.with_transform(apply_transforms_test)

        GLOBAL_TORCH_GENERATOR = torch.Generator()
        GLOBAL_TORCH_GENERATOR.manual_seed(fold_id)

        random.seed(fold_id)
        np.random.seed(fold_id)
        torch.manual_seed(fold_id)
        torch.cuda.manual_seed_all(fold_id)

        trainloader = DataLoader(
            partition_train, batch_size=batch_size, shuffle=True, generator=GLOBAL_TORCH_GENERATOR, num_workers=0
        )
        testloader = DataLoader(partition_test, batch_size=batch_size, generator=GLOBAL_TORCH_GENERATOR, num_workers=0)
        return trainloader, testloader

    except Exception as e:
        print("load_data error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def train(model, trainloader, valloader, optimizer, epochs, learning_rate, device, client_id, t, dataset_name, n_classes, concept_drift_window=0):
    try:
        """Train the utils on the training set."""
        model.to(device)  # move utils to GPU if available
        criterion = torch.nn.CrossEntropyLoss().to(device)
        model.train()
        key = DATASET_INPUT_MAP[dataset_name]
        for _ in range(epochs):
            loss_total = 0
            correct = 0
            y_true = []
            y_prob = []
            for batch in trainloader:
                # logger.info("""dentro {} labels {}""".format(images, labels))
                x = batch[key]
                labels = batch["label"]
                # logger.info("""tamanho images {} tamanho labels {}""".format(images.shape, labels.shape))
                x = x.to(device)
                labels = labels.to(device)
                if concept_drift_window > 0:
                    labels = (labels + concept_drift_window) % n_classes

                optimizer.zero_grad()
                outputs = model(x)
                # print("""saida: {} true: {}""".format(outputs, labels))
                loss = criterion(outputs, labels)
                loss.backward()
                loss_total += loss.item() * labels.shape[0]
                y_true.append(label_binarize(labels.detach().cpu().numpy().tolist(), classes=np.arange(n_classes)))
                y_prob.append(outputs.detach().cpu().numpy().tolist())
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                optimizer.step()
        accuracy = correct / len(trainloader.dataset)
        loss = loss_total / len(trainloader.dataset)
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        y_prob = y_prob.argmax(axis=1)
        y_true = y_true.argmax(axis=1)
        balanced_accuracy = float(metrics.balanced_accuracy_score(y_true, y_prob))

        train_metrics = {"Train accuracy": accuracy, "Train balanced accuracy": balanced_accuracy, "Train loss": loss, "Train round (t)": t}
        # print(train_metrics)

        val_loss, test_metrics = test(model, valloader, device, client_id, t, dataset_name, n_classes)
        results = {
            "val_loss": val_loss,
            "val_accuracy": test_metrics["Accuracy"],
            "val_balanced_accuracy": test_metrics["Balanced accuracy"],
            "train_loss": train_metrics["Train loss"],
            "train_accuracy": train_metrics["Train accuracy"],
            "train_balanced_accuracy": train_metrics["Train balanced accuracy"],
            "Round (t)": t
        }
        return results

    except Exception as e:
        print("train error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def train_fedkd(model, trainloader, valloader, epochs, learning_rate, device, client_id, t, dataset_name, n_classes):
    """Train the utils on the training set."""
    try:
        model.to(device)  # move utils to GPU if available
        # utils.teacher.to(device)
        # utils.student.to(device)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        model.train()
        feature_dim = 512
        W_h = torch.nn.Linear(feature_dim, feature_dim, bias=False).to(device)
        MSE = torch.nn.MSELoss().to(device)
        key = DATASET_INPUT_MAP[dataset_name]
        logger.info("""Inicio train_fedkd client {}""".format(client_id))
        for _ in range(epochs):
            loss_total = 0
            correct = 0
            y_true = []
            y_prob = []
            for batch in trainloader:
                # logger.info("""dentro {} labels {}""".format(images, labels))
                x = batch[key]
                labels = batch["label"]
                x = x.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                output_student, rep_g, output_teacher, rep = model(x)
                outputs_S1 = F.log_softmax(output_student, dim=1)
                outputs_S2 = F.log_softmax(output_teacher, dim=1)
                outputs_T1 = F.softmax(output_student, dim=1)
                outputs_T2 = F.softmax(output_teacher, dim=1)

                loss_student = criterion(output_student, labels)
                loss_teacher = criterion(output_teacher, labels)
                loss_1 = torch.nn.KLDivLoss()(outputs_S1, outputs_T2) / (loss_student + loss_teacher)
                loss_2 = torch.nn.KLDivLoss()(outputs_S2, outputs_T1) / (loss_student + loss_teacher)
                L_h = MSE(rep, W_h(rep_g)) / (loss_student + loss_teacher)
                # loss = loss_student + loss_teacher
                loss = loss_teacher + loss_student + L_h + loss_1 + loss_2
                loss.backward()
                optimizer.step()
                loss_total += loss.item() * labels.shape[0]
                y_true.append(label_binarize(labels.detach().cpu().numpy(), classes=np.arange(n_classes)))
                y_prob.append(output_teacher.detach().cpu().numpy())
                correct += (torch.max(output_teacher.data, 1)[1] == labels).sum().item()

        accuracy = correct / len(trainloader.dataset)
        loss = loss_total / len(trainloader.dataset)
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        y_prob = y_prob.argmax(axis=1)
        y_true = y_true.argmax(axis=1)
        balanced_accuracy = float(metrics.balanced_accuracy_score(y_true, y_prob))

        train_metrics = {"Train accuracy": accuracy, "Train balanced accuracy": balanced_accuracy, "Train loss": loss, "Train round (t)": t}
        logger.info(train_metrics)

        val_loss, test_metrics = test_fedkd(model, valloader, device, client_id, t, dataset_name, n_classes)
        results = {
            "val_loss": val_loss,
            "val_accuracy": test_metrics["Accuracy"],
            "val_balanced_accuracy": test_metrics["Balanced accuracy"],
            "train_loss": train_metrics["Train loss"],
            "train_accuracy": train_metrics["Train accuracy"],
            "train_balanced_accuracy": train_metrics["Train balanced accuracy"]
        }
        return results

    except Exception as e:
        print("""Error on train_fedkd""")
        print('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def test(model, testloader, device, client_id, t, dataset_name, n_classes, concept_drift_window=0):
    try:
        """Validate the utils on the test set."""
        g = torch.Generator()
        g.manual_seed(t)
        torch.manual_seed(t)
        model.eval()
        model.to(device)  # move utils to GPU if available
        criterion = torch.nn.CrossEntropyLoss().to(device)
        correct, loss = 0, 0.0
        y_prob = []
        y_true = []
        key = DATASET_INPUT_MAP[dataset_name]
        with torch.no_grad():
            for batch in testloader:
                x = batch[key]
                labels = batch["label"]
                x = x.to(device)
                labels = labels.to(device)
                if concept_drift_window > 0:
                    labels = (labels + concept_drift_window) % n_classes
                y_true.append(label_binarize(labels.detach().cpu().numpy(), classes=np.arange(n_classes)))
                outputs = model(x)
                y_prob.append(outputs.detach().cpu().numpy())
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        loss = loss / len(testloader.dataset)
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        # test_auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        y_prob = y_prob.argmax(axis=1)
        y_true = y_true.argmax(axis=1)
        balanced_accuracy = float(metrics.balanced_accuracy_score(y_true, y_prob))

        test_metrics = {"Accuracy": accuracy, "Balanced accuracy": balanced_accuracy, "Loss": loss, "Round (t)": t}
        # logger.info("""metricas cliente {} valores {}""".format(client_id, test_metrics))
        return loss, test_metrics

    except Exception as e:
        print("test error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def test_fedkd(model, testloader, device, client_id, t, dataset_name, n_classes):
        try:
            model.to(device)  # move utils to GPU if available
            # utils.teacher.to(device)
            # utils.student.to(device)
            model.eval()
            criterion = torch.nn.CrossEntropyLoss().to(device)

            correct = 0
            loss = 0
            y_prob = []
            y_true = []

            key = DATASET_INPUT_MAP[dataset_name]
            with torch.no_grad():
                for batch in testloader:
                    x = batch[key]
                    labels = batch["label"]
                    x = x.to(device)
                    labels = labels.to(device)
                    y_true.append(label_binarize(labels.detach().cpu().numpy(), classes=np.arange(n_classes)))
                    output, proto_student, output_teacher, proto_teacher = model(x)
                    y_prob.append(output_teacher.detach().cpu().numpy())
                    loss += criterion(output_teacher, labels).item()
                    correct += (torch.sum(torch.argmax(output_teacher, dim=1) == labels)).item()

            accuracy = correct / len(testloader.dataset)
            loss = loss / len(testloader.dataset)
            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            # test_auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

            y_prob = y_prob.argmax(axis=1)
            y_true = y_true.argmax(axis=1)
            balanced_accuracy = float(metrics.balanced_accuracy_score(y_true, y_prob))

            test_metrics = {"Accuracy": accuracy, "Balanced accuracy": balanced_accuracy, "Loss": loss, "Round (t)": t}
            # logger.info("""metricas cliente {} valores {}""".format(client_id, test_metrics))
            return loss, test_metrics
        except Exception as e:
            print("Error test_fedkd")
            print('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))

def test_fedpredict(model, testloader, device, client_id, t, dataset_name, n_classes, s, p, concept_drift_window=0):
    try:
        """Validate the utils on the test set."""
        g = torch.Generator()
        g.manual_seed(t)
        torch.manual_seed(t)
        model.eval()
        model.to(device)  # move utils to GPU if available
        criterion = torch.nn.CrossEntropyLoss().to(device)
        correct, loss = 0, 0.0
        y_prob = []
        y_true = []
        key = DATASET_INPUT_MAP[dataset_name]
        with torch.no_grad():
            for batch in testloader:
                x = batch[key]
                labels = batch["label"]
                x = x.to(device)
                labels = labels.to(device)
                if concept_drift_window > 0:
                    labels = (labels + concept_drift_window) % n_classes
                y_true.append(label_binarize(labels.detach().cpu().numpy(), classes=np.arange(n_classes)))
                outputs = model(x).to(device)
                if round(s, 2) != 1:
                    print(f"similaridade {s}")
                    outputs = fedpredict_client_weight_predictions_torch(output=outputs, t=t,
                                                                        current_proportion=p,
                                                                        similarity=s)
                y_prob.append(outputs.detach().cpu().numpy())
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        loss = loss / len(testloader.dataset)
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        # test_auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        y_prob = y_prob.argmax(axis=1)
        y_true = y_true.argmax(axis=1)
        balanced_accuracy = float(metrics.balanced_accuracy_score(y_true, y_prob))

        test_metrics = {"Accuracy": accuracy, "Balanced accuracy": balanced_accuracy, "Loss": loss, "Round (t)": t}
        # logger.info("""metricas cliente {} valores {}""".format(client_id, test_metrics))
        return loss, test_metrics

    except Exception as e:
        print("teest fedpredict error")
        print("""Error on line {} {} {}""".format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))


def test_fedkd_fedpredict(lt, model, testloader, device, client_id, t, dataset_name, n_classes):
    try:
        model.to(device)  # move utils to GPU if available
        # utils.teacher.to(device)
        # utils.student.to(device)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss().to(device)

        correct = 0
        loss = 0
        y_prob = []
        y_true = []

        key = DATASET_INPUT_MAP[dataset_name]
        with torch.no_grad():
            for batch in testloader:
                x = batch[key]
                labels = batch["label"]
                x = x.to(device)
                labels = labels.to(device)
                y_true.append(label_binarize(labels.detach().cpu().numpy(), classes=np.arange(n_classes)))
                output, proto_student, output_teacher, proto_teacher = model(x)
                if lt == 0:
                    output_teacher = output
                y_prob.append(output_teacher.detach().cpu().numpy())
                loss += criterion(output_teacher, labels).item()
                correct += (torch.sum(torch.argmax(output_teacher, dim=1) == labels)).item()

        accuracy = correct / len(testloader.dataset)
        loss = loss / len(testloader.dataset)
        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        # test_auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        y_prob = y_prob.argmax(axis=1)
        y_true = y_true.argmax(axis=1)
        balanced_accuracy = float(metrics.balanced_accuracy_score(y_true, y_prob))

        test_metrics = {"Accuracy": accuracy, "Balanced accuracy": balanced_accuracy, "Loss": loss, "Round (t)": t}
        # logger.info("""metricas cliente {} valores {}""".format(client_id, test_metrics))
        return loss, test_metrics
    except Exception as e:
        print("Error test_fedkd_fedpredict")
        print('Error on line {} {} {}'.format(sys.exc_info()[-1].tb_lineno, type(e).__name__, e))
