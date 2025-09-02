# Multi-model Federated Learning (MEFL)

This project extends [PFLlib (Personalized Federated Learning Algorithm Library)](https://github.com/TsingZ0/PFLlib) to train multiple models simultaneously. For simulations using the Flower framework, please consider visiting our second project that simulates MEFL using the Flower framework - [FL-HIAAC_docker](https://github.com/claudiocapanema/FL-HIAAC_docker).

[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html) [![arXiv](https://img.shields.io/badge/arXiv-2312.04992-b31b1b.svg)](https://arxiv.org/abs/2312.04992)

## Quick start

### FL example

To execute a single model training use the following example:
```bash

python main.py --total_clients=20 --number_of_rounds=100 --strategy="MultiFedAvg+MFP" --dataset="CIFAR10"  --model="CNN_3" --fraction_fit=0.3 --alpha=0.1 --experiment_id="2"

```

### MEFL example

You can execute a simulation with only one command line. The command below executes an experiment with three models.
```bash

python main.py --total_clients=30 --number_of_rounds=100 --strategy="MultiFedAvg" --dataset="WISDM-W" --dataset="ImageNet10"  --dataset="Gowalla" --model="gru" --model="CNN" --model="lstm" --fraction_fit=0.3 --alpha=0.1 --alpha=0.1 --alpha=1.0 --experiment_id="2"

```

### MEFL label shift example

```bash

python main.py --total_clients=20 --number_of_rounds=10 --strategy="MultiFedAvg" --dataset="WISDM-W" --dataset="ImageNet10"  --dataset="Gowalla" --model="gru" --model="CNN" --fraction_fit=0.3 --alpha=0.1 --alpha=0.1 --experiment_id="label_shift#1"

```

### MEFL concept_drift example

```bash

python main.py --total_clients=20 --number_of_rounds=10 --strategy="MultiFedAvg" --dataset="WISDM-W" --dataset="ImageNet10"  --dataset="Gowalla" --model="gru" --model="CNN" --fraction_fit=0.3 --alpha=0.1 --alpha=0.1 --experiment_id="concept_drift#1"

```

## Available strategies

The available strategies are: MultifedAvg, MultiFedAvgRR, FedFairMMFL, MultiFedAvg+MFP (MultiFedPredict plugin), MultiFedAvg+FPD (FedPredict-Dynamic plugin), and MultiFedAvg+FP (FedPredict plugin).

## Project scope

Currently, this project supports the development of MEFL solutions that explores data heterogeneity between models.  

## Citing

If this project has been useful to you, please cite our paper.

[Data Shift Under Delayed Labeling in Multi-Model Federated Learning](https://ieeexplore.ieee.org/document/11096152) (MultiFedPredict)

```text
@INPROCEEDINGS{capanema2025@data,
  author={Capanema, Cláudio G. S. and Silva, Fabrício A. and Villas, Leandro A. and Loureiro, Antonio A. F.},
  booktitle={2025 21st International Conference on Distributed Computing in Smart Systems and the Internet of Things (DCOSS-IoT)}, 
  title={Data Shift Under Delayed Labeling in Multi-Model Federated Learning}, 
  year={2025},
  volume={},
  number={},
  pages={570-577},
  keywords={Training;Global navigation satellite system;Costs;Federated learning;Smart systems;Data models;Labeling;Servers;Smart phones;Vehicles;Multi-model Federated Learning (MEFL);Data Shift;Delayed Labeling},
  doi={10.1109/DCOSS-IoT65416.2025.00092}}

```
