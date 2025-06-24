# Multi-model Federated Learning

This project extends [PFLlib (Personalized Federated Learning Algorithm Library)](https://github.com/TsingZ0/PFLlib) to train multiple models simultaneously.

[![License: GPL v2](https://img.shields.io/badge/License-GPL_v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html) [![arXiv](https://img.shields.io/badge/arXiv-2312.04992-b31b1b.svg)](https://arxiv.org/abs/2312.04992)

## MEFL example

You can execute a simulation with only one command line. The command below executes an experiment with three models.
```bash

python main.py --total_clients=30 --number_of_rounds=100 --strategy="MultiFedAvg" --dataset="WISDM-W" --dataset="ImageNet10"  --dataset="Gowalla" --model="gru" --model="CNN" --model="lstm" --fraction_fit=0.3 --alpha=0.1 --alpha=0.1 --alpha=1.0 --experiment_id=2

```

## Available strategies

The available strategies are: MultifedAvg, MultiFedAvgRR, FedFairMMFL, and MultiFedAvg-MDH.

### Citing

If FedPredict has been useful to you, please cite our papers.

[FedPredict: Combining Global and Local Parameters in the Prediction Step of Federated Learning](https://ieeexplore.ieee.org/abstract/document/10257293) (original paper):

```
@INPROCEEDINGS{capanema2023fedpredict,
  author={Capanema, Cláudio G. S. and de Souza, Allan M. and Silva, Fabrício A. and Villas, Leandro A. and Loureiro, Antonio A. F.},
  booktitle={2023 19th International Conference on Distributed Computing in Smart Systems and the Internet of Things (DCOSS-IoT)}, 
  title={FedPredict: Combining Global and Local Parameters in the Prediction Step of Federated Learning}, 
  year={2023},
  volume={},
  number={},
  pages={17-24},
  keywords={Federated learning;Computational modeling;Neural networks;Mathematical models;Internet of Things;Distributed computing;Personalized Federated Learning;Neural Networks;Federated Learning Plugin},
  doi={10.1109/DCOSS-IoT58021.2023.00012}}
```
[A Novel Prediction Technique for Federated Learning](https://ieeexplore.ieee.org/abstract/document/10713874) (extended journal paper):
```
@ARTICLE{capanema2025@novel,
  author={Capanema, Cláudio G. S. and de Souza, Allan M. and da Costa, Joahannes B. D. and Silva, Fabrício A. and Villas, Leandro A. and Loureiro, Antonio A. F.},
  journal={IEEE Transactions on Emerging Topics in Computing}, 
  title={A Novel Prediction Technique for Federated Learning}, 
  year={2025},
  volume={13},
  number={1},
  pages={5-21},
  keywords={Servers;Costs;Training;Downlink;Adaptation models;Computational modeling;Federated learning;Quantization (signal);Context modeling;Accuracy;Federated learning plugin;neural networks;personalized federated learning},
  doi={10.1109/TETC.2024.3471458}}
```

[A Modular Plugin for Concept Drift in Federated Learning](https://ieeexplore.ieee.org/abstract/document/10621488) (FedPredict-Dynamic):
```
@INPROCEEDINGS{capanema2024@modular,
  author={Capanema, Cláudio G. S. and Da Costa, Joahannes B. D. and Silva, Fabrício A. and Villas, Leandro A. and Loureiro, Antonio A. F.},
  booktitle={2024 20th International Conference on Distributed Computing in Smart Systems and the Internet of Things (DCOSS-IoT)}, 
  title={A Modular Plugin for Concept Drift in Federated Learning}, 
  year={2024},
  volume={},
  number={},
  pages={101-108},
  keywords={Training;Accuracy;Federated learning;Geology;Concept drift;Data models;Internet of Things;Concept Drift;Personalized Federated Learning;Federated Learning Plugin;Neural Networks},
  doi={10.1109/DCOSS-IoT61029.2024.00024}}
```