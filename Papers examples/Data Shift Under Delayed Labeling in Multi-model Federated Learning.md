


## MEFL label shift example

```bash

python main.py --total_clients=30 --number_of_rounds=100 --strategy="MultiFedAvg" --dataset="WISDM-W" --dataset="ImageNet10"  --dataset="Gowalla" --model="gru" --model="CNN" --model="lstm" --fraction_fit=0.3 --alpha=0.1 --alpha=0.1 --alpha=0.1 --experiment_id="label_shift#1"

```



## MEFL cocnept_drift example

```bash

python main.py --total_clients=30 --number_of_rounds=100 --strategy="MultiFedAvg" --dataset="WISDM-W" --dataset="ImageNet10"  --dataset="Gowalla" --model="gru" --model="CNN" --model="lstm" --fraction_fit=0.3 --alpha=0.1 --alpha=0.1 --alpha=0.1 --experiment_id="concept_drift#1"

```