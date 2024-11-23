import pandas as pd
from pathlib import Path

rounds = 60

datasets = ["ImageNet", "ImageNet_v2"]


# Experiment 1
experiment = 1
alpha_initial = {"ImageNet": 0.1, "WISDM-W": 0.1, "ImageNet_v2": 0.1}
percent = {"ImageNet": 0.7, "WISDM-W": 0.7, "ImageNet_v2": 0.7}
rounds_alpha = {"ImageNet": int(rounds * percent["ImageNet"]), "WISDM-W": int(rounds * percent["WISDM-W"]), "ImageNet_v2": int(rounds * percent["ImageNet_v2"])}
alpha_new = {"ImageNet": 1.0, "WISDM-W": 1.0, "ImageNet_v2": 1.0}

# Experiment 2
# experiment = 2
# alpha_initial = {"ImageNet": 1.0, "WISDM-W": 1.0, "ImageNet_v2": 1.0}
# percent = {"ImageNet": 0.7, "WISDM-W": 0.7, "ImageNet_v2": 0.7}
# rounds_alpha = {"ImageNet": int(rounds * percent["ImageNet"]), "WISDM-W": int(rounds * percent["WISDM-W"]), "ImageNet_v2": int(rounds * percent["ImageNet_v2"])}
# alpha_new = {"ImageNet": 0.1, "WISDM-W": 0.1, "ImageNet_v2": 0.1}

round_list = []
dataset_list = []
alpha_list = []

for dataset in datasets:

    for i in range(1, rounds + 1):

        alpha = alpha_initial[dataset]

        if i > rounds_alpha[dataset]:
            alpha = alpha_new[dataset]

        round_list.append(i)
        dataset_list.append(dataset)
        alpha_list.append(alpha)

df = pd.DataFrame({"Dataset": dataset_list, "Round": round_list, "Alpha": alpha_list})
Path("""rounds_{}/datasets_{}/""".format(rounds, datasets)).mkdir(parents=True, exist_ok=True)
df.to_csv("""rounds_{}/datasets_{}/experiment_{}.csv""".format(rounds, datasets, experiment), index=False)