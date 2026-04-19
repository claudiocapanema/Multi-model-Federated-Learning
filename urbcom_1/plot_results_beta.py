import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# CONFIG
# =====================================================

LAMBDA_CAPACITY = 0.3
LAMBDA_INTRA = 0.3

FRAC = 0.3
ALPHA = 1.0

BETA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

PROPOSTA_NAME = "fair_resource_k"
DISPLAY_NAME = "DPFS"

COST_RATIO_STR = "4.0x"

RESULTS_DIR = f"results/gtsrb_{COST_RATIO_STR}_cifar"
RESULTS_DIR_WRITE = f"{RESULTS_DIR}/plots/alpha_{ALPHA}"

os.makedirs(RESULTS_DIR_WRITE, exist_ok=True)

FAIRNESS_METRICS = [
    "inter_client_fairness",
    "intra_client_fairness",
    "inter_model_fairness"
]


# =====================================================
# SAVE
# =====================================================

def save_figure(name):
    path = f"{RESULTS_DIR_WRITE}/{name}.png"
    plt.savefig(path, dpi=300)
    print(f"✅ {path}")


# =====================================================
# LOAD
# =====================================================

def load_proposta_results():

    dfs = []

    for beta in BETA_VALUES:

        for dataset in ["cifar", "gtsrb"]:

            file = (
                f"{RESULTS_DIR}/proposta_k_{dataset}"
                f"_frac_{FRAC}"
                f"_alpha_{ALPHA}"
                f"_alphaEff_{beta}"
                f"_lambdaCap_{LAMBDA_CAPACITY}"
                f"_lambdaIntra_{LAMBDA_INTRA}.csv"
            )

            if not os.path.exists(file):
                print(f"⚠️ {file}")
                continue

            df = pd.read_csv(file)
            df["beta"] = beta

            dfs.append(df)

    if not dfs:
        raise ValueError("Nenhum CSV encontrado.")

    return pd.concat(dfs, ignore_index=True)


# =====================================================
# PLOT ÚNICO (🔥 NOVO)
# =====================================================

def plot_beta_analysis():

    df = load_proposta_results()

    markers = ['o', 's', '^', 'D', 'P', 'X']

    for metric in FAIRNESS_METRICS:

        plt.figure(figsize=(6, 5))

        placed_labels = []
        points = []

        for i, beta in enumerate(BETA_VALUES):

            df_beta = df[df["beta"] == beta]
            subset = df_beta[df_beta["algorithm"] == PROPOSTA_NAME]

            if subset.empty:
                continue

            final = (
                subset
                .sort_values("round")
                .groupby(["dataset"])
                .tail(1)
            )

            fairness_val = final[metric].mean() * 100

            cifar = final[final["dataset"] == "cifar"]["global_acc"].mean()
            gtsrb = final[final["dataset"] == "gtsrb"]["global_acc"].mean()

            if np.isnan(cifar) or np.isnan(gtsrb):
                continue

            acc_mean = ((cifar + gtsrb) / 2) * 100

            # 🔥 ponto por beta
            plt.scatter(
                fairness_val,
                acc_mean,
                marker=markers[i % len(markers)],
                s=100,
                label=rf"$\beta={beta}$"
            )

            points.append((fairness_val, acc_mean))

            # 🔥 score
            f = fairness_val / 100
            a = acc_mean / 100
            score = 2 * f * a / (f + a) if (f + a) > 0 else 0

            # 🔥 label inteligente
            directions = [(4,4), (-4,4), (4,-4), (-4,-4), (6,0), (0,6)]

            for dx, dy in directions:
                new_x = fairness_val + dx
                new_y = acc_mean + dy

                if not any(abs(new_x - px) < 4 and abs(new_y - py) < 4 for px, py in points):
                    plt.text(new_x, new_y, f"{score:.2f}", fontsize=9)
                    break

        # =====================================================
        # 🔥 CONECTA OS PONTOS (MUITO IMPORTANTE)
        # =====================================================
        sorted_points = []

        for beta in BETA_VALUES:
            df_beta = df[df["beta"] == beta]
            subset = df_beta[df_beta["algorithm"] == PROPOSTA_NAME]

            if subset.empty:
                continue

            final = (
                subset
                .sort_values("round")
                .groupby(["dataset"])
                .tail(1)
            )

            fairness_val = final[metric].mean() * 100
            cifar = final[final["dataset"] == "cifar"]["global_acc"].mean()
            gtsrb = final[final["dataset"] == "gtsrb"]["global_acc"].mean()

            if np.isnan(cifar) or np.isnan(gtsrb):
                continue

            acc_mean = ((cifar + gtsrb) / 2) * 100

            sorted_points.append((fairness_val, acc_mean))

        if len(sorted_points) > 1:
            xs, ys = zip(*sorted_points)
            plt.plot(xs, ys, linestyle='--', alpha=0.7)

        # =====================================================
        # VISUAL
        # =====================================================
        plt.xlabel("Fairness (%)")
        plt.ylabel("Accuracy (%)")
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.grid()

        plt.legend(title=DISPLAY_NAME)

        plt.title(f"{DISPLAY_NAME}: Fairness vs Accuracy ({metric})")

        plt.tight_layout()

        save_figure(f"beta_curve_{metric}")

        plt.close()


# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    plot_beta_analysis()