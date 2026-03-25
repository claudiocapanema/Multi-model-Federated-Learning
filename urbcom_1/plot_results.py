import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

FRAC = 0.3
ALPHA = 1.0

BASELINE_FRACS = [0.2, 0.3, 0.4, 0.5]

RESOURCE_METRICS = [
    "resource_usage_cifar",
    "resource_usage_gtsrb"
]

ALGORITHM_ORDER = [
    "fair_resource",
    "fairhetero",
    "fedbalancer",
    "baseline_f0.2",
    "baseline_f0.3",
    "baseline_f0.4",
    "baseline_f0.5"
]

def get_algorithm_order(df):
    return [alg for alg in ALGORITHM_ORDER if alg in df["algorithm"].unique()]

# =====================================================
# MÉTRICAS
# =====================================================

FAIRNESS_METRICS = [
    "inter_client_fairness",
    "intra_client_fairness",
    "inter_model_fairness"
]

METRIC_LABEL = {
    "global_acc": "Accuracy",
    "inter_client_fairness": "Inter-Client Fairness",
    "intra_client_fairness": "Intra-Client Fairness",
    "inter_model_fairness": "Inter-Model Fairness",
}

METRIC_FILENAME = {
    "global_acc": "accuracy",
    "inter_client_fairness": "inter_client_fairness",
    "intra_client_fairness": "intra_client_fairness",
    "inter_model_fairness": "inter_model_fairness",
}

def plot_clients_selected(df):

    plt.figure(figsize=(6,4))

    for alg in get_algorithm_order(df):

        curve = (
            df[df["algorithm"] == alg]
            .groupby("round")["clients_selected_total"]
            .mean()
        )

        plt.plot(curve.index, curve.values, label=alg)

    plt.title("Clients Selected")
    plt.xlabel("Round")
    plt.ylabel("Clients")

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(f"{RESULTS_DIR}/clients_selected.png")
    plt.close()

def plot_cumulative_clients(df):

    plt.figure(figsize=(6,4))

    for alg in get_algorithm_order(df):

        curve = (
            df[df["algorithm"] == alg]
            .groupby("round")["clients_selected_total"]
            .mean()
            .cumsum()
        )

        plt.plot(curve.index, curve.values, label=alg)

    plt.title("Cumulative Clients Selected")
    plt.xlabel("Round")
    plt.ylabel("Clients")

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(f"{RESULTS_DIR}/cumulative_clients_selected.png")
    plt.close()

def plot_resource_usage(df):

    for metric in RESOURCE_METRICS:

        plt.figure(figsize=(6,4))

        for alg in get_algorithm_order(df):

            curve = (
                df[df["algorithm"] == alg]
                .groupby("round")[metric]
                .mean()
            )

            plt.plot(curve.index, curve.values, label=alg)

        title_name = metric.replace("_", " ").title()

        plt.title(title_name)
        plt.xlabel("Round")
        plt.ylabel("Resource Usage")

        plt.legend()
        plt.grid()
        plt.tight_layout()

        plt.savefig(f"{RESULTS_DIR}/{metric}.png")
        plt.close()

def plot_cumulative_resource_usage(df):

    for metric in RESOURCE_METRICS:

        plt.figure(figsize=(6,4))

        for alg in get_algorithm_order(df):

            curve = (
                df[df["algorithm"] == alg]
                .groupby("round")[metric]
                .mean()
                .cumsum()
            )

            plt.plot(curve.index, curve.values, label=alg)

        title_name = metric.replace("_", " ").title()

        plt.title(f"Cumulative {title_name}")
        plt.xlabel("Round")
        plt.ylabel("Resource Usage")

        plt.legend()
        plt.grid()
        plt.tight_layout()

        plt.savefig(f"{RESULTS_DIR}/cumulative_{metric}.png")
        plt.close()

def load_results():

    dfs = []

    for file in os.listdir(RESULTS_DIR):

        if not file.endswith(".csv"):
            continue

        path = os.path.join(RESULTS_DIR, file)

        # -----------------------------
        # BASELINE (MultiFedAvg)
        # -----------------------------
        if file.startswith("baseline_"):

            if f"alpha_{ALPHA}" not in file:
                continue

            for frac in BASELINE_FRACS:
                if f"frac_{frac}" in file:

                    df = pd.read_csv(path)

                    df["algorithm"] = f"baseline_f{frac}"
                    df["frac"] = frac

                    dfs.append(df)
                    break

        # -----------------------------
        # PROPOSTA
        # -----------------------------
        elif file.startswith("proposta_"):

            df = pd.read_csv(path)

            df["algorithm"] = "fair_resource"
            df["frac"] = FRAC

            dfs.append(df)

        # -----------------------------
        # FAIRHETERO
        # -----------------------------
        elif file.startswith("fairhetero_"):

            if f"frac_{FRAC}" in file and f"alpha_{ALPHA}" in file:

                df = pd.read_csv(path)

                df["algorithm"] = "fairhetero"
                df["frac"] = FRAC

                dfs.append(df)

        # -----------------------------
        # FEDBALANCER
        # -----------------------------
        elif file.startswith("fedbalancer_"):

            if f"frac_{FRAC}" in file and f"alpha_{ALPHA}" in file:

                df = pd.read_csv(path)

                df["algorithm"] = "fedbalancer"
                df["frac"] = FRAC

                dfs.append(df)

    if len(dfs) == 0:
        raise ValueError("Nenhum CSV encontrado em results/")

    df = pd.concat(dfs, ignore_index=True)

    # ordenar para consistência
    df = df.sort_values(["algorithm", "dataset", "round"])

    return df

def plot_accuracy(df):

    for dataset in df["dataset"].unique():

        subset = df[df["dataset"] == dataset]

        plt.figure(figsize=(6,4))

        for alg in get_algorithm_order(df):
            subset = df[df["algorithm"] == alg]

            print(f"\nALG: {alg}")
            print(subset.head())

            curve = (
                subset[subset["algorithm"] == alg]
                .groupby("round")["global_acc"]
                .mean()
            )

            print("curva: ", alg, curve)

            plt.plot(curve.index, curve.values, label=alg)

        plt.title(f"Accuracy ({dataset})")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")

        plt.legend()
        plt.grid()
        plt.tight_layout()

        plt.savefig(f"{RESULTS_DIR}/accuracy_{dataset}.png")
        plt.close()

def plot_fairness(df):

    for metric in FAIRNESS_METRICS:

        plt.figure(figsize=(6,4))

        for alg in get_algorithm_order(df):

            curve = (
                df[df["algorithm"] == alg]
                .groupby("round")[metric]
                .mean()
            )

            plt.plot(curve.index, curve.values, label=alg)

        plt.title(METRIC_LABEL[metric])
        plt.xlabel("Round")
        plt.ylabel(METRIC_LABEL[metric])

        plt.legend()
        plt.grid()
        plt.tight_layout()

        plt.savefig(f"{RESULTS_DIR}/{METRIC_FILENAME[metric]}.png")
        plt.close()

def plot_cumulative_fairness(df):

    for metric in FAIRNESS_METRICS:

        plt.figure(figsize=(6,4))

        for alg in get_algorithm_order(df):

            curve = (
                df[df["algorithm"] == alg]
                .groupby("round")[metric]
                .mean()
                .expanding()
                .mean()
            )

            plt.plot(curve.index, curve.values, label=alg)

        plt.title(f"Cumulative {METRIC_LABEL[metric]}")
        plt.xlabel("Round")
        plt.ylabel(METRIC_LABEL[metric])

        plt.legend()
        plt.grid()
        plt.tight_layout()

        plt.savefig(f"{RESULTS_DIR}/cumulative_{METRIC_FILENAME[metric]}.png")
        plt.close()

def main():
    df = load_results()

    print("\nALGORITHMS FOUND:")
    print(df["algorithm"].unique())

    print("\nFILES LOADED:")
    print(len(df))
    # accuracy por dataset
    plot_accuracy(df)

    # fairness
    plot_fairness(df)
    plot_cumulative_fairness(df)

    # clientes
    plot_clients_selected(df)
    plot_cumulative_clients(df)

    # recursos
    plot_resource_usage(df)
    plot_cumulative_resource_usage(df)

    print("\n✔ Todos os plots gerados")


if __name__ == "__main__":
    main()