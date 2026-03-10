import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

REGIME = "realistic"
FRAC = 0.3
ALPHA = 1.0

DATASETS = ["cifar", "gtsrb"]

# =====================================================
# DIREÇÃO DAS MÉTRICAS
# =====================================================

METRIC_DIRECTION = {
    "global_acc": "↑ higher is better",
    "fairness_resource": "↓ lower is better",
    "client_model_fairness": "↓ lower is better",
    "client_capacity_fairness": "↑ higher is better",
    "client_fairness_cifar": "↑ higher is better",
    "client_fairness_gtsrb": "↑ higher is better",
    "clients_selected_total": "↑ higher is better"
}


# =====================================================
# CARREGAR RESULTADOS
# =====================================================

def load_results():

    dfs = []

    for dataset in DATASETS:

        proposta_file = f"{RESULTS_DIR}/proposta_{dataset}_regime_{REGIME}_frac_{FRAC}_alpha_{ALPHA}.csv"
        baseline_file = f"{RESULTS_DIR}/baseline_{dataset}_regime_{REGIME}_frac_{FRAC}_alpha_{ALPHA}.csv"

        if os.path.exists(proposta_file):

            df = pd.read_csv(proposta_file)
            df["algorithm"] = "proposta"
            dfs.append(df)

        if os.path.exists(baseline_file):

            df = pd.read_csv(baseline_file)
            df["algorithm"] = "baseline"
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# =====================================================
# FUNÇÃO GENÉRICA DE PLOT
# =====================================================

def plot_metric(df, metric):

    plt.figure(figsize=(6,4))

    for alg in df["algorithm"].unique():

        curve = (
            df[df["algorithm"] == alg]
            .groupby("round")[metric]
            .mean()
        )

        plt.plot(curve.index, curve.values, label=alg)

    direction = METRIC_DIRECTION.get(metric, "")

    plt.title(f"{metric} ({direction})")
    plt.xlabel("Round")
    plt.ylabel(metric)

    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.savefig(f"results/{metric}.png")
    plt.close()


# =====================================================
# PLOTS ESPECÍFICOS
# =====================================================

def plot_accuracy(df):

    for dataset in DATASETS:

        subset = df[df["dataset"] == dataset]

        plt.figure(figsize=(6,4))

        for alg in subset["algorithm"].unique():

            curve = (
                subset[subset["algorithm"] == alg]
                .groupby("round")["global_acc"]
                .mean()
            )

            plt.plot(curve.index, curve.values, label=alg)

        direction = METRIC_DIRECTION["global_acc"]

        plt.title(f"Accuracy ({dataset}) ({direction})")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")

        plt.legend()
        plt.grid()

        plt.tight_layout()

        plt.savefig(f"results/accuracy_{dataset}.png")
        plt.close()


def plot_all_metrics(df):

    metrics = [
        "fairness_resource",
        "client_model_fairness",
        "client_capacity_fairness",
        "client_fairness_cifar",
        "client_fairness_gtsrb",
        "clients_selected_total"
    ]

    for metric in metrics:
        plot_metric(df, metric)


# =====================================================
# TABELA RESUMO
# =====================================================

def summary_table(df):

    metrics = [
        "global_acc",
        "fairness_resource",
        "client_model_fairness",
        "client_capacity_fairness",
        "client_fairness_cifar",
        "client_fairness_gtsrb"
    ]

    summary = (
        df.groupby(["algorithm", "dataset"])[metrics]
        .mean()
        .round(4)
    )

    print("\n===== MÉDIAS FINAIS =====\n")
    print(summary)

    summary.to_csv("results/summary_comparison.csv")


# =====================================================
# MAIN
# =====================================================

def main():

    df = load_results()

    summary_table(df)

    plot_accuracy(df)

    plot_all_metrics(df)

    print("\n✔ Análise concluída.")
    print("Gráficos salvos em /results")


if __name__ == "__main__":
    main()