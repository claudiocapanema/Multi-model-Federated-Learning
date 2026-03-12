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
    "client_capacity_fairness": "↑ higher is better",
    "clients_selected_total": "↑ higher is better",
    "resource_usage_cifar": "resource usage",
    "resource_usage_gtsrb": "resource usage"
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

    plt.savefig(f"{RESULTS_DIR}/{metric}.png")
    plt.close()


# =====================================================
# PLOT DE ACURÁCIA
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

        plt.savefig(f"{RESULTS_DIR}/accuracy_{dataset}.png")
        plt.close()


# =====================================================
# PLOTS DE RECURSO POR MODELO
# =====================================================

def plot_resource_usage(df):

    metrics = ["resource_usage_cifar", "resource_usage_gtsrb"]

    for metric in metrics:

        plt.figure(figsize=(6,4))

        for alg in df["algorithm"].unique():

            curve = (
                df[df["algorithm"] == alg]
                .groupby("round")[metric]
                .mean()
            )

            plt.plot(curve.index, curve.values, label=alg)

        plt.title(metric)
        plt.xlabel("Round")
        plt.ylabel(metric)

        plt.legend()
        plt.grid()

        plt.tight_layout()

        plt.savefig(f"{RESULTS_DIR}/{metric}.png")
        plt.close()


# =====================================================
# PLOTS PRINCIPAIS
# =====================================================

def plot_all_metrics(df):

    metrics = [
        "fairness_resource",
        "client_capacity_fairness",
        "clients_selected_total"
    ]

    for metric in metrics:
        plot_metric(df, metric)

# =====================================================
# ACCURACY vs RESOURCE USAGE
# =====================================================

def plot_accuracy_vs_resource(df):

    for dataset in DATASETS:

        subset = df[df["dataset"] == dataset]

        plt.figure(figsize=(6,4))

        for alg in subset["algorithm"].unique():

            alg_df = subset[subset["algorithm"] == alg].copy()

            # recurso acumulado ao longo das rodadas
            resource = (
                alg_df["resource_usage_cifar"] +
                alg_df["resource_usage_gtsrb"]
            )

            cumulative_resource = resource.cumsum()

            acc_curve = (
                alg_df
                .groupby("round")["global_acc"]
                .mean()
            )

            resource_curve = (
                cumulative_resource
                .groupby(alg_df["round"])
                .mean()
            )

            plt.plot(resource_curve.values, acc_curve.values, label=alg)

        plt.title(f"Accuracy vs Resource Usage ({dataset})")
        plt.xlabel("Cumulative Resource Usage")
        plt.ylabel("Accuracy")

        plt.legend()
        plt.grid()

        plt.tight_layout()

        plt.savefig(f"{RESULTS_DIR}/accuracy_vs_resource_{dataset}.png")
        plt.close()


# =====================================================
# TABELA RESUMO
# =====================================================

def summary_table(df):

    acc = (
        df.groupby(["algorithm", "dataset"])["global_acc"]
        .mean()
        .rename("acc_mean")
    )

    clients = (
        df.groupby(["algorithm", "dataset"])["clients_selected_total"]
        .mean()
        .rename("clients_mean")
    )

    # fairness_resource
    fairness_res_mean = (
        df.groupby(["algorithm", "dataset"])["fairness_resource"]
        .mean()
        .rename("fairness_resource_mean")
    )

    fairness_res_sum = (
        df.groupby(["algorithm", "dataset"])["fairness_resource"]
        .sum()
        .rename("fairness_resource_sum")
    )

    # client_capacity_fairness
    cap_mean = (
        df.groupby(["algorithm", "dataset"])["client_capacity_fairness"]
        .mean()
        .rename("capacity_fairness_mean")
    )

    cap_sum = (
        df.groupby(["algorithm", "dataset"])["client_capacity_fairness"]
        .sum()
        .rename("capacity_fairness_sum")
    )

    summary = pd.concat([
        acc,
        clients,
        fairness_res_mean,
        fairness_res_sum,
        cap_mean,
        cap_sum
    ], axis=1).round(4)

    print("\n===== MÉTRICAS AGREGADAS =====\n")
    print(summary)

    summary.to_csv(f"{RESULTS_DIR}/summary_comparison.csv")

# =====================================================
# MAIN
# =====================================================

def main():

    df = load_results()

    summary_table(df)

    plot_accuracy(df)

    plot_all_metrics(df)

    plot_resource_usage(df)

    plot_accuracy_vs_resource(df)

    print("\n✔ Análise concluída.")
    print("Gráficos salvos em /results")


if __name__ == "__main__":
    main()