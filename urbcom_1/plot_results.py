import os
import pandas as pd
import matplotlib.pyplot as plt

FRAC = 0.3
ALPHA = 0.1
# ALPHA = 1.0

RESULTS_DIR_WIRTE = "results/alpha_"+str(ALPHA)

RESULTS_DIR = "results/"

BASELINE_FRACS = [0.2, 0.3, 0.4, 0.5]

RESOURCE_METRICS = [
    "resource_usage_cifar",
    "resource_usage_gtsrb"
]

ALGORITHM_ORDER = [
    "fair_resource_k",
    "fair_resource_budget",
    "oort",
    "fairhetero",
    "fedbalancer",
    "baseline_f0.2",
    "baseline_f0.3",
    "baseline_f0.4",
    "baseline_f0.5"
]

os.makedirs(RESULTS_DIR_WIRTE, exist_ok=True)

def get_algorithm_order(df):

    algs = list(df["algorithm"].unique())

    import re

    def sort_key(a):

        # 🔥 fair_resource_kX
        match = re.search(r'fair_resource_k(\d+\.\d+)', a)
        if match:
            return (0, float(match.group(1)))

        # budget
        if a == "fair_resource_budget":
            return (1, 0)

        # outros métodos
        if a == "oort":
            return (2, 0)
        if a == "fairhetero":
            return (3, 0)
        if a == "fedbalancer":
            return (4, 0)

        # baselines
        match = re.search(r'baseline_f(\d+\.\d+)', a)
        if match:
            return (5, float(match.group(1)))

        return (6, 0)

    return sorted(algs, key=sort_key)

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

    plt.savefig(f"{RESULTS_DIR_WIRTE}/clients_selected.png")
    plt.close()

def plot_cumulative_clients(df):

    plt.figure(figsize=(6,4))

    for alg in get_algorithm_order(df):

        subset = df[df["algorithm"] == alg]

        # 🔥 pega um único valor por rodada (evita duplicação CIFAR/GTSRB)
        curve = (
            subset
            .groupby("round")["clients_selected_total"]
            .first()   # ou .max() (ambos funcionam aqui)
            .sort_index()
        )

        # 🔥 soma acumulada correta
        cumulative = curve.cumsum()

        plt.plot(
            cumulative.index,
            cumulative.values,
            label=alg
        )

    plt.title("Cumulative Clients Selected")
    plt.xlabel("Round")
    plt.ylabel("Clients")

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(f"{RESULTS_DIR_WIRTE}/cumulative_clients_selected.png")
    plt.close()

def plot_mean_fairness(df):

    # 🔥 cria coluna nova com média das fairness
    df["mean_fairness"] = df[FAIRNESS_METRICS].mean(axis=1)

    plt.figure(figsize=(6,4))

    for alg in get_algorithm_order(df):

        curve = (
            df[df["algorithm"] == alg]
            .groupby("round")["mean_fairness"]
            .mean()
        )

        plt.plot(curve.index, curve.values, label=alg)

    plt.title("Mean Fairness")
    plt.xlabel("Round")
    plt.ylabel("Fairness (avg)")

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(f"{RESULTS_DIR_WIRTE}/mean_fairness.png")
    plt.close()

def plot_cumulative_mean_fairness(df):

    df["mean_fairness"] = df[FAIRNESS_METRICS].mean(axis=1)

    plt.figure(figsize=(6,4))

    for alg in get_algorithm_order(df):

        curve = (
            df[df["algorithm"] == alg]
            .groupby("round")["mean_fairness"]
            .mean()
            .expanding()
            .mean()
        )

        plt.plot(curve.index, curve.values, label=alg)

    plt.title("Cumulative Mean Fairness")
    plt.xlabel("Round")
    plt.ylabel("Fairness (avg)")

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(f"{RESULTS_DIR_WIRTE}/cumulative_mean_fairness.png")
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

        plt.savefig(f"{RESULTS_DIR_WIRTE}/{metric}.png")
        plt.close()

def plot_cumulative_resource_usage(df):

    for metric in RESOURCE_METRICS:

        plt.figure(figsize=(6,4))

        for alg in get_algorithm_order(df):

            subset = df[df["algorithm"] == alg]

            # 🔥 soma por rodada (CIFAR + GTSRB corretamente)
            curve = (
                subset
                .groupby("round")[metric]
                .sum()
                .sort_index()
            )

            # 🔥 acumulado correto
            cumulative = curve.cumsum()

            plt.plot(
                cumulative.index,
                cumulative.values,
                label=alg
            )

        title_name = metric.replace("_", " ").title()

        plt.title(f"Cumulative {title_name}")
        plt.xlabel("Round")
        plt.ylabel("Resource Usage")

        plt.legend()
        plt.grid()
        plt.tight_layout()

        plt.savefig(f"{RESULTS_DIR_WIRTE}/cumulative_{metric}.png")
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
        # -----------------------------
        # PROPOSTA - K
        # -----------------------------
        elif file.startswith("proposta_k_"):

            if f"alpha_{ALPHA}" in file:

                df = pd.read_csv(path)

                # 🔥 extrai frac do nome do arquivo
                frac_value = None

                for frac in BASELINE_FRACS + [FRAC]:
                    if f"frac_{frac}" in file:
                        frac_value = frac
                        break

                # fallback (caso não esteja na lista)
                if frac_value is None:
                    import re
                    match = re.search(r'frac_(\d+\.\d+)', file)
                    if match:
                        frac_value = float(match.group(1))

                # 🔥 nome do algoritmo com K (= frac)
                alg_name = f"fair_resource_k{frac_value}"

                df["algorithm"] = alg_name
                df["frac"] = frac_value

                dfs.append(df)

        # -----------------------------
        # PROPOSTA - BUDGET
        # -----------------------------
        elif file.startswith("proposta_budget_"):

            if f"alpha_{ALPHA}" in file:
                df = pd.read_csv(path)

                df["algorithm"] = "fair_resource_budget"
                df["frac"] = FRAC

                dfs.append(df)

        # -----------------------------
        # OORT
        # -----------------------------
        elif file.startswith("oort_"):

            if f"frac_{FRAC}" in file and f"alpha_{ALPHA}" in file:
                df = pd.read_csv(path)

                df["algorithm"] = "oort"
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
        raise ValueError("Nenhum CSV encontrado em results/alpha_"+str(ALPHA)+"/")

    df = pd.concat(dfs, ignore_index=True)

    # ordenar para consistência
    df = df.sort_values(["algorithm", "dataset", "round"])

    return df

def plot_accuracy(df):

    # =====================================================
    # 1) PLOT POR DATASET (CIFAR / GTSRB)
    # =====================================================
    for dataset in df["dataset"].unique():

        subset_dataset = df[df["dataset"] == dataset]

        plt.figure(figsize=(6,4))

        for alg in get_algorithm_order(df):

            subset_alg = subset_dataset[subset_dataset["algorithm"] == alg]

            if len(subset_alg) == 0:
                continue

            curve = (
                subset_alg
                .groupby("round")["global_acc"]
                .mean()
            )

            plt.plot(curve.index, curve.values, label=alg)

        plt.title(f"Accuracy ({dataset})")
        plt.xlabel("Round")
        plt.ylabel("Accuracy")

        plt.legend()
        plt.grid()
        plt.tight_layout()

        plt.savefig(f"{RESULTS_DIR_WIRTE}/accuracy_{dataset}.png")
        plt.close()

    # =====================================================
    # 2) PLOT MÉDIO (CIFAR + GTSRB)
    # =====================================================
    plt.figure(figsize=(6,4))

    for alg in get_algorithm_order(df):

        subset_alg = df[df["algorithm"] == alg]

        if len(subset_alg) == 0:
            continue

        # 🔥 média entre datasets por rodada
        curve = (
            subset_alg
            .groupby(["round"])["global_acc"]
            .mean()
        )

        plt.plot(curve.index, curve.values, label=alg)

    plt.title("Mean Accuracy (CIFAR + GTSRB)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig(f"{RESULTS_DIR_WIRTE}/accuracy_mean.png")
    plt.close()

import numpy as np
from scipy import stats
import os

def compute_ci(series, confidence=0.95):
    mean = np.mean(series)
    sem = stats.sem(series)
    h = sem * stats.t.ppf((1 + confidence) / 2., len(series)-1)
    return mean, h


def build_latex_table(df, save_path="results/alpha_"+str(ALPHA)+"/accuracy_table.tex"):

    results = []

    for alg in get_algorithm_order(df):

        subset = df[df["algorithm"] == alg]

        cifar = subset[subset["dataset"] == "cifar"]["global_acc"]
        gtsrb = subset[subset["dataset"] == "gtsrb"]["global_acc"]

        if len(cifar) == 0 or len(gtsrb) == 0:
            continue

        mean_cifar, ci_cifar = compute_ci(cifar)
        mean_gtsrb, ci_gtsrb = compute_ci(gtsrb)

        mean_total = (mean_cifar + mean_gtsrb) / 2
        ci_total = (ci_cifar + ci_gtsrb) / 2

        results.append({
            "alg": alg,
            "cifar_mean": mean_cifar,
            "cifar_ci": ci_cifar,
            "gtsrb_mean": mean_gtsrb,
            "gtsrb_ci": ci_gtsrb,
            "total_mean": mean_total,
            "total_ci": ci_total,
        })

    df_res = pd.DataFrame(results)

    # =====================================================
    # IDENTIFICAR MELHORES + IC
    # =====================================================

    def mark_best(col_mean, col_ci):

        best_idx = df_res[col_mean].idxmax()
        best_mean = df_res.loc[best_idx, col_mean]
        best_ci = df_res.loc[best_idx, col_ci]

        selected = []

        for i, row in df_res.iterrows():

            lower = row[col_mean] - row[col_ci]
            upper = row[col_mean] + row[col_ci]

            best_lower = best_mean - best_ci
            best_upper = best_mean + best_ci

            overlap = not (upper < best_lower or lower > best_upper)

            if overlap:
                selected.append(i)

        return selected

    best_total = mark_best("total_mean", "total_ci")
    best_cifar = mark_best("cifar_mean", "cifar_ci")
    best_gtsrb = mark_best("gtsrb_mean", "gtsrb_ci")

    # =====================================================
    # GERAR LATEX
    # =====================================================

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Algorithm & Mean Acc (\\%) & CIFAR (\\%) & GTSRB (\\%) \\\\")
    lines.append("\\midrule")

    for i, row in df_res.iterrows():

        def fmt(mean, ci, bold):
            # 🔥 converter para porcentagem
            mean *= 100
            ci *= 100

            text = f"{mean:.2f} $\\pm$ {ci:.2f}"
            return f"\\textbf{{{text}}}" if bold else text

        total = fmt(row["total_mean"], row["total_ci"], i in best_total)
        cifar = fmt(row["cifar_mean"], row["cifar_ci"], i in best_cifar)
        gtsrb = fmt(row["gtsrb_mean"], row["gtsrb_ci"], i in best_gtsrb)

        lines.append(f"{row['alg']} & {total} & {cifar} & {gtsrb} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Accuracy comparison (\\%) with 95\\% confidence intervals.}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    # =====================================================
    # SALVAR
    # =====================================================

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        f.write(latex)

    print(f"\n✅ Tabela salva em: {save_path}")

    return latex

def build_fairness_latex_table(df, save_path="results/alpha_"+str(ALPHA)+"/fairness_table.tex"):

    results = []

    for alg in get_algorithm_order(df):

        subset = df[df["algorithm"] == alg]

        if len(subset) == 0:
            continue

        inter_client = subset["inter_client_fairness"]
        intra_client = subset["intra_client_fairness"]
        inter_model = subset["inter_model_fairness"]

        if len(inter_client) == 0:
            continue

        mean_inter_client, ci_inter_client = compute_ci(inter_client)
        mean_intra_client, ci_intra_client = compute_ci(intra_client)
        mean_inter_model, ci_inter_model = compute_ci(inter_model)

        mean_total = np.mean([
            mean_inter_client,
            mean_intra_client,
            mean_inter_model
        ])

        ci_total = np.mean([
            ci_inter_client,
            ci_intra_client,
            ci_inter_model
        ])

        results.append({
            "alg": alg,

            "mean_total": mean_total,
            "ci_total": ci_total,

            "inter_client_mean": mean_inter_client,
            "inter_client_ci": ci_inter_client,

            "intra_client_mean": mean_intra_client,
            "intra_client_ci": ci_intra_client,

            "inter_model_mean": mean_inter_model,
            "inter_model_ci": ci_inter_model,
        })

    df_res = pd.DataFrame(results)

    # =====================================================
    # IDENTIFICAR MELHORES + IC
    # =====================================================

    def mark_best(col_mean, col_ci):

        best_idx = df_res[col_mean].idxmax()
        best_mean = df_res.loc[best_idx, col_mean]
        best_ci = df_res.loc[best_idx, col_ci]

        selected = []

        for i, row in df_res.iterrows():

            lower = row[col_mean] - row[col_ci]
            upper = row[col_mean] + row[col_ci]

            best_lower = best_mean - best_ci
            best_upper = best_mean + best_ci

            overlap = not (upper < best_lower or lower > best_upper)

            if overlap:
                selected.append(i)

        return selected

    best_total = mark_best("mean_total", "ci_total")
    best_inter_client = mark_best("inter_client_mean", "inter_client_ci")
    best_intra_client = mark_best("intra_client_mean", "intra_client_ci")
    best_inter_model = mark_best("inter_model_mean", "inter_model_ci")

    # =====================================================
    # GERAR LATEX
    # =====================================================

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Algorithm & Mean Fairness & Inter-Client & Intra-Client & Inter-Model \\\\")
    lines.append("\\midrule")

    for i, row in df_res.iterrows():

        def fmt(mean, ci, bold):
            text = f"{mean:.2f} $\\pm$ {ci:.2f}"
            return f"\\textbf{{{text}}}" if bold else text

        total = fmt(row["mean_total"], row["ci_total"], i in best_total)
        inter_c = fmt(row["inter_client_mean"], row["inter_client_ci"], i in best_inter_client)
        intra_c = fmt(row["intra_client_mean"], row["intra_client_ci"], i in best_intra_client)
        inter_m = fmt(row["inter_model_mean"], row["inter_model_ci"], i in best_inter_model)

        lines.append(f"{row['alg']} & {total} & {inter_c} & {intra_c} & {inter_m} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Fairness comparison with 95\\% confidence intervals.}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    # =====================================================
    # SALVAR
    # =====================================================

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        f.write(latex)

    print(f"\n✅ Tabela de fairness salva em: {save_path}")

    return latex

def build_resource_latex_table(df, save_path="results/alpha_"+str(ALPHA)+"/resource_table.tex"):

    results = []

    for alg in get_algorithm_order(df):

        subset = df[df["algorithm"] == alg]

        if len(subset) == 0:
            continue

        # =====================================================
        # 🔥 SOMATÓRIO REAL POR RODADA (evita duplicação)
        # =====================================================
        per_round = (
            subset
            .groupby("round")[["resource_usage_cifar", "resource_usage_gtsrb"]]
            .mean()
        )

        # =====================================================
        # 🔥 SOMATÓRIO TOTAL (CORRETO)
        # =====================================================
        total_cifar = per_round["resource_usage_cifar"].sum()
        total_gtsrb = per_round["resource_usage_gtsrb"].sum()

        total_all = total_cifar + total_gtsrb

        results.append({
            "alg": alg,
            "cifar_total": total_cifar,
            "gtsrb_total": total_gtsrb,
            "total_all": total_all
        })

    df_res = pd.DataFrame(results)

    # =====================================================
    # 🔥 MELHOR = MENOR USO TOTAL
    # =====================================================

    best_idx = df_res["total_all"].idxmin()

    # =====================================================
    # GERAR LATEX
    # =====================================================

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Algorithm & Total Resource & CIFAR & GTSRB \\\\")
    lines.append("\\midrule")

    for i, row in df_res.iterrows():

        def fmt(val, bold):
            text = f"{val:.2f}"
            return f"\\textbf{{{text}}}" if bold else text

        is_best = (i == best_idx)

        total = fmt(row["total_all"], is_best)
        cifar = fmt(row["cifar_total"], False)
        gtsrb = fmt(row["gtsrb_total"], False)

        lines.append(f"{row['alg']} & {total} & {cifar} & {gtsrb} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Total resource usage (sum across all rounds). Lower is better.}")
    lines.append("\\end{table}")

    latex = "\n".join(lines)

    # =====================================================
    # SALVAR
    # =====================================================

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w") as f:
        f.write(latex)

    print(f"\n✅ Tabela de resource usage salva em: {save_path}")

    return latex

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

        plt.savefig(f"{RESULTS_DIR_WIRTE}/{METRIC_FILENAME[metric]}.png")
        plt.close()

def plot_cumulative_fairness(df):

    for metric in FAIRNESS_METRICS:

        plt.figure(figsize=(6,4))

        for alg in get_algorithm_order(df):

            subset = df[df["algorithm"] == alg].copy()

            # 🔥 agrega por rodada (evita duplicação CIFAR/GTSRB distorcer)
            curve = (
                subset
                .groupby("round")[metric]
                .mean()
                .sort_index()
            )

            # 🔥 soma acumulada REAL (CORRETO)
            cumulative = curve.cumsum()

            plt.plot(
                cumulative.index,
                cumulative.values,
                label=alg
            )

        plt.title(f"Cumulative {METRIC_LABEL[metric]}")
        plt.xlabel("Round")
        plt.ylabel(f"Cumulative {METRIC_LABEL[metric]}")

        plt.legend()
        plt.grid()
        plt.tight_layout()

        plt.savefig(f"{RESULTS_DIR_WIRTE}/cumulative_{METRIC_FILENAME[metric]}.png")
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

    # fairness médio
    plot_mean_fairness(df)
    # plot_cumulative_mean_fairness(df)

    build_latex_table(df)
    build_fairness_latex_table(df)  # novo
    build_resource_latex_table(df)  # 🔥 NOVO

    print("\n✔ Todos os plots gerados")


if __name__ == "__main__":
    main()