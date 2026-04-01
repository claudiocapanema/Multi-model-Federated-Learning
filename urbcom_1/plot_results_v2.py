import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# =====================================================
# CONFIGURAÇÕES
# =====================================================

FRAC = 0.3
# ALPHA = 0.1
ALPHA = 1.0

RESULTS_DIR_WRITE = f"results/alpha_{ALPHA}"
RESULTS_DIR = "results/"

os.makedirs(RESULTS_DIR_WRITE, exist_ok=True)

# =====================================================
# 🔥 NOVO: SUBPASTAS PARA FIGURAS
# =====================================================

PNG_DIR = os.path.join(RESULTS_DIR_WRITE, "png")
PDF_DIR = os.path.join(RESULTS_DIR_WRITE, "pdf")

os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

# =====================================================
# 🔥 NOVO: SELEÇÃO DE ALGORITMOS
# =====================================================

SELECTED_ALGORITHMS = None

SELECTED_ALGORITHMS = [
    # "fair_resource_k0.2",
    # "fair_resource_k0.3",
    # "fair_resource_k0.4",
    # "fair_resource_k0.5",
    "oort",
    "fairhetero",
    "fedbalancer",
    # "fedfairmmfl_f0.2",
    "fedfairmmfl_f0.3",
    # "fedfairmmfl_f0.4",
    # "fedfairmmfl_f0.5",   # 🔥 ADICIONAR
    # "baseline_f0.2",
    "baseline_f0.3",
    # "baseline_f0.4",
    # "baseline_f0.5"
]

def filter_algorithms(df):
    if SELECTED_ALGORITHMS is None:
        return df
    return df[df["algorithm"].isin(SELECTED_ALGORITHMS)]

# =====================================================
# 🔥 FORMATADOR DE NOMES (METRICS → HUMAN READABLE)
# =====================================================

def format_metric_name(metric):

    # substitui "_" por espaço
    name = metric.replace("_", " ")

    # capitaliza palavras
    name = name.title()

    # ajustes específicos (opcional, mas recomendado)
    name = name.replace("Cifar", "CIFAR")
    name = name.replace("Gtsrb", "GTSRB")

    return name

# =====================================================
# 🔥 DISPLAY NAMES (PARA PLOTS E TABELAS)
# =====================================================

def get_display_name(alg):

    mapping = {
        "oort": "Oort",
        "fairhetero": "FairHetero",
        "fedbalancer": "FedBalancer",
        "fair_resource_budget": "Fair Resource"
    }

    if alg in mapping:
        return mapping[alg]

    # 🔹 BASELINE
    if alg.startswith("baseline_f"):
        frac = float(alg.split("baseline_f")[-1])
        pct = int(frac * 100)
        return f"MultiFedAvg ({pct}\\%)"

    # 🔹 FEDFAIRMMFL
    if alg.startswith("fedfairmmfl_f"):
        frac = float(alg.split("fedfairmmfl_f")[-1])
        pct = int(frac * 100)
        return f"FedFairMMFL ({pct}\\%)"

    # 🔹 PROPOSTA
    if alg.startswith("fair_resource_k"):
        frac = float(alg.split("fair_resource_k")[-1])
        pct = int(frac * 100)
        return f"Fair Resource ({pct}\\%)"

    return alg

def save_figure(filename):
    """
    Salva a figura em PNG e PDF automaticamente
    """
    plt.savefig(os.path.join(PNG_DIR, f"{filename}.png"))
    plt.savefig(os.path.join(PDF_DIR, f"{filename}.pdf"))

# =====================================================
# 🔥 NOVO: SUPORTE A IDIOMAS
# =====================================================

LANGUAGES = ["en", "pt"]

TEXTS = {

    "en": {
        "accuracy": "Accuracy",
        "mean_accuracy": "Mean Accuracy (CIFAR + GTSRB)",
        "round": "Round",
        "clients": "Clients",
        "clients_selected": "Clients Selected",
        "cumulative_clients": "Cumulative Clients Selected",
        "resource_usage": "Resource Usage",
        "mean_fairness": "Mean Fairness",
        "cumulative": "Cumulative",
        "fairness": "Fairness",

        # tabelas
        "table_accuracy_caption": "Accuracy comparison (\\%) with 95\\% confidence intervals.",
        "table_fairness_caption": "Fairness comparison with 95\\% confidence intervals.",
        "table_resource_caption": "Total resource usage (sum across all rounds). Lower is better.",

        "algorithm": "Algorithm",
        "mean_acc": "Mean Acc (\\%)",
        "cifar": "CIFAR",
        "gtsrb": "GTSRB",
        "mean_fairness_col": "Mean Fairness",
        "inter_client": "Inter-Client",
        "intra_client": "Intra-Client",
        "inter_model": "Inter-Model",
        "total_resource": "Total Resource",
    },

    "pt": {
        "accuracy": "Acurácia",
        "mean_accuracy": "Acurácia Média (CIFAR + GTSRB)",
        "round": "Rodada",
        "clients": "Clientes",
        "clients_selected": "Clientes Selecionados",
        "cumulative_clients": "Clientes Acumulados",
        "resource_usage": "Uso de Recurso",
        "mean_fairness": "Fairness Médio",
        "cumulative": "Acumulado",
        "fairness": "Fairness",

        # tabelas
        "table_accuracy_caption": "Comparação de acurácia (\\%) com intervalos de confiança de 95\\%.",
        "table_fairness_caption": "Comparação de fairness com intervalos de confiança de 95\\%.",
        "table_resource_caption": "Uso total de recursos (soma em todas as rodadas). Menor é melhor.",

        "algorithm": "Algoritmo",
        "mean_acc": "Acurácia Média (\\%)",
        "cifar": "CIFAR",
        "gtsrb": "GTSRB",
        "mean_fairness_col": "Fairness Médio",
        "inter_client": "Inter-Cliente",
        "intra_client": "Intra-Cliente",
        "inter_model": "Inter-Modelo",
        "total_resource": "Recurso Total",
    }
}

# =====================================================
# UTILIDADES
# =====================================================

def get_algorithm_order(df):

    algs = list(df["algorithm"].unique())

    import re

    def sort_key(a):

        match = re.search(r'fair_resource_k(\d+\.\d+)', a)
        if match:
            return (0, float(match.group(1)))

        if a.lower() == "fair_resource_budget":
            return (1, 0)

        if a.lower() == "oort":
            return (2, 0)
        if a.lower() == "fairhetero":
            return (3, 0)
        if a.lower() == "fedbalancer":
            return (4, 0)

        match = re.search(r'baseline_f(\d+\.\d+)', a)
        if match:
            return (6, float(match.group(1)))

        match = re.search(r'fedfairmmfl_f(\d+\.\d+)', a)
        if match:
            return (4, float(match.group(1)))

        return (6, 0)

    return sorted(algs, key=sort_key)

def compute_ci(series, confidence=0.95):
    mean = np.mean(series)
    sem = stats.sem(series)
    h = sem * stats.t.ppf((1 + confidence) / 2., len(series)-1)
    return mean, h

# =====================================================
# LOAD RESULTADOS (COM FILTRO)
# =====================================================

BASELINE_FRACS = [0.2, 0.3, 0.4, 0.5]

def load_results():

    dfs = []

    for file in os.listdir(RESULTS_DIR):

        if not file.endswith(".csv"):
            continue

        path = os.path.join(RESULTS_DIR, file)

        # BASELINE
        if file.startswith("baseline_"):

            if f"alpha_{ALPHA}" not in file:
                continue

            for frac in BASELINE_FRACS:
                if f"frac_{frac}" in file:

                    df = pd.read_csv(path)
                    df["algorithm"] = f"baseline_f{frac}"
                    dfs.append(df)
                    break

        # PROPOSTA K
        elif file.startswith("proposta_k_"):

            if f"alpha_{ALPHA}" in file:

                df = pd.read_csv(path)

                import re
                match = re.search(r'frac_(\d+\.\d+)', file)
                frac_value = float(match.group(1)) if match else FRAC

                df["algorithm"] = f"fair_resource_k{frac_value}"
                dfs.append(df)

        elif file.startswith("fedfairmmfl_"):

            if f"alpha_{ALPHA}" not in file:
                continue

            for frac in BASELINE_FRACS:
                if f"frac_{frac}" in file:
                    df = pd.read_csv(path)
                    df["algorithm"] = f"fedfairmmfl_f{frac}"
                    dfs.append(df)
                    break

        # BUDGET
        elif file.startswith("proposta_budget_"):

            if f"alpha_{ALPHA}" in file:
                df = pd.read_csv(path)
                df["algorithm"] = "fair_resource_budget"
                dfs.append(df)

        # OORT
        elif file.startswith("oort_"):

            if f"alpha_{ALPHA}" in file:
                df = pd.read_csv(path)
                df["algorithm"] = "oort"
                dfs.append(df)

        # FAIRHETERO
        elif file.startswith("fairhetero_"):

            if f"alpha_{ALPHA}" in file:
                df = pd.read_csv(path)
                df["algorithm"] = "fairhetero"
                dfs.append(df)

        # FEDBALANCER
        elif file.startswith("fedbalancer_"):

            if f"alpha_{ALPHA}" in file:
                df = pd.read_csv(path)
                df["algorithm"] = "fedbalancer"
                dfs.append(df)

    if len(dfs) == 0:
        raise ValueError("Nenhum CSV encontrado.")

    df = pd.concat(dfs, ignore_index=True)

    # 🔥 aplicar filtro
    df = filter_algorithms(df)

    return df

# =====================================================
# MÉTRICAS
# =====================================================

FAIRNESS_METRICS = [
    "inter_client_fairness",
    "intra_client_fairness",
    "inter_model_fairness"
]

RESOURCE_METRICS = [
    "resource_usage_cifar",
    "resource_usage_gtsrb"
]

# =====================================================
# 🔥 FUNÇÃO BASE DE PLOT (BILÍNGUE)
# =====================================================

def finalize_plot(title_key, xlabel_key, ylabel_key, filename):

    for lang in LANGUAGES:

        texts = TEXTS[lang]

        plt.title(texts[title_key])
        plt.xlabel(texts[xlabel_key])
        plt.ylabel(texts[ylabel_key])

        plt.legend()
        plt.grid()
        plt.tight_layout()

        plt.savefig(f"{RESULTS_DIR_WRITE}/{filename}_{lang}")
        plt.clf()


# =====================================================
# ACCURACY
# =====================================================

def plot_accuracy(df):

    # -------------------------
    # POR DATASET
    # -------------------------
    for dataset in df["dataset"].unique():

        for lang in LANGUAGES:

            plt.figure(figsize=(6,4))

            texts = TEXTS[lang]

            for alg in get_algorithm_order(df):

                subset = df[
                    (df["dataset"] == dataset) &
                    (df["algorithm"] == alg)
                ]

                if len(subset) == 0:
                    continue

                curve = (
                    subset.groupby("round")["global_acc"].mean()
                )

                plt.plot(curve.index, curve.values, label=get_display_name(alg))

            plt.title(f"{texts['accuracy']} ({dataset})")
            plt.xlabel(texts["round"])
            plt.ylabel(texts["accuracy"])

            plt.legend()
            plt.ylim(0, 100)
            plt.grid()
            plt.tight_layout()

            plt.savefig(f"{RESULTS_DIR_WRITE}/accuracy_{dataset}_{lang}")
            plt.close()

    # -------------------------
    # MÉDIA
    # -------------------------
    for lang in LANGUAGES:

        plt.figure(figsize=(6,4))
        texts = TEXTS[lang]

        for alg in get_algorithm_order(df):

            subset = df[df["algorithm"] == alg]

            if len(subset) == 0:
                continue

            curve = subset.groupby("round")["global_acc"].mean()

            plt.plot(curve.index, curve.values, label=get_display_name(alg))

        plt.title(texts["mean_accuracy"])
        plt.xlabel(texts["round"])
        plt.ylabel(texts["accuracy"])

        plt.legend()
        plt.ylim(0, 100)
        plt.grid()
        plt.tight_layout()

        save_figure(f"accuracy_mean_{lang}")
        plt.close()


# =====================================================
# FAIRNESS
# =====================================================

def plot_fairness(df):

    for metric in FAIRNESS_METRICS:

        for lang in LANGUAGES:

            plt.figure(figsize=(6,4))
            texts = TEXTS[lang]

            for alg in get_algorithm_order(df):

                curve = (
                    df[df["algorithm"] == alg]
                    .groupby("round")[metric]
                    .mean()
                )

                plt.plot(curve.index, curve.values, label=get_display_name(alg))

            plt.title(texts["fairness"] + f" ({format_metric_name(metric)})")
            plt.xlabel(texts["round"])
            plt.ylabel(texts["fairness"])

            plt.legend()
            plt.grid()
            plt.tight_layout()

            save_figure(f"{metric}_{lang}")
            plt.close()


def plot_cumulative_fairness(df):

    for metric in FAIRNESS_METRICS:

        for lang in LANGUAGES:

            plt.figure(figsize=(6,4))
            texts = TEXTS[lang]

            for alg in get_algorithm_order(df):

                subset = df[df["algorithm"] == alg]

                curve = (
                    subset.groupby("round")[metric]
                    .mean()
                    .sort_index()
                )

                cumulative = curve.cumsum()

                plt.plot(cumulative.index, cumulative.values, label=get_display_name(alg))

            plt.title(f"{texts['cumulative']} {texts['fairness']} ({format_metric_name(metric)})")
            plt.xlabel(texts["round"])
            plt.ylabel(texts["fairness"])

            plt.legend()
            plt.ylim(0, 100)
            plt.grid()
            plt.tight_layout()

            save_figure(f"cumulative_{metric}_{lang}")
            plt.close()


# =====================================================
# CLIENTES
# =====================================================

def plot_clients_selected(df):

    for lang in LANGUAGES:

        plt.figure(figsize=(6,4))
        texts = TEXTS[lang]

        for alg in get_algorithm_order(df):

            curve = (
                df[df["algorithm"] == alg]
                .groupby("round")["clients_selected_total"]
                .mean()
            )

            plt.plot(curve.index, curve.values, label=get_display_name(alg))

        plt.title(texts["clients_selected"])
        plt.xlabel(texts["round"])
        plt.ylabel(texts["clients"])

        plt.legend()
        plt.grid()
        plt.tight_layout()

        save_figure(f"clients_selected_{lang}")
        plt.close()


def plot_cumulative_clients(df):

    for lang in LANGUAGES:

        plt.figure(figsize=(6,4))
        texts = TEXTS[lang]

        for alg in get_algorithm_order(df):

            subset = df[df["algorithm"] == alg]

            curve = (
                subset.groupby("round")["clients_selected_total"]
                .first()
                .sort_index()
            )

            cumulative = curve.cumsum()

            plt.plot(cumulative.index, cumulative.values, label=get_display_name(alg))

        plt.title(texts["cumulative_clients"])
        plt.xlabel(texts["round"])
        plt.ylabel(texts["clients"])

        plt.legend()
        plt.grid()
        plt.tight_layout()

        save_figure(f"cumulative_clients_selected_{lang}")
        plt.close()


# =====================================================
# RESOURCE USAGE
# =====================================================

def plot_resource_usage(df):

    for metric in RESOURCE_METRICS:

        for lang in LANGUAGES:

            plt.figure(figsize=(6,4))
            texts = TEXTS[lang]

            for alg in get_algorithm_order(df):

                curve = (
                    df[df["algorithm"] == alg]
                    .groupby("round")[metric]
                    .mean()
                )

                plt.plot(curve.index, curve.values, label=get_display_name(alg))

            plt.title(f"{texts['resource_usage']} ({format_metric_name(metric)})")
            plt.xlabel(texts["round"])
            plt.ylabel(texts["resource_usage"])

            plt.legend()
            plt.grid()
            plt.tight_layout()

            save_figure(f"{metric}_{lang}")
            plt.close()


def plot_cumulative_resource_usage(df):

    for metric in RESOURCE_METRICS:

        for lang in LANGUAGES:

            plt.figure(figsize=(6,4))
            texts = TEXTS[lang]

            for alg in get_algorithm_order(df):

                subset = df[df["algorithm"] == alg]

                curve = (
                    subset.groupby("round")[metric]
                    .sum()
                    .sort_index()
                )

                cumulative = curve.cumsum()

                plt.plot(cumulative.index, cumulative.values, label=get_display_name(alg))

            plt.title(f"{texts['cumulative']} {texts['resource_usage']} ({format_metric_name(metric)})")
            plt.xlabel(texts["round"])
            plt.ylabel(texts["resource_usage"])

            plt.legend()
            plt.grid()
            plt.tight_layout()

            save_figure(f"cumulative_{metric}_{lang}")
            plt.close()


# =====================================================
# FAIRNESS MÉDIO
# =====================================================

def plot_mean_fairness(df):

    df["mean_fairness"] = df[FAIRNESS_METRICS].mean(axis=1)

    for lang in LANGUAGES:

        plt.figure(figsize=(6,4))
        texts = TEXTS[lang]

        for alg in get_algorithm_order(df):

            curve = (
                df[df["algorithm"] == alg]
                .groupby("round")["mean_fairness"]
                .mean()
            )

            plt.plot(curve.index, curve.values, label=get_display_name(alg))

        plt.title(texts["mean_fairness"])
        plt.xlabel(texts["round"])
        plt.ylabel(texts["fairness"])

        plt.legend()
        plt.ylim(0, 100)
        plt.grid()
        plt.tight_layout()

        save_figure(f"mean_fairness_{lang}")
        plt.close()

# =====================================================
# 🔥 TABELA DE ACURÁCIA (BILÍNGUE)
# =====================================================

def build_accuracy_table(df):

    for lang in LANGUAGES:

        texts = TEXTS[lang]
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

        if df_res.empty:
            print(f"⚠️ Nenhum dado para tabela de accuracy ({lang}) — pulando.")
            continue

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

        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{lccc}")
        lines.append("\\toprule")

        lines.append(
            f"{texts['algorithm']} & {texts['mean_acc']} & {texts['cifar']} & {texts['gtsrb']} \\\\"
        )

        lines.append("\\midrule")

        for i, row in df_res.iterrows():

            def fmt(mean, ci, bold):
                mean *= 100
                ci *= 100
                text = f"{mean:.2f} $\\pm$ {ci:.2f}"
                return f"\\textbf{{{text}}}" if bold else text

            total = fmt(row["total_mean"], row["total_ci"], i in best_total)
            cifar = fmt(row["cifar_mean"], row["cifar_ci"], i in best_cifar)
            gtsrb = fmt(row["gtsrb_mean"], row["gtsrb_ci"], i in best_gtsrb)

            alg_name = get_display_name(row['alg'])
            lines.append(f"{alg_name} & {total} & {cifar} & {gtsrb} \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")

        # 🔥 caption com alpha
        lines.append(
            f"\\caption{{{texts['table_accuracy_caption']} $\\alpha={ALPHA}$.}}"
        )

        # 🔥 label
        lines.append("\\label{tab:accuracy}")

        lines.append("\\end{table}")

        latex = "\n".join(lines)

        path = f"{RESULTS_DIR_WRITE}/accuracy_table_{lang}.tex"
        with open(path, "w") as f:
            f.write(latex)

        print(f"✅ Accuracy table ({lang}) salva em: {path}")

def build_final_accuracy_table(df):

    for lang in LANGUAGES:

        texts = TEXTS[lang]
        results = []

        for alg in get_algorithm_order(df):

            subset = df[df["algorithm"] == alg]

            if len(subset) == 0:
                continue

            # 🔥 pega apenas última rodada por execução/dataset
            final = (
                subset
                .sort_values("round")
                .groupby(["dataset"])
                .tail(1)
            )

            cifar = final[final["dataset"] == "cifar"]["global_acc"].mean()
            gtsrb = final[final["dataset"] == "gtsrb"]["global_acc"].mean()

            if np.isnan(cifar) or np.isnan(gtsrb):
                continue

            mean_total = (cifar + gtsrb) / 2

            results.append({
                "alg": alg,
                "cifar": cifar,
                "gtsrb": gtsrb,
                "total": mean_total
            })

        df_res = pd.DataFrame(results)

        if df_res.empty:
            print(f"⚠️ Nenhum dado final de accuracy ({lang})")
            continue

        # 🔥 melhor = maior
        best_total = df_res["total"].idxmax()
        best_cifar = df_res["cifar"].idxmax()
        best_gtsrb = df_res["gtsrb"].idxmax()

        # LATEX
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{lccc}")
        lines.append("\\toprule")

        lines.append(
            f"{texts['algorithm']} & {texts['mean_acc']} & {texts['cifar']} & {texts['gtsrb']} \\\\"
        )

        lines.append("\\midrule")

        for i, row in df_res.iterrows():

            def fmt(val, bold):
                val *= 100
                text = f"{val:.2f}"
                return f"\\textbf{{{text}}}" if bold else text

            total = fmt(row["total"], i == best_total)
            cifar = fmt(row["cifar"], i == best_cifar)
            gtsrb = fmt(row["gtsrb"], i == best_gtsrb)

            alg_name = get_display_name(row["alg"])
            lines.append(f"{alg_name} & {total} & {cifar} & {gtsrb} \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\caption{Final accuracy (last round).}")
        lines.append("\\end{table}")

        path = f"{RESULTS_DIR_WRITE}/final_accuracy_table_{lang}.tex"
        with open(path, "w") as f:
            f.write("\n".join(lines))

        print(f"✅ Final Accuracy table ({lang}) salva em: {path}")

# =====================================================
# 🔥 TABELA DE FAIRNESS
# =====================================================

def build_fairness_table(df):

    for lang in LANGUAGES:

        texts = TEXTS[lang]
        results = []

        for alg in get_algorithm_order(df):

            subset = df[df["algorithm"] == alg]

            if len(subset) == 0:
                continue

            inter_client = subset["inter_client_fairness"]
            intra_client = subset["intra_client_fairness"]

            mean_inter_client, ci_inter_client = compute_ci(inter_client)
            mean_intra_client, ci_intra_client = compute_ci(intra_client)

            mean_total = np.mean([
                mean_inter_client,
                mean_intra_client
            ])

            ci_total = np.mean([
                ci_inter_client,
                ci_intra_client
            ])

            results.append({
                "alg": alg,
                "mean_total": mean_total,
                "ci_total": ci_total,
                "inter_client_mean": mean_inter_client,
                "inter_client_ci": ci_inter_client,
                "intra_client_mean": mean_intra_client,
                "intra_client_ci": ci_intra_client,
            })

        df_res = pd.DataFrame(results)

        if df_res.empty:
            print(f"⚠️ Nenhum dado para tabela de fairness ({lang}) — pulando.")
            continue

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

        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{lccc}")
        lines.append("\\toprule")

        lines.append(
            f"{texts['algorithm']} & {texts['mean_fairness_col']} & {texts['inter_client']} & {texts['intra_client']} \\\\"
        )

        lines.append("\\midrule")

        for i, row in df_res.iterrows():

            def fmt(mean, ci, bold):
                text = f"{mean:.2f} $\\pm$ {ci:.2f}"
                return f"\\textbf{{{text}}}" if bold else text

            total = fmt(row["mean_total"], row["ci_total"], i in best_total)
            inter_c = fmt(row["inter_client_mean"], row["inter_client_ci"], i in best_inter_client)
            intra_c = fmt(row["intra_client_mean"], row["intra_client_ci"], i in best_intra_client)

            alg_name = get_display_name(row['alg'])
            lines.append(f"{alg_name} & {total} & {inter_c} & {intra_c} \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")

        # 🔥 caption com alpha
        lines.append(
            f"\\caption{{{texts['table_fairness_caption']} $\\alpha={ALPHA}$.}}"
        )

        # 🔥 label
        lines.append("\\label{tab:fairness}")

        lines.append("\\end{table}")

        latex = "\n".join(lines)

        path = f"{RESULTS_DIR_WRITE}/fairness_table_{lang}.tex"
        with open(path, "w") as f:
            f.write(latex)

        print(f"✅ Fairness table ({lang}) salva em: {path}")

def build_final_fairness_table(df):

    for lang in LANGUAGES:

        texts = TEXTS[lang]
        results = []

        for alg in get_algorithm_order(df):

            subset = df[df["algorithm"] == alg]

            if len(subset) == 0:
                continue

            final = (
                subset
                .sort_values("round")
                .groupby(["dataset"])
                .tail(1)
            )

            inter_client = final["inter_client_fairness"].mean()
            intra_client = final["intra_client_fairness"].mean()

            mean_total = np.mean([
                inter_client,
                intra_client
            ])

            results.append({
                "alg": alg,
                "total": mean_total,
                "inter_client": inter_client,
                "intra_client": intra_client,
            })

        df_res = pd.DataFrame(results)

        if df_res.empty:
            print(f"⚠️ Nenhum dado final de fairness ({lang})")
            continue

        best_total = df_res["total"].idxmax()
        best_inter_client = df_res["inter_client"].idxmax()
        best_intra_client = df_res["intra_client"].idxmax()

        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{lccc}")
        lines.append("\\toprule")

        lines.append(
            f"{texts['algorithm']} & {texts['mean_fairness_col']} & {texts['inter_client']} & {texts['intra_client']} \\\\"
        )

        lines.append("\\midrule")

        for i, row in df_res.iterrows():

            def fmt(val, bold):
                text = f"{val:.2f}"
                return f"\\textbf{{{text}}}" if bold else text

            total = fmt(row["total"], i == best_total)
            inter_c = fmt(row["inter_client"], i == best_inter_client)
            intra_c = fmt(row["intra_client"], i == best_intra_client)

            alg_name = get_display_name(row["alg"])
            lines.append(f"{alg_name} & {total} & {inter_c} & {intra_c} \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")

        # 🔥 caption com alpha
        lines.append(
            f"\\caption{{Final fairness (last round, without inter-model fairness). $\\alpha={ALPHA}$.}}"
        )

        # 🔥 label
        lines.append("\\label{tab:final_fairness}")

        lines.append("\\end{table}")

        latex = "\n".join(lines)

        path = f"{RESULTS_DIR_WRITE}/final_fairness_table_{lang}.tex"
        with open(path, "w") as f:
            f.write(latex)

        print(f"✅ Final Fairness table ({lang}) salva em: {path}")

# =====================================================
# 🔥 TABELA DE RESOURCE (SOMATÓRIO)
# =====================================================

def build_resource_table(df):

    for lang in LANGUAGES:

        texts = TEXTS[lang]
        results = []

        for alg in get_algorithm_order(df):

            subset = df[df["algorithm"] == alg]

            if len(subset) == 0:
                continue

            per_round = (
                subset
                .groupby("round")[["resource_usage_cifar", "resource_usage_gtsrb"]]
                .mean()
            )

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

        if df_res.empty:
            print(f"⚠️ Nenhum dado para tabela de accuracy ({lang}) — pulando.")
            continue

        df_res = df_res.sort_values(
            by="alg",
            key=lambda col: col.map(lambda x: get_algorithm_order(pd.DataFrame({"algorithm": [x]}))[0])
        )

        best_idx = df_res["total_all"].idxmin()

        # LATEX
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{lccc}")
        lines.append("\\toprule")

        lines.append(
            f"{texts['algorithm']} & {texts['total_resource']} & {texts['cifar']} & {texts['gtsrb']} \\\\"
        )

        lines.append("\\midrule")

        for i, row in df_res.iterrows():

            def fmt(val, bold):
                text = f"{val:.2f}"
                return f"\\textbf{{{text}}}" if bold else text

            total = fmt(row["total_all"], i == best_idx)
            cifar = fmt(row["cifar_total"], False)
            gtsrb = fmt(row["gtsrb_total"], False)

            alg_name = get_display_name(row['alg'])
            lines.append(f"{alg_name} & {total} & {cifar} & {gtsrb} \\\\")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append(f"\\caption{{{texts['table_resource_caption']}}}")
        lines.append("\\end{table}")

        latex = "\n".join(lines)

        path = f"{RESULTS_DIR_WRITE}/resource_table_{lang}.tex"
        with open(path, "w") as f:
            f.write(latex)

        print(f"✅ Resource table ({lang}) salva em: {path}")

def main():

    df = load_results()

    print("\nALGORITHMS USED:")
    print(df["algorithm"].unique())

    # PLOTS
    plot_accuracy(df)
    plot_fairness(df)
    plot_cumulative_fairness(df)
    plot_clients_selected(df)
    plot_cumulative_clients(df)
    plot_resource_usage(df)
    plot_cumulative_resource_usage(df)
    plot_mean_fairness(df)

    # TABELAS
    build_accuracy_table(df)
    build_fairness_table(df)
    build_resource_table(df)

    build_final_accuracy_table(df)
    build_final_fairness_table(df)

    print("\n✔ Tudo gerado com sucesso (EN + PT)")

main()