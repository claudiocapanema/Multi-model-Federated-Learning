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
ALPHAS = [0.1, 1.0]  # 🔥 pode adicionar mais aqui

COST_RATIO_STR = "2,4x"
# COST_RATIO_STR = "2.0x"
COST_RATIO_STR = "4.0x"
# COST_RATIO_STR = "10.0x"
RESULTS_DIR_WRITE = f"results/gtsrb_{COST_RATIO_STR}_cifar/plots/alpha_{ALPHAS}"
RESULTS_DIR = f"results/gtsrb_{COST_RATIO_STR}_cifar/"

os.makedirs(RESULTS_DIR_WRITE, exist_ok=True)

# =====================================================
# 🔥 NOVO: SUBPASTAS PARA FIGURAS
# =====================================================

PNG_DIR = os.path.join(RESULTS_DIR_WRITE, "png")
PDF_DIR = os.path.join(RESULTS_DIR_WRITE, "pdf")

os.makedirs(PNG_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

def extract_alpha(file):
    import re
    match = re.search(r'alpha_(\d+\.?\d*)', file)
    return float(match.group(1)) if match else None

# =====================================================
# 🔥 NOVO: SELEÇÃO DE ALGORITMOS
# =====================================================

SELECTED_ALGORITHMS = None

SELECTED_ALGORITHMS = [
    # "fair_resource_k0.2",
    "dpfs_k0.3_b0.5",  # padrão
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

# =====================================================
# 🔥 MARKERS POR BETA (GLOBAL)
# =====================================================

BETA_MARKERS = {
    0.1: 'o',
    0.5: 's',
    1.0: '^'
}

BASE_MARKERS = ['D', 'P', 'X', 'v', '*']  # fallback

def filter_algorithms(df):

    if SELECTED_ALGORITHMS is None:
        return df

    selected = set(SELECTED_ALGORITHMS)

    import re

    def keep(alg):

        # DPFS → sempre mantém
        if alg.startswith("dpfs_k"):
            return True

        # BASELINE → sempre mantém
        if alg.startswith("baseline_f"):
            return True

        # 🔥 remove alpha e beta
        base_alg = re.sub(r'_a\d+\.?\d*', '', alg)
        base_alg = re.sub(r'_b\d+\.?\d*', '', base_alg)

        return base_alg in selected

    return df[df["algorithm"].apply(keep)]

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

    # 🔹 BASELINE COM BETA
    if alg.startswith("baseline_f"):
        import re
        match = re.search(r'b(\d+\.?\d*)', alg)
        if match:
            beta = float(match.group(1))
            return f"MultiFedAvg ($\\beta={beta}$)"
        return "MultiFedAvg"

    # 🔹 FEDFAIRMMFL
    if alg.startswith("fedfairmmfl_f"):
        frac = float(alg.split("fedfairmmfl_f")[-1])
        pct = int(frac * 100)
        return f"FedFairMMFL ({pct}\\%)"

    # 🔹 DPFS
    if alg.startswith("dpfs_k"):
        import re
        beta = float(re.search(r'b(\d+\.?\d*)', alg).group(1))
        return f"DPFS ($\\beta={beta}$)"

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
        "inter_client": "(EPF-Inter)",
        "intra_client": "(EPF-Intra)",
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

        # DPFS
        match = re.search(r'dpfs_k(\d+\.\d+)_b(\d+\.?\d*)', a)
        if match:
            frac = float(match.group(1))
            beta = float(match.group(2))
            beta_priority = 0 if beta == 0.5 else 1
            return (0, frac, beta_priority, beta)

        # BASELINE COM BETA
        match = re.search(r'baseline_f(\d+\.\d+)_b(\d+\.?\d*)', a)
        if match:
            frac = float(match.group(1))
            beta = float(match.group(2))
            return (1, frac, beta)

        if a.lower() == "fair_resource_budget":
            return (2, 0)

        if a.lower() == "oort":
            return (3, 0)
        if a.lower() == "fairhetero":
            return (4, 0)
        if a.lower() == "fedbalancer":
            return (5, 0)

        match = re.search(r'fedfairmmfl_f(\d+\.\d+)', a)
        if match:
            return (6, float(match.group(1)))

        return (7, 0)

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
    import re

    for root, dirs, files in os.walk(RESULTS_DIR):

        for file in files:

            if not file.endswith(".csv"):
                continue

            path = os.path.join(root, file)

            # =========================================
            # EXTRAÇÃO (ROBUSTA)
            # =========================================

            frac = None
            alpha = None
            beta = None

            frac_match = re.search(r'frac_(\d+\.?\d*)', root)
            alpha_match = re.search(r'alpha_dirichlet_(\d+\.?\d*)', root)
            beta_match = re.search(r'beta_(\d+\.?\d*)', root)

            if frac_match:
                frac = float(frac_match.group(1))

            if alpha_match:
                alpha = float(alpha_match.group(1))

            if beta_match:
                beta = float(beta_match.group(1))

            # fallback filename
            if frac is None:
                m = re.search(r'f(\d+\.?\d*)', file)
                if m:
                    frac = float(m.group(1))

            if beta is None:
                m = re.search(r'b(\d+\.?\d*)', file)
                if m:
                    beta = float(m.group(1))

            if alpha is None:
                print(f"⚠️ Ignorado (sem alpha): {path}")
                continue

            if alpha not in ALPHAS:
                continue

            if frac is None:
                frac = FRAC

            # =========================================
            # LEITURA
            # =========================================
            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"❌ Erro lendo {path}: {e}")
                continue

            # =========================================
            # DATASET
            # =========================================
            fname = file.lower()

            if "cifar" in fname:
                df["dataset"] = "cifar"
            elif "gtsrb" in fname:
                df["dataset"] = "gtsrb"
            else:
                print(f"⚠️ Dataset não identificado: {file}")
                continue

            # =========================================
            # 🔥 IDENTIFICAÇÃO + ALPHA EMBUTIDO
            # =========================================

            name = file.replace(".csv", "").lower()
            name = name.replace("_cifar", "").replace("_gtsrb", "")

            # DPFS
            if name.startswith("proposta_k"):
                alg = f"dpfs_k{frac}_b{beta}"

            # BASELINE
            elif name.startswith("baseline"):
                alg = f"baseline_f{frac}_b{beta}" if beta is not None else f"baseline_f{frac}"

            elif name.startswith("fedbalancer"):
                alg = f"fedbalancer_b{beta}_a{alpha}"

            elif name.startswith("fairhetero"):
                alg = f"fairhetero_b{beta}_a{alpha}"

            elif name.startswith("oort"):
                alg = f"oort_b{beta}_a{alpha}"

            # FEDFAIRMMFL
            elif name.startswith("fedfairmmfl"):
                alg = f"fedfairmmfl_b{beta}_a{alpha}"

            else:
                alg = name
                print(f"⚠️ Algoritmo desconhecido: {name}")

            # =========================================
            # METADADOS
            # =========================================
            df["algorithm"] = alg
            df["alpha"] = alpha
            df["beta"] = beta
            df["frac"] = frac

            dfs.append(df)

    if len(dfs) == 0:
        raise ValueError("❌ Nenhum CSV encontrado.")

    df = pd.concat(dfs, ignore_index=True)

    print("\n==============================")
    print("ALGORITHMS ENCONTRADOS:")
    print(sorted(df["algorithm"].unique()))
    print("==============================\n")

    df = filter_algorithms(df)

    print("\n==============================")
    print("APÓS FILTRO:")
    print(sorted(df["algorithm"].unique()))
    print("==============================\n")

    return df

def build_accuracy_table_multi_alpha(df):

    texts = TEXTS["pt"]

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\renewcommand{\\arraystretch}{1.2}")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\begin{tabular}{l|c|cc}")
    lines.append("\\toprule")

    lines.append(
        f"{texts['algorithm']} & {texts['mean_acc']} & {texts['cifar']} & {texts['gtsrb']} \\\\"
    )

    for alpha in sorted(df["alpha"].unique()):

        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{4}}{{c}}{{$\\alpha = {alpha}$}} \\\\")
        lines.append("\\midrule")

        df_alpha = df[df["alpha"] == alpha]

        results = []

        for alg in get_algorithm_order(df_alpha):

            subset = df_alpha[df_alpha["algorithm"] == alg]

            cifar = subset[subset["dataset"] == "cifar"]["global_acc"]
            gtsrb = subset[subset["dataset"] == "gtsrb"]["global_acc"]

            if len(cifar) == 0 or len(gtsrb) == 0:
                continue

            mean_cifar, ci_cifar = compute_ci(cifar)
            mean_gtsrb, ci_gtsrb = compute_ci(gtsrb)

            # 🔥 MÉDIA CORRETA
            mean_total = (mean_cifar + mean_gtsrb) / 2

            # 🔥 IC CORRETO (assumindo independência)
            ci_total = np.sqrt((ci_cifar ** 2 + ci_gtsrb ** 2)) / 2

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
            continue

        # 🔥 mesma lógica de overlap que você já usa
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

        for i, row in df_res.iterrows():

            def fmt(mean, ci, bold):
                text = f"{mean*100:.2f} $\\pm$ {ci*100:.2f}"
                return f"\\textbf{{{text}}}" if bold else text

            total = fmt(row["total_mean"], row["total_ci"], i in best_total)
            cifar = fmt(row["cifar_mean"], row["cifar_ci"], i in best_cifar)
            gtsrb = fmt(row["gtsrb_mean"], row["gtsrb_ci"], i in best_gtsrb)

            alg_name = get_display_name(row["alg"])

            lines.append(f"{alg_name} & {total} & {cifar} & {gtsrb} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Comparação de acurácia (\\%) para diferentes valores de $\\alpha$.}")
    lines.append("\\label{tab:accuracy}")
    lines.append("\\end{table}")

    with open(f"{RESULTS_DIR_WRITE}/accuracy_multi_alpha.tex", "w") as f:
        f.write("\n".join(lines))

    print("✅ Tabela multi-alpha gerada (com destaque correto)")

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

    metric_display = {
        "inter_client_fairness": "(EPF-Inter)",
        "intra_client_fairness": "(EPF-Intra)",
        "inter_model_fairness": "Inter-Model"
    }

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

            metric_name = metric_display.get(metric, format_metric_name(metric))

            plt.title(texts["fairness"] + f" {metric_name}")
            plt.xlabel(texts["round"])
            plt.ylabel(texts["fairness"])

            plt.legend()
            plt.grid()
            plt.tight_layout()

            save_figure(f"{metric}_{lang}")
            plt.close()


def plot_cumulative_fairness(df):

    metric_display = {
        "inter_client_fairness": "(EPF-Inter)",
        "intra_client_fairness": "(EPF-Intra)",
        "inter_model_fairness": "Inter-Model"
    }

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

            metric_name = metric_display.get(metric, format_metric_name(metric))

            plt.title(f"{texts['cumulative']} {texts['fairness']} {metric_name}")
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

def compute_efficiency_score(fairness, accuracy):
    """
    fairness: [0,100]
    accuracy: [0,100]

    Score baseado na distância ao ponto ideal (1,1).
    Retorna valor em [0,1], onde maior é melhor.
    """

    f = fairness / 100
    a = accuracy / 100

    # distância euclidiana ao ideal (1,1)
    dist = np.sqrt((1 - f)**2 + (1 - a)**2)

    # normalização (distância máxima = sqrt(2))
    score = 1 - (dist / np.sqrt(2))

    return score

def get_pareto_front(points):
    """
    points: lista de (fairness, accuracy)

    retorna lista de pontos no Pareto front
    """
    pareto = []

    for i, (f1, a1) in enumerate(points):
        dominated = False

        for j, (f2, a2) in enumerate(points):
            if j == i:
                continue

            if (f2 >= f1 and a2 >= a1) and (f2 > f1 or a2 > a1):
                dominated = True
                break

        if not dominated:
            pareto.append((f1, a1))

    return pareto

def distance_to_pareto(point, pareto_front):
    """
    menor distância euclidiana até o Pareto front
    """
    f, a = point

    distances = [
        np.sqrt((f - pf)**2 + (a - pa)**2)
        for (pf, pa) in pareto_front
    ]

    return min(distances)

def compute_pareto_score(point, pareto_front):
    """
    score normalizado [0,1]
    maior = melhor (mais próximo do Pareto front)
    """

    dist = distance_to_pareto(point, pareto_front)

    # normalização (máx possível no espaço 0-100)
    max_dist = np.sqrt(100**2 + 100**2)

    score = 1 - (dist / max_dist)

    return score

def get_base_algorithm(alg):

    if alg.startswith("dpfs_k"):
        return "dpfs"

    if alg.startswith("baseline_f"):
        return "baseline"

    if alg.startswith("fedfairmmfl"):
        return "fedfairmmfl"

    if alg.startswith("fedbalancer"):
        return "fedbalancer"

    if alg.startswith("fairhetero"):
        return "fairhetero"

    if alg.startswith("oort"):
        return "oort"

    return alg

def plot_fairness_vs_accuracy(df):

    DATASETS = ["cifar", "gtsrb"]
    LAST_N_ROUNDS = 5  # ajuste aqui (ex: 5, 10, etc.)

    import re
    import matplotlib.lines as mlines

    for metric in FAIRNESS_METRICS:

        for lang in LANGUAGES:

            texts = TEXTS[lang]

            fig, axes = plt.subplots(
                len(ALPHAS),
                len(DATASETS),
                figsize=(10, 8),
                sharex=True,
                sharey=True
            )

            if len(ALPHAS) == 1:
                axes = [axes]

            # ordem correta dos algoritmos
            ordered_algs = get_algorithm_order(df)

            # 🔥 extrai base_algs preservando ordem
            base_algs = []
            seen = set()

            for alg in ordered_algs:
                base = get_base_algorithm(alg)
                if base not in seen:
                    base_algs.append(base)
                    seen.add(base)

            color_map = {
                base_alg: plt.cm.tab10(i % 10)
                for i, base_alg in enumerate(base_algs)
            }

            # =====================================================
            # 🔹 MARKERS FIXOS POR ALGORITMO
            # =====================================================
            base_marker_map = {
                base_alg: BASE_MARKERS[i % len(BASE_MARKERS)]
                for i, base_alg in enumerate(base_algs)
            }

            # =====================================================
            # 🔹 PLOT
            # =====================================================
            for i, alpha in enumerate(sorted(ALPHAS)):
                for j, dataset in enumerate(DATASETS):

                    ax = axes[i][j]

                    df_alpha = df[df["alpha"] == alpha]

                    algs = sorted(df_alpha["algorithm"].unique())

                    for alg in algs:

                        subset = df_alpha[df_alpha["algorithm"] == alg]

                        if len(subset) == 0:
                            continue

                        # =====================================================
                        # 🔥 NOVO: MÉDIA DAS ÚLTIMAS X RODADAS
                        # =====================================================

                        subset_ds = subset[subset["dataset"] == dataset]

                        if len(subset_ds) == 0:
                            continue

                        # ordenar por rodada
                        subset_ds = subset_ds.sort_values("round")

                        # pegar rounds únicos
                        rounds = subset_ds["round"].unique()

                        if len(rounds) == 0:
                            continue

                        # últimas X rodadas
                        last_rounds = rounds[-LAST_N_ROUNDS:]

                        final_ds = subset_ds[
                            subset_ds["round"].isin(last_rounds)
                        ]

                        if len(final_ds) == 0:
                            continue

                        # média (rodadas + execuções)
                        fairness_val = final_ds[metric].mean() * 100
                        acc = final_ds["global_acc"].mean() * 100

                        if np.isnan(acc):
                            continue

                        # -------------------------
                        # β (se existir)
                        # -------------------------
                        match = re.search(r'b(\d+\.?\d*)', alg)
                        beta = float(match.group(1)) if match else None

                        # -------------------------
                        # COR
                        # -------------------------
                        base_alg = get_base_algorithm(alg)
                        color = color_map[base_alg]

                        # -------------------------
                        # MARKER
                        # -------------------------
                        if beta in BETA_MARKERS:
                            marker = BETA_MARKERS[beta]
                        else:
                            marker = base_marker_map[base_alg]

                        ax.scatter(
                            fairness_val,
                            acc,
                            marker=marker,
                            color=color,
                            edgecolors='black',
                            linewidths=0.5,
                            s=90
                        )

                    # -------------------------
                    # EIXOS
                    # -------------------------
                    ax.set_title(
                        f"$\\alpha={alpha}$ | {dataset.upper()}",
                        fontsize=14
                    )

                    if i == len(ALPHAS) - 1:
                        ax.set_xlabel(texts["fairness"] + " (%)", fontsize=14)

                    if j == 0:
                        ax.set_ylabel(texts["accuracy"] + " (%)", fontsize=14)

                    ax.set_xlim(0, 100)
                    ax.set_ylim(0, 100)
                    ax.grid()

            # =====================================================
            # 🔹 LEGENDA
            # =====================================================
            legend_handles = []
            legend_labels = []

            # β → shapes
            for beta, marker in BETA_MARKERS.items():
                handle = mlines.Line2D(
                    [], [],
                    color="black",
                    marker=marker,
                    linestyle='None',
                    markersize=10,
                    markerfacecolor='none',
                    markeredgewidth=1.5
                )
                legend_handles.append(handle)
                legend_labels.append(f"$\\beta={beta}$")

            # algoritmos → cores
            for base_alg in base_algs:

                handle = mlines.Line2D(
                    [], [],
                    color=color_map[base_alg],
                    marker='o',
                    linestyle='None',
                    markersize=10
                )

                legend_handles.append(handle)

                if base_alg == "dpfs":
                    legend_labels.append("DPFS")
                elif base_alg == "baseline":
                    legend_labels.append("MultiFedAvg")
                elif base_alg == "fedfairmmfl":
                    legend_labels.append("FedFairMMFL")
                else:
                    legend_labels.append(base_alg.capitalize())

            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.94),
                ncol=4,
                fontsize=12
            )

            metric_display = {
                "inter_client_fairness": "EPF-Inter",
                "intra_client_fairness": "EPF-Intra",
                "inter_model_fairness": "Inter-Model"
            }

            plt.suptitle(
                f"{texts['fairness']} vs {texts['accuracy']} "
                f"({metric_display.get(metric, metric)})",
                y=0.97,
                fontsize=16
            )

            plt.tight_layout(rect=[0, 0, 1, 0.90])

            save_figure(f"fairness_vs_accuracy_{metric}_{lang}")
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
            ci_total = np.sqrt((ci_cifar**2 + ci_gtsrb**2)) / 2

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
            f"\\caption{{{texts['table_accuracy_caption']} $\\alpha={ALPHAS}$.}}"
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

def build_fairness_table_multi_alpha(df):

    texts = TEXTS["pt"]

    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\renewcommand{\\arraystretch}{1.2}")
    lines.append("\\setlength{\\tabcolsep}{6pt}")
    lines.append("\\begin{tabular}{l|c|cc}")
    lines.append("\\toprule")

    lines.append(
        f"{texts['algorithm']} & {texts['mean_fairness_col']} & "
        f"{texts['inter_client']} & {texts['intra_client']} \\\\"
    )

    for alpha in sorted(df["alpha"].unique()):

        lines.append("\\midrule")
        lines.append(f"\\multicolumn{{4}}{{c}}{{$\\alpha = {alpha}$}} \\\\")
        lines.append("\\midrule")

        df_alpha = df[df["alpha"] == alpha]

        results = []

        for alg in get_algorithm_order(df_alpha):

            subset = df_alpha[df_alpha["algorithm"] == alg]

            if len(subset) == 0:
                continue

            inter = subset["inter_client_fairness"]
            intra = subset["intra_client_fairness"]

            mean_inter, ci_inter = compute_ci(inter)
            mean_intra, ci_intra = compute_ci(intra)

            mean_total = np.mean([mean_inter, mean_intra])
            ci_total = np.mean([ci_inter, ci_intra])

            results.append({
                "alg": alg,
                "inter_mean": mean_inter,
                "inter_ci": ci_inter,
                "intra_mean": mean_intra,
                "intra_ci": ci_intra,
                "total_mean": mean_total,
                "total_ci": ci_total,
            })

        df_res = pd.DataFrame(results)

        if df_res.empty:
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
        best_inter = mark_best("inter_mean", "inter_ci")
        best_intra = mark_best("intra_mean", "intra_ci")

        for i, row in df_res.iterrows():

            def fmt(mean, ci, bold):
                text = f"{mean:.2f} $\\pm$ {ci:.2f}"
                return f"\\textbf{{{text}}}" if bold else text

            total = fmt(row["total_mean"], row["total_ci"], i in best_total)
            inter = fmt(row["inter_mean"], row["inter_ci"], i in best_inter)
            intra = fmt(row["intra_mean"], row["intra_ci"], i in best_intra)

            alg_name = get_display_name(row["alg"])

            lines.append(f"{alg_name} & {total} & {inter} & {intra} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Comparação de fairness para diferentes valores de $\\alpha$.}")
    lines.append("\\label{tab:fairness}")
    lines.append("\\end{table}")

    with open(f"{RESULTS_DIR_WRITE}/fairness_multi_alpha.tex", "w") as f:
        f.write("\n".join(lines))

    print("✅ Tabela multi-alpha de fairness gerada (com destaque correto)")

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
            f"\\caption{{Final fairness (last round, without inter-model fairness). $\\alpha={ALPHAS}$.}}"
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
    plot_fairness_vs_accuracy(df)

    # TABELAS
    build_accuracy_table_multi_alpha(df)
    build_fairness_table_multi_alpha(df)
    build_resource_table(df)

    build_final_accuracy_table(df)
    build_final_fairness_table(df)

    print("\n✔ Tudo gerado com sucesso (EN + PT)")

main()