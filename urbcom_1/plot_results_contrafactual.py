# =====================================================
# PARTE 1 — IMPORTS E CONFIGURAÇÃO
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# CONFIGURAÇÃO DO EXPERIMENTO
# =====================================================

VERSIONS = [
    "traditional",
    "remove_random",
    "remove_top_gamma_inter",
    "remove_top_gamma_intra"
]

REMOVAL_FRACTIONS = [0.0, 0.1, 0.2, 0.3]

MODELS = ["cifar", "gtsrb"]

# 🔥 MULTI-ALPHA
DIRICHLET_ALPHAS = [0.1, 1.0]

# 🔥 FRAÇÕES A EXIBIR NOS PLOTS (controle manual)
PLOT_REMOVAL_FRACTIONS = [0.0, 0.3]
# exemplos:
# [0.0, 0.3]
# [0.0, 0.1, 0.2]
# REMOVAL_FRACTIONS  # para todas

# 🔥 caminho base (ajuste se necessário)
BASE_ROOT = "results/gtsrb_4.0x_cifar/frac_0.3/"

# 🔥 saída dos plots
OUTPUT_DIR = Path("results/contrafactual/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# estilo visual
plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (8, 5),
    "axes.grid": True
})

# =====================================================
# PARTE 2 — LEITURA DOS CSVs
# =====================================================

def load_all_data():

    data = {}

    for alpha in DIRICHLET_ALPHAS:

        base_dir = f"{BASE_ROOT}/alpha_dirichlet_{alpha}/beta_1.0/"

        data[alpha] = {}

        for version in VERSIONS:
            data[alpha][version] = {}

            for frac in REMOVAL_FRACTIONS:
                data[alpha][version][frac] = {}

                for model in MODELS:

                    file = Path(base_dir) / f"baseline_contrafactual_{version}_frac{frac}_{model}.csv"

                    if file.exists():
                        df = pd.read_csv(file)

                        # média por rodada (caso tenha múltiplos folds)
                        df = df.groupby("round").mean(numeric_only=True).reset_index()

                        data[alpha][version][frac][model] = df
                    else:
                        print(f"[WARNING] Arquivo não encontrado: {file}")
                        data[alpha][version][frac][model] = None

    return data

# =====================================================
# ACCURACY VS TIME
# =====================================================

# =====================================================
# ACCURACY VS TIME — 1 GRÁFICO POR VERSÃO
# =====================================================

def plot_accuracy_vs_time_by_version(data, model):

    # 🔥 fallback: se não definir, usa todas
    fracs_to_plot = PLOT_REMOVAL_FRACTIONS if PLOT_REMOVAL_FRACTIONS else REMOVAL_FRACTIONS

    for version in VERSIONS:

        n = len(DIRICHLET_ALPHAS)
        fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharey=True)

        if n == 1:
            axes = [axes]

        for i, alpha in enumerate(DIRICHLET_ALPHAS):

            ax = axes[i]

            for frac in fracs_to_plot:

                df = data[alpha][version][frac][model]

                if df is None:
                    continue

                # 🔥 agregação correta
                df = df.groupby("round").agg({
                    "global_acc": "mean",
                    "total_training_time": "max"
                }).reset_index()

                # 🔥 ordenação correta
                df = df.sort_values("total_training_time")

                ax.plot(
                    df["total_training_time"],
                    df["global_acc"],
                    label=f"f={frac}",
                    linewidth=2
                )

            ax.set_title(f"{version} | α={alpha}")
            ax.set_xlabel("Tempo acumulado")

            if i == 0:
                ax.set_ylabel("Acurácia")

            ax.grid(True)

        # legenda dinâmica
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(fracs_to_plot))

        plt.tight_layout(rect=[0, 0.1, 1, 1])

        filepath = OUTPUT_DIR / f"accuracy_vs_time_{model}_{version}.png"
        plt.savefig(filepath, dpi=300)
        plt.close()

        print(f"[SALVO] {filepath}")

# =====================================================
# EFICIÊNCIA
# =====================================================

def plot_efficiency_all_alpha(data, model):

    n = len(DIRICHLET_ALPHAS)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharey=True)

    if n == 1:
        axes = [axes]

    for i, alpha in enumerate(DIRICHLET_ALPHAS):

        ax = axes[i]

        for version in VERSIONS:

            values = []

            for frac in REMOVAL_FRACTIONS:

                df = data[alpha][version][frac][model]

                if df is None:
                    values.append(np.nan)
                else:
                    if model == "cifar":
                        values.append(df["efficiency_mean_cifar"].iloc[-1])
                    else:
                        values.append(df["efficiency_mean_gtsrb"].iloc[-1])

            ax.plot(REMOVAL_FRACTIONS, values, marker='o', label=version)

        ax.set_title(f"α = {alpha}")
        ax.set_xlabel("Fração removida")

        if i == 0:
            ax.set_ylabel("Eficiência")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    filepath = OUTPUT_DIR / f"efficiency_{model}.png"
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"[SALVO] {filepath}")

def plot_accuracy_vs_removal_all_alpha(data, model):

    n = len(DIRICHLET_ALPHAS)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharey=True)

    if n == 1:
        axes = [axes]

    for i, alpha in enumerate(DIRICHLET_ALPHAS):

        ax = axes[i]

        for version in VERSIONS:

            values = []

            for frac in REMOVAL_FRACTIONS:

                df = data[alpha][version][frac][model]

                if df is None:
                    values.append(np.nan)
                else:
                    # 🔥 pega acurácia FINAL
                    final_acc = df["global_acc"].iloc[-1]
                    values.append(final_acc)

            ax.plot(
                REMOVAL_FRACTIONS,
                values,
                marker='o',
                label=version
            )

        ax.set_title(f"α = {alpha}")
        ax.set_xlabel("Fração removida")

        if i == 0:
            ax.set_ylabel("Acurácia final")

        ax.grid(True)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    filepath = OUTPUT_DIR / f"accuracy_vs_removal_{model}.png"
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"[SALVO] {filepath}")

# =====================================================
# ACCURACY DROP (CAUSAL)
# =====================================================

# =====================================================
# FAIRNESS
# =====================================================

def plot_inter_client_fairness(data):

    n = len(DIRICHLET_ALPHAS)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharey=True)

    if n == 1:
        axes = [axes]

    for i, alpha in enumerate(DIRICHLET_ALPHAS):

        ax = axes[i]

        for version in VERSIONS:

            values = []

            for frac in REMOVAL_FRACTIONS:

                df = data[alpha][version][frac]["cifar"]

                if df is None:
                    values.append(np.nan)
                else:
                    values.append(df["inter_client_fairness"].iloc[-1])

            ax.plot(REMOVAL_FRACTIONS, values, marker='o', label=version)

        ax.set_title(f"α = {alpha}")
        ax.set_xlabel("Fração removida")

        if i == 0:
            ax.set_ylabel("Inter-client fairness")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    plt.savefig(OUTPUT_DIR / "fairness_inter_client.png", dpi=300)
    plt.close()

def plot_intra_client_fairness(data):

    n = len(DIRICHLET_ALPHAS)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharey=True)

    if n == 1:
        axes = [axes]

    for i, alpha in enumerate(DIRICHLET_ALPHAS):

        ax = axes[i]

        for version in VERSIONS:

            values = []

            for frac in REMOVAL_FRACTIONS:

                df = data[alpha][version][frac]["cifar"]

                if df is None:
                    values.append(np.nan)
                else:
                    values.append(df["intra_client_fairness"].iloc[-1])

            ax.plot(REMOVAL_FRACTIONS, values, marker='o', label=version)

        ax.set_title(f"α = {alpha}")
        ax.set_xlabel("Fração removida")

        if i == 0:
            ax.set_ylabel("Intra-client fairness")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    plt.savefig(OUTPUT_DIR / "fairness_intra_client.png", dpi=300)
    plt.close()

def plot_inter_model_fairness(data):

    n = len(DIRICHLET_ALPHAS)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharey=True)

    if n == 1:
        axes = [axes]

    for i, alpha in enumerate(DIRICHLET_ALPHAS):

        ax = axes[i]

        for version in VERSIONS:

            values = []

            for frac in REMOVAL_FRACTIONS:

                df = data[alpha][version][frac]["cifar"]

                if df is None:
                    values.append(np.nan)
                else:
                    values.append(df["inter_model_fairness"].iloc[-1])

            ax.plot(REMOVAL_FRACTIONS, values, marker='o', label=version)

        ax.set_title(f"α = {alpha}")
        ax.set_xlabel("Fração removida")

        if i == 0:
            ax.set_ylabel("Inter-model fairness")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    plt.savefig(OUTPUT_DIR / "fairness_inter_model.png", dpi=300)
    plt.close()

def plot_efficiency_global(data):

    n = len(DIRICHLET_ALPHAS)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharey=True)

    if n == 1:
        axes = [axes]

    for i, alpha in enumerate(DIRICHLET_ALPHAS):

        ax = axes[i]

        for version in VERSIONS:

            values = []

            for frac in REMOVAL_FRACTIONS:

                df = data[alpha][version][frac]["cifar"]

                if df is None:
                    values.append(np.nan)
                else:
                    values.append(df["efficiency_global_mean"].iloc[-1])

            ax.plot(REMOVAL_FRACTIONS, values, marker='o', label=version)

        ax.set_title(f"α = {alpha}")
        ax.set_xlabel("Fração removida")

        if i == 0:
            ax.set_ylabel("Eficiência global")

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    plt.savefig(OUTPUT_DIR / "efficiency_global.png", dpi=300)
    plt.close()

# =====================================================
# MAIN
# =====================================================
def main():

    data = load_all_data()

    for model in MODELS:

        print(f"\n================ {model.upper()} ================\n")

        # 🔥 NOVO PLOT (principal)
        plot_accuracy_vs_time_by_version(data, model)

        # 🔹 mantidos
        plot_accuracy_vs_removal_all_alpha(data, model)

    # 🔹 métricas globais
    plot_efficiency_global(data)

    plot_inter_client_fairness(data)
    plot_intra_client_fairness(data)
    # plot_inter_model_fairness(data)


if __name__ == "__main__":
    main()

