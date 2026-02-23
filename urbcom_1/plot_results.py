# =====================================================
# PLOTS COMPLETOS — BASELINE vs RAWCS (ROBUSTO)
# =====================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

REGIME = "realistic"
REGIME = "severe"
ALPHA = 0.1
FRAC = 0.3

DATASETS = ["cifar", "gtsrb"]

FIGURES_DIR = f"figures/regime_{REGIME}_alpha_{ALPHA}"
os.makedirs(FIGURES_DIR, exist_ok=True)


# =====================================================
# Carregar CSV com tolerância
# =====================================================

def load_results(algorithm, dataset):

    if algorithm == "proposta":
        path = os.path.join(
            RESULTS_DIR,
            f"{algorithm}_{dataset}_regime_{REGIME}_alpha_{ALPHA}.csv"
        )
    else:
        path = os.path.join(
            RESULTS_DIR,
            f"{algorithm}_{dataset}_regime_{REGIME}_frac_{FRAC}_alpha_{ALPHA}.csv"
        )

    if not os.path.exists(path):
        print(f"⚠️ Arquivo não encontrado: {path}")
        return None

    df = pd.read_csv(path)

    if df.empty:
        print(f"⚠️ Arquivo vazio: {path}")
        return None

    return df


# =====================================================
# Agregação por rodada (tolerante)
# =====================================================

def aggregate_by_round(df):

    if df is None:
        return None

    # Apenas colunas que existirem
    agg_columns = {
        "global_acc": "mean",
        "clients_selected": "mean",
        "clients_selected_total": "mean",
        "viable_clients_total": "mean",
        "viable_pairs_total": "mean",
        "energy_consumed_round": "mean",
        "cumulative_energy": "mean",
        "drained_clients_round": "mean",
        "avg_battery_remaining": "mean"
    }

    existing_columns = {
        col: agg_columns[col]
        for col in agg_columns
        if col in df.columns
    }

    grouped = (
        df.groupby("round")
        .agg(existing_columns)
        .reset_index()
    )

    return grouped


# =====================================================
# Plot tolerante a rodadas faltantes
# =====================================================

def plot_metric(baseline, rawcs, column, ylabel, title, filename):

    if baseline is None and rawcs is None:
        print(f"⚠️ Nada para plotar: {title}")
        return

    plt.figure(figsize=(8, 5))

    # Determinar todas as rodadas possíveis
    rounds = set()

    if baseline is not None and column in baseline.columns:
        rounds.update(baseline["round"])

    if rawcs is not None and column in rawcs.columns:
        rounds.update(rawcs["round"])

    if not rounds:
        print(f"⚠️ Coluna ausente em ambos: {column}")
        return

    rounds = sorted(rounds)

    # Plot baseline
    if baseline is not None and column in baseline.columns:
        baseline_interp = (
            baseline.set_index("round")
            .reindex(rounds)
        )
        plt.plot(rounds, baseline_interp[column], label="Baseline")

    # Plot rawcs
    if rawcs is not None and column in rawcs.columns:
        rawcs_interp = (
            rawcs.set_index("round")
            .reindex(rounds)
        )
        plt.plot(rounds, rawcs_interp[column], label="RAWCS")

    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"✅ Salvo: {save_path}")


# =====================================================
# EXECUÇÃO
# =====================================================

def main():

    for dataset in DATASETS:

        print(f"\n📊 Gerando gráficos para {dataset.upper()}")

        baseline_df = load_results("baseline", dataset)
        rawcs_df = load_results("proposta", dataset)

        baseline_agg = aggregate_by_round(baseline_df)
        rawcs_agg = aggregate_by_round(rawcs_df)

        metrics = [
            ("global_acc", "Accuracy", "accuracy"),
            ("clients_selected", "Clients Selected", "clients_selected"),
            ("viable_clients_total", "Viable Clients", "viable_clients"),
            ("viable_pairs_total", "Viable Pairs", "viable_pairs"),
            ("drained_clients_round", "Drained Clients", "drained_clients"),
            ("energy_consumed_round", "Energy per Round", "energy_per_round"),
            ("cumulative_energy", "Cumulative Energy", "cumulative_energy"),
            ("avg_battery_remaining", "Average Battery", "avg_battery")
        ]

        for column, ylabel, fname in metrics:

            plot_metric(
                baseline_agg,
                rawcs_agg,
                column,
                ylabel,
                f"{ylabel} per Round — {dataset.upper()}",
                f"{dataset}_{fname}.png"
            )


if __name__ == "__main__":
    main()