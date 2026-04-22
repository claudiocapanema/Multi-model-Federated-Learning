# =====================================================
# IMPORTS
# =====================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =====================================================
# CONFIG
# =====================================================

BASE_DIR = "results/"
MODE = "full"

SOLUTION_NAME = ["baseline_contrafactual_", "baseline_contrafactual_removed_flops", "baseline_contrafactual_uk_participacao", "baseline_contrafactual_uk_participacao_referencia_eficiency"][3]

METRICS = {
    "inter": "inter_client_fairness",
    "intra": "intra_client_fairness"
}

VERSION_COLOR = {
    "remove_random_inter": "blue",
    "remove_top_gamma_inter": "green",
    "remove_random_intra": "orange",
    "remove_top_gamma_intra": "red",
}

MODE_TITLE = {
    "data_only": "Data Heterogeneity",
    "speed_only": "System Heterogeneity",
    "cost_only": "Model Cost Heterogeneity",
    "full": "" # Joint Heterogeneity"
}

VERSION_LABEL = {
    "remove_random_inter": "Random Removal (Inter)",
    "remove_top_gamma_inter": "Efficiency-Based Removal (Inter)",
    "remove_random_intra": "Random Removal (Intra)",
    "remove_top_gamma_intra": "Efficiency-Based Removal (Intra)",
}

VERSION_LABEL_LATEX = {
    "remove_random_inter": "Random inter",
    "remove_top_gamma_inter": r"Top $\gamma$ inter",
    "remove_random_intra": "Random intra",
    "remove_top_gamma_intra": r"Top $\gamma$ intra",
}

VALID_FRACS = [0.1, 0.3]

MODES = ["data_only", "speed_only", "cost_only", "full"]


def get_baseline_version(version):
    if "inter" in version:
        return "remove_random_inter"
    elif "intra" in version:
        return "remove_random_intra"
    return None


def get_linestyle(frac):
    return {
        0.1: "--",
        0.2: "-.",
        0.3: ":"
    }.get(frac, "-")


# =====================================================
# LOAD DATA (ROBUSTO)
# =====================================================

def load_all_modes():

    data = {}

    for mode in MODES:

        mode_dir = os.path.join(BASE_DIR, mode)

        if not os.path.exists(mode_dir):
            print(f"⚠️ missing: {mode}")
            continue

        dfs = []

        for root, _, files in os.walk(mode_dir):

            for file in files:

                if not file.startswith(SOLUTION_NAME):
                    continue

                path = os.path.join(root, file)

                try:
                    df = pd.read_csv(path)

                    if df.empty:
                        continue

                    if "dataset" in df.columns:
                        df = df[df["dataset"].isin(["cifar", "gtsrb"])]

                    if df.empty:
                        continue

                    # version fallback
                    if "version" not in df.columns:
                        for v in VERSION_COLOR.keys():
                            if v in root:
                                df["version"] = v

                    # removal fallback
                    if "removal_fraction" not in df.columns:
                        for p in root.split(os.sep):
                            if p.startswith("removal_"):
                                df["removal_fraction"] = float(p.split("_")[1])

                    df["mode"] = mode
                    dfs.append(df)

                except:
                    continue

        if len(dfs) > 0:
            data[mode] = pd.concat(dfs, ignore_index=True)
            print(f"✔ loaded: {mode}")

    return data

# =====================================================
# COMPUTE RATIOS (CENTRAL)
# =====================================================

def compute_ratios(df):

    results = []

    for frac in sorted([f for f in df["removal_fraction"].unique() if f in VALID_FRACS]):

        df_frac = df[df["removal_fraction"] == frac]

        for version in df_frac["version"].unique():

            if "random" in version:
                continue

            baseline = get_baseline_version(version)

            df_base = df_frac[df_frac["version"] == baseline]
            df_v = df_frac[df_frac["version"] == version]

            if df_base.empty or df_v.empty:
                continue

            base_last = df_base[df_base["round"] == df_base["round"].max()]
            v_last = df_v[df_v["round"] == df_v["round"].max()]

            delta_inter = abs(
                base_last["inter_client_fairness"].mean() -
                v_last["inter_client_fairness"].mean()
            )

            delta_intra = abs(
                base_last["intra_client_fairness"].mean() -
                v_last["intra_client_fairness"].mean()
            )

            ratio = delta_inter / (delta_intra + 1e-12)

            results.append({
                "removal_fraction": frac,
                "version": version,
                "baseline": baseline,
                "delta_inter": delta_inter,
                "delta_intra": delta_intra,
                "ratio": ratio
            })

    return pd.DataFrame(results)

def plot_all_modes(data):

    # 🔥 filtra só modos que realmente têm dados
    available_modes = [m for m in MODES if m in data]

    n_cols = len(available_modes)

    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 6), sharex=True)

    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, mode in enumerate(available_modes):

        df = data[mode]

        for row, (name, col_metric) in enumerate(METRICS.items()):

            ax = axes[row, col]

            grouped = (
                df.groupby(["round", "removal_fraction", "version"])[col_metric]
                .mean()
                .reset_index()
            )

            for (frac, version), sub in grouped.groupby(["removal_fraction", "version"]):

                sub = sub.sort_values("round")

                y = sub[col_metric].rolling(5, min_periods=1).mean()

                ax.plot(
                    sub["round"],
                    y,
                    color=VERSION_COLOR.get(version, "gray"),
                    linestyle=get_linestyle(frac),
                    linewidth=2
                )

            # títulos
            if row == 0:
                ax.set_title(MODE_TITLE.get(mode, mode), fontsize=12)

            # y labels melhores
            if col == 0:
                if name == "inter":
                    ax.set_ylabel("EPF-Inter", fontsize=11)
                else:
                    ax.set_ylabel("EPF-Intra", fontsize=11)

            ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D

    # =============================
    # HANDLES
    # =============================

    solution_handles = [
        Line2D([0], [0], color=color, lw=3,
               label=VERSION_LABEL.get(version, version))
        for version, color in VERSION_COLOR.items()
    ]

    fraction_handles = [
        Line2D([0], [0], color="black", lw=2,
               linestyle=get_linestyle(f), label=f"f={f}")
        for f in [0.1, 0.2, 0.3]
    ]

    # separadores visuais (truque importante)
    separator = [Line2D([0], [0], color="none", label="")]

    # =============================
    # LEGENDA ÚNICA ORGANIZADA
    # =============================

    fig.legend(
        handles=solution_handles + separator + fraction_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.1),
        fontsize=9,
        title="Intervention (color) and Removal Fraction (line style)"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    # 🔥 espaço real para legenda
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(f"results/{SOLUTION_NAME}/", exist_ok=True)
    plt.savefig(
        f"results/{SOLUTION_NAME}/full_multimode_plot.png",
        dpi=300,
        bbox_inches="tight"
    )

def latex_main_table(df):

    lines = [r"""
\begin{table}[h]
\centering
\caption{Sensitivity and specificity of EPF metrics (mode = full)}
\label{tab:epf_main}
\begin{tabular}{lccccc}
\hline
Intervention & Baseline & $f$ & $\Delta$ Inter & $\Delta$ Intra & Effect \\
\hline
"""]

    for _, r in df.iterrows():
        name = VERSION_LABEL_LATEX.get(r["version"], r["version"])
        base = VERSION_LABEL_LATEX.get(r["baseline"], r["baseline"])

        lines.append(
            f"{name} & {base} & {r['removal_fraction']} & "
            f"{r['delta_inter']:.3f} & "
            f"{r['delta_intra']:.3f} & "
            f"{classify(r)} \\\\"
        )

    lines.append(r"""
\hline
\end{tabular}
\end{table}
""")

    return "\n".join(lines)

def latex_monotonic_table(df):

    lines = [r"""
\begin{table}[h]
\centering
\caption{Monotonic behavior under increasing removal (mode = full)}
\label{tab:epf_monotonic}
\begin{tabular}{lccc}
\hline
Intervention & $\Delta_{0.1}$ & $\Delta_{0.3}$ & Trend \\
\hline
"""]

    for v in df["version"].unique():

        sub = df[df["version"] == v]

        d01 = sub[sub["removal_fraction"] == 0.1]["delta_inter"]
        d03 = sub[sub["removal_fraction"] == 0.3]["delta_inter"]

        if d01.empty or d03.empty:
            continue

        d01 = d01.values[0]
        d03 = d03.values[0]

        if d03 > d01:
            trend = "Increasing"
        elif d03 < d01:
            trend = "Decreasing"
        else:
            trend = "Flat"

        lines.append(
            f"{VERSION_LABEL_LATEX.get(v, v)} & {d01:.3f} & {d03:.3f} & {trend} \\\\"
        )

    lines.append(r"""
\hline
\end{tabular}
\end{table}
""")

    return "\n".join(lines)

def classify(row):
    r = row["ratio"]

    if r > 2:
        return "Inter-dominant"
    elif r < 0.5:
        return "Intra-dominant"
    else:
        return "Coupled"


# =====================================================
# PIPELINE
# =====================================================

data = load_all_modes()

# plot (todos os modos)
plot_all_modes(data)

# tabelas (somente FULL)
if "full" not in data:
    raise ValueError("Modo 'full' não encontrado nos dados")

df_full = data["full"]

df_ratios = compute_ratios(df_full)

main_table = latex_main_table(df_ratios)
mono_table = latex_monotonic_table(df_ratios)

os.makedirs(f"results/{SOLUTION_NAME}/", exist_ok=True)

with open(f"results/{SOLUTION_NAME}/table_main.tex", "w") as f:
    f.write(main_table)

with open(f"results/{SOLUTION_NAME}/table_monotonic.tex", "w") as f:
    f.write(mono_table)

print(main_table)
print(mono_table)