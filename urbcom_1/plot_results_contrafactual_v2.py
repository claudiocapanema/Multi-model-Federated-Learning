import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# VISUAL ENCODING (UPGRADE)
# =====================================================

VERSION_COLOR = {
    "traditional": "black",
    "remove_random_inter": "blue",
    "remove_top_gamma_inter": "green",
    "remove_random_intra": "orange",
    "remove_top_gamma_intra": "red",
}

def get_linestyle_by_fraction(frac):
    if frac == 0.0:
        return "-"
    elif frac == 0.1:
        return "--"
    elif frac == 0.2:
        return "-."
    elif frac == 0.3:
        return ":"
    else:
        return "-"

# =====================================================
# CONFIG
# =====================================================

BASE_DIR = "results/"

MODES = ["data_only", "speed_only", "cost_only", "full"]

METRICS = {
    "inter": "inter_client_fairness",
    "intra": "intra_client_fairness"
}

# cores por removal_fraction
COLOR_MAP = {
    0.0: "black",
    0.1: "blue",
    0.2: "orange",
    0.3: "red",
}

dataset = "cifar"

# estilo por versão
def get_linestyle(version):
    if "random" in version:
        return "--"
    elif "top_gamma" in version:
        return "-"
    else:
        return ":"

# =====================================================
# LOAD DATA (NOVO - COMPATÍVEL COM SUA ESTRUTURA)
# =====================================================

def load_mode_data(mode):

    mode_dir = os.path.join(BASE_DIR, mode)
    dfs = []

    print(f"\n🔍 Carregando modo: {mode}")

    for root, _, files in os.walk(mode_dir):
        for f in files:

            if not f.endswith(".csv"):
                continue

            path = os.path.join(root, f)

            try:
                df = pd.read_csv(path)

                # 🔥 CORREÇÃO CRÍTICA
                if "dataset" in df.columns:
                    df = df[df["dataset"] == dataset]  # ou "gtsrb"

                # garantir colunas
                if "version" not in df.columns or "removal_fraction" not in df.columns:
                    continue

                df["removal_fraction"] = df["removal_fraction"].astype(float)
                df["mode"] = mode

                dfs.append(df)

            except Exception as e:
                print(f"⚠️ Erro ao ler {path}: {e}")

    if len(dfs) == 0:
        print(f"❌ Nenhum dado encontrado para {mode}")
        return None

    df_all = pd.concat(dfs, ignore_index=True)

    # DEBUG IMPORTANTE
    print("✔ versões encontradas:", df_all["version"].unique())
    print("✔ fractions encontradas:", sorted(df_all["removal_fraction"].unique()))

    return df_all

# =====================================================
# LOAD ALL
# =====================================================

data = {}

for mode in MODES:
    df = load_mode_data(mode)
    if df is not None:
        data[mode] = df

# =====================================================
# AGGREGATION (mean + std)
# =====================================================

def aggregate(df, metric):

    grouped = (
        df
        .groupby(["round", "removal_fraction", "version"])[metric]
        .agg(["mean", "std"])
        .reset_index()
    )

    return grouped

# =====================================================
# PLOT
# =====================================================

fig, axes = plt.subplots(
    2, 4,
    figsize=(24, 10),
    sharex=True,
    sharey="row"
)

for col, mode in enumerate(MODES):

    if mode not in data:
        for row in range(2):
            axes[row, col].set_title(f"{mode}\n(no data)")
        continue

    df_mode = data[mode]

    for row, (metric_name, metric_col) in enumerate(METRICS.items()):

        ax = axes[row, col]

        df_agg = aggregate(df_mode, metric_col)

        if df_agg.empty:
            ax.set_title(f"{mode}\n(empty)")
            continue

        # -----------------------------------
        # loop: (fraction, version)
        # -----------------------------------
        for (frac, version), df_sub in df_agg.groupby(["removal_fraction", "version"]):

            df_sub = df_sub.sort_values("round")

            x = df_sub["round"].values
            y = df_sub["mean"].values
            std = df_sub["std"].fillna(0).values

            # suavização
            y_smooth = pd.Series(y).rolling(5, min_periods=1).mean().values

            color = VERSION_COLOR.get(version, "gray")
            linestyle = get_linestyle_by_fraction(frac)

            label = None  # 🔥 NÃO usamos mais label aqui

            ax.plot(
                x,
                y_smooth,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=2
            )

            # error band
            ax.fill_between(
                x,
                y_smooth - std,
                y_smooth + std,
                color=color,
                alpha=0.1
            )

        # títulos
        if row == 0:
            ax.set_title(f"{mode}", fontsize=12)

        if col == 0:
            ax.set_ylabel(metric_name, fontsize=11)

        if row == 1:
            ax.set_xlabel("Round")

        ax.grid(True)

from matplotlib.lines import Line2D

# =====================================================
# LEGENDAS SEPARADAS
# =====================================================

# -------- Legenda de VERSION (cores) --------
version_handles = [
    Line2D([0], [0], color=color, lw=3, label=version)
    for version, color in VERSION_COLOR.items()
]

# -------- Legenda de FRACTION (linestyle) --------
fraction_values = sorted({0.0, 0.1, 0.2, 0.3})

fraction_handles = [
    Line2D(
        [0], [0],
        color="black",
        linestyle=get_linestyle_by_fraction(f),
        lw=2,
        label=f"f = {f}"
    )
    for f in fraction_values
]

# -------- posicionamento --------
legend1 = fig.legend(
    handles=version_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=5,
    fontsize=10,
    title="Version"
)

legend2 = fig.legend(
    handles=fraction_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.96),
    ncol=4,
    fontsize=10,
    title="Removal Fraction"
)

fig.add_artist(legend1)

plt.suptitle("Fairness vs Round (GLOBAL)", fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# =====================================================
# SAVE
# =====================================================

OUTPUT_DIR_PDF = "results/contrafactual/pdf"
OUTPUT_DIR_PNG = "results/contrafactual/png"

os.makedirs(OUTPUT_DIR_PDF, exist_ok=True)
os.makedirs(OUTPUT_DIR_PNG, exist_ok=True)

pdf_path = os.path.join(OUTPUT_DIR_PDF, "fairness_ablation.pdf")
png_path = os.path.join(OUTPUT_DIR_PNG, "fairness_ablation.png")

plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
plt.savefig(png_path, dpi=300, bbox_inches="tight")

print(f"✅ PDF salvo em: {pdf_path}")
print(f"✅ PNG salvo em: {png_path}")

# plt.show()