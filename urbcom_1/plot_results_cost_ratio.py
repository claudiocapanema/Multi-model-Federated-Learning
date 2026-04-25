import os
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================

# ALPHA = 1.0
ALPHA = 0.1

COST_RATIOS = ["1.0x", "2.0x", "4.0x", "6.0x", "8.0x", "10.0x"]

RESULTS_BASE_DIR = "results"
OUTPUT_DIR = "results/plots_cost_ratio"

os.makedirs(OUTPUT_DIR, exist_ok=True)

SELECTED_ALGORITHMS = [
    "fair_resource_k0.3",
    "oort",
    "fairhetero",
    "fedbalancer",
    "fedfairmmfl_f0.3",
    "baseline_f0.3",
]

BASELINE_FRACS = [0.3]

# =====================================================
# HELPERS
# =====================================================

def parse_cost_ratio(s):
    return float(s.replace("x", "").replace(",", "."))

def get_display_name(alg):
    if alg == "oort":
        return "Oort"
    if alg == "fairhetero":
        return "FairHetero"
    if alg == "fedbalancer":
        return "FedBalancer"

    if alg.startswith("baseline_f"):
        return f"MultiFedAvg"

    if alg.startswith("fedfairmmfl_f"):
        return f"FedFairMMFL"

    if alg.startswith("fair_resource_k"):
        return "DPFS"

    return alg

def get_algorithm_order(df):
    return [alg for alg in SELECTED_ALGORITHMS if alg in df["algorithm"].unique()]

# =====================================================
# LOAD
# =====================================================

def load_results(cost_ratio):

    path = os.path.join(
        RESULTS_BASE_DIR,
        f"gtsrb_{cost_ratio}_cifar/frac_0.3/alpha_dirichlet_0.1/beta_1.0/"
    )

    if not os.path.exists(path):
        return None

    dfs = []

    for file in os.listdir(path):

        if not file.endswith(".csv"):
            continue

        full = os.path.join(path, file)
        print(full)

        df = pd.read_csv(full)

        # =========================
        # 🔥 IDENTIFICA ALGORITMO
        # =========================
        if file.startswith("fairhetero_"):
            df["algorithm"] = "fairhetero"

        elif file.startswith("fedfairmmfl_"):
            df["algorithm"] = "fedfairmmfl_f0.3"

        elif file.startswith("baseline_"):
            df["algorithm"] = "baseline_f0.3"

        elif file.startswith("oort_"):
            df["algorithm"] = "oort"

        elif file.startswith("fedbalancer_"):
            df["algorithm"] = "fedbalancer"

        elif file.startswith("proposta_"):
            df["algorithm"] = "fair_resource_k0.3"

        else:
            print(f"⚠️ Ignorando arquivo: {file}")
            continue

        # =========================
        # 🔥 GARANTE DATASET CORRETO
        # =========================
        if "dataset" not in df.columns:

            if "_cifar" in file:
                df["dataset"] = "cifar"

            elif "_gtsrb" in file:
                df["dataset"] = "gtsrb"

            else:
                raise ValueError(f"Dataset não identificado: {file}")

        dfs.append(df)

    if len(dfs) == 0:
        return None

    df = pd.concat(dfs, ignore_index=True)

    return df[df["algorithm"].isin(SELECTED_ALGORITHMS)]

# =====================================================
# PLOT (1 COLUNA, 3 LINHAS)
# =====================================================

def plot_all():

    fig, axes = plt.subplots(3, 1, figsize=(6, 10), sharex=True)

    metrics = [
        ("accuracy", "Accuracy (%)"),
        ("inter", "IRCPF-Inter (%)"),
        ("intra", "IRCPF-Intra (%)"),
    ]

    for i, (metric, ylabel) in enumerate(metrics):

        ax = axes[i]

        for alg in get_algorithm_order(df_all):

            sub = df_all[df_all["algorithm"] == alg]

            if len(sub) == 0:
                continue

            ax.plot(
                sub["cost"],
                sub[metric],
                marker="o",
                label=get_display_name(alg)
            )

        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 100)
        ax.grid(True)

    # eixo X com explicação clara
    axes[-1].set_xlabel("Cost Ratio (GTSRB / CIFAR-10)")

    # legenda única bem posicionada
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.975)
    )

    fig.suptitle(
        "Accuracy and Fairness vs Cost Ratio",
        fontsize=12,
        y=0.995
    )

    # ajuste fino de layout (evita sobreposição)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(os.path.join(OUTPUT_DIR, "combined_plot.png"))
    plt.savefig(os.path.join(OUTPUT_DIR, "combined_plot.pdf"))
    plt.close()

# =====================================================
# GERAR FIGURA
# =====================================================

# =====================================================
# EXTRAÇÃO (ÚLTIMA RODADA)
# =====================================================

def extract_metrics(df):

    # 🔥 evitar SettingWithCopyWarning
    df = df.copy()

    # =========================
    # CLEAN ROUND
    # =========================
    df["round"] = pd.to_numeric(df["round"], errors="coerce")
    df = df.dropna(subset=["round"])
    df["round"] = df["round"].astype(int)

    # =========================
    # CLEAN METRICS
    # =========================
    numeric_cols = [
        "global_acc",
        "inter_client_fairness",
        "intra_client_fairness"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # remove linhas inválidas
    df = df.dropna(subset=numeric_cols)

    rows = []

    # df["round"] = df["round"].astype(int)

    for alg in df["algorithm"].unique():

        sub = df[df["algorithm"] == alg]

        final = (
            sub.sort_values("round")
            .groupby("dataset")
            .tail(1)
        )

        rows.append({
            "algorithm": alg,
            "accuracy": final["global_acc"].mean() * 100,  # 🔥 só aqui
            "inter": final["inter_client_fairness"].mean() * 100,
            "intra": final["intra_client_fairness"].mean() * 100
        })

    return pd.DataFrame(rows)

# =====================================================
# BUILD DATASET GLOBAL
# =====================================================

all_data = []

for cost in COST_RATIOS:

    df = load_results(cost)

    print(df)
    # exit()

    if df is None:
        print(f"Sem dados: {cost}")
        continue

    metrics = extract_metrics(df)
    metrics["cost"] = parse_cost_ratio(cost)

    all_data.append(metrics)

df_all = pd.concat(all_data)
df_all = df_all.sort_values("cost")

# =====================================================
# PLOT (LINHAS = ALGORITMOS)
# =====================================================

def plot(metric, ylabel):

    plt.figure(figsize=(6,4))

    for alg in get_algorithm_order(df_all):

        sub = df_all[df_all["algorithm"] == alg]

        if len(sub) == 0:
            continue

        plt.plot(
            sub["cost"],
            sub[metric],
            marker="o",
            label=get_display_name(alg)
        )

    plt.xlabel("Cost Ratio")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Cost Ratio")

    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}.png"))
    plt.savefig(os.path.join(OUTPUT_DIR, f"{metric}.pdf"))
    plt.close()

    print(OUTPUT_DIR)

# =====================================================
# GERAR OS 3 GRÁFICOS
# =====================================================

plot("accuracy", "Accuracy")
plot("inter", "Inter-Client Fairness")
plot("intra", "Intra-Client Fairness")

print("✅ Pronto: gráficos com linhas por algoritmo!")

plot_all()

print("✅ Pronto: figura única com 3 subplots (1 coluna x 3 linhas)!")