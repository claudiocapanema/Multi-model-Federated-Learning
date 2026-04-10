import os
import pandas as pd
import matplotlib.pyplot as plt

# =====================================================
# CONFIG
# =====================================================

ALPHA = 1.0
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
        return f"MultiFedAvg ({int(float(alg.split('f')[-1])*100)}%)"

    if alg.startswith("fedfairmmfl_f"):
        return f"FedFairMMFL ({int(float(alg.split('f')[-1])*100)}%)"

    if alg.startswith("fair_resource_k"):
        return f"Fair Resource ({int(float(alg.split('k')[-1])*100)}%)"

    return alg

def get_algorithm_order(df):
    return sorted(df["algorithm"].unique())

# =====================================================
# LOAD
# =====================================================

def load_results(cost_ratio):

    path = os.path.join(RESULTS_BASE_DIR, f"gtsrb_{cost_ratio}_cifar")

    if not os.path.exists(path):
        return None

    dfs = []

    for file in os.listdir(path):

        if not file.endswith(".csv"):
            continue

        full = os.path.join(path, file)

        if file.startswith("baseline_") and f"alpha_{ALPHA}" in file:
            for frac in BASELINE_FRACS:
                if f"frac_{frac}" in file:
                    df = pd.read_csv(full)
                    df["algorithm"] = f"baseline_f{frac}"
                    dfs.append(df)

        elif file.startswith("proposta_k_") and f"alpha_{ALPHA}" in file:
            df = pd.read_csv(full)
            df["algorithm"] = "fair_resource_k0.3"
            dfs.append(df)

        elif file.startswith("fedfairmmfl_") and f"alpha_{ALPHA}" in file:
            df = pd.read_csv(full)
            df["algorithm"] = "fedfairmmfl_f0.3"
            dfs.append(df)

        elif file.startswith("oort_") and f"alpha_{ALPHA}" in file:
            df = pd.read_csv(full)
            df["algorithm"] = "oort"
            dfs.append(df)

        elif file.startswith("fairhetero_") and f"alpha_{ALPHA}" in file:
            df = pd.read_csv(full)
            df["algorithm"] = "fairhetero"
            dfs.append(df)

        elif file.startswith("fedbalancer_") and f"alpha_{ALPHA}" in file:
            df = pd.read_csv(full)
            df["algorithm"] = "fedbalancer"
            dfs.append(df)

    if len(dfs) == 0:
        return None

    df = pd.concat(dfs, ignore_index=True)

    return df[df["algorithm"].isin(SELECTED_ALGORITHMS)]

# =====================================================
# EXTRAÇÃO (ÚLTIMA RODADA)
# =====================================================

def extract_metrics(df):

    rows = []

    for alg in df["algorithm"].unique():

        sub = df[df["algorithm"] == alg]

        final = (
            sub.sort_values("round")
            .groupby("dataset")
            .tail(1)
        )

        rows.append({
            "algorithm": alg,
            "accuracy": final["global_acc"].mean(),
            "inter": final["inter_client_fairness"].mean(),
            "intra": final["intra_client_fairness"].mean()
        })

    return pd.DataFrame(rows)

# =====================================================
# BUILD DATASET GLOBAL
# =====================================================

all_data = []

for cost in COST_RATIOS:

    df = load_results(cost)

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

# =====================================================
# GERAR OS 3 GRÁFICOS
# =====================================================

plot("accuracy", "Accuracy")
plot("inter", "Inter-Client Fairness")
plot("intra", "Intra-Client Fairness")

print("✅ Pronto: gráficos com linhas por algoritmo!")