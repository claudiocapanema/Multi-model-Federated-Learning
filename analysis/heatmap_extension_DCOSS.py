import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
import copy


# ===============================
# 🔹 EXTRAIR ALPHA DO NOME
# ===============================

def extract_alpha_from_experiment(experiment_id):
    alpha_part = experiment_id.split("#")[1]
    first_alpha = alpha_part.split("-")[0]
    return float(first_alpha)


# ===============================
# 🔹 MÉDIA + IC
# ===============================

def mean_ci(values, ci=0.95):

    values = values.dropna().to_numpy()

    if len(values) == 0:
        return 0.0, 0.0

    mean = np.mean(values)

    if len(values) > 1:
        interval = st.t.interval(
            confidence=ci,
            df=len(values) - 1,
            loc=mean,
            scale=st.sem(values)
        )
        margin = mean - interval[0]
    else:
        margin = 0.0

    return mean, margin

def read_data(read_solutions, read_dataset_order):

    df_concat = None

    solution_strategy_version = {
        "MultiFedAvg+MFP_v2": {"Strategy": "MultiFedAvg", "Version": "MFP_v2", "Table": "$MultiFedAvg+MFP_{v2}$"},
        "MultiFedAvg+MFP_v2_dh": {"Strategy": "MultiFedAvg", "Version": "MFP_v2_dh", "Table": "$MultiFedAvg+MFP_{v2dh}$"},
        "MultiFedAvg+MFP_v2_iti": {"Strategy": "MultiFedAvg", "Version": "MFP_v2_iti", "Table": "$MultiFedAvg+MFP_{v2iti}$"},
        "MultiFedAvg+MFP": {"Strategy": "MultiFedAvg", "Version": "MFP", "Table": "MultiFedAvg+MFP"},
        "MultiFedAvg+FPD": {"Strategy": "MultiFedAvg", "Version": "FPD", "Table": "MultiFedAvg+FPD"},
        "MultiFedAvg+FP": {"Strategy": "MultiFedAvg", "Version": "FP", "Table": "MultiFedAvg+FP"},
        "DMA-FL": {"Strategy": "DMA-FL", "Version": "Original", "Table": "DMA-FL"},
        "AdaptiveFedAvg": {"Strategy": "AdaptiveFedAvg", "Version": "Original", "Table": "AdaptiveFedAvg"},
        "MultiFedAvg": {"Strategy": "MultiFedAvg", "Version": "Original", "Table": "MultiFedAvg"},
    }

    for solution in read_solutions:

        paths = read_solutions[solution]

        for i in range(len(paths)):

            try:
                dataset = read_dataset_order[i]
                path = paths[i]

                df = pd.read_csv(path)

                df["Solution"] = solution
                df["Accuracy (%)"] = df["Accuracy"] * 100
                df["Balanced accuracy (%)"] = df["Balanced accuracy"] * 100
                df["Dataset"] = dataset
                df["Table"] = solution_strategy_version[solution]["Table"]

                if df_concat is None:
                    df_concat = df
                else:
                    df_concat = pd.concat([df_concat, df])

            except Exception as e:
                print("Arquivo faltando:", paths[i])
                print(e)

    return df_concat

def read_data_multi_experiments(
    experiment_ids,
    solutions,
    datasets,
    total_clients,
    model_name,
    fraction_fit,
    number_of_rounds,
    local_epochs,
    train_test
):

    df_concat = None

    for experiment_id in experiment_ids:

        alpha_value = extract_alpha_from_experiment(experiment_id)
        alphas = [alpha_value] * len(datasets)

        read_solutions = {solution: [] for solution in solutions}
        read_dataset_order = []

        for solution in solutions:
            for dt in datasets:

                read_path = """../system/results/experiment_id_{}/clients_{}/alpha_{}/{}/{}/fc_{}/rounds_{}/epochs_{}/{}/""".format(
                    experiment_id,
                    total_clients,
                    alphas,
                    datasets,
                    model_name,
                    fraction_fit,
                    number_of_rounds,
                    local_epochs,
                    train_test
                )

                read_dataset_order.append(dt)

                read_solutions[solution].append(
                    "{}{}_{}.csv".format(read_path, dt, solution)
                )

        df_exp = read_data(read_solutions, read_dataset_order)

        if df_exp is None:
            continue

        df_exp["Scenario"] = experiment_id

        if df_concat is None:
            df_concat = df_exp
        else:
            df_concat = pd.concat([df_concat, df_exp])

    return df_concat


# =====================================================
# 🔹 PARSE TRANSIÇÃO
# =====================================================

def parse_transition(scenario):
    match = re.search(r"#(.*)_", scenario)
    alpha_part = match.group(1)
    a1, a2 = alpha_part.split("-")
    return float(a1), float(a2)


# =====================================================
# 🔹 COMPUTAR GANHOS
# =====================================================

def compute_all_gains(df, metric, baseline="MultiFedAvg"):

    records = []

    for scenario in df["Scenario"].unique():

        a1, a2 = parse_transition(scenario)
        delta = abs(np.log10(a2) - np.log10(a1))
        direction = "increase" if a2 > a1 else "decrease"

        for dataset in df["Dataset"].unique():

            base = df.query(
                f"Dataset == '{dataset}' and Solution == '{baseline}' and Scenario == '{scenario}'"
            )[metric].mean()

            for sol in df["Solution"].unique():

                val = df.query(
                    f"Dataset == '{dataset}' and Solution == '{sol}' and Scenario == '{scenario}'"
                )[metric].mean()

                if pd.isna(base) or base == 0:
                    gain = 0.0
                else:
                    gain = ((val - base) / base) * 100

                records.append({
                    "Dataset": dataset,
                    "Solution": sol,
                    "Scenario": scenario,
                    "Gain": gain,
                    "Delta": delta,
                    "Direction": direction
                })

    return pd.DataFrame(records)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats as st
import re

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import scipy.stats as st
import re


# =====================================================
# 🔹 MÉDIA + IC
# =====================================================

def mean_ci(values, ci=0.95):

    values = values.dropna().to_numpy()

    if len(values) == 0:
        return 0.0, 0.0

    mean = np.mean(values)

    if len(values) > 1:
        interval = st.t.interval(
            confidence=ci,
            df=len(values) - 1,
            loc=mean,
            scale=st.sem(values)
        )
        margin = mean - interval[0]
    else:
        margin = 0.0

    return mean, margin


# =====================================================
# 🔹 PARSE TRANSIÇÃO
# =====================================================

def parse_transition(scenario):
    match = re.search(r"#(.*)_", scenario)
    alpha_part = match.group(1)
    a1, a2 = alpha_part.split("-")
    return float(a1), float(a2)


# =====================================================
# 🔥 HEATMAP ATUALIZADO
# =====================================================

def generate_rich_heatmaps(df, metric, output_path, baseline="MultiFedAvg"):

    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path
    import scipy.stats as st
    import re

    def mean_ci(values, ci=0.95):
        values = values.dropna().to_numpy()
        if len(values) == 0:
            return 0.0, 0.0

        mean = np.mean(values)

        if len(values) > 1:
            interval = st.t.interval(
                confidence=ci,
                df=len(values) - 1,
                loc=mean,
                scale=st.sem(values)
            )
            margin = mean - interval[0]
        else:
            margin = 0.0

        return mean, margin

    def parse_transition(scenario):
        match = re.search(r"#(.*)_", scenario)
        alpha_part = match.group(1)
        a1, a2 = alpha_part.split("-")
        return float(a1), float(a2)

    save_dir = Path(output_path) / "heatmaps"
    save_dir.mkdir(parents=True, exist_ok=True)

    for dataset in df["Dataset"].unique():

        scenarios = df["Scenario"].unique()

        transitions = []
        for sc in scenarios:
            a1, a2 = parse_transition(sc)
            label = f"{a1:g}→{a2:g}"
            transitions.append((sc, a1, a2, label))

        transitions = sorted(transitions, key=lambda x: (x[1], x[2]))

        ordered_scenarios = [t[0] for t in transitions]
        x_labels = [t[3] for t in transitions]

        solutions = df["Table"].unique()

        # =====================================================
        # 🔹 MAPA DE NOMES CURTOS PARA O EIXO Y
        # =====================================================

        label_map = {
            "MultiFedAvg+MFP_v2": "MFPv2",
            "MultiFedAvg+MFP_v2_dh": "MFPv2-dh",
            "MultiFedAvg+MFP_v2_iti": "MFPv2-iti",
            "MultiFedAvg+MFP": "MFP",
            "MultiFedAvg+FPD": "FPD",
            "MultiFedAvg+FP": "FP",
            "DMA-FL": "DMA-FL",
            "AdaptiveFedAvg": "AdaptiveFedAvg",
            "MultiFedAvg": "MultiFedAvg"
        }

        solutions = df["Table"].unique()

        def shorten_label(name):
            name = name.replace("MultiFedAvg + ", "")
            name = name.replace("MultiFedAvg+", "")
            name = name.replace("MultiFedAvg", "MultiFedAvg")
            return name

        short_solutions = [shorten_label(s) for s in solutions]

        matrix_gain = []
        matrix_mean = []
        matrix_ci = []

        for sol in solutions:

            row_gain = []
            row_mean = []
            row_ci = []

            for scenario in ordered_scenarios:

                subset = df.query(
                    f"Dataset == '{dataset}' and Table == '{sol}' and Scenario == '{scenario}'"
                )[metric]

                base_subset = df.query(
                    f"Dataset == '{dataset}' and Table == '{baseline}' and Scenario == '{scenario}'"
                )[metric]

                mean_val, ci_val = mean_ci(subset)
                base_mean, _ = mean_ci(base_subset)

                gain = ((mean_val - base_mean) / base_mean) * 100 if base_mean != 0 else 0

                row_gain.append(gain)
                row_mean.append(mean_val)
                row_ci.append(ci_val)

            matrix_gain.append(row_gain)
            matrix_mean.append(row_mean)
            matrix_ci.append(row_ci)

        matrix_gain = np.array(matrix_gain)
        matrix_mean = np.array(matrix_mean)
        matrix_ci = np.array(matrix_ci)

        matrix_text = []

        for row in range(len(solutions)):
            matrix_text.append([""] * len(ordered_scenarios))

        for col in range(len(ordered_scenarios)):

            col_means = matrix_mean[:, col]
            best_idx = np.argmax(col_means)

            best_mean = matrix_mean[best_idx, col]
            best_ci = matrix_ci[best_idx, col]

            best_lower = best_mean - best_ci
            best_upper = best_mean + best_ci

            for row in range(len(solutions)):

                mean_val = matrix_mean[row, col]
                ci_val = matrix_ci[row, col]
                gain = matrix_gain[row, col]

                lower = mean_val - ci_val
                upper = mean_val + ci_val

                overlap = not (upper < best_lower or lower > best_upper)

                text = f"{mean_val:.2f}±{ci_val:.2f}\n({gain:+.2f}%)"

                # ✅ NÍVEL 2 CORRETO:
                # destacar se estatisticamente indistinguível do melhor
                if overlap:
                    text = "★ " + text

                matrix_text[row][col] = text

        matrix_text = np.array(matrix_text)

        # =====================================================
        # 🔥 PLOT OTIMIZADO PARA LATEX (MELHOR LEGIBILIDADE)
        # =====================================================

        plt.figure(figsize=(14, 8))

        ax = sns.heatmap(
            matrix_gain,
            annot=matrix_text,
            fmt="",
            cmap="coolwarm",
            center=0,
            xticklabels=x_labels,
            yticklabels=short_solutions,
            cbar_kws={"label": "Gain (%)"},
            annot_kws={"size": 13}
        )

        # Ajustes finos de fonte
        ax.set_xticklabels(x_labels, rotation=0, fontsize=12)
        ax.set_yticklabels(short_solutions, rotation=0, fontsize=12)

        plt.xlabel("Label Shift Transition (α₁→α₂)", fontsize=13)
        plt.ylabel("")  # removido para ganhar espaço
        plt.title(f"{dataset} - {metric}", fontsize=14)

        # Ajuste da colorbar
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=11)
        cbar.set_label("Gain (%)", fontsize=12)

        plt.tight_layout()

        import matplotlib.pyplot as plt

        plt.rcParams.update({
            "font.size": 14,  # tamanho base
            "axes.titlesize": 16,  # título
            "axes.labelsize": 15,  # labels dos eixos
            "xtick.labelsize": 13,  # eixo X
            "ytick.labelsize": 13,  # eixo Y
            "legend.fontsize": 13,
            "figure.titlesize": 17
        })

        filename_png = save_dir / f"{dataset}_{metric.replace(' ', '_')}_rich_heatmap.png"
        filename_pdf = save_dir / f"{dataset}_{metric.replace(' ', '_')}_rich_heatmap.pdf".replace("_(%)", "")

        plt.savefig(filename_png, dpi=300, bbox_inches="tight")
        plt.savefig(filename_pdf, format="pdf", bbox_inches="tight")

        plt.close()

        print("Heatmap salvo em:")
        print(" -", filename_png)
        print(" -", filename_pdf)
# =====================================================
# 📊 MAGNITUDE PLOT
# =====================================================

def generate_magnitude_plot(df_gain, metric, save_root):

    save_dir = save_root / "magnitude"
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=df_gain,
        x="Delta",
        y="Gain",
        hue="Direction"
    )

    plt.axhline(0, linestyle="--")
    plt.title("Gain (%) grouped by |Δ log10 α|")
    plt.tight_layout()

    filename = save_dir / f"{metric.replace(' ', '_')}_magnitude.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    print("Magnitude plot salvo em:", filename)


# =====================================================
# 📋 TABELA COMPACTA
# =====================================================
def generate_summary_table(df_gain, metric, output_path, baseline="MultiFedAvg"):

    import numpy as np
    import pandas as pd
    from pathlib import Path
    import re

    save_dir = Path(output_path) / "summary"
    save_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # 🔹 PADRONIZAÇÃO DOS NOMES
    # =====================================================

    name_map = {
        "MultiFedAvg+MFP_v2": "MFPv2",
        "MultiFedAvg+MFP_v2_dh": "MFPv2-dh",
        "MultiFedAvg+MFP_v2_iti": "MFPv2-iti",
        "MultiFedAvg+MFP": "MFP",
        "MultiFedAvg+FPD": "FPD",
        "MultiFedAvg+FP": "FP",
        "DMA-FL": "DMA-FL",
        # "AdaptiveFedAvg": "AdaptiveFedAvg",
        "MultiFedAvg": "MultiFedAvg"
    }

    df_gain["Solution"] = df_gain["Solution"].map(name_map)

    # =====================================================
    # 🔹 ORDENAR TRANSIÇÕES
    # =====================================================

    transition_info = []

    for sc in df_gain["Scenario"].unique():
        match = re.search(r"#(.*)_", sc)
        alpha_part = match.group(1)
        a1, a2 = alpha_part.split("-")
        label = f"{float(a1):g}→{float(a2):g}"
        transition_info.append((sc, float(a1), float(a2), label))

    transition_info = sorted(transition_info, key=lambda x: (x[1], x[2]))

    ordered_scenarios = [t[0] for t in transition_info]
    transition_labels = [t[3] for t in transition_info]

    # =====================================================
    # 🔹 SOLUÇÕES (baseline por último)
    # =====================================================

    all_solutions = list(name_map.values())
    non_baseline = [s for s in all_solutions if s != "MultiFedAvg"]
    expected_solutions = non_baseline + ["MultiFedAvg"]

    rows = []

    for sol in expected_solutions:

        row = {"Solution": sol}
        gains_all = []

        for sc, label in zip(ordered_scenarios, transition_labels):

            dataset_means = []

            for dataset in df_gain["Dataset"].unique():

                subset = df_gain[
                    (df_gain["Solution"] == sol) &
                    (df_gain["Scenario"] == sc) &
                    (df_gain["Dataset"] == dataset)
                ]["Gain"]

                if len(subset) > 0:
                    dataset_means.append(subset.mean())

            if len(dataset_means) == 0:
                mean_gain = 0.0
            else:
                mean_gain = np.mean(dataset_means)

            # 🔹 substituir 0 por "-"
            if round(mean_gain, 2) == 0.00:
                row[label] = "-"
            else:
                row[label] = f"{mean_gain:.2f}"

            gains_all.append(mean_gain)

        mean_global = np.mean(gains_all)

        # 🔹 substituir 0 por "-"
        if round(mean_global, 2) == 0.00:
            row["Mean Gain"] = "-"
        else:
            row["Mean Gain"] = f"{mean_global:.2f}"

        rows.append(row)

    df_summary = pd.DataFrame(rows)

    # =====================================================
    # 🔹 DESTACAR MELHOR POR COLUNA (exceto baseline)
    # =====================================================

    for label in transition_labels:

        means_only = []

        for _, row in df_summary.iterrows():
            if row["Solution"] == "MultiFedAvg":
                means_only.append(-np.inf)
            elif row[label] == "-":
                means_only.append(-np.inf)
            else:
                means_only.append(float(row[label]))

        max_value = max(means_only)

        for idx in df_summary.index:
            if df_summary.loc[idx, "Solution"] != "MultiFedAvg":
                cell = df_summary.loc[idx, label]

                if cell != "-" and float(cell) == max_value:
                    df_summary.loc[idx, label] = "\\textbf{" + cell + "}"

    # =====================================================
    # 🔹 GERAR LATEX
    # =====================================================

    n_cols = len(df_summary.columns)
    col_format = "l" + "c" * (n_cols - 1)

    header = " & ".join(df_summary.columns) + " \\\\"
    body = "\n".join(
        " & ".join(row.astype(str)) + " \\\\"
        for _, row in df_summary.iterrows()
    )

    latex_code = (
        "\\begin{table*}[t]\n"
        f"\\caption{{Average gain (\\%) over MultiFedAvg aggregated across datasets under label shift transitions.}}\n"
        f"\\label{{tab:gain_{metric.lower().replace(' ', '_').replace('(%)', '')}}}\n"
        "\\centering\n"
        f"\\begin{{tabular}}{{{col_format}}}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        
        "\\end{table*}"
    )

    latex_path = save_dir / f"{metric.replace(' ', '_')}_aggregated_summary.tex"

    with open(latex_path, "w") as f:
        f.write(latex_code)

    print("Tabela LaTeX salva em:", latex_path)

    return df_summary

# =====================================================
# 🚀 EXECUÇÃO COMPLETA
# =====================================================

def run_transition_analysis(df, metric, output_path):

    save_root = Path(output_path)
    save_root.mkdir(parents=True, exist_ok=True)

    df_gain = compute_all_gains(df, metric)

    generate_rich_heatmaps(
        df,
        "Accuracy (%)",
        analysis_path
    )
    generate_magnitude_plot(df_gain, metric, save_root)
    generate_summary_table(df_gain, metric, save_root)

    print("\n✅ Análise completa finalizada.\n")


# =====================================================
# 🔹 COMO USAR NO SEU MAIN
# =====================================================

if __name__ == "__main__":

    experiment_ids = [
        "label_shift#0.1-1.0_sudden",
        "label_shift#0.1-10.0_sudden",
        "label_shift#1.0-0.1_sudden",
        "label_shift#1.0-10.0_sudden",
        "label_shift#10.0-0.1_sudden",
        "label_shift#10.0-1.0_sudden"
    ]

    total_clients = 40
    datasets = ["WISDM-W", "ImageNet10", "Foursquare"]
    model_name = ["gru", "CNN", "lstm"]
    fraction_fit = 0.3
    number_of_rounds = 100
    local_epochs = 1
    train_test = "test"

    solutions = [
        "MultiFedAvg+MFP_v2",
        "MultiFedAvg+MFP_v2_dh",
        "MultiFedAvg+MFP_v2_iti",
        "MultiFedAvg+MFP",
        "MultiFedAvg+FPD",
        "MultiFedAvg+FP",
        "DMA-FL",
        # "AdaptiveFedAvg",
        "MultiFedAvg"
    ]

    write_path = "plots/MEFL/multi_experiments/"

    df = read_data_multi_experiments(
        experiment_ids,
        solutions,
        datasets,
        total_clients,
        model_name,
        fraction_fit,
        number_of_rounds,
        local_epochs,
        train_test
    )

    # Exemplo:
    analysis_path = "plots/MEFL/multi_experiments/analysis/"
    # run_transition_analysis(df, "Balanced accuracy (%)", analysis_path)
    run_transition_analysis(df, "Accuracy (%)", analysis_path)