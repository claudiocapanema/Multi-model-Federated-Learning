from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
import os

import copy

from base_plots import bar_plot, line_plot, ecdf_plot
import matplotlib.pyplot as plt

def read_data(
    read_solutions,
    solution_names=None,
    experiment_id=None,
    alpha_value=None
):
    """
    Lê os CSVs de shift detection metrics de todas as soluções
    e concatena os resultados em um único DataFrame.

    Parameters
    ----------
    read_solutions : dict
        Dicionário no formato:

        {
            "FedConD": [path1, path2, ...],
            "FedDrift": [path1, path2, ...],
            ...
        }

    solution_names : dict, optional
        Mapeamento entre nome interno e nome apresentado na tabela.

    experiment_id : str, optional
        Identificador do experimento.

    alpha_value : float, optional
        Valor de alpha utilizado no experimento.

    Returns
    -------
    pd.DataFrame
        Dados concatenados de todas as soluções.
    """

    df_concat = None

    if solution_names is None:
        solution_names = {
            solution: solution
            for solution in read_solutions.keys()
        }

    for solution, paths in read_solutions.items():

        for path in paths:

            try:

                if not os.path.exists(path):
                    print("\n#########")
                    print(f"Arquivo não encontrado: {path}")
                    continue

                df = pd.read_csv(path)
                print(df)

                if df.empty:
                    print(f"\nArquivo vazio: {path}")
                    continue

                # --------------------------------------------------
                # Identificação da solução
                # --------------------------------------------------

                df["Detector"] = solution

                df["Table"] = solution_names.get(
                    solution,
                    solution
                )

                # --------------------------------------------------
                # Metadados do experimento
                # --------------------------------------------------

                if experiment_id is not None:
                    df["Experiment ID"] = experiment_id

                if alpha_value is not None:

                    if isinstance(alpha_value, (tuple, list)):

                        # Label Shift: alpha_before -> alpha_after
                        df["Alpha Before"] = alpha_value[0]
                        df["Alpha After"] = alpha_value[1]

                    else:

                        # Concept Drift
                        df["Alpha"] = alpha_value

                # --------------------------------------------------
                # Tipos numéricos
                # --------------------------------------------------

                numeric_columns = [
                    "Fold ID",
                    "Round",
                    "Model",
                    "Precision",
                    "Recall",
                    "F1",
                    "Detection Delay",
                    "False Alarms",
                    "First Detection Round",
                    "Shift Round",
                    "Alpha",
                    "Alpha Before",
                    "Alpha After"
                ]

                for col in numeric_columns:

                    if col in df.columns:

                        df[col] = pd.to_numeric(
                            df[col],
                            errors="coerce"
                        )

                # --------------------------------------------------
                # Tipos categóricos
                # --------------------------------------------------

                if "Shift Configuration" in df.columns:

                    df["Shift Configuration"] = (
                        df["Shift Configuration"]
                        .astype(str)
                    )

                if "Shift Type" in df.columns:

                    df["Shift Type"] = (
                        df["Shift Type"]
                        .astype(str)
                    )

                if "Dataset" in df.columns:

                    df["Dataset"] = (
                        df["Dataset"]
                        .astype(str)
                    )

                # --------------------------------------------------
                # Adicionar ao DataFrame global
                # --------------------------------------------------

                if df_concat is None:

                    df_concat = df.copy()

                else:

                    df_concat = pd.concat(
                        [df_concat, df],
                        ignore_index=True
                    )

            except Exception as e:

                print("\n#########")
                print(f"Erro lendo: {path}")
                print(e)

    if df_concat is None:

        print("\nNenhum arquivo foi carregado.")

        return pd.DataFrame()

    return df_concat

def format_shift_configuration(shift_type, experiment_id):
    """
    Formata a configuração do shift a partir do Experiment ID.

    Concept:
        concept_drift#0.1_sudden
        -> $\\alpha=0.1$

    Label:
        label_shift#0.1-1.0_sudden
        -> $0.1 \\rightarrow 1.0$
    """

    if pd.isna(experiment_id):
        return "--"

    experiment_id = str(experiment_id).strip()

    if "#" not in experiment_id:
        return experiment_id.replace("_", r"\_")

    config = experiment_id.split("#", 1)[1]

    # Remove "_sudden"
    config = config.split("_sudden", 1)[0]

    # ------------------------------------------------------------
    # Concept Drift
    # ------------------------------------------------------------

    if str(shift_type).lower() == "concept":

        try:
            alpha = float(config)
            return rf"$\alpha={alpha:g}$"
        except ValueError:
            return config.replace("_", r"\_")

    # ------------------------------------------------------------
    # Label Shift
    # ------------------------------------------------------------

    if str(shift_type).lower() == "label":

        if "-" in config:

            alpha_before, alpha_after = config.split(
                "-",
                1
            )

            try:

                alpha_before = float(alpha_before)
                alpha_after = float(alpha_after)

                return (
                    rf"${alpha_before:g}"
                    rf"\rightarrow "
                    rf"{alpha_after:g}$"
                )

            except ValueError:
                pass

    return config.replace("_", r"\_")

def select_final_detection_results(df):
    """
    Seleciona o resultado final de cada unidade experimental.

    A unidade experimental é:

        Detector
        Dataset
        Fold ID
        Model
        Shift Type
        Shift Configuration

    Para cada unidade experimental, seleciona a linha correspondente
    à maior rodada disponível.

    IMPORTANTE
    ----------
    Os arquivos de detecção do MultiFedAvg+MFP contêm métricas
    cumulativas: cada rodada representa o estado acumulado do detector
    até aquela rodada.

    Portanto, para o MFP, a linha da maior rodada é justamente o
    resultado final do experimento e deve ser utilizada diretamente,
    sem recalcular as métricas a partir das rodadas anteriores.
    """

    required_columns = [
        "Detector",
        "Dataset",
        "Fold ID",
        "Round",
        "Model",
        "Shift Type",
        "Shift Configuration"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "Colunas obrigatórias ausentes: "
            + ", ".join(missing)
        )

    if df.empty:
        return df.copy()

    df_work = df.copy()

    # ============================================================
    # 1. Normalizar Round
    # ============================================================

    df_work["Round"] = pd.to_numeric(
        df_work["Round"],
        errors="coerce"
    )

    # Remover linhas sem rodada válida
    df_work = df_work[
        df_work["Round"].notna()
    ].copy()

    if df_work.empty:
        return df_work

    # ============================================================
    # 2. Normalizar identificadores da unidade experimental
    # ============================================================

    string_columns = [
        "Detector",
        "Dataset",
        "Shift Type",
        "Shift Configuration"
    ]

    for column in string_columns:
        df_work[column] = (
            df_work[column]
            .astype(str)
            .str.strip()
        )

    # ============================================================
    # 3. Model e Fold ID
    # ============================================================

    # Não converter Model para string, pois os CSVs atuais utilizam
    # o índice inteiro do modelo.

    # Fold ID pode eventualmente vir como string numérica.
    if "Fold ID" in df_work.columns:
        fold_numeric = pd.to_numeric(
            df_work["Fold ID"],
            errors="coerce"
        )

        # Só substituir quando a conversão for possível.
        mask = fold_numeric.notna()

        df_work.loc[
            mask,
            "Fold ID"
        ] = fold_numeric[mask]

    # ============================================================
    # 4. Unidade experimental
    # ============================================================

    group_columns = [
        "Detector",
        "Dataset",
        "Fold ID",
        "Model",
        "Shift Type",
        "Shift Configuration"
    ]

    # ============================================================
    # 5. Ordenar por rodada
    # ============================================================

    df_work = df_work.sort_values(
        by=group_columns + ["Round"],
        kind="mergesort"
    ).copy()

    # ============================================================
    # 6. Selecionar somente a última rodada
    # ============================================================

    df_final = (
        df_work
        .groupby(
            group_columns,
            as_index=False,
            sort=False
        )
        .tail(1)
        .copy()
    )

    # ============================================================
    # 7. Restaurar índice
    # ============================================================

    df_final.reset_index(
        drop=True,
        inplace=True
    )

    # ============================================================
    # 8. DEBUG específico do MultiFedPredict
    # ============================================================

    mfp_mask = (
        df_final["Detector"]
        .astype(str)
        .str.strip()
        .eq("MultiFedAvg+MFP")
    )

    if mfp_mask.any():

        print(
            "\n"
            + "=" * 80
        )

        print(
            "DEBUG - RESULTADOS FINAIS DO MULTIFEDAVG+MFP"
        )

        print(
            "=" * 80
        )

        debug_columns = [
            "Detector",
            "Dataset",
            "Fold ID",
            "Round",
            "Model",
            "Shift Type",
            "Shift Configuration"
        ]

        debug_metrics = [
            column
            for column in [
                "Precision",
                "Recall",
                "F1",
                "Detection Delay",
                "False Alarms"
            ]
            if column in df_final.columns
        ]

        print(
            df_final.loc[
                mfp_mask,
                debug_columns + debug_metrics
            ].to_string(
                index=False
            )
        )

        print(
            "=" * 80
            + "\n"
        )

    return df_final

def mean_ci(values, ci=0.95, bounded=False):
    """
    Calcula a média e a margem do intervalo de confiança.

    Parameters
    ----------
    values : array-like
        Valores numéricos da métrica.

    ci : float
        Nível de confiança.

    bounded : bool
        Se True, limita o intervalo ao domínio [0, 1].
        Usado para Precision, Recall e F1.

    Returns
    -------
    mean, margin

    Notes
    -----
    Esta função é estatística e não contém regras específicas
    de nenhuma métrica.

    Em particular, valores -1 não são removidos aqui.
    O tratamento de valores especiais deve ser feito antes
    da chamada desta função, de acordo com a semântica da métrica.
    """

    values = pd.to_numeric(
        values,
        errors="coerce"
    ).dropna().to_numpy(dtype=float)

    if len(values) == 0:
        return np.nan, np.nan

    mean = np.mean(values)

    # Apenas uma observação
    if len(values) == 1:
        return round(mean, 2), 0.00

    # Todos os valores são iguais
    if np.allclose(values, values[0]):
        return round(mean, 2), 0.00

    # Erro padrão da média
    sem = st.sem(values)

    # Intervalo de confiança baseado na distribuição t
    interval = st.t.interval(
        confidence=ci,
        df=len(values) - 1,
        loc=mean,
        scale=sem
    )

    lower, upper = interval

    # Métricas limitadas a [0, 1]
    if bounded:
        lower = max(0.0, lower)
        upper = min(1.0, upper)

    # A tabela utiliza:
    #
    #     mean ± margin
    #
    # Portanto, usamos a maior distância entre a média
    # e os limites do intervalo.
    margin = max(
        mean - lower,
        upper - mean
    )

    return round(mean, 2), round(margin, 2)

def prepare_detection_metric_values(
    df,
    metric
):
    """
    Prepara os valores de uma métrica de detecção para
    agregação estatística.

    Parameters
    ----------
    df : pd.DataFrame
        Resultados finais dos experimentos.

    metric : str
        Nome da métrica.

    Returns
    -------
    pd.Series
        Valores válidos para a métrica.

    Notes
    -----
    Detection Delay:
        -1 significa que o shift não foi detectado.
        Esses valores NÃO participam do cálculo do delay.

    Undetected Shift Rate:
        Não retorna valores diretamente. Essa métrica deve
        ser calculada pela função calculate_undetected_shift_rate().
    """

    values = pd.to_numeric(
        df[metric],
        errors="coerce"
    )

    if metric == "Detection Delay":
        values = values[
            values >= 0
        ]

    return values.dropna()

def calculate_undetected_shift_rate(df):
    """
    Calcula a taxa de shifts não detectados.

    Definition
    ----------
        Undetected Shift Rate =
            N_undetected / N_shifts

    No CSV:
        Detection Delay >= 0 -> shift detectado
        Detection Delay == -1 -> shift não detectado

    Returns
    -------
    float
        Valor entre 0 e 1.

    Notes
    -----
    Cada linha de df representa uma unidade experimental:

        Detector × Dataset × Fold × Model × Shift Configuration

    Portanto, cada linha corresponde a um shift ground-truth
    que deve ser classificado como detectado ou não detectado.
    """

    if df.empty:
        return np.nan

    delays = pd.to_numeric(
        df["Detection Delay"],
        errors="coerce"
    ).dropna()

    if len(delays) == 0:
        return np.nan

    undetected = np.sum(
        delays < 0
    )

    total = len(delays)

    return float(
        undetected / total
    )

def calculate_detection_metric(
    df,
    metric,
    ci=0.95
):
    """
    Calcula média e intervalo de confiança para uma métrica
    de detecção.

    Parameters
    ----------
    df : pd.DataFrame
        Resultados finais dos experimentos.

    metric : str
        Métrica a ser calculada.

    ci : float
        Nível de confiança.

    Returns
    -------
    mean, margin

    Notes
    -----
    Detection Delay:
        calculado somente sobre shifts detectados.

    Undetected Shift Rate:
        calculado sobre todos os shifts.

    Precision, Recall e F1:
        calculados sobre todos os experimentos.

    False Alarms:
        calculado sobre todos os experimentos.
    """

    if df.empty:
        return np.nan, np.nan

    if metric == "Undetected Shift Rate":

        rate = calculate_undetected_shift_rate(df)

        if pd.isna(rate):
            return np.nan, np.nan

        return round(rate, 2), 0.00

    if metric == "Detection Delay":

        values = prepare_detection_metric_values(
            df,
            metric
        )

    else:

        if metric not in df.columns:
            return np.nan, np.nan

        values = pd.to_numeric(
            df[metric],
            errors="coerce"
        ).dropna()

    bounded = metric in [
        "Precision",
        "Recall",
        "F1",
        "Undetected Shift Rate"
    ]

    return mean_ci(
        values,
        ci=ci,
        bounded=bounded
    )

def table_detection_quality(
    df,
    write_path,
    solutions_order,
    metrics=None,
    ci=0.95
):
    """
    Gera tabelas quantitativas de qualidade da detecção.

    Métricas:

        Precision
        Recall
        F1
        Detection Delay
        Undetected Shift Rate
        False Alarms

    Detection Delay:
        calculado somente sobre shifts detectados.

    Undetected Shift Rate:
        proporção de shifts ground-truth que não foram
        detectados.

    A unidade experimental é:

        Detector × Dataset × Fold × Model × Shift Configuration

    O CSV contém várias rodadas, mas somente a última rodada
    de cada experimento é utilizada na tabela.
    """

    if metrics is None:
        metrics = [
            "Precision",
            "Recall",
            "F1",
            "Detection Delay",
            "Undetected Shift Rate",
            "False Alarms"
        ]

    Path(write_path).mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # 1. Selecionar somente o resultado final de cada experimento
    # ------------------------------------------------------------

    df_final = select_final_detection_results(
        df
    )

    print("\nResultados finais selecionados:")

    print(
        df_final[
            [
                "Detector",
                "Dataset",
                "Fold ID",
                "Model",
                "Shift Type",
                "Shift Configuration",
                "Round"
            ]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------
    # 2. Tipos de shift
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Padronizar nomes dos tipos de shift para a tabela
    # ------------------------------------------------------------
    if "Shift Type" in df_final.columns:
        df_final["Shift Type"] = (
            df_final["Shift Type"]
            .apply(normalize_shift_type_for_table)
        )

    shift_types = sorted(
        df_final["Shift Type"]
        .dropna()
        .unique()
    )

    # ------------------------------------------------------------
    # 3. Soluções existentes na ordem solicitada
    # ------------------------------------------------------------

    solutions = [
        solution
        for solution in solutions_order
        if solution in df_final["Detector"].unique()
    ]

    # ------------------------------------------------------------
    # 4. Gerar uma tabela para cada tipo de shift e métrica
    # ------------------------------------------------------------

    for shift_type in shift_types:

        df_shift = df_final[
            df_final["Shift Type"] == shift_type
        ].copy()

        if df_shift.empty:
            continue

        for metric in metrics:

            if (
                metric != "Undetected Shift Rate"
                and metric not in df_shift.columns
            ):
                print(
                    f"\nMétrica ausente: {metric}"
                )
                continue

            # ====================================================
            # 5. Calcular média e IC
            # ====================================================

            rows_raw = {}

            for solution in solutions:

                filtered_solution = df_shift[
                    df_shift["Detector"] == solution
                ]

                if filtered_solution.empty:
                    continue

                mean, ci_margin = calculate_detection_metric(
                    filtered_solution,
                    metric,
                    ci=ci
                )

                rows_raw[solution] = {
                    "mean": mean,
                    "ci": ci_margin
                }

            # ====================================================
            # 6. Determinar direção da métrica
            # ====================================================

            higher_is_better = metric in [
                "Precision",
                "Recall",
                "F1"
            ]

            # Detection Delay,
            # Undetected Shift Rate e
            # False Alarms:
            #
            # menor é melhor.

            valid_solutions = [
                solution
                for solution in solutions
                if solution in rows_raw
                and not pd.isna(
                    rows_raw[solution]["mean"]
                )
            ]

            if not valid_solutions:
                continue

            # ====================================================
            # 7. Identificar melhor resultado
            # ====================================================

            if higher_is_better:

                best_solution = max(
                    valid_solutions,
                    key=lambda solution:
                        rows_raw[solution]["mean"]
                )

            else:

                best_solution = min(
                    valid_solutions,
                    key=lambda solution:
                        rows_raw[solution]["mean"]
                )

            best_mean = rows_raw[
                best_solution
            ]["mean"]

            best_ci = rows_raw[
                best_solution
            ]["ci"]

            best_lower = best_mean - best_ci
            best_upper = best_mean + best_ci

            # ====================================================
            # 8. Verificar sobreposição dos ICs
            # ====================================================

            for solution in valid_solutions:

                mean_val = rows_raw[
                    solution
                ]["mean"]

                ci_val = rows_raw[
                    solution
                ]["ci"]

                for solution in valid_solutions:
                    mean_val = rows_raw[
                        solution
                    ]["mean"]

                    rows_raw[
                        solution
                    ]["bold"] = (
                            best_lower
                            <= mean_val
                            <= best_upper
                    )

            # ====================================================
            # 9. Construir DataFrame da tabela
            # ====================================================

            rows_final = []

            for solution in solutions:

                if solution not in rows_raw:
                    continue

                mean_val = rows_raw[
                    solution
                ]["mean"]

                ci_val = rows_raw[
                    solution
                ]["ci"]

                if pd.isna(mean_val):
                    continue

                bold = rows_raw[
                    solution
                ].get(
                    "bold",
                    False
                )

                safe_solution = (
                    solution.replace(
                        "_",
                        r"\_"
                    )
                )

                value_str = (
                    f"{mean_val:.2f}"
                    f"$\\pm$"
                    f"{ci_val:.2f}"
                )

                if bold:
                    value_str = (
                        f"\\textbf{{{value_str}}}"
                    )

                rows_final.append(
                    {
                        "Detector": safe_solution,
                        metric: value_str
                    }
                )

            if not rows_final:
                continue

            df_table = pd.DataFrame(
                rows_final
            )

            df_table.set_index(
                "Detector",
                inplace=True
            )

            # ====================================================
            # 10. LaTeX
            # ====================================================

            latex = df_table.to_latex(
                escape=False,
                column_format="lc",
                index_names=False
            )

            shift_label = (
                shift_type
                .replace("_", " ")
                .title()
            )

            metric_label = metric
            latex_complete = f"""
\\begin{{table}}[t]
\\centering
\\caption{{Detection quality for {shift_label} -- {metric_label}.}}
\\label{{tab:detection_{shift_type.lower().replace(" ", "_")}_{metric.lower().replace(" ", "_")}}}
\\resizebox{{\\columnwidth}}{{!}}{{%
{latex}
}}
\\end{{table}}
""".replace(" "
            "Concept ", " Concept drift ").replace(" Label ", " Label shift ")

            # ====================================================
            # 11. Salvar
            # ====================================================

            filename = (
                f"{write_path}/"
                f"latex_table_detection_"
                f"{shift_type.lower()}_"
                f"{metric.lower().replace(' ', '_')}.tex"
            )

            with open(
                filename,
                "w",
                encoding="utf-8"
            ) as f:

                f.write(
                    latex_complete
                )

            print(
                f"\nTabela salva:"
                f"\n{filename}"
            )

def normalize_configuration_for_table(configuration):
    if pd.isna(configuration):
        return configuration

    configuration = str(configuration)

    if "#" in configuration:
        configuration = configuration.split("#", 1)[1]

    if configuration.endswith("_sudden"):
        configuration = configuration[:-len("_sudden")]

    return configuration

def normalize_shift_type_for_table(shift_type):
    """
    Padroniza os nomes dos tipos de data shift
    exclusivamente para apresentação nas tabelas.
    """

    if pd.isna(shift_type):
        return shift_type

    shift_type = str(shift_type).strip()

    normalized = {
        "Concept": "Concept Drift",
        "Concept Drift": "Concept Drift",
        "CONCEPT": "Concept Drift",
        "CONCEPT_DRIFT": "Concept Drift",

        "Label": "Label Shift",
        "Label Shift": "Label Shift",
        "LABEL": "Label Shift",
        "LABEL_SHIFT": "Label Shift",
    }

    return normalized.get(
        shift_type,
        shift_type
    )

def format_configuration(shift_type, configuration):
    """
    Normalize a shift configuration for the final table.

    Examples
    --------
    Concept drift:
        concept_drift#0.1_sudden -> 0.1
        0.1                     -> 0.1

    Label shift:
        label_shift#0.1-1.0_sudden -> 0.1-1.0
        0.1-1.0                   -> 0.1-1.0

    The function intentionally returns ONLY the configuration
    values, because the table should display:

        Concept drift | 0.1
        Label shift   | 0.1-1.0
    """

    if pd.isna(configuration):
        return "N/A"

    configuration = str(
        configuration
    ).strip()

    # ------------------------------------------------------------
    # Remove experiment prefix
    #
    # concept_drift#0.1_sudden
    #        -> 0.1_sudden
    #
    # label_shift#0.1-1.0_sudden
    #        -> 0.1-1.0_sudden
    # ------------------------------------------------------------

    if "#" in configuration:

        configuration = configuration.split(
            "#",
            1
        )[1]

    # ------------------------------------------------------------
    # Remove temporal suffix
    #
    # 0.1_sudden
    #        -> 0.1
    #
    # 0.1-1.0_sudden
    #        -> 0.1-1.0
    # ------------------------------------------------------------

    if "_" in configuration:

        configuration = configuration.split(
            "_",
            1
        )[0]

    configuration = configuration.strip()

    # ------------------------------------------------------------
    # Normalize numeric representation
    #
    # 0.10 -> 0.1
    # 1.00 -> 1
    # ------------------------------------------------------------

    if "-" in configuration:

        parts = configuration.split(
            "-",
            1
        )

        try:

            first = float(
                parts[0]
            )

            second = float(
                parts[1]
            )

            return (
                f"{first:g}-{second:g}"
            )

        except (
            ValueError,
            TypeError
        ):

            pass

    else:

        try:

            value = float(
                configuration
            )

            return f"{value:g}"

        except (
            ValueError,
            TypeError
        ):

            pass

    # ------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------

    return configuration.replace(
        "_",
        r"\_"
    )

def format_shift_type(shift_type):
    """
    Normalize all representations of shift type.

    Concept-drift representations:
        CONCEPT_DRIFT
        concept_drift
        Concept
        Concept Drift

    become:

        Concept drift

    Label-shift representations:
        LABEL_SHIFT
        label_shift
        Label
        Label Shift

    become:

        Label shift
    """

    if pd.isna(shift_type):
        return "N/A"

    shift_type = str(
        shift_type
    ).strip()

    normalized = (
        shift_type
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    # ------------------------------------------------------------
    # Concept drift
    # ------------------------------------------------------------

    if (
        "concept" in normalized
        or "concept drift" in normalized
    ):

        return "Concept drift"

    # ------------------------------------------------------------
    # Label shift
    # ------------------------------------------------------------

    if (
        "label" in normalized
        or "label shift" in normalized
    ):

        return "Label shift"

    # ------------------------------------------------------------
    # Unknown type
    # ------------------------------------------------------------

    return shift_type

def generate_latex_table(
    df_table,
    filename,
    caption,
    label,
    column_format,
):
    """
    Generate the final LaTeX table.

    The dataframe already contains the desired LaTeX markup,
    including \\textbf{...}. Therefore escape=False is mandatory.
    """

    latex = df_table.to_latex(
        index=False,                 # REMOVE 0, 1, 2, ...
        escape=False,
        caption=caption,
        label=label,
        column_format=column_format,
    )

    # ============================================================
    # RESTORE / PRESERVE EXISTING LATEX REPLACEMENTS
    # ============================================================

    latex = (
        latex
        .replace(
            "MFP\\_v2\\_dh",
            "$\\textit{MFP}_{\\textit{DDH}}$"
        )
        .replace(
            "MFP\\_v2\\_iti",
            "$\\textit{MFP}_{\\textit{ITI}}$"
        )
        .replace(
            "MFP\\_v2",
            "$\\textit{MFP}$"
        )
    )

    # ============================================================
    # NORMALIZE SHIFT TYPE
    # ============================================================

    latex = (
        latex
        .replace(
            "CONCEPT_DRIFT",
            "Concept drift"
        )
        .replace(
            "concept_drift",
            "Concept drift"
        )
        .replace(
            "Concept",
            "Concept drift"
        )
        .replace(
            "LABEL_SHIFT",
            "Label shift"
        )
        .replace(
            "label_shift",
            "Label shift"
        )
        .replace(
            "Label",
            "Label shift"
        )
    )

    # ============================================================
    # REMOVE ACCIDENTAL DUPLICATED TOPRULE
    # ============================================================

    latex = latex.replace(
        "\\toprule\n\\toprule",
        "\\toprule"
    )

    # ============================================================
    # RESTORE AMPERSANDS
    # ============================================================

    latex = latex.replace(
        r"\&",
        "&"
    )

    # ============================================================
    # WRITE FILE
    # ============================================================

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            latex
        )

def table_detection_quality_combined(
    df_final,
    write_path,
    solutions,
    metrics,
    higher_is_better_metrics,
    ci=0.95,
):
    """
    Generate the combined detection-quality table.

    Grouping:
        Shift Type × Configuration

    The comparison is performed independently for:

        Concept drift
        Label shift

    For each:
        Shift Type × Configuration × Metric

    the best mean is identified according to the metric direction.

    Higher is better:
        Precision
        Recall
        F1

    Lower is better:
        Detection Delay
        Undetected Shift Rate
        False Alarms

    Bold criterion
    --------------
    A method is bold when its MEAN lies inside the confidence
    interval of the BEST MEAN.

    IMPORTANT
    ---------
    The confidence interval of the competing method is NOT used.

    Example:

        Best = 0.50 ± 0.06
        Best interval = [0.44, 0.56]

        0.50 -> bold
        0.30 -> normal
        0.18 -> normal
        0.21 -> normal

    Concept drift and Label shift are always evaluated independently.
    """

    # ============================================================
    # 1. COPY DATAFRAME
    # ============================================================

    df_final = df_final.copy()

    if df_final.empty:

        print(
            "\nWARNING: "
            "table_detection_quality_combined received "
            "an empty dataframe."
        )

        return

    # ============================================================
    # 2. NORMALIZE SHIFT TYPE
    #
    # This MUST happen BEFORE grouping.
    # ============================================================

    df_final["Shift Type"] = (
        df_final["Shift Type"]
        .apply(format_shift_type)
    )

    # ============================================================
    # 3. NORMALIZE CONFIGURATION
    #
    # This MUST happen BEFORE grouping.
    # ============================================================

    df_final["Shift Configuration"] = (
        df_final["Shift Configuration"]
        .apply(
            lambda value:
                format_configuration(
                    None,
                    value
                )
        )
    )

    # ============================================================
    # 4. REMOVE INVALID SHIFT TYPES
    # ============================================================

    df_final = df_final[
        df_final["Shift Type"].isin(
            [
                "Concept drift",
                "Label shift",
            ]
        )
    ].copy()

    if df_final.empty:

        raise RuntimeError(
            "No valid Concept drift or Label shift "
            "records remain after normalization."
        )

    # ============================================================
    # 5. DEBUG CANONICAL DATA
    # ============================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "DEBUG - CANONICAL DATA USED BY "
        "TABLE_DETECTION_QUALITY_COMBINED"
    )

    print(
        "=" * 90
    )

    debug_groups = (
        df_final[
            [
                "Shift Type",
                "Shift Configuration",
                "Detector",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "Shift Type",
                "Shift Configuration",
                "Detector",
            ]
        )
    )

    print(
        debug_groups.to_string(
            index=False
        )
    )

    print(
        "\nRows by Shift Type:"
    )

    print(
        df_final[
            "Shift Type"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nConfigurations by Shift Type:"
    )

    for current_shift_type in [
        "Concept drift",
        "Label shift",
    ]:

        configs = (
            df_final.loc[
                df_final["Shift Type"]
                == current_shift_type,
                "Shift Configuration",
            ]
            .dropna()
            .unique()
            .tolist()
        )

        print(
            f"  {current_shift_type}: "
            f"{configs}"
        )

    print(
        "=" * 90
        + "\n"
    )

    # ============================================================
    # 6. INITIALIZE TABLE
    # ============================================================

    detailed_rows = []

    # ============================================================
    # 7. FIXED SHIFT TYPE ORDER
    #
    # This guarantees Concept drift appears before Label shift.
    # ============================================================

    shift_types = [
        "Concept drift",
        "Label shift",
    ]

    shift_types = [
        shift_type
        for shift_type in shift_types
        if shift_type in
        df_final["Shift Type"].unique()
    ]

    # ============================================================
    # 8. PROCESS EACH SHIFT TYPE
    # ============================================================

    for shift_type in shift_types:

        df_shift = df_final[
            df_final["Shift Type"]
            == shift_type
        ].copy()

        if df_shift.empty:
            continue

        # ========================================================
        # 9. GET CONFIGURATIONS
        # ========================================================

        configurations = (
            df_shift[
                "Shift Configuration"
            ]
            .dropna()
            .unique()
            .tolist()
        )

        # --------------------------------------------------------
        # Configuration sorting
        # --------------------------------------------------------

        def configuration_sort_key(
            value
        ):

            value = str(
                value
            ).strip()

            # --------------------------------------------
            # Label shift: 0.1-1.0
            # --------------------------------------------

            if "-" in value:

                try:

                    parts = value.split(
                        "-",
                        1
                    )

                    return (
                        0,
                        float(parts[0]),
                        float(parts[1]),
                    )

                except (
                    ValueError,
                    TypeError
                ):

                    pass

            # --------------------------------------------
            # Concept drift: 0.1
            # --------------------------------------------

            try:

                return (
                    0,
                    float(value),
                    0.0,
                )

            except (
                ValueError,
                TypeError
            ):

                return (
                    1,
                    value,
                    0.0,
                )

        configurations = sorted(
            configurations,
            key=configuration_sort_key
        )

        # ========================================================
        # 10. PROCESS EACH CONFIGURATION
        # ========================================================

        for configuration in configurations:

            df_config = df_shift[
                df_shift[
                    "Shift Configuration"
                ]
                == configuration
            ].copy()

            if df_config.empty:
                continue

            # ====================================================
            # 11. CALCULATE RAW METRICS
            # ====================================================

            rows_raw = {}

            for solution in solutions:

                df_solution = df_config[
                    df_config["Detector"]
                    == solution
                ]

                if df_solution.empty:
                    continue

                rows_raw[
                    solution
                ] = {}

                for metric in metrics:

                    # ------------------------------------------------
                    # Metric existence
                    # ------------------------------------------------

                    if (
                        metric
                        !=
                        "Undetected Shift Rate"
                        and
                        metric
                        not in
                        df_solution.columns
                    ):

                        rows_raw[
                            solution
                        ][metric] = {
                            "mean": np.nan,
                            "ci": np.nan,
                            "bold": False,
                        }

                        continue

                    # ------------------------------------------------
                    # Calculate metric
                    # ------------------------------------------------

                    mean_value, ci_value = (
                        calculate_detection_metric(
                            df_solution,
                            metric,
                            ci=ci
                        )
                    )

                    rows_raw[
                        solution
                    ][metric] = {
                        "mean": mean_value,
                        "ci": ci_value,
                        "bold": False,
                    }

            # ====================================================
            # 12. DETERMINE BOLD VALUES
            # ====================================================

            for metric in metrics:

                valid_solutions = []

                # ------------------------------------------------
                # Collect valid methods
                # ------------------------------------------------

                for solution in solutions:

                    if solution not in rows_raw:
                        continue

                    if metric not in rows_raw[
                        solution
                    ]:
                        continue

                    mean_value = rows_raw[
                        solution
                    ][metric]["mean"]

                    if pd.isna(mean_value):
                        continue

                    valid_solutions.append(
                        solution
                    )

                if not valid_solutions:
                    continue

                # ------------------------------------------------
                # Determine metric direction
                # ------------------------------------------------

                higher_is_better = (
                    metric
                    in
                    higher_is_better_metrics
                )

                # ------------------------------------------------
                # Find BEST MEAN
                # ------------------------------------------------

                if higher_is_better:

                    best_solution = max(
                        valid_solutions,
                        key=lambda solution:
                            rows_raw[
                                solution
                            ][metric]["mean"]
                    )

                else:

                    best_solution = min(
                        valid_solutions,
                        key=lambda solution:
                            rows_raw[
                                solution
                            ][metric]["mean"]
                    )

                best_mean = rows_raw[
                    best_solution
                ][metric]["mean"]

                best_ci = rows_raw[
                    best_solution
                ][metric]["ci"]

                # ------------------------------------------------
                # Safety
                # ------------------------------------------------

                if pd.isna(best_mean):
                    continue

                # ------------------------------------------------
                # If CI unavailable:
                # only exact best mean is bold.
                # ------------------------------------------------

                if pd.isna(best_ci):

                    for solution in valid_solutions:

                        mean_value = rows_raw[
                            solution
                        ][metric]["mean"]

                        rows_raw[
                            solution
                        ][metric]["bold"] = (
                            np.isclose(
                                mean_value,
                                best_mean,
                                rtol=1e-12,
                                atol=1e-12,
                            )
                        )

                    continue

                # ------------------------------------------------
                # BEST METHOD CI
                # ------------------------------------------------

                best_lower = (
                    best_mean
                    - best_ci
                )

                best_upper = (
                    best_mean
                    + best_ci
                )

                # ------------------------------------------------
                # DEBUG
                # ------------------------------------------------

                print(
                    f"\nShift Type: {shift_type}"
                )

                print(
                    f"Configuration: "
                    f"{configuration}"
                )

                print(
                    f"Metric: {metric}"
                )

                print(
                    f"Best solution: "
                    f"{best_solution}"
                )

                print(
                    f"Best mean: "
                    f"{best_mean:.6f}"
                )

                print(
                    f"Best CI: "
                    f"{best_ci:.6f}"
                )

                print(
                    f"Best interval: "
                    f"[{best_lower:.6f}, "
                    f"{best_upper:.6f}]"
                )

                # ------------------------------------------------
                # FINAL BOLD RULE
                #
                # Compare ONLY the competing MEAN against
                # the BEST METHOD'S confidence interval.
                # ------------------------------------------------

                for solution in valid_solutions:

                    mean_value = rows_raw[
                        solution
                    ][metric]["mean"]

                    is_bold = (
                        best_lower
                        <= mean_value
                        <= best_upper
                    )

                    rows_raw[
                        solution
                    ][metric]["bold"] = bool(
                        is_bold
                    )

                    print(
                        f"    {solution}: "
                        f"mean={mean_value:.6f}, "
                        f"bold={is_bold}"
                    )

            # ====================================================
            # 13. BUILD FINAL ROWS
            # ====================================================

            for solution in solutions:

                if solution not in rows_raw:
                    continue

                safe_solution = (
                    solution.replace(
                        "_",
                        r"\_"
                    )
                )

                row = {
                    "Shift Type":
                        shift_type,

                    "Configuration":
                        format_configuration(
                            shift_type,
                            configuration
                        ),

                    "Detector":
                        safe_solution,
                }

                # =================================================
                # Metrics
                # =================================================

                for metric in metrics:

                    if metric not in rows_raw[
                        solution
                    ]:

                        row[
                            metric
                        ] = "--"

                        continue

                    mean_value = rows_raw[
                        solution
                    ][metric]["mean"]

                    ci_value = rows_raw[
                        solution
                    ][metric]["ci"]

                    is_bold = rows_raw[
                        solution
                    ][metric].get(
                        "bold",
                        False
                    )

                    # ---------------------------------------------
                    # N/A
                    # ---------------------------------------------

                    if pd.isna(
                        mean_value
                    ):

                        if (
                            metric
                            ==
                            "Detection Delay"
                        ):

                            value_str = "N/A"

                        else:

                            value_str = "--"

                    else:

                        # -----------------------------------------
                        # Mean ± CI
                        # -----------------------------------------

                        if pd.isna(
                            ci_value
                        ):

                            value_str = (
                                f"{mean_value:.2f}"
                            )

                        else:

                            value_str = (
                                f"{mean_value:.2f}"
                                f"$\\pm$"
                                f"{ci_value:.2f}"
                            )

                        # -----------------------------------------
                        # Bold
                        # -----------------------------------------

                        if is_bold:

                            value_str = (
                                "\\textbf{"
                                + value_str
                                + "}"
                            )

                    row[
                        metric
                    ] = value_str

                detailed_rows.append(
                    row
                )

    # ============================================================
    # 14. SAFETY CHECK
    # ============================================================

    if not detailed_rows:

        raise RuntimeError(
            "No rows were generated for the "
            "detection-quality table."
        )

    # ============================================================
    # 15. CREATE DATAFRAME
    # ============================================================

    df_detailed = pd.DataFrame(
        detailed_rows
    )

    # ============================================================
    # 16. FORCE EXPECTED COLUMN ORDER
    # ============================================================

    latex_columns = [
        "Shift Type",
        "Configuration",
        "Detector",
        "Precision",
        "Recall",
        "F1",
        "Detection Delay",
        "Undetected Shift Rate",
        "False Alarms",
    ]

    df_detailed = df_detailed[
        latex_columns
    ]

    # ============================================================
    # 17. FINAL DEBUG
    # ============================================================

    print(
        "\n"
        + "=" * 90
    )

    print(
        "FINAL DETECTION QUALITY TABLE"
    )

    print(
        "=" * 90
    )

    print(
        df_detailed.to_string(
            index=True
        )
    )

    print(
        "=" * 90
        + "\n"
    )

    # ============================================================
    # 18. GENERATE LATEX
    # ============================================================

    filename_detailed = (
        f"{write_path}"
        "latex_table_detection_quality_"
        "by_configuration.tex"
    )

    generate_latex_table(
        df_table=df_detailed,
        filename=filename_detailed,
        caption=(
            "Quantitative comparison of "
            "data-shift detection quality "
            "under different shift "
            "configurations. Values are "
            "reported as mean $\\pm$ 95\\% "
            "confidence interval across "
            "datasets, models, and folds."
        ),
        label=(
            "tab:detection_quality_"
            "by_configuration"
        ),
        column_format=(
            "lll"
            + "c" * len(metrics)
        ),
    )

    print(
        "\nTabela salva em:"
    )

    print(
        filename_detailed
    )

def table_per_dataset(df, write_path, metric, solutions_order, ci=0.95):

    datasets = sorted(df["Dataset"].unique().tolist())
    alphas = sorted(df["Alpha"].unique().tolist())
    solutions = [
        df[df["Solution"] == s]["Table"].iloc[0]
        for s in solutions_order
        if s in df["Solution"].values
    ]

    Path(write_path).mkdir(parents=True, exist_ok=True)

    for dataset in datasets:

        rows_raw = {}

        # ==============================
        # 1️⃣ CALCULAR MÉDIA E CI
        # ==============================

        for solution in solutions:
            rows_raw[solution] = {}

            for alpha in alphas:

                filtered = df.query(
                    f"Dataset == '{dataset}' and Table == '{solution}' and Alpha == {alpha}"
                )

                mean, ci_margin = mean_ci(filtered[metric], ci=ci)

                rows_raw[solution][alpha] = {
                    "mean": mean,
                    "ci": ci_margin
                }

        # ==============================
        # 2️⃣ IDENTIFICAR MELHORES COM IC
        # ==============================

        for alpha in alphas:

            # coletar valores da coluna
            col_values = {
                sol: rows_raw[sol][alpha]
                for sol in solutions
            }

            # encontrar maior média
            best_sol = max(col_values, key=lambda x: col_values[x]["mean"])
            best_mean = col_values[best_sol]["mean"]
            best_ci = col_values[best_sol]["ci"]

            best_lower = best_mean - best_ci
            best_upper = best_mean + best_ci

            # verificar sobreposição
            for sol in solutions:
                mean_val = col_values[sol]["mean"]
                ci_val = col_values[sol]["ci"]

                lower = mean_val - ci_val
                upper = mean_val + ci_val

                overlap = not (upper < best_lower or lower > best_upper)

                rows_raw[sol][alpha]["bold"] = overlap

        # ==============================
        # 3️⃣ FORMATAR TABELA FINAL
        # ==============================

        rows_final = []

        for solution in solutions:
            safe_solution = solution.replace("_", r"\_")
            row = {"Solution": safe_solution}

            for alpha in alphas:

                mean_val = rows_raw[solution][alpha]["mean"]
                ci_val = rows_raw[solution][alpha]["ci"]
                bold = rows_raw[solution][alpha]["bold"]

                # CORREÇÃO AQUI
                value_str = f"{mean_val:.2f}$\\pm${ci_val:.2f}"

                if bold:
                    value_str = f"\\textbf{{{value_str}}}"

                row[f"$\\alpha={alpha}$"] = value_str

            rows_final.append(row)

        df_dataset = pd.DataFrame(rows_final)
        df_dataset.set_index("Solution", inplace=True)

        # ==============================
        # 4️⃣ GERAR LATEX
        # ==============================

        latex = df_dataset.to_latex(
            escape=False,
            column_format="l" + "c" * len(alphas),
            index_names=False
        ).replace("MFP\_v2\_dh", "$\\textit{MFP}_{\\textit{DDH}}$").replace("MFP\_v2\_iti", "$\\textit{MFP}_{\\textit{ITI}}$").replace("MFP\_v2", "$\\textit{MFP}$").replace(" Concept ", " Concept drift ").replace(" Label ", " Label shift ")
        print("Latexx: ", latex)
        latex_complete = f"""
        \\begin{{table}}[t]
        \\centering
        \\caption{{Concept Drift -- {dataset} - {metric.replace("%", "\%")}}}
        \\label{{tab:concept_drift_{dataset}_{metric.replace(' ', '_').replace('_(%)', '')}}}
        \\resizebox{{\\columnwidth}}{{!}}{{%
        {latex}
        }}
        \\end{{table}}
        """.replace(" Concept ", " Concept drift ").replace(" Label ", " Label shift ")

        filename = f"{write_path}/latex_table_concept_dirft_{dataset}_{metric.replace(' ','_')}.tex".replace("_(%)", "")

        with open(filename, "w") as f:
            f.write(latex_complete)

        print(f"\nTabela salva para {dataset} em:")
        print(filename)

def extract_alpha_from_experiment(experiment_id):
    """
    Extrai os valores de alpha do Experiment ID.

    Concept Drift:
        concept_drift#0.1_sudden
        -> 0.1

    Label Shift:
        label_shift#0.1-1.0_sudden
        -> (0.1, 1.0)
    """

    experiment_id = str(experiment_id).strip()

    if "#" not in experiment_id:
        raise ValueError(
            f"Experiment ID inválido: {experiment_id}"
        )

    shift_type, config = experiment_id.split(
        "#",
        1
    )

    config = config.split(
        "_sudden",
        1
    )[0]

    # ============================================================
    # CONCEPT DRIFT
    # ============================================================

    if shift_type == "concept_drift":

        try:
            return float(config)

        except ValueError as e:
            raise ValueError(
                f"Alpha inválido no Experiment ID: "
                f"{experiment_id}"
            ) from e

    # ============================================================
    # LABEL SHIFT
    # ============================================================

    if shift_type == "label_shift":

        if "-" not in config:
            raise ValueError(
                f"Configuração de Label Shift inválida: "
                f"{experiment_id}"
            )

        alpha_before, alpha_after = config.split(
            "-",
            1
        )

        try:

            return (
                float(alpha_before),
                float(alpha_after)
            )

        except ValueError as e:

            raise ValueError(
                f"Valores de alpha inválidos no "
                f"Experiment ID: {experiment_id}"
            ) from e

    raise ValueError(
        f"Tipo de shift desconhecido: {shift_type}"
    )


import matplotlib.pyplot as plt
import seaborn as sns


def plot_per_dataset_alpha(df, solutions_order, metric="Accuracy (%)", save_path=None):

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # ==============================
    # 1️⃣ garantir diretório
    # ==============================
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    # ==============================
    # 2️⃣ coluna de rodada
    # ==============================
    round_col = "Round (t)"

    # ==============================
    # 3️⃣ ORDEM IGUAL À TABELA
    # ==============================
    table_order = [
        df[df["Solution"] == s]["Table"].iloc[0]
        for s in solutions_order
        if s in df["Solution"].values
    ]

    print("✔️ Ordem das soluções no plot:")
    print(table_order)

    # ==============================
    # 4️⃣ setup geral
    # ==============================
    datasets = sorted(df["Dataset"].unique())
    alphas = sorted(df["Alpha"].unique())

    sns.set(style="whitegrid")

    # ==============================
    # 5️⃣ loop por dataset
    # ==============================
    for dataset in datasets:

        fig, axes = plt.subplots(
            len(alphas), 1,
            figsize=(8, 5 * len(alphas)),
            sharex=True
        )

        if len(alphas) == 1:
            axes = [axes]

        for i, alpha in enumerate(alphas):
            ax = axes[i]

            filtered = df[
                (df["Dataset"] == dataset) &
                (df["Alpha"] == alpha)
            ]

            # ==============================
            # 6️⃣ agregação (se necessário)
            # ==============================
            if filtered.duplicated(subset=[round_col, "Table"]).any():
                filtered = (
                    filtered
                    .groupby([round_col, "Table"], as_index=False)[metric]
                    .mean()
                )

            # ==============================
            # 7️⃣ plot com ORDEM FIXA
            # ==============================
            sns.lineplot(
                data=filtered,
                x=round_col,
                y=metric,
                hue="Table",
                hue_order=table_order,   # 🔥 AQUI ESTÁ O SEGREDO
                linewidth=2,
                ax=ax
            )

            ax.set_title(f"{dataset} | α={alpha}")
            ax.set_ylabel(metric)

            ax.legend().remove()

        # ==============================
        # 8️⃣ legenda única (ordenada)
        # ==============================
        handles, labels = axes[-1].get_legend_handles_labels()

        # reordenar legenda manualmente (garantia extra)
        label_to_handle = dict(zip(labels, handles))
        ordered_handles = [label_to_handle[l] for l in table_order if l in label_to_handle]

        fig.legend(
            ordered_handles,
            table_order,
            loc="upper center",
            ncol=min(5, len(table_order)),
            frameon=False
        )

        plt.tight_layout(rect=[0, 0, 1, 0.9])

        # ==============================
        # 9️⃣ salvar
        # ==============================
        if save_path:
            filename = f"{save_path}/plot_{dataset}_{metric.replace(' ', '_').replace('(%)','')}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"📊 Plot salvo em: {filename}")

        plt.close(fig)


if __name__ == "__main__":

    # ============================================================
    # CONFIGURAÇÕES
    # ============================================================

    total_clients = 40

    dataset = [
        "WISDM-W",
        "ImageNet10",
        "Foursquare"
    ]

    model_name = [
        "gru",
        "CNN",
        "lstm"
    ]

    fraction_fit = 0.375
    number_of_rounds = 100
    local_epochs = 1
    train_test = "test"

    solutions = [
        "MultiFedAvg+MFP",
        "FedConD",
        "FedDCA",
        "CDA-FedAvg"
        # adicionar demais soluções aqui
    ]

    concept_experiments = [
        "concept_drift#0.1_sudden",
        "concept_drift#1.0_sudden",
        "concept_drift#10.0_sudden"
    ]

    label_experiments = [
        "label_shift#0.1-1.0_sudden",
        "label_shift#0.1-10.0_sudden",
        "label_shift#1.0-0.1_sudden",
        "label_shift#1.0-10.0_sudden",
        "label_shift#10.0-0.1_sudden",
        "label_shift#10.0-1.0_sudden"
    ]

    experiment_ids = (
            concept_experiments
            + label_experiments
    )

    df_all = None

    # ============================================================
    # LEITURA DOS RESULTADOS
    # ============================================================

    for experiment_id in experiment_ids:

        # ============================================================
        # EXTRAIR CONFIGURAÇÃO DE ALPHA
        # ============================================================

        alpha_config = extract_alpha_from_experiment(
            experiment_id
        )

        # ============================================================
        # CONCEPT DRIFT
        # ============================================================

        if experiment_id.startswith("concept_drift#"):

            alpha_value = alpha_config

            alphas = [alpha_value] * len(dataset)

        # ============================================================
        # LABEL SHIFT
        # ============================================================

        elif experiment_id.startswith("label_shift#"):

            alpha_before, alpha_after = alpha_config

            # Para o diretório do experimento, utilizamos
            # os dois valores da transição.
            #
            # Exemplo:
            #
            # label_shift#0.1-1.0_sudden
            #
            # -> [0.1, 1.0, 0.1]
            #
            # ATENÇÃO:
            # esta lista deve reproduzir exatamente a configuração
            # utilizada pelo servidor FedConD.

            alphas = [
                alpha_before,
                alpha_before,
                alpha_before
            ]

            # Não usamos alpha_value como um único float para
            # Label Shift.
            alpha_value = alpha_config

        else:

            raise ValueError(
                f"Experiment ID não suportado: {experiment_id}"
            )

        read_solutions = {
            solution: []
            for solution in solutions
        }

        read_dataset_order = []

        for solution in solutions:
            read_path = (
                "../system/results/"
                "experiment_id_{}/"
                "clients_{}/"
                "alpha_{}/"
                "{}/"
                "{}/"
                "fc_{}/"
                "rounds_{}/"
                "epochs_{}/"
                "{}/"
            ).format(
                experiment_id,
                total_clients,
                alphas,
                dataset,
                model_name,
                fraction_fit,
                number_of_rounds,
                local_epochs,
                train_test
            )

            detection_file = (
                f"{read_path}"
                f"shift_detection_metrics_{solution.replace("MultiFedAvg+MFP", "MultiFedAvg+MFP_v2")}.csv"
            )

            read_solutions[solution].append(
                detection_file
            )

            print(
                f"\nLendo métricas de detecção para {solution}:"
                f"\n{detection_file}"
            )

        print("\n" + "=" * 80)
        print("DEBUG - DIRETÓRIO DE LEITURA")
        print("=" * 80)

        print(f"Experiment ID : {experiment_id}")
        print(f"Alphas        : {alphas}")
        print(f"Read path     : {read_path}")
        print(f"Detection file: {detection_file}")

        print("\nDiretório pai:")
        print(os.path.dirname(detection_file))

        if os.path.exists(os.path.dirname(detection_file)):

            print("\nArquivos encontrados:")

            for filename in sorted(
                    os.listdir(os.path.dirname(detection_file))
            ):
                print(f"  - {filename}")

        else:

            print("\n*** DIRETÓRIO NÃO EXISTE ***")

            # Mostrar também o que existe nos diretórios anteriores
            parent = os.path.dirname(
                os.path.dirname(detection_file)
            )

            print(f"\nTentando listar diretório anterior:")
            print(parent)

            if os.path.exists(parent):

                for filename in sorted(os.listdir(parent)):
                    print(f"  - {filename}")

            else:
                print("*** DIRETÓRIO ANTERIOR TAMBÉM NÃO EXISTE ***")

        print("=" * 80 + "\n")

        # --------------------------------------------------------
        # Ler os CSVs
        # --------------------------------------------------------

        df = read_data(
            read_solutions,
            experiment_id=experiment_id,
            alpha_value=alpha_value
        )

        print(df)

        if df is None or df.empty:
            continue

        if df_all is None:
            df_all = df.copy()
        else:
            df_all = pd.concat(
                [df_all, df],
                ignore_index=True
            )

    # ============================================================
    # VERIFICAÇÃO
    # ============================================================

    if df_all is None or df_all.empty:

        raise RuntimeError(
            "Nenhum CSV de shift detection foi encontrado."
        )

    print(
        "\n====================================="
    )

    print(
        "Dados carregados:"
    )

    print(
        df_all.shape
    )

    print(df_all.columns)

    print(
        df_all[
            [
                "Detector",
                "Dataset",
                "Fold ID",
                "Round",
                "Model",
                "Shift Type",
                "Shift Configuration",
                "Precision",
                "Recall",
                "F1",
                "Detection Delay",
                "False Alarms",
                "First Detection Round",
                "Shift Round",
            ]
        ].head(20)
    )

    # ============================================================
    # ESCRITA DA TABELA PRINCIPAL
    # ============================================================

    write_path = "plots/MEFL/multi_experiments/"

    Path(write_path).mkdir(
        parents=True,
        exist_ok=True
    )

    # ============================================================
    # TABELA PRINCIPAL
    # ============================================================

    metrics = [
        "Precision",
        "Recall",
        "F1",
        "Detection Delay",
        "Undetected Shift Rate",
        "False Alarms",
    ]

    higher_is_better_metrics = {
        "Precision",
        "Recall",
        "F1",
    }

    table_detection_quality_combined(
        df_final=df_all,
        write_path=write_path,
        solutions=solutions,
        metrics=metrics,
        higher_is_better_metrics=higher_is_better_metrics,
        ci=0.95
    )