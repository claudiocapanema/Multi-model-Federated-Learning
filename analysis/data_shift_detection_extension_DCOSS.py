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

    alpha_value : float or tuple/list, optional
        Configuração de alpha utilizada no experimento.

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

                # ==================================================
                # IDENTIFICAÇÃO DA SOLUÇÃO
                # ==================================================

                df["Detector"] = solution

                df["Table"] = solution_names.get(
                    solution,
                    solution
                )

                # ==================================================
                # METADADOS DO EXPERIMENTO
                # ==================================================

                if experiment_id is not None:

                    df["Experiment ID"] = experiment_id

                if alpha_value is not None:

                    if isinstance(
                        alpha_value,
                        (tuple, list)
                    ):

                        # Label Shift:
                        # alpha_before -> alpha_after

                        df["Alpha Before"] = (
                            alpha_value[0]
                        )

                        df["Alpha After"] = (
                            alpha_value[1]
                        )

                    else:

                        # Concept Drift

                        df["Alpha"] = alpha_value

                # ==================================================
                # TIPOS NUMÉRICOS
                # ==================================================

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

                    # IMPORTANTE:
                    # Detection Round é um evento temporal
                    # utilizado na avaliação.

                    "Detection Round",

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

                # ==================================================
                # TIPOS CATEGÓRICOS
                # ==================================================

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

                # ==================================================
                # ADICIONAR AO DATAFRAME GLOBAL
                # ==================================================

                if df_concat is None:

                    df_concat = df.copy()

                else:

                    df_concat = pd.concat(
                        [
                            df_concat,
                            df
                        ],
                        ignore_index=True
                    )

            except Exception as e:

                print("\n#########")
                print(f"Erro lendo: {path}")
                print(e)

    if df_concat is None:

        print(
            "\nNenhum arquivo foi carregado."
        )

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
    Prepara os resultados temporais para avaliação de detecção
    de shift.

    IMPORTANTE
    ----------
    Este método NÃO seleciona somente a última rodada.

    Ele agrega os eventos temporais de cada unidade experimental:

        Detector
        Dataset
        Fold ID
        Model
        Shift Type
        Shift Configuration

    A avaliação depende de:

        - Shift Round
        - First Detection Round
        - Detection Round

    Valores especiais
    -----------------
    First Detection Round = -1

        significa que o detector NÃO detectou o shift.

        Portanto, -1 é convertido para NaN e NÃO é tratado
        como uma rodada de detecção.

    Returns
    -------
    pd.DataFrame
        Uma linha por unidade experimental.
    """

    required_columns = [

        "Detector",

        "Dataset",

        "Fold ID",

        "Round",

        "Model",

        "Shift Type",

        "Shift Configuration",

        "Shift Round",

        "First Detection Round"
    ]

    missing = [

        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing required columns for temporal detection "
            "evaluation: "
            + ", ".join(missing)
        )

    if df.empty:

        return df.copy()

    df_work = df.copy()

    # ============================================================
    # NORMALIZE NUMERIC COLUMNS
    # ============================================================

    numeric_columns = [

        "Fold ID",

        "Round",

        "Model",

        "Shift Round",

        "First Detection Round"
    ]

    if "Detection Round" in df_work.columns:

        numeric_columns.append(
            "Detection Round"
        )

    for column in numeric_columns:

        df_work[column] = pd.to_numeric(
            df_work[column],
            errors="coerce"
        )

    # ============================================================
    # NORMALIZE IDENTIFIERS
    # ============================================================

    for column in [

        "Detector",

        "Dataset",

        "Shift Type",

        "Shift Configuration"
    ]:

        df_work[column] = (
            df_work[column]
            .astype(str)
            .str.strip()
        )

    # ============================================================
    # EXPERIMENTAL UNIT
    # ============================================================

    group_columns = [

        "Detector",

        "Dataset",

        "Fold ID",

        "Model",

        "Shift Type",

        "Shift Configuration"
    ]

    rows = []

    for group_key, group in df_work.groupby(
        group_columns,
        dropna=False
    ):

        group = group.sort_values(
            "Round"
        ).copy()

        # ========================================================
        # GROUND-TRUTH SHIFT ROUND
        # ========================================================

        shift_rounds = (
            pd.to_numeric(
                group["Shift Round"],
                errors="coerce"
            )
            .dropna()
            .unique()
        )

        if len(shift_rounds) == 0:

            shift_round = np.nan

        else:

            shift_round = float(
                shift_rounds[0]
            )

        # ========================================================
        # FIRST DETECTION ROUND
        # ========================================================

        first_detection_values = (
            pd.to_numeric(
                group["First Detection Round"],
                errors="coerce"
            )
            .dropna()
        )

        # --------------------------------------------------------
        # -1 significa "não detectado".
        #
        # Portanto, NÃO pode ser considerado uma detecção.
        # --------------------------------------------------------

        first_detection_values = (
            first_detection_values[
                first_detection_values >= 0
            ]
        )

        if len(first_detection_values) == 0:

            first_detection_round = np.nan

        else:

            first_detection_round = float(
                first_detection_values.min()
            )

        # ========================================================
        # DETECTION ROUND
        # ========================================================

        detection_rounds = []

        if "Detection Round" in group.columns:

            values = (
                pd.to_numeric(
                    group["Detection Round"],
                    errors="coerce"
                )
                .dropna()
                .tolist()
            )

            # ----------------------------------------------------
            # Também descartar valores negativos.
            #
            # Se Detection Round utilizar -1 para indicar
            # ausência de detecção, ele não deve ser tratado
            # como alarme.
            # ----------------------------------------------------

            values = [
                float(x)
                for x in values
                if float(x) >= 0
            ]

            detection_rounds.extend(
                values
            )

        # ========================================================
        # FIRST DETECTION ROUND
        # ========================================================

        # First Detection Round é uma informação de detecção.
        #
        # Só adicionamos se for realmente uma detecção válida.

        if not pd.isna(
            first_detection_round
        ):

            detection_rounds.append(
                first_detection_round
            )

        # ========================================================
        # REMOVE INVALID / DUPLICATE ROUNDS
        # ========================================================

        detection_rounds = sorted(
            set(
                float(x)
                for x in detection_rounds
                if not pd.isna(x)
                and float(x) >= 0
            )
        )

        # ========================================================
        # LAST EVALUATED ROUND
        # ========================================================

        evaluated_rounds = (
            pd.to_numeric(
                group["Round"],
                errors="coerce"
            )
            .dropna()
        )

        if len(evaluated_rounds) > 0:

            last_round = float(
                evaluated_rounds.max()
            )

        else:

            last_round = np.nan

        # ========================================================
        # CREATE RESULT
        # ========================================================

        row = dict(
            zip(
                group_columns,
                group_key
            )
        )

        row["Shift Round"] = (
            shift_round
        )

        row["First Detection Round"] = (
            first_detection_round
        )

        row["Detection Rounds"] = (
            detection_rounds
        )

        row["Last Round"] = (
            last_round
        )

        rows.append(row)

    result = pd.DataFrame(
        rows
    )

    return result

def _get_valid_detection_rounds(
    row,
    max_detection_delay=None
):
    """
    Retorna os alarmes considerados válidos para o shift.

    Parameters
    ----------
    row : pd.Series
        Unidade experimental.

    max_detection_delay : int or None
        Janela máxima aceitável de detecção após o shift.

        None:
            qualquer detecção após o shift e antes do fim
            da avaliação é considerada válida.

    Returns
    -------
    list
        Rodadas de detecção válidas.
    """

    shift_round = row["Shift Round"]

    if pd.isna(shift_round):
        return []

    detection_rounds = row.get(
        "Detection Rounds",
        []
    )

    if not isinstance(
        detection_rounds,
        (list, tuple)
    ):
        return []

    valid = []

    for detection_round in detection_rounds:

        if pd.isna(detection_round):
            continue

        detection_round = float(
            detection_round
        )

        # Detection before the actual shift
        # is a false alarm.
        if detection_round < shift_round:
            continue

        # Optional maximum detection window.
        if max_detection_delay is not None:

            if (
                detection_round
                > shift_round
                + max_detection_delay
            ):
                continue

        valid.append(detection_round)

    return sorted(set(valid))


def calculate_detection_rate(
    df,
    max_detection_delay=None
):
    """
    Detection Rate (DR).

    DR = N_detected_shifts / N_actual_shifts

    A shift é considerado detectado se existir pelo menos
    uma detecção válida após o Shift Round.

    Higher is better.
    """

    if df.empty:
        return np.nan

    detected = 0
    total = 0

    for _, row in df.iterrows():

        shift_round = row["Shift Round"]

        if pd.isna(shift_round):
            continue

        total += 1

        valid_detections = (
            _get_valid_detection_rounds(
                row,
                max_detection_delay
            )
        )

        if len(valid_detections) > 0:
            detected += 1

    if total == 0:
        return np.nan

    return detected / total


def calculate_missed_detection_rate(
    df,
    max_detection_delay=None
):
    """
    Missed Detection Rate (MDR).

    MDR = N_missed / N_actual_shifts

    Lower is better.
    """

    detection_rate = calculate_detection_rate(
        df,
        max_detection_delay
    )

    if pd.isna(detection_rate):
        return np.nan

    return 1.0 - detection_rate


def calculate_average_detection_delay(
    df,
    max_detection_delay=None
):
    """
    Average Detection Delay (ADD / MTD).

    Para cada shift detectado:

        delay =
            First Valid Detection Round
            - Shift Round

    Shifts não detectados não participam do cálculo.

    Lower is better.
    """

    if df.empty:
        return np.nan

    delays = []

    for _, row in df.iterrows():

        shift_round = row["Shift Round"]

        if pd.isna(shift_round):
            continue

        valid_detections = (
            _get_valid_detection_rounds(
                row,
                max_detection_delay
            )
        )

        if not valid_detections:
            continue

        first_detection = valid_detections[0]

        delay = (
            first_detection
            - float(shift_round)
        )

        if delay >= 0:
            delays.append(delay)

    if not delays:
        return np.nan

    return float(
        np.mean(delays)
    )


def calculate_episode_f1(
    df,
    max_detection_delay=None
):
    """
    Episode-level F1.

    Cada shift ground-truth pode produzir no máximo
    um verdadeiro positivo.

    Isso evita contar várias detecções do mesmo shift
    como múltiplos TP.

    TP = shift com pelo menos uma detecção válida
    FN = shift sem detecção válida
    FP = alarmes que ocorreram fora das janelas dos shifts

    Higher is better.
    """

    if df.empty:
        return np.nan

    tp = 0
    fn = 0
    fp = 0

    for _, row in df.iterrows():

        shift_round = row["Shift Round"]

        if pd.isna(shift_round):
            continue

        detection_rounds = row.get(
            "Detection Rounds",
            []
        )

        if not isinstance(
            detection_rounds,
            (list, tuple)
        ):
            detection_rounds = []

        detection_rounds = sorted(
            set(
                float(x)
                for x in detection_rounds
                if not pd.isna(x)
            )
        )

        valid_detections = (
            _get_valid_detection_rounds(
                row,
                max_detection_delay
            )
        )

        # --------------------------------------------------------
        # TP / FN
        # --------------------------------------------------------

        if valid_detections:
            tp += 1
        else:
            fn += 1

        # --------------------------------------------------------
        # False alarms
        # --------------------------------------------------------

        valid_set = set(
            valid_detections
        )

        for detection_round in detection_rounds:

            if detection_round < shift_round:
                fp += 1
                continue

            if (
                max_detection_delay is not None
                and detection_round
                > shift_round
                + max_detection_delay
            ):
                fp += 1
                continue

            # A detection inside the valid episode
            # is not counted as FP.
            if detection_round in valid_set:
                continue

    precision_denominator = (
        tp + fp
    )

    recall_denominator = (
        tp + fn
    )

    if precision_denominator == 0:
        precision = 0.0
    else:
        precision = (
            tp
            / precision_denominator
        )

    if recall_denominator == 0:
        recall = 0.0
    else:
        recall = (
            tp
            / recall_denominator
        )

    if (
        precision + recall
    ) == 0:
        return 0.0

    return (
        2.0
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


def calculate_alarm_rate(df):
    """
    Alarm Rate.

    Número total de alarmes dividido pelo número
    total de rodadas avaliadas.

    AR = N_alarms / N_evaluated_rounds

    Lower is better.

    Essa métrica mede a tendência do detector de gerar
    alarmes excessivos.
    """

    if df.empty:
        return np.nan

    total_alarms = 0
    total_rounds = 0

    for _, row in df.iterrows():

        detection_rounds = row.get(
            "Detection Rounds",
            []
        )

        if isinstance(
            detection_rounds,
            (list, tuple)
        ):
            total_alarms += len(
                set(detection_rounds)
            )

        last_round = row.get(
            "Last Round",
            np.nan
        )

        if not pd.isna(last_round):
            total_rounds += int(
                last_round
            )

    if total_rounds <= 0:
        return np.nan

    return (
        total_alarms
        / total_rounds
    )


def calculate_detection_metric_values(
    df,
    metric,
    max_detection_delay=None
):
    """
    Retorna uma observação da métrica por unidade experimental.

    Cada linha de `df` representa uma unidade experimental.

    As métricas são calculadas a partir dos eventos temporais:

        - Shift Round
        - Detection Rounds

    Métricas:

        Detection Rate
        Missed Detection Rate
        Average Detection Delay
        Episode F1
        Alarm Rate
    """

    if df is None or df.empty:
        return pd.Series(dtype=float)

    values = []

    for _, row in df.iterrows():

        shift_round = row.get(
            "Shift Round",
            np.nan
        )

        if pd.isna(shift_round):
            continue

        shift_round = float(
            shift_round
        )

        # ============================================================
        # DETECTION ROUNDS
        # ============================================================

        detection_rounds = row.get(
            "Detection Rounds",
            []
        )

        if not isinstance(
            detection_rounds,
            (list, tuple, np.ndarray)
        ):
            detection_rounds = []

        normalized_detection_rounds = []

        for detection_round in detection_rounds:

            try:
                detection_round = float(
                    detection_round
                )
            except (
                ValueError,
                TypeError
            ):
                continue

            # -1 e outros valores negativos significam
            # ausência de detecção.
            if detection_round < 0:
                continue

            normalized_detection_rounds.append(
                detection_round
            )

        detection_rounds = sorted(
            set(normalized_detection_rounds)
        )

        # ============================================================
        # VALID DETECTIONS
        # ============================================================

        valid_detections = (
            _get_valid_detection_rounds(
                row,
                max_detection_delay=(
                    max_detection_delay
                )
            )
        )

        # ============================================================
        # DETECTION RATE
        # ============================================================

        if metric == "Detection Rate":

            values.append(
                1.0
                if valid_detections
                else 0.0
            )

        # ============================================================
        # MISSED DETECTION RATE
        # ============================================================

        elif metric == "Missed Detection Rate":

            values.append(
                0.0
                if valid_detections
                else 1.0
            )

        # ============================================================
        # AVERAGE DETECTION DELAY
        # ============================================================

        elif metric == "Average Detection Delay":

            # Shifts não detectados não entram no ADD.
            if not valid_detections:
                continue

            first_detection = float(
                valid_detections[0]
            )

            delay = (
                first_detection
                - shift_round
            )

            if delay >= 0:
                values.append(
                    delay
                )

        # ============================================================
        # EPISODE F1
        # ============================================================

        elif metric == "Episode F1":

            # --------------------------------------------------------
            # Um shift representa um único episódio.
            # --------------------------------------------------------

            tp = (
                1
                if valid_detections
                else 0
            )

            fn = (
                0
                if valid_detections
                else 1
            )

            # --------------------------------------------------------
            # False alarms
            #
            # Alarmes antes do shift ou fora da janela máxima
            # são considerados FP.
            # --------------------------------------------------------

            fp = 0

            valid_set = set(
                valid_detections
            )

            for detection_round in detection_rounds:

                # ----------------------------------------------------
                # Alarme antes do shift
                # ----------------------------------------------------

                if detection_round < shift_round:
                    fp += 1
                    continue

                # ----------------------------------------------------
                # Alarme depois da janela válida
                # ----------------------------------------------------

                if (
                    max_detection_delay is not None
                    and detection_round
                    > (
                        shift_round
                        + max_detection_delay
                    )
                ):
                    fp += 1
                    continue

                # ----------------------------------------------------
                # Detecção válida.
                # ----------------------------------------------------

                if detection_round in valid_set:
                    continue

            # --------------------------------------------------------
            # Episode F1
            # --------------------------------------------------------

            precision_denominator = (
                tp + fp
            )

            recall_denominator = (
                tp + fn
            )

            if precision_denominator == 0:
                precision = 0.0
            else:
                precision = (
                    tp
                    / precision_denominator
                )

            if recall_denominator == 0:
                recall = 0.0
            else:
                recall = (
                    tp
                    / recall_denominator
                )

            if (
                precision + recall
            ) == 0:

                f1 = 0.0

            else:

                f1 = (
                    2.0
                    * precision
                    * recall
                    / (
                        precision
                        + recall
                    )
                )

            values.append(
                f1
            )

        # ============================================================
        # ALARM RATE
        # ============================================================

        elif metric == "Alarm Rate":

            last_round = row.get(
                "Last Round",
                np.nan
            )

            if pd.isna(last_round):
                continue

            last_round = float(
                last_round
            )

            if last_round <= 0:
                continue

            # Cada Detection Round único corresponde a um alarme.
            number_of_alarms = len(
                detection_rounds
            )

            alarm_rate = (
                number_of_alarms
                / last_round
            )

            values.append(
                alarm_rate
            )

        else:

            raise ValueError(
                f"Unsupported detection metric: "
                f"'{metric}'"
            )

    return pd.Series(
        values,
        dtype=float
    )

def mean_ci(
    values,
    ci=0.95,
    bounded=False
):
    """
    Calcula a média e a margem do intervalo de confiança.

    Parameters
    ----------
    values : array-like
        Valores individuais da métrica.

    ci : float
        Nível de confiança.

    bounded : bool
        Se True, limita os limites do IC ao intervalo [0, 1].

    Returns
    -------
    mean : float
        Média.

    margin : float
        Margem do intervalo de confiança.

    A tabela apresenta:

        mean ± margin
    """

    # ============================================================
    # CONVERTER PARA SERIES
    # ============================================================

    values = pd.Series(
        values,
        dtype="float64"
    )

    # ============================================================
    # REMOVER VALORES INVÁLIDOS
    # ============================================================

    values = pd.to_numeric(
        values,
        errors="coerce"
    ).dropna()

    values = values.to_numpy(
        dtype=float
    )

    # ============================================================
    # SEM DADOS
    # ============================================================

    if len(values) == 0:
        return np.nan, np.nan

    # ============================================================
    # MÉDIA
    # ============================================================

    mean = np.mean(
        values
    )

    # ============================================================
    # UMA OBSERVAÇÃO
    # ============================================================

    if len(values) == 1:

        return (
            round(mean, 2),
            0.00
        )

    # ============================================================
    # TODOS OS VALORES IGUAIS
    # ============================================================

    if np.allclose(
        values,
        values[0]
    ):

        return (
            round(mean, 2),
            0.00
        )

    # ============================================================
    # STANDARD ERROR
    # ============================================================

    sem = st.sem(
        values
    )

    # ============================================================
    # CONFIDENCE INTERVAL
    # ============================================================

    lower, upper = st.t.interval(
        confidence=ci,
        df=len(values) - 1,
        loc=mean,
        scale=sem
    )

    # ============================================================
    # BOUND [0, 1]
    # ============================================================

    if bounded:

        lower = max(
            0.0,
            lower
        )

        upper = min(
            1.0,
            upper
        )

    # ============================================================
    # IC REPRESENTADO COMO MARGEM
    # ============================================================

    margin = max(
        mean - lower,
        upper - mean
    )

    return (
        round(mean, 2),
        round(margin, 2)
    )

def table_detection_quality(
    df,
    write_path,
    solutions_order,
    metrics=None,
    ci=0.95,
    max_detection_delay=None
):
    """
    Gera uma única tabela com o desempenho dos detectores.

    Estrutura:

        Data shift type | Solution | Detection Rate |
        Episode F1 | Average Detection Delay |
        Missed Detection Rate | Alarm Rate

    Para cada tipo de shift:

        Concept drift
            Solution 1
            Solution 2
            Solution 3
            Solution 4

        Label shift
            Solution 1
            Solution 2
            Solution 3
            Solution 4

    O tipo de shift é apresentado uma única vez utilizando
    LaTeX \\multirow.

    Cada métrica é apresentada como:

        mean ± 95% CI

    O IC é calculado sobre as unidades experimentais.
    """

    if metrics is None:

        metrics = [
            "Detection Rate",
            "Episode F1",
            "Average Detection Delay",
            "Missed Detection Rate",
            "Alarm Rate",
        ]

    # ============================================================
    # CREATE OUTPUT DIRECTORY
    # ============================================================

    Path(
        write_path
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    # ============================================================
    # PREPARE TEMPORAL DATA
    # ============================================================

    df_eval = select_final_detection_results(
        df
    )

    if df_eval.empty:

        print(
            "\nWARNING: no temporal detection "
            "records available."
        )

        return df_eval

    # ============================================================
    # NORMALIZE SHIFT TYPE
    # ============================================================

    df_eval["Shift Type"] = (
        df_eval["Shift Type"]
        .apply(format_shift_type)
    )

    # ============================================================
    # VALID SHIFT TYPES
    # ============================================================

    valid_shift_types = [
        "Concept drift",
        "Label shift",
    ]

    df_eval = df_eval[
        df_eval["Shift Type"].isin(
            valid_shift_types
        )
    ].copy()

    if df_eval.empty:

        print(
            "\nWARNING: no valid shift types."
        )

        return df_eval

    # ============================================================
    # SOLUTION ORDER
    # ============================================================

    available_solutions = (
        df_eval["Detector"]
        .dropna()
        .unique()
        .tolist()
    )

    solutions = [
        solution
        for solution in solutions_order
        if solution in available_solutions
    ]

    # ============================================================
    # METRIC DISPLAY NAMES
    # ============================================================

    metric_names = {
        "Detection Rate":
            "Detection Rate",

        "Episode F1":
            "Episode F1",

        "Average Detection Delay":
            "Average Detection Delay",

        "Missed Detection Rate":
            "Missed Detection Rate",

        "Alarm Rate":
            "Alarm Rate",
    }

    # ============================================================
    # METRIC DIRECTION
    # ============================================================

    higher_is_better = {
        "Detection Rate",
        "Episode F1",
    }

    # ============================================================
    # STORE RAW RESULTS
    # ============================================================

    results = {}

    for shift_type in valid_shift_types:

        results[shift_type] = {}

        df_shift = df_eval[
            df_eval["Shift Type"]
            == shift_type
        ].copy()

        if df_shift.empty:
            continue

        for solution in solutions:

            df_solution = df_shift[
                df_shift["Detector"]
                == solution
            ].copy()

            if df_solution.empty:
                continue

            results[
                shift_type
            ][solution] = {}

            for metric in metrics:

                mean_value, ci_value = (
                    calculate_detection_metric(
                        df_solution,
                        metric,
                        ci=ci,
                        max_detection_delay=(
                            max_detection_delay
                        )
                    )
                )

                results[
                    shift_type
                ][solution][metric] = {
                    "mean": mean_value,
                    "ci": ci_value,
                }

    # ============================================================
    # DETERMINE BEST RESULTS
    # ============================================================

    best_results = {}

    for shift_type in valid_shift_types:

        best_results[
            shift_type
        ] = {}

        if shift_type not in results:
            continue

        for metric in metrics:

            valid_solutions = [
                solution
                for solution in solutions
                if solution in results[
                    shift_type
                ]
                and metric in results[
                    shift_type
                ][solution]
                and not pd.isna(
                    results[
                        shift_type
                    ][solution][metric]["mean"]
                )
            ]

            if not valid_solutions:
                continue

            if metric in higher_is_better:

                best_solution = max(
                    valid_solutions,
                    key=lambda solution:
                        results[
                            shift_type
                        ][solution][metric]["mean"]
                )

            else:

                best_solution = min(
                    valid_solutions,
                    key=lambda solution:
                        results[
                            shift_type
                        ][solution][metric]["mean"]
                )

            best_results[
                shift_type
            ][metric] = best_solution

    # ============================================================
    # FORMAT RESULT
    # ============================================================

    def format_metric_value(
        shift_type,
        solution,
        metric
    ):

        if (
            shift_type not in results
            or solution not in results[
                shift_type
            ]
            or metric not in results[
                shift_type
            ][solution]
        ):

            return "--"

        mean_value = results[
            shift_type
        ][solution][metric]["mean"]

        ci_value = results[
            shift_type
        ][solution][metric]["ci"]

        if pd.isna(mean_value):
            return "--"

        if pd.isna(ci_value):

            text = (
                f"{mean_value:.2f}"
            )

        else:

            text = (
                f"{mean_value:.2f}"
                f" $\\pm$ "
                f"{ci_value:.2f}"
            )

        # ========================================================
        # BOLD BEST
        # ========================================================

        best_solution = (
            best_results
            .get(shift_type, {})
            .get(metric)
        )

        if (
            best_solution is not None
            and solution == best_solution
        ):

            text = (
                "\\textbf{"
                + text
                + "}"
            )

        return text

    # ============================================================
    # BUILD TABLE ROWS
    # ============================================================

    table_rows = []

    for shift_type in valid_shift_types:

        if shift_type not in results:
            continue

        # --------------------------------------------------------
        # Somente as soluções que realmente possuem resultados.
        # --------------------------------------------------------

        shift_solutions = [
            solution
            for solution in solutions
            if solution in results[
                shift_type
            ]
        ]

        if not shift_solutions:
            continue

        number_of_rows = len(
            shift_solutions
        )

        for index, solution in enumerate(
            shift_solutions
        ):

            # ----------------------------------------------------
            # multirow somente na primeira linha do grupo.
            # ----------------------------------------------------

            if index == 0:

                shift_label = (
                    "\\multirow{"
                    f"{number_of_rows}"
                    "}{*}{"
                    f"{shift_type}"
                    "}"
                )

            else:

                shift_label = ""

            row = {
                "Data shift type":
                    shift_label,

                "Solution":
                    solution,
            }

            for metric in metrics:

                row[
                    metric_names.get(
                        metric,
                        metric
                    )
                ] = format_metric_value(
                    shift_type,
                    solution,
                    metric
                )

            table_rows.append(
                row
            )

    # ============================================================
    # CREATE DATAFRAME
    # ============================================================

    columns = [
        "Data shift type",
        "Solution",
    ] + [
        metric_names.get(
            metric,
            metric
        )
        for metric in metrics
    ]

    df_table = pd.DataFrame(
        table_rows,
        columns=columns
    )

    # ============================================================
    # OUTPUT
    # ============================================================

    filename = os.path.join(
        write_path,
        "detection_quality_summary.tex"
    )

    # ============================================================
    # LATEX
    # ============================================================

    generate_latex_table(
        df_table=df_table,
        filename=filename,
        caption=(
            "Data-shift detection performance "
            "with 95\\% confidence intervals."
        ),
        label=(
            "tab:detection_quality"
        ),
        column_format=(
            "llccccc"
        ),
    )

    print(
        "\nGenerated unified detection-quality table:"
    )

    print(
        filename
    )

    return df_eval

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
    Gera uma tabela LaTeX.

    O dataframe pode conter comandos LaTeX como:

        \\textbf{...}
        \\multirow{...}{*}{...}

    Portanto, escape=False é obrigatório.
    """

    latex = df_table.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        column_format=column_format,
    )

    # ============================================================
    # PRESERVE EXISTING MODEL REPLACEMENTS
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
            "CONCEPT",
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
            "LABEL",
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
    # ADD REQUIRED MULTIROW PACKAGE
    # ============================================================

    # Como a tabela usa \\multirow, o documento principal
    # precisa carregar o pacote multirow.
    #
    # Não inserimos \\usepackage diretamente na tabela porque
    # isso não pertence ao ambiente table.
    #
    # Adicionamos apenas um comentário informativo caso o
    # pacote ainda não esteja sendo carregado.

    latex = (
        "% Requires: \\usepackage{multirow}\n"
        + latex
    )

    # ============================================================
    # WRITE FILE
    # ============================================================

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            latex
        )

def calculate_detection_metric(
    df,
    metric,
    ci=0.95,
    max_detection_delay=None
):
    values = calculate_detection_metric_values(
        df,
        metric,
        max_detection_delay=max_detection_delay
    )

    bounded = metric in {
        "Detection Rate",
        "Episode F1",
        "Missed Detection Rate",
        "Alarm Rate",
    }

    return mean_ci(
        values,
        ci=ci,
        bounded=bounded
    )

def table_detection_quality_by_shift_type(
    df_final,
    write_path,
    solutions,
    metrics,
    higher_is_better_metrics,
    ci=0.95,
    max_detection_delay=None,
):
    """
    Gera a tabela geral de desempenho dos detectores
    por tipo de shift.

    A avaliação utiliza os eventos temporais de detecção.

    Aggregation
    -----------
        Shift Type × Detector

    Metrics
    -------
        Detection Rate
        Episode F1
        Average Detection Delay
        Missed Detection Rate
        Alarm Rate

    Higher is better
    -----------------
        Detection Rate
        Episode F1

    Lower is better
    ----------------
        Average Detection Delay
        Missed Detection Rate
        Alarm Rate
    """

    if df_final.empty:
        print(
            "\nWARNING: empty dataframe passed "
            "to table_detection_quality_by_shift_type."
        )
        return

    df_eval = df_final.copy()

    # ============================================================
    # NORMALIZE SHIFT TYPE
    # ============================================================

    df_eval["Shift Type"] = (
        df_eval["Shift Type"]
        .apply(format_shift_type)
    )

    valid_shift_types = [
        "Concept drift",
        "Label shift",
    ]

    df_eval = df_eval[
        df_eval["Shift Type"].isin(
            valid_shift_types
        )
    ].copy()

    if df_eval.empty:
        return

    # ============================================================
    # METRIC DIRECTION
    # ============================================================

    higher_is_better = {
        "Detection Rate",
        "Episode F1",
    }

    # ============================================================
    # BUILD AGGREGATED RESULTS
    # ============================================================

    rows_raw = {}

    for shift_type in valid_shift_types:

        df_shift = df_eval[
            df_eval["Shift Type"] == shift_type
        ].copy()

        if df_shift.empty:
            continue

        rows_raw[shift_type] = {}

        for solution in solutions:

            df_solution = df_shift[
                df_shift["Detector"] == solution
            ].copy()

            if df_solution.empty:
                continue

            rows_raw[
                shift_type
            ][solution] = {}

            for metric in metrics:

                mean_value, ci_value = (
                    calculate_detection_metric(
                        df_solution,
                        metric,
                        ci=ci,
                        max_detection_delay=(
                            max_detection_delay
                        ),
                    )
                )

                rows_raw[
                    shift_type
                ][solution][metric] = {
                    "mean": mean_value,
                    "ci": ci_value,
                }

    # ============================================================
    # BUILD TABLE
    # ============================================================

    for shift_type in valid_shift_types:

        if shift_type not in rows_raw:
            continue

        for metric in metrics:

            valid_solutions = [
                solution
                for solution in solutions
                if solution in rows_raw[
                    shift_type
                ]
                and not pd.isna(
                    rows_raw[
                        shift_type
                    ][solution][metric]["mean"]
                )
            ]

            if not valid_solutions:
                continue

            metric_is_higher_better = (
                metric in higher_is_better
            )

            if metric_is_higher_better:

                best_solution = max(
                    valid_solutions,
                    key=lambda solution:
                        rows_raw[
                            shift_type
                        ][solution][metric]["mean"]
                )

            else:

                best_solution = min(
                    valid_solutions,
                    key=lambda solution:
                        rows_raw[
                            shift_type
                        ][solution][metric]["mean"]
                )

            best_mean = rows_raw[
                shift_type
            ][best_solution][metric]["mean"]

            best_ci = rows_raw[
                shift_type
            ][best_solution][metric]["ci"]

            # ----------------------------------------------------
            # BUILD ROWS
            # ----------------------------------------------------

            table_rows = []

            for solution in solutions:

                if solution not in valid_solutions:
                    continue

                mean_value = rows_raw[
                    shift_type
                ][solution][metric]["mean"]

                ci_value = rows_raw[
                    shift_type
                ][solution][metric]["ci"]

                if pd.isna(mean_value):
                    continue

                # ------------------------------------------------
                # Formatting
                # ------------------------------------------------

                if pd.isna(ci_value):

                    text = (
                        f"{mean_value:.2f}"
                    )

                else:

                    text = (
                        f"{mean_value:.2f}"
                        f" $\\pm$ "
                        f"{ci_value:.2f}"
                    )

                # ------------------------------------------------
                # BOLD
                # ------------------------------------------------

                if not pd.isna(best_ci):

                    best_lower = (
                        best_mean
                        - best_ci
                    )

                    best_upper = (
                        best_mean
                        + best_ci
                    )

                    is_bold = (
                        best_lower
                        <= mean_value
                        <= best_upper
                    )

                else:

                    is_bold = np.isclose(
                        mean_value,
                        best_mean,
                        rtol=1e-12,
                        atol=1e-12,
                    )

                if is_bold:

                    text = (
                        "\\textbf{"
                        + text
                        + "}"
                    )

                table_rows.append(
                    {
                        "Detector": solution,
                        metric: text,
                    }
                )

            if not table_rows:
                continue

            df_table = pd.DataFrame(
                table_rows
            )

            filename = os.path.join(
                write_path,
                (
                    "overall_detection_"
                    f"{metric.lower().replace(' ', '_')}_"
                    f"{shift_type.lower().replace(' ', '_')}.tex"
                )
            )

            generate_latex_table(
                df_table=df_table,
                filename=filename,
                caption=(
                    f"{metric} for "
                    f"{shift_type} detection."
                ),
                label=(
                    "tab:overall_detection_"
                    f"{metric.lower().replace(' ', '_')}_"
                    f"{shift_type.lower().replace(' ', '_')}"
                ),
                column_format="lr",
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
        "Detection Rate",
        "Episode F1",
        "Average Detection Delay",
        "Missed Detection Rate",
        "Alarm Rate",
    ]

    higher_is_better_metrics = {
        "Detection Rate",
        "Episode F1",
    }

    print("\n" + "=" * 100)
    print("VALIDAÇÃO DOS CSVs UTILIZADOS NA TABELA")
    print("=" * 100)

    for detector in solutions:

        detector_df = df_all[
            df_all["Detector"] == detector
            ].copy()

        print(f"\nDetector: {detector}")

        if detector_df.empty:
            print("  *** NENHUM DADO CARREGADO ***")
            continue

        print(f"  Número de linhas: {len(detector_df)}")
        print(f"  Datasets: {detector_df['Dataset'].unique()}")
        print(f"  Models: {detector_df['Model'].unique()}")
        print(f"  Shift Types: {detector_df['Shift Type'].unique()}")
        print(f"  Configurations: {detector_df['Shift Configuration'].unique()}")
        print(f"  Rounds: {detector_df['Round'].min()} -> {detector_df['Round'].max()}")

    # ============================================================
    # SELECIONAR SOMENTE A ÚLTIMA RODADA DE CADA UNIDADE EXPERIMENTAL
    # ============================================================

    df_final = select_final_detection_results(
        df_all
    )

    print("\n" + "=" * 100)
    print("VALIDAÇÃO - DADOS FINAIS USADOS NA TABELA")
    print("=" * 100)

    print(f"Total de linhas originais: {len(df_all)}")
    print(f"Total de linhas finais:   {len(df_final)}")

    print("\nLinhas por detector:")
    print(
        df_final["Detector"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nLinhas por detector e tipo de shift:")
    print(
        df_final
        .groupby(
            ["Detector", "Shift Type"]
        )
        .size()
        .to_string()
    )

    print("\nDados temporais selecionados para detecção:")
    print(
        df_final[
            [
                "Detector",
                "Dataset",
                "Fold ID",
                "Model",
                "Shift Type",
                "Shift Configuration",
                "Shift Round",
                "First Detection Round",
                "Detection Rounds",
                "Last Round",
            ]
        ]
        .sort_values(
            [
                "Detector",
                "Shift Type",
                "Dataset",
                "Model",
            ]
        )
        .to_string(index=False)
    )

    print("=" * 100)

    # ============================================================
    # GERAR TABELA COM OS DADOS FINAIS
    # ============================================================

    table_detection_quality(
        df=df_all,
        write_path=write_path,
        solutions_order=solutions,
        metrics=metrics,
        ci=0.95,
        max_detection_delay=None,
    )