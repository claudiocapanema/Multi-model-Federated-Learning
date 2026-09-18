from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
import os
import re

import copy

from base_plots import bar_plot, line_plot, ecdf_plot
import matplotlib.pyplot as plt

# Registro de TODOS os arquivos/configurações esperados para verificar,
# ao final da execução, arquivos ausentes, ilegíveis ou incompletos.
#
# Regra de completude:
#   - somente 1 fold distinto -> exatamente 100 rodadas completas;
#   - mais de 1 fold distinto -> exatamente 500 rodadas completas.
#
# "Rodadas completas" significa que, para CADA fold, as rodadas esperadas
# estão presentes de 1 até o número esperado, sem faltar nenhuma rodada.
EXPERIMENT_VALIDATION = []

def read_data(
    read_solutions,
    solution_names=None,
    experiment_id=None,
    alpha_value=None,
    dataset_shift_rounds=None,
    transition_window=None
):
    """
    Lê os CSVs específicos de cada dataset e solução.

    Cada CSV principal deve conter:
        Fold ID, Round (t), Accuracy

    A coluna Data shift é opcional. Quando ausente (como nos baselines
    sem detector, por exemplo MultiFedAvg e MultiFedAvgRR), ela é
    preenchida com NO_SHIFT exclusivamente para manter o dataframe
    compatível com as métricas de detecção.

    São adicionadas:
        Solution
        Dataset
        Shift Round
        Transition Window
    """

    df_list = []

    if solution_names is None:
        solution_names = {
            solution: solution
            for solution in read_solutions.keys()
        }

    if dataset_shift_rounds is None:
        dataset_shift_rounds = {}

    for solution, paths in read_solutions.items():

        for path in paths:

            try:

                if not os.path.exists(path):
                    print("\n#########")
                    print(f"Arquivo não encontrado: {path}")

                    EXPERIMENT_VALIDATION.append({
                        "path": os.path.abspath(path),
                        "solution": solution_names.get(solution, solution),
                        "dataset": None,
                        "experiment_id": experiment_id,
                        "transition_window": transition_window,
                        "fold_count": 0,
                        "fold_ids": [],
                        "expected_rounds": None,
                        "fold_details": {},
                        "complete": False,
                        "status": "missing",
                    })
                    continue

                # ========================================================
                # LEITURA EXCLUSIVA DOS CSVs PRINCIPAIS
                # ========================================================
                # Os arquivos de resultados usados por este script seguem:
                #
                #     {dataset}_{solution}.csv
                #
                # NUNCA utilizar:
                #
                #     {dataset}_{solution}_metrics.csv
                #
                # Alguns baselines (em particular MultiFedAvg e
                # MultiFedAvgRR) não possuem a coluna "Data shift", pois
                # não executam um detector de data shift. Essa coluna é
                # necessária apenas para as métricas de detecção e, por
                # isso, deve ser tratada como opcional no carregamento.
                # Accuracy, Fold ID e Round (t) continuam sendo os dados
                # obrigatórios para as tabelas de desempenho.
                if os.path.basename(path).endswith("_metrics.csv"):
                    raise ValueError(
                        "Arquivo _metrics.csv não pode ser utilizado: "
                        f"{path}"
                    )

                # Ler apenas o cabeçalho primeiro para evitar que a
                # ausência de "Data shift" descarte MultiFedAvg /
                # MultiFedAvgRR do dataframe.
                header = pd.read_csv(
                    path,
                    nrows=0
                ).columns.tolist()

                required_columns = [
                    "Fold ID",
                    "Round (t)",
                    "Accuracy",
                ]

                missing_required = [
                    column
                    for column in required_columns
                    if column not in header
                ]

                if missing_required:
                    raise ValueError(
                        "Colunas obrigatórias ausentes: "
                        + ", ".join(missing_required)
                    )

                columns_to_read = required_columns.copy()

                has_data_shift = "Data shift" in header

                if has_data_shift:
                    columns_to_read.append("Data shift")

                df = pd.read_csv(
                    path,
                    usecols=columns_to_read
                )

                # Baselines sem detector não possuem "Data shift".
                # Para eles, todas as rodadas são consideradas como
                # ausência de alarme. Isso preserva seus resultados de
                # Accuracy sem atribuir-lhes detecções inexistentes.
                if not has_data_shift:
                    df["Data shift"] = "NO_SHIFT"

                if df.empty:
                    print(f"\nArquivo vazio: {path}")

                    EXPERIMENT_VALIDATION.append({
                        "path": os.path.abspath(path),
                        "solution": solution_names.get(solution, solution),
                        "dataset": None,
                        "experiment_id": experiment_id,
                        "transition_window": transition_window,
                        "fold_count": 0,
                        "fold_ids": [],
                        "expected_rounds": None,
                        "fold_details": {},
                        "complete": False,
                        "status": "empty",
                    })
                    continue

                # Manter somente as colunas necessárias do CSV.
                # Accuracy é utilizada adicionalmente nas tabelas de
                # desempenho, enquanto as demais colunas continuam sendo
                # utilizadas para as métricas de detecção.
                df = df[
                    ["Fold ID", "Round (t)", "Data shift", "Accuracy"]
                ].copy()

                # ----------------------------------------------------
                # DATASET
                # ----------------------------------------------------
                filename = os.path.basename(path)

                dataset_name = None

                # O nome do CSV segue:
                # {dataset_name}_{solution}.csv
                for candidate_dataset in dataset_shift_rounds.keys():
                    if filename.startswith(
                        f"{candidate_dataset}_"
                    ):
                        dataset_name = candidate_dataset
                        break

                if dataset_name is None:
                    raise ValueError(
                        f"Não foi possível identificar o dataset "
                        f"a partir do arquivo: {filename}"
                    )

                df["Dataset"] = dataset_name

                # Preserve the experiment alpha in the dataframe.
                # This is required by the accuracy tables, which compare
                # solutions separately for each alpha/configuration.
                if isinstance(alpha_value, (tuple, list)):
                    df["Alpha"] = float(alpha_value[0])
                else:
                    df["Alpha"] = float(alpha_value) if alpha_value is not None else np.nan

                # ----------------------------------------------------
                # SOLUTION
                # ----------------------------------------------------
                df["Solution"] = solution_names.get(
                    solution,
                    solution
                )

                # ----------------------------------------------------
                # SHIFT ROUND
                # ----------------------------------------------------
                df["Shift Round"] = dataset_shift_rounds[
                    dataset_name
                ]

                # Gradual shifts have an explicit transition window in
                # the result-directory path. Sudden shifts use None.
                df["Transition Window"] = transition_window

                # --------------------------------------------------------
                # VALIDAÇÃO DE COMPLETUDE DO CSV
                # --------------------------------------------------------
                # A quantidade de rodadas esperada depende da quantidade
                # de folds encontrados no arquivo:
                #
                #   1 fold  -> 100 rodadas
                #   >1 fold -> 500 rodadas
                #
                # A validação é feita por fold, garantindo que cada fold
                # contenha todas as rodadas de 1 até o limite esperado.
                fold_numeric = pd.to_numeric(
                    df["Fold ID"],
                    errors="coerce"
                )
                round_numeric = pd.to_numeric(
                    df["Round (t)"],
                    errors="coerce"
                )

                valid_mask = (
                    fold_numeric.notna()
                    & round_numeric.notna()
                )

                fold_round_df = pd.DataFrame({
                    "Fold ID": fold_numeric[valid_mask],
                    "Round (t)": round_numeric[valid_mask],
                })

                fold_ids = sorted(
                    fold_round_df["Fold ID"].unique().tolist()
                )

                fold_count = len(fold_ids)
                expected_rounds = (
                    100 if fold_count <= 1 else 500
                )

                fold_details = {}

                for fold_id in fold_ids:
                    rounds = sorted(
                        set(
                            fold_round_df.loc[
                                fold_round_df["Fold ID"] == fold_id,
                                "Round (t)"
                            ].astype(int).tolist()
                        )
                    )

                    expected_round_set = set(
                        range(1, expected_rounds + 1)
                    )
                    actual_round_set = set(rounds)

                    missing_rounds = sorted(
                        expected_round_set - actual_round_set
                    )
                    extra_rounds = sorted(
                        actual_round_set - expected_round_set
                    )

                    fold_details[fold_id] = {
                        "round_count": len(rounds),
                        "min_round": min(rounds) if rounds else None,
                        "max_round": max(rounds) if rounds else None,
                        "missing_rounds": missing_rounds,
                        "extra_rounds": extra_rounds,
                        "complete": (
                            actual_round_set == expected_round_set
                        ),
                    }

                file_complete = (
                    fold_count >= 1
                    and all(
                        details["complete"]
                        for details in fold_details.values()
                    )
                )

                EXPERIMENT_VALIDATION.append({
                    "path": os.path.abspath(path),
                    "solution": df["Solution"].iloc[0],
                    "dataset": dataset_name,
                    "experiment_id": experiment_id,
                    "transition_window": transition_window,
                    "fold_count": fold_count,
                    "fold_ids": fold_ids,
                    "expected_rounds": expected_rounds,
                    "fold_details": fold_details,
                    "complete": file_complete,
                    "status": "complete" if file_complete else "incomplete",
                })

                df_list.append(df)

                print(f"CSV lido: {path}")
                print(f"Dataset: {dataset_name}")
                print(f"Solução: {df['Solution'].iloc[0]}")
                print(
                    f"Shift Round: "
                    f"{df['Shift Round'].iloc[0]}"
                )
                print(f"Linhas: {len(df)}")

            except ValueError as e:
                print(
                    f"\nColunas/dados esperados não encontrados "
                    f"em: {path}"
                )
                print(f"Erro: {e}")

                EXPERIMENT_VALIDATION.append({
                    "path": os.path.abspath(path),
                    "solution": solution_names.get(solution, solution),
                    "dataset": None,
                    "experiment_id": experiment_id,
                    "transition_window": transition_window,
                    "fold_count": 0,
                    "fold_ids": [],
                    "expected_rounds": None,
                    "fold_details": {},
                    "complete": False,
                    "status": "invalid",
                    "error": str(e),
                })
                continue

            except Exception as e:
                print(f"\nErro ao ler {path}: {e}")

                EXPERIMENT_VALIDATION.append({
                    "path": os.path.abspath(path),
                    "solution": solution_names.get(solution, solution),
                    "dataset": None,
                    "experiment_id": experiment_id,
                    "transition_window": transition_window,
                    "fold_count": 0,
                    "fold_ids": [],
                    "expected_rounds": None,
                    "fold_details": {},
                    "complete": False,
                    "status": "error",
                    "error": str(e),
                })
                continue

    if not df_list:
        return pd.DataFrame(
            columns=[
                "Fold ID",
                "Round (t)",
                "Data shift",
                "Accuracy",
                "Solution",
                "Dataset",
                "Shift Round"
            ]
        )

    return pd.concat(
        df_list,
        ignore_index=True
    )

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

def _get_shift_detection_bounds(
    row,
    max_detection_delay=None,
    use_transition_window_for_gradual=True,
):
    """
    Determina os limites da janela válida de detecção.

    Sudden:
        O shift ocorre instantaneamente em ``Shift Round``.
        A janela válida é:
            [Shift Round, Shift Round + max_detection_delay]

    Gradual (``use_transition_window_for_gradual=True``):
        O shift ocorre durante ``Transition Window`` rodadas, iniciando em
        ``Shift Round``. Portanto, se o shift começa em t_s e possui janela
        W, o último round da transição é:

            t_e = t_s + W - 1

        A janela válida de detecção é:

            [t_s, t_e + max_detection_delay]

    Gradual avaliado como sudden (``use_transition_window_for_gradual=False``):
        ``Transition Window`` é ignorada para a validade da detecção e a
        janela é:

            [t_s, t_s + max_detection_delay]

    Esta separação é importante: ``Transition Window`` representa a duração
    do fenômeno, enquanto ``max_detection_delay`` representa a tolerância
    para o detector reagir após o término do fenômeno.

    Returns
    -------
    tuple(float, float or None)
        Início e fim da janela válida de detecção.
    """

    shift_round = row.get("Shift Round", np.nan)

    if pd.isna(shift_round):
        return np.nan, None

    shift_round = float(shift_round)

    temporal_shift_type = str(
        row.get("Temporal Shift Type", "")
    ).strip().lower()

    transition_window = row.get(
        "Transition Window",
        np.nan
    )

    # ------------------------------------------------------------
    # END OF THE GROUND-TRUTH SHIFT
    # ------------------------------------------------------------

    if (
        temporal_shift_type == "gradual"
        and use_transition_window_for_gradual
        and not pd.isna(transition_window)
    ):
        try:
            transition_window = int(
                float(transition_window)
            )
        except (TypeError, ValueError):
            transition_window = None

        if transition_window is not None and transition_window > 0:
            shift_end = (
                shift_round
                + transition_window
                - 1
            )
        else:
            # Defensive fallback: if the temporal type is gradual but
            # the transition window is unavailable/invalid, preserve the
            # instantaneous interpretation rather than inventing a window.
            shift_end = shift_round
    else:
        # Sudden shifts are instantaneous.
        shift_end = shift_round

    # ------------------------------------------------------------
    # END OF THE VALID DETECTION WINDOW
    # ------------------------------------------------------------

    if max_detection_delay is None:
        detection_window_end = None
    else:
        detection_window_end = (
            shift_end
            + float(max_detection_delay)
        )

    return shift_round, detection_window_end


def _get_valid_detection_rounds(
    row,
    max_detection_delay=None,
    use_transition_window_for_gradual=True,
):
    """
    Retorna os alarmes considerados válidos para o shift.

    A definição é baseada no episódio ground-truth:

    * Sudden: o episódio contém apenas ``Shift Round``.
    * Gradual: o episódio contém
      ``Shift Round`` até ``Shift Round + Transition Window - 1``.

    Uma detecção em qualquer ponto do episódio é válida. Após o fim do
    episódio, a mesma margem ``max_detection_delay`` é aplicada tanto a
    sudden quanto a gradual.

    Assim, para um gradual que começa em 30 com Transition Window = 5,
    o episódio é [30, 34]. Com max_detection_delay = 10, a janela válida
    de detecção é [30, 44].

    ``Transition Window`` e ``max_detection_delay`` têm papéis distintos:

    * Transition Window = duração do fenômeno de shift;
    * max_detection_delay = tolerância para o detector reagir após o
      término do fenômeno.

    Parameters
    ----------
    row : pd.Series
        Unidade experimental.

    max_detection_delay : int or None
        Janela máxima aceitável de detecção após o término do shift.

        None:
            qualquer detecção após o início do shift e antes do fim
            da avaliação é considerada válida.

    Returns
    -------
    list
        Rodadas de detecção válidas.
    """

    shift_round, detection_window_end = _get_shift_detection_bounds(
        row,
        max_detection_delay=max_detection_delay,
        use_transition_window_for_gradual=use_transition_window_for_gradual,
    )

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

        # If a maximum detection delay is configured, the deadline is
        # measured from the END of the ground-truth episode. This makes
        # the post-transition behavior identical for sudden and gradual.
        if (
            detection_window_end is not None
            and detection_round > detection_window_end
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
    max_detection_delay=None,
    use_transition_window_for_gradual=True,
):
    """
    Retorna uma observação da métrica por unidade experimental.

    Cada linha de `df` representa uma unidade experimental.

    As métricas são calculadas a partir dos eventos temporais:

        - Shift Round
        - Detection Rounds

    Métricas:

        Detection Rate
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
                max_detection_delay=max_detection_delay,
                use_transition_window_for_gradual=use_transition_window_for_gradual,
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
                        _get_shift_detection_bounds(
                            row,
                            max_detection_delay=max_detection_delay,
                            use_transition_window_for_gradual=use_transition_window_for_gradual,
                        )[1]
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
        Mantido por compatibilidade. O IC t não é truncado, mesmo
        para métricas bounded ([0, 1]).

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
    # IC REPRESENTADO COMO MARGEM
    # ============================================================
    #
    # Para métricas bounded ([0, 1]), o IC t é mantido sem
    # truncamento. Isso evita transformar um intervalo truncado
    # em uma margem simétrica que não representa o IC real.
    # ============================================================

    margin = max(
        mean - lower,
        upper - mean
    )

    return (
        round(mean, 2),
        round(margin, 2)
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

        "Combined": "Combined Shift",
        "Combined Shift": "Combined Shift",
        "COMBINED": "Combined Shift",
        "COMBINED_SHIFT": "Combined Shift",
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

    Combined-shift representations:
        COMBINED_SHIFT
        combined_shift
        Combined
        Combined Shift

    become:

        Combined shift
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
    # Combined shift
    # ------------------------------------------------------------
    if (
        "combined" in normalized
        or "combined shift" in normalized
    ):
        return "Combined shift"

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
        "% Requires: \\usepackage{booktabs}\n"
        "% Requires: \\usepackage{multirow}\n"
        "% Requires: \\usepackage{graphicx}\n"
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
    max_detection_delay=None,
    use_transition_window_for_gradual=True,
):
    values = calculate_detection_metric_values(
        df,
        metric,
        max_detection_delay=max_detection_delay,
        use_transition_window_for_gradual=use_transition_window_for_gradual,
    )

    bounded = metric in {
        "Detection Rate",
        "Episode F1",
        "Alarm Rate",
    }

    return mean_ci(
        values,
        ci=ci,
        bounded=bounded
    )

def _extract_detection_rounds_from_unit(df_experiment):
    """
    Extrai as rodadas em que o detector realmente gerou DATA_SHIFT.

    Os CSVs atuais não possuem uma coluna "Detection Rounds".
    Eles registram o resultado do detector por rodada na coluna
    "Data shift":

        DATA_SHIFT -> alarme/detecção
        NO_SHIFT   -> ausência de alarme

    Portanto, Detection Rounds deve ser reconstruído diretamente
    a partir de "Data shift".
    """

    if "Data shift" not in df_experiment.columns:
        return []

    shift_values = (
        df_experiment["Data shift"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    rounds = pd.to_numeric(
        df_experiment["Round (t)"],
        errors="coerce"
    )

    detection_rounds = rounds[
        shift_values == "DATA_SHIFT"
    ].dropna().astype(float).tolist()

    return sorted(set(detection_rounds))


def _calculate_experiment_detection_metrics(
    df_experiment,
    max_detection_delay=None,
    use_transition_window_for_gradual=True,
):
    """
    Calcula as métricas para UMA unidade experimental:

        Solution × Dataset × Experiment × Fold ID

    IMPORTANTE:
    Os CSVs atuais armazenam a decisão do detector por rodada na
    coluna "Data shift". Portanto, as rodadas de detecção são
    reconstruídas dessa coluna, em vez de procurar uma coluna
    inexistente chamada "Detection Rounds".
    """

    if df_experiment.empty:
        return {
            "Detection Rate": np.nan,
            "Episode F1": np.nan,
            "Average Detection Delay": np.nan,
            "False Alarm Rate": np.nan,
        }

    row = df_experiment.iloc[0].copy()

    # ============================================================
    # GROUND TRUTH
    # ============================================================

    shift_round = row.get("Shift Round", np.nan)

    if pd.isna(shift_round):
        return {
            "Detection Rate": np.nan,
            "Episode F1": np.nan,
            "Average Detection Delay": np.nan,
            "False Alarm Rate": np.nan,
        }

    shift_round = float(shift_round)

    # ============================================================
    # DETECTIONS
    # ============================================================
    #
    # ESTA É A CORREÇÃO PRINCIPAL.
    #
    # O CSV não contém "Detection Rounds". A detecção está codificada
    # diretamente em:
    #
    #     Data shift == DATA_SHIFT
    #
    # Portanto:
    #
    #     Detection Rounds =
    #         Round (t) onde Data shift == DATA_SHIFT
    # ============================================================

    detection_rounds = _extract_detection_rounds_from_unit(
        df_experiment
    )

    # Validity is defined from the ground-truth episode:
    #   sudden  -> [Shift Round, Shift Round]
    #   gradual -> [Shift Round,
    #               Shift Round + Transition Window - 1]
    #
    # The same max_detection_delay is then applied AFTER the end of
    # that episode for both sudden and gradual.
    detection_row = row.copy()
    detection_row["Temporal Shift Type"] = (
        df_experiment["Temporal Shift Type"].iloc[0]
        if "Temporal Shift Type" in df_experiment.columns
        else ""
    )
    detection_row["Transition Window"] = (
        df_experiment["Transition Window"].iloc[0]
        if "Transition Window" in df_experiment.columns
        else np.nan
    )
    detection_row["Detection Rounds"] = detection_rounds

    valid_detections = _get_valid_detection_rounds(
        detection_row,
        max_detection_delay=max_detection_delay,
        use_transition_window_for_gradual=use_transition_window_for_gradual,
    )

    # ============================================================
    # DETECTION RATE
    # ============================================================

    detection_rate = (
        1.0 if valid_detections else 0.0
    )

    # ============================================================
    # DETECTION DELAY
    # ============================================================

    if valid_detections:

        first_detection = float(
            valid_detections[0]
        )

        average_detection_delay = (
            first_detection
            - shift_round
        )

    else:
        average_detection_delay = np.nan

    # ============================================================
    # FALSE POSITIVES
    # ============================================================

    false_alarms = []
    pre_shift_false_alarms = []

    for detection_round in detection_rounds:

        # Alarm before the actual shift.
        # It is an FP for Episode F1 and also contributes to FAR.
        if detection_round < shift_round:
            false_alarms.append(
                detection_round
            )
            pre_shift_false_alarms.append(
                detection_round
            )
            continue

        # Alarm after the allowed detection window.
        # It is an FP for Episode F1, but NOT for FAR because FAR
        # measures false alarms during the stable pre-shift regime.
        # Alarm after the valid detection window.
        # The deadline is measured from the END of the ground-truth
        # episode, so gradual and sudden use the same post-transition
        # detection tolerance.
        _, detection_window_end = _get_shift_detection_bounds(
            detection_row,
            max_detection_delay=max_detection_delay,
            use_transition_window_for_gradual=use_transition_window_for_gradual,
        )

        if (
            detection_window_end is not None
            and detection_round > detection_window_end
        ):
            false_alarms.append(
                detection_round
            )

    # ============================================================
    # EPISODE F1
    # ============================================================

    tp = 1 if valid_detections else 0
    fn = 0 if valid_detections else 1
    fp = len(false_alarms)

    precision_denominator = tp + fp
    recall_denominator = tp + fn

    precision = (
        tp / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        tp / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    if precision + recall == 0:
        episode_f1 = 0.0
    else:
        episode_f1 = (
            2.0
            * precision
            * recall
            / (precision + recall)
        )

    # ============================================================
    # FALSE ALARM RATE
    # ============================================================
    #
    # FAR = FP / number of stable rounds
    #
    # Como o experimento possui um shift conhecido em Shift Round,
    # as rodadas estáveis anteriores ao shift são:
    #
    #     1, ..., Shift Round - 1
    #
    # Alarmes após a janela máxima são FP para o Episode F1, mas
    # não entram no FAR. O FAR mede exclusivamente falsos alarmes
    # durante o regime estável pré-shift. Assim, MAX_DETECTION_DELAY
    # afeta DR/MTD/F1, mas não altera o FAR.
    # ============================================================

    stable_rounds = max(
        shift_round - 1.0,
        0.0
    )

    if stable_rounds > 0:
        false_alarm_rate = (
            len(pre_shift_false_alarms)
            / stable_rounds
        )
    else:
        false_alarm_rate = np.nan

    return {
        "Detection Rate": detection_rate,
        "Episode F1": episode_f1,
        "Average Detection Delay": average_detection_delay,
        "False Alarm Rate": false_alarm_rate,
    }


def table_detection_quality_by_shift_type(
    df_final,
    write_path,
    solutions,
    metrics=None,
    higher_is_better_metrics=None,
    ci=0.95,
    max_detection_delay=None,
    use_transition_window_for_gradual=True,
):
    """
    Gera a tabela consolidada de qualidade da detecção.

    Para gradual shifts, ``Transition Window`` é um parâmetro experimental
    explícito e é extraído do diretório de resultados.

    A avaliação trata o shift como um episódio ground-truth:
        * sudden: [Shift Round, Shift Round]
        * gradual: [Shift Round,
                    Shift Round + Transition Window - 1]

    A margem ``max_detection_delay`` é aplicada após o fim do episódio
    quando ``use_transition_window_for_gradual=True``.

    Quando ``use_transition_window_for_gradual=False``, gradual é avaliado
    exatamente como sudden para as métricas: ``Transition Window`` não é
    usada para decidir se uma detecção ocorreu dentro da janela válida.
    Nesse modo, a janela válida é ``[Shift Round,
    Shift Round + max_detection_delay]``. A ``Transition Window`` continua
    sendo preservada apenas como dimensão experimental da tabela.

        transition_window_2
        transition_window_5
        ...

    Portanto, gradual/window=2 e gradual/window=5 são tratados como
    configurações experimentais distintas e nunca são agregados juntos.

    A unidade experimental é:

        Solution × Dataset × Experiment × Fold ID × Shift Type
        × Temporal Shift Type × Transition Window

    As métricas são calculadas primeiro por unidade experimental e só depois
    agregadas. Os melhores resultados são determinados separadamente dentro
    de cada combinação Shift × Temporal Shift Type × Transition Window.
    """
    if df_final is None or df_final.empty:
        print("\nWARNING: empty dataframe passed to table_detection_quality_by_shift_type.")
        return

    if metrics is None:
        metrics = [
            "Episode F1",
            "Detection Rate",
            "False Alarm Rate",
            "Average Detection Delay",
        ]

    if higher_is_better_metrics is None:
        higher_is_better_metrics = {"Detection Rate", "Episode F1"}

    df_eval = df_final.copy()

    # ------------------------------------------------------------
    # Shift family
    # ------------------------------------------------------------
    if "Experiment ID" in df_eval.columns:
        exp = df_eval["Experiment ID"].astype(str).str.strip()
        df_eval["Shift Type"] = np.select(
            [
                exp.str.startswith("concept_drift#"),
                exp.str.startswith("label_shift#"),
                exp.str.startswith("combined_shift#"),
            ],
            ["Concept drift", "Label shift", "Combined shift"],
            default=df_eval["Shift Type"] if "Shift Type" in df_eval.columns else "N/A",
        )

    if "Shift Type" not in df_eval.columns:
        raise KeyError("The dataframe must contain 'Shift Type' or 'Experiment ID'.")

    df_eval["Shift Type"] = df_eval["Shift Type"].apply(format_shift_type)

    # ------------------------------------------------------------
    # Experiment identifier and temporal type
    # ------------------------------------------------------------
    experiment_column = next(
        (c for c in ["Experiment ID", "experiment_id", "Experiment"] if c in df_eval.columns),
        None,
    )
    if experiment_column is None:
        raise KeyError(
            "The dataframe must contain an experiment identifier "
            "('Experiment ID', 'experiment_id' or 'Experiment')."
        )

    df_eval["_Experiment"] = df_eval[experiment_column].astype(str).str.strip()
    exp_lower = df_eval[experiment_column].astype(str).str.strip().str.lower()
    df_eval["Temporal Shift Type"] = np.select(
        [exp_lower.str.endswith("_sudden"), exp_lower.str.endswith("_gradual")],
        ["sudden", "gradual"],
        default="N/A",
    )

    # ------------------------------------------------------------
    # Transition window
    # ------------------------------------------------------------
    # The value is supplied by read_data from the directory path. Do not
    # infer it from Experiment ID because the ID intentionally does not
    # contain transition_window_N.
    if "Transition Window" not in df_eval.columns:
        df_eval["Transition Window"] = np.nan

    def normalize_transition_window(value):
        if pd.isna(value):
            return np.nan
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return np.nan

    df_eval["Transition Window"] = df_eval["Transition Window"].apply(
        normalize_transition_window
    )

    print("\n" + "=" * 100)
    print("DEBUG - SHIFT / TEMPORAL TYPE / TRANSITION WINDOW")
    print("=" * 100)
    print(
        df_eval[
            ["Shift Type", "Temporal Shift Type", "Transition Window"]
        ].value_counts(dropna=False)
    )

    valid_shift_types = ["Concept drift", "Label shift", "Combined shift"]
    valid_temporal_types = ["sudden", "gradual"]

    df_eval = df_eval[
        df_eval["Shift Type"].isin(valid_shift_types)
        & df_eval["Temporal Shift Type"].isin(valid_temporal_types)
    ].copy()

    if df_eval.empty:
        print("\nWARNING: no valid shift/temporal types found.")
        return

    # ------------------------------------------------------------
    # Experimental units
    # ------------------------------------------------------------
    group_columns = [
        "Solution",
        "Dataset",
        "_Experiment",
        "Fold ID",
        "Shift Type",
        "Temporal Shift Type",
        "Transition Window",
    ]

    experimental_rows = []
    for group_key, df_unit in df_eval.groupby(group_columns, dropna=False):
        (
            solution,
            dataset,
            experiment,
            fold_id,
            shift_type,
            temporal_shift_type,
            transition_window,
        ) = group_key

        metric_values = _calculate_experiment_detection_metrics(
            df_unit,
            max_detection_delay=max_detection_delay,
            use_transition_window_for_gradual=use_transition_window_for_gradual,
        )

        # Debug: show the effective ground-truth episode and the final
        # valid detection deadline for this experimental configuration.
        debug_row = df_unit.iloc[0].copy()
        debug_start, debug_end = _get_shift_detection_bounds(
            debug_row,
            max_detection_delay=max_detection_delay,
            use_transition_window_for_gradual=use_transition_window_for_gradual,
        )
        debug_window = debug_row.get("Transition Window", np.nan)
        debug_temporal = debug_row.get("Temporal Shift Type", "N/A")

        if max_detection_delay is None:
            debug_deadline = "unlimited"
        else:
            debug_deadline = (
                f"{debug_end:g}"
                if debug_end is not None
                else "unlimited"
            )

        print(
            f"DEBUG - {temporal_shift_type} | "
            f"Shift start={debug_start:g} | "
            f"Transition Window={debug_window} | "
            f"valid detection through={debug_deadline}"
        )

        row = {
            "Solution": solution,
            "Dataset": dataset,
            "Experiment": experiment,
            "Fold ID": fold_id,
            "Shift Type": shift_type,
            "Temporal Shift Type": temporal_shift_type,
            "Transition Window": transition_window,
        }
        row.update(metric_values)
        experimental_rows.append(row)

    df_experimental = pd.DataFrame(experimental_rows)
    if df_experimental.empty:
        print("\nWARNING: no experimental units were created.")
        return

    # ------------------------------------------------------------
    # Aggregation helper
    # ------------------------------------------------------------
    valid_transition_windows = sorted(
        df_experimental.loc[
            df_experimental["Temporal Shift Type"] == "gradual",
            "Transition Window",
        ].dropna().astype(int).unique().tolist()
    )

    print("\n" + "=" * 100)
    print("DEBUG - EXPERIMENTAL UNITS BY SHIFT / TEMPORAL TYPE / WINDOW")
    print("=" * 100)
    print(
        df_experimental.groupby(
            ["Shift Type", "Temporal Shift Type", "Transition Window"],
            dropna=False,
        ).size()
    )

    # Keep the same aggregation principle as before, but add transition
    # window as an explicit grouping dimension.
    aggregated = {}
    for (shift_type, temporal_type, window), df_group in df_experimental.groupby(
        ["Shift Type", "Temporal Shift Type", "Transition Window"],
        dropna=False,
    ):
        aggregated[(shift_type, temporal_type, window)] = {}
        for solution in solutions:
            df_solution = df_group[df_group["Solution"] == solution]
            if df_solution.empty:
                continue
            aggregated[(shift_type, temporal_type, window)][solution] = {}
            for metric in metrics:
                values = pd.to_numeric(
                    df_solution[metric], errors="coerce"
                ).dropna()
                mean_value, ci_value = mean_ci(
                    values,
                    ci=ci,
                    bounded=metric in {
                        "Detection Rate",
                        "Episode F1",
                                },
                )
                aggregated[(shift_type, temporal_type, window)][solution][metric] = {
                    "mean": mean_value,
                    "ci": ci_value,
                    "n": len(values),
                }

            # JS-Drift was not designed for Concept Drift. If its DR is zero,
            # its Concept Drift performance metrics are not applicable.
            # Keep the raw experimental values intact, but suppress these
            # metrics in the final table and exclude them from "best" ranking.
            if (
                str(shift_type).strip().lower() == "concept drift"
                and str(solution).strip().lower() == "js-drift"
                and "Detection Rate"
                in aggregated[(shift_type, temporal_type, window)][solution]
                and np.isclose(
                    aggregated[(shift_type, temporal_type, window)][solution]
                    ["Detection Rate"]["mean"],
                    0.0,
                    atol=1e-12,
                )
            ):
                for metric in metrics:
                    aggregated[(shift_type, temporal_type, window)][solution][metric] = {
                        "mean": np.nan,
                        "ci": np.nan,
                        "n": 0,
                    }

    # ------------------------------------------------------------
    # Build LaTeX rows
    # ------------------------------------------------------------
    shift_labels = {
        "Concept drift": "Concept Drift",
        "Label shift": "Label Shift",
        "Combined shift": "Combined Shift",
    }

    temporal_labels = {
        "sudden": "Sudden",
        "gradual": "Gradual",
    }

    table_rows = []

    for shift_type in valid_shift_types:

        keys = [
            key
            for key in aggregated
            if key[0] == shift_type
        ]

        if not keys:
            continue

        # Stable ordering:
        #   1. Sudden
        #   2. Gradual, ordered by transition window
        keys.sort(
            key=lambda k: (
                0 if k[1] == "sudden" else 1,
                -1 if k[2] is None else k[2],
            )
        )

        # Number of rows occupied by the complete shift family.
        shift_row_count = sum(
            len(aggregated[k])
            for k in keys
        )

        shift_first = True

        # --------------------------------------------------------
        # IMPORTANT:
        # Gradual spans ALL transition-window groups belonging
        # to the same shift family.
        # --------------------------------------------------------
        temporal_row_counts = {}

        for temporal_type in ["sudden", "gradual"]:

            temporal_keys = [
                k
                for k in keys
                if k[1] == temporal_type
            ]

            temporal_row_counts[temporal_type] = sum(
                len(aggregated[k])
                for k in temporal_keys
            )

        temporal_first = {
            "sudden": True,
            "gradual": True,
        }

        # --------------------------------------------------------
        # Generate table rows
        # --------------------------------------------------------
        for key in keys:

            _, temporal_type, transition_window = key
            temporal_rows = aggregated[key]

            solutions_present = [
                s
                for s in solutions
                if s in temporal_rows
            ]

            if not solutions_present:
                continue

            # ----------------------------------------------------
            # Best solution is determined independently for each
            # Shift × Temporal Type × Transition Window group.
            # ----------------------------------------------------
            best_by_metric = {}

            for metric in metrics:

                candidates = [
                    sol
                    for sol in solutions_present
                    if not pd.isna(
                        temporal_rows[sol][metric]["mean"]
                    )
                ]

                if not candidates:
                    continue

                if metric in higher_is_better_metrics:

                    best_by_metric[metric] = max(
                        candidates,
                        key=lambda sol:
                        temporal_rows[sol][metric]["mean"],
                    )

                else:

                    best_by_metric[metric] = min(
                        candidates,
                        key=lambda sol:
                        temporal_rows[sol][metric]["mean"],
                    )

            # ----------------------------------------------------
            # One row per solution
            # ----------------------------------------------------
            for solution in solutions_present:

                row = {}

                # ==================================================
                # SHIFT
                # ==================================================
                if shift_first:

                    row["Shift"] = (
                        f"\\multirow{{{shift_row_count}}}"
                        f"{{*}}{{{shift_labels[shift_type]}}}"
                    )

                    shift_first = False

                else:

                    row["Shift"] = ""

                # ==================================================
                # SHIFT TYPE
                #
                # Gradual is ONE multirow covering all transition
                # windows of the gradual condition.
                # ==================================================
                if temporal_first[temporal_type]:

                    temporal_row_count = (
                        temporal_row_counts[temporal_type]
                    )

                    row["Shift Type"] = (
                        f"\\multirow{{{temporal_row_count}}}"
                        f"{{*}}{{{temporal_labels[temporal_type]}}}"
                    )

                    temporal_first[temporal_type] = False

                else:

                    row["Shift Type"] = ""

                # ==================================================
                # TRANSITION WINDOW
                # ==================================================
                if (
                    transition_window is None
                    or pd.isna(transition_window)
                ):

                    row["Transition Window"] = "N/A"

                else:

                    row["Transition Window"] = (
                        str(int(float(transition_window)))
                    )

                # ==================================================
                # SOLUTION
                # ==================================================
                solution_display = str(solution)

                if solution_display == "MFP_v2_dh":

                    solution_display = (
                        "$\\textit{MFP}_{\\textit{DDH}}$"
                    )

                elif solution_display == "MFP_v2_iti":

                    solution_display = (
                        "$\\textit{MFP}_{\\textit{ITI}}$"
                    )

                elif solution_display == "MFP_v2":

                    solution_display = "MFP"

                elif solution_display == "MultiFedAvg+MFP_v2":

                    solution_display = "MultiFedAvg+MFP"

                else:

                    solution_display = solution_display.replace(
                        "_",
                        r"\_",
                    )

                row["Solution"] = solution_display

                # ==================================================
                # METRICS
                # ==================================================
                for metric in metrics:

                    result = temporal_rows[solution].get(metric)

                    if (
                        result is None
                        or pd.isna(result["mean"])
                    ):

                        row[metric] = "--"
                        if metric == "Average Detection Delay":
                            row["MTD n"] = "--"
                        continue

                    mean_value = result["mean"]
                    ci_value = result["ci"]

                    text = (
                        f"{mean_value:.2f}"
                        if pd.isna(ci_value)
                        else (
                            f"{mean_value:.2f} "
                            f"$\\pm$ {ci_value:.2f}"
                        )
                    )

                    best_solution = best_by_metric.get(metric)

                    is_bold = False

                    if best_solution is not None:

                        best = temporal_rows[
                            best_solution
                        ][metric]

                        if not pd.isna(best["mean"]):

                            if pd.isna(best["ci"]):

                                is_bold = np.isclose(
                                    mean_value,
                                    best["mean"],
                                    rtol=1e-12,
                                    atol=1e-12,
                                )

                            else:

                                is_bold = (
                                    best["mean"] - best["ci"]
                                    <= mean_value
                                    <= best["mean"] + best["ci"]
                                )

                    if is_bold:

                        text = f"\\textbf{{{text}}}"

                    row[metric] = text

                    if metric == "Average Detection Delay":

                        row["MTD n"] = str(result["n"])

                table_rows.append(row)

    # ------------------------------------------------------------
    # DataFrame
    # ------------------------------------------------------------
    df_table = pd.DataFrame(
        table_rows,
        columns=[
            "Shift",
            "Shift Type",
            "Transition Window",
            "Solution",
        ] + metrics + ["MTD n"],
    )

    df_table = df_table.rename(
        columns={
            "Detection Rate":
                "DR $\\uparrow$",

            "Episode F1":
                "Episode F1 $\\uparrow$",

            "Average Detection Delay":
                "MTD $\\downarrow$",

            "MTD n":
                "$n_{\\mathrm{MTD}}$",

            "False Alarm Rate":
                "FAR $\\downarrow$",
        }
    )

    filename = os.path.join(
        write_path,
        "overall_detection_quality.tex"
    )

    # ------------------------------------------------------------
    # LaTeX table
    #
    # Transition Window:
    #   - fixed width = 2 cm
    #   - horizontally centered
    #
    # The complete tabular is wrapped in resizebox so that the
    # table fits the page width.
    # ------------------------------------------------------------
    latex = df_table.to_latex(
        index=False,
        escape=False,
        caption=(
            "Performance of data-shift detection methods. Results are "
            "separated by shift family (Concept Drift, Label Shift, and "
            "Combined Shift), temporal shift type (Sudden or Gradual), "
            "and, for gradual shifts, the transition window. A gradual "
            "shift is evaluated as an episode from Shift Round through "
            "Shift Round + Transition Window - 1; the same maximum detection "
            "delay is then applied after the episode ends for both sudden "
            "and gradual shifts. Each metric is first computed per "
            "Solution $\\times$ Dataset $\\times$ Experiment $\\times$ "
            "Fold ID and then aggregated. The best result for each metric "
            "is determined independently within each Shift $\\times$ "
            "Shift Type $\\times$ Transition Window group. Results are "
            "reported as mean $\\pm$ 95\\% confidence interval."
        ),
        label="tab:overall_detection_quality",
        column_format=(
            "ll"
            ">{\\centering\\arraybackslash}p{2cm}"
            "l"
            "ccccc"
            "c"
        ),
    )

    # ------------------------------------------------------------
    # Visual separators between shift families and temporal types
    # ------------------------------------------------------------
    # The first row of each shift family contains its \multirow label.
    # We insert:
    #   - \midrule before each new shift family;
    #   - \cline{2-10} between Sudden and Gradual inside
    #     the same shift family.
    #
    # This keeps the hierarchy visually clear:
    #
    #   Concept Drift
    #       Sudden
    #       Gradual
    #           window = 2
    #           window = 5
    #   --------------------------------
    #   Label Shift
    #       Sudden
    #       Gradual
    #   --------------------------------
    #   Combined Shift
    #       Sudden
    #       Gradual
    # ------------------------------------------------------------

    lines = latex.splitlines()
    processed_lines = []
    previous_shift = None
    previous_temporal = None

    for line in lines:
        stripped = line.strip()

        # Detect the first row of each shift family.
        current_shift = None
        if "\\multirow" in line:
            if "Concept Drift" in line:
                current_shift = "Concept Drift"
            elif "Label Shift" in line:
                current_shift = "Label Shift"
            elif "Combined Shift" in line:
                current_shift = "Combined Shift"

        # Detect the first row of each temporal type.
        current_temporal = None
        if "\\multirow" in line:
            if "{*}{Sudden}" in line:
                current_temporal = "Sudden"
            elif "{*}{Gradual}" in line:
                current_temporal = "Gradual"

        # Keep the active shift family across all rows.
        # The Shift cell is a multirow, so it only appears on the
        # first row of the family. On subsequent rows current_shift
        # is therefore inferred from previous_shift.
        active_shift = (
            current_shift
            if current_shift is not None
            else previous_shift
        )

        # Separator between different shift families.
        if (
            current_shift is not None
            and previous_shift is not None
            and current_shift != previous_shift
        ):
            processed_lines.append("\\midrule")

        # Separator between Sudden and Gradual within the same shift.
        # Since Shift is a multirow, use the active shift rather than
        # requiring a Shift value on the Gradual row itself.
        if (
            current_temporal == "Gradual"
            and previous_temporal == "Sudden"
            and active_shift == previous_shift
        ):
            # Columns 2--10 correspond to Shift Type through the metrics.
            processed_lines.append("\\cline{2-9}")

        processed_lines.append(line)

        if current_shift is not None:
            previous_shift = current_shift
        if current_temporal is not None:
            previous_temporal = current_temporal

    latex = "\n".join(processed_lines) + "\n"

    # ------------------------------------------------------------
    # Convert table to table*
    # ------------------------------------------------------------
    latex = latex.replace(
        "\\begin{table}",
        "\\begin{table*}",
        1,
    )

    latex = latex.replace(
        "\\end{table}",
        "\\end{table*}",
        1,
    )

    # ------------------------------------------------------------
    # Make the complete table fit the page width.
    # ------------------------------------------------------------
    latex = latex.replace(
        "\\begin{tabular}",
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}",
        1,
    )

    latex = latex.replace(
        "\\end{tabular}",
        "\\end{tabular}%\n}",
        1,
    )

    # ------------------------------------------------------------
    # Required LaTeX packages
    # ------------------------------------------------------------
    latex = (
        "% Requires: \\usepackage{booktabs}\n"
        "% Requires: \\usepackage{multirow}\n"
        "% Requires: \\usepackage{graphicx}\n"
        "% Requires: \\usepackage{array}\n"
        + latex
    )

    Path(write_path).mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(latex)

    print("\n" + "=" * 100)
    print("DETECTION TABLE - EXPERIMENTAL UNITS")
    print("=" * 100)
    print("\nEach metric was first calculated for:")
    print("  Solution × Dataset × Experiment × Fold ID × Shift Type × Temporal Shift Type × Transition Window")
    print("\nNumber of experimental units by shift / temporal type / window:")
    print(
        df_experimental.groupby(
            ["Shift Type", "Temporal Shift Type", "Transition Window"],
            dropna=False,
        ).size()
    )
    print(f"\nLaTeX table written to:\n{filename}")
    print("=" * 100)

    print("=" * 100)

    # ============================================================
    # SECOND TABLE: OVERALL AVERAGE ACROSS ALL SHIFTS
    # ============================================================
    #
    # The original table above is intentionally preserved unchanged.
    # This second table aggregates the valid experimental units for each
    # solution, regardless of:
    #   - shift family;
    #   - sudden/gradual temporal type;
    #   - transition window;
    #   - dataset;
    #   - experiment/configuration;
    #   - fold.
    #
    # The aggregation is performed on the already computed
    # Solution x Dataset x Experiment x Fold x Shift unit values.
    # Therefore, each valid experimental unit contributes equally.
    # For JS-Drift, Concept Drift units with DR = 0 are excluded because
    # JS-Drift was not designed for Concept Drift and those results are N/A.
    # ============================================================

    overall_rows = []

    for solution in solutions:

        df_solution = df_experimental[
            df_experimental["Solution"] == solution
        ].copy()

        # JS-Drift was not designed for Concept Drift. In the overall
        # average, Concept Drift results for JS-Drift with DR = 0 must
        # not contribute to any metric. These are the same results
        # represented as N/A/-- in the detailed table above.
        if str(solution).strip().lower() == "js-drift":
            concept_invalid = (
                df_solution["Shift Type"].astype(str).str.strip().str.lower()
                == "concept drift"
            ) & (
                pd.to_numeric(
                    df_solution["Detection Rate"],
                    errors="coerce"
                ).fillna(0.0) <= 0.0
            )
            df_solution = df_solution.loc[~concept_invalid].copy()

        if df_solution.empty:
            continue

        row = {
            "Solution": solution,
        }

        for metric in metrics:

            values = pd.to_numeric(
                df_solution[metric],
                errors="coerce"
            ).dropna()

            mean_value, ci_value = mean_ci(
                values,
                ci=ci,
                bounded=metric in {
                    "Detection Rate",
                    "Episode F1",
                    "False Alarm Rate",
                },
            )

            row[metric] = {
                "mean": mean_value,
                "ci": ci_value,
                "n": len(values),
            }

        overall_rows.append(row)

    # ------------------------------------------------------------
    # Determine the best solution for each metric globally.
    # ------------------------------------------------------------

    overall_best_by_metric = {}

    for metric in metrics:

        candidates = []

        for row in overall_rows:

            result = row.get(metric)

            if (
                result is None
                or pd.isna(result["mean"])
            ):
                continue

            candidates.append(
                (
                    row["Solution"],
                    result["mean"],
                )
            )

        if not candidates:
            continue

        if metric in higher_is_better_metrics:
            overall_best_by_metric[metric] = max(
                candidates,
                key=lambda x: x[1]
            )[0]
        else:
            overall_best_by_metric[metric] = min(
                candidates,
                key=lambda x: x[1]
            )[0]

    # ------------------------------------------------------------
    # Build the second table.
    # ------------------------------------------------------------

    overall_table_rows = []

    for row_data in overall_rows:

        solution = row_data["Solution"]

        solution_display = str(solution)

        if solution_display == "MFP_v2_dh":
            solution_display = (
                "$\\textit{MFP}_{\\textit{DDH}}$"
            )

        elif solution_display == "MFP_v2_iti":
            solution_display = (
                "$\\textit{MFP}_{\\textit{ITI}}$"
            )

        elif solution_display == "MFP_v2":
            solution_display = "MFP"

        elif solution_display == "MultiFedAvg+MFP_v2":
            solution_display = "MultiFedAvg+MFP"

        else:
            solution_display = solution_display.replace(
                "_",
                r"\_",
            )

        output_row = {
            "Solution": solution_display,
        }

        for metric in metrics:

            result = row_data.get(metric)

            if (
                result is None
                or pd.isna(result["mean"])
            ):
                output_row[metric] = "--"
                continue

            mean_value = result["mean"]
            ci_value = result["ci"]

            text = (
                f"{mean_value:.2f}"
                if pd.isna(ci_value)
                else (
                    f"{mean_value:.2f} "
                    f"$\\pm$ {ci_value:.2f}"
                )
            )

            best_solution = overall_best_by_metric.get(metric)
            is_bold = False

            if best_solution == solution:

                best_result = next(
                    (
                        r[metric]
                        for r in overall_rows
                        if r["Solution"] == best_solution
                    ),
                    None,
                )

                if (
                    best_result is not None
                    and not pd.isna(best_result["mean"])
                ):

                    if pd.isna(best_result["ci"]):
                        is_bold = np.isclose(
                            mean_value,
                            best_result["mean"],
                            rtol=1e-12,
                            atol=1e-12,
                        )

                    else:
                        is_bold = (
                            best_result["mean"]
                            - best_result["ci"]
                            <= mean_value
                            <=
                            best_result["mean"]
                            + best_result["ci"]
                        )

            if is_bold:
                text = f"\\textbf{{{text}}}"

            output_row[metric] = text

        overall_table_rows.append(output_row)

    df_overall_table = pd.DataFrame(
        overall_table_rows,
        columns=["Solution"] + metrics,
    )

    df_overall_table = df_overall_table.rename(
        columns={
            "Detection Rate":
                "DR $\\uparrow$",

            "Episode F1":
                "Episode F1 $\\uparrow$",

            "Average Detection Delay":
                "MTD $\\downarrow$",

            "False Alarm Rate":
                "FAR $\\downarrow$",
        }
    )

    overall_filename = os.path.join(
        write_path,
        "overall_detection_quality_average.tex"
    )

    overall_latex = df_overall_table.to_latex(
        index=False,
        escape=False,
        caption=(
            "Average performance of data-shift detection methods "
            "across all evaluated shifts. Results are aggregated for "
            "each solution over all shift families, temporal shift "
            "types, transition windows, datasets, experiment "
            "configurations, and folds. Each experimental unit "
            "contributes equally. Results are reported as mean "
            "$\\pm$ 95\\% confidence interval."
        ),
        label="tab:overall_detection_quality_average",
        column_format=(
            "l"
            + "c" * len(metrics)
        ),
    )

    overall_latex = overall_latex.replace(
        "\\begin{table}",
        "\\begin{table*}",
        1,
    )

    overall_latex = overall_latex.replace(
        "\\end{table}",
        "\\end{table*}",
        1,
    )

    overall_latex = overall_latex.replace(
        "\\begin{tabular}",
        "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}",
        1,
    )

    overall_latex = overall_latex.replace(
        "\\end{tabular}",
        "\\end{tabular}%\n}",
        1,
    )

    overall_latex = (
        "% Requires: \\usepackage{booktabs}\n"
        "% Requires: \\usepackage{graphicx}\n"
        + overall_latex
    )

    with open(
        overall_filename,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(overall_latex)

    print("\n" + "=" * 100)
    print("DETECTION TABLE - AVERAGE ACROSS ALL SHIFTS")
    print("=" * 100)
    print(
        "\nEach solution is averaged across ALL experimental units:"
    )
    print(
        "  Solution × Dataset × Experiment × Fold ID × Shift Type "
        "× Temporal Shift Type × Transition Window"
    )
    print(
        f"\nLaTeX table written to:\n{overall_filename}"
    )
    print("=" * 100)

    return df_experimental, df_table, df_overall_table



def _format_solution_for_accuracy_table(solution):
    """
    Formata o nome da solução para apresentação nas tabelas LaTeX.
    Mantém a mesma nomenclatura utilizada na tabela de detecção.
    """
    solution_display = str(solution)

    if solution_display == "MFP_v2_dh":
        return "$\\textit{MFP}_{\\textit{DDH}}$"

    if solution_display == "MFP_v2_iti":
        return "$\\textit{MFP}_{\\textit{ITI}}$"

    if solution_display == "MFP_v2":
        return "MFP"

    if solution_display == "MultiFedAvg+MFP_v2":
        return "MultiFedAvg+MFP"

    return solution_display.replace("_", r"\_")


def _accuracy_ci(values, ci=0.95):
    """
    Calcula média e IC de 95% da acurácia.

    A acurácia é convertida para porcentagem antes da agregação.
    """
    values = pd.to_numeric(values, errors="coerce").dropna()

    if values.empty:
        return np.nan, np.nan, 0

    values = values.to_numpy(dtype=float) * 100.0

    mean_value = float(np.mean(values))

    if len(values) == 1 or np.allclose(values, values[0]):
        return round(mean_value, 2), 0.00, len(values)

    sem = st.sem(values)

    lower, upper = st.t.interval(
        confidence=ci,
        df=len(values) - 1,
        loc=mean_value,
        scale=sem,
    )

    margin = max(
        mean_value - lower,
        upper - mean_value,
    )

    return round(mean_value, 2), round(margin, 2), len(values)


def _accuracy_is_statistically_superior_candidate(candidate, all_results):
    """
    Verifica se uma solução pertence ao grupo estatisticamente superior
    dentro de UMA configuração experimental.

    A regra para o destaque é baseada no maior valor médio observado:

      1. identifica-se a maior média de acurácia;
      2. todas as soluções com essa maior média são candidatas ao destaque;
      3. uma solução com média menor também é destacada quando seu IC de
         95% se sobrepõe ao IC de pelo menos uma das soluções com a maior
         média.

    Assim, a tabela não destaca somente um máximo pontual. Ela destaca
    o maior valor e também os valores que, considerando os ICs de 95%,
    não podem ser distinguidos do maior valor.

    Esta regra é aplicada independentemente em cada configuração
    experimental (dataset × temporal type × transition window × alpha).
    """
    if candidate is None or not all_results:
        return False

    candidate_mean = candidate.get("mean", np.nan)
    candidate_ci = candidate.get("ci", np.nan)

    if pd.isna(candidate_mean) or pd.isna(candidate_ci):
        return False

    valid_results = []
    for result in all_results:
        if result is None:
            continue

        mean_value = result.get("mean", np.nan)
        ci_value = result.get("ci", np.nan)

        if pd.isna(mean_value) or pd.isna(ci_value):
            continue

        valid_results.append(result)

    if not valid_results:
        return False

    # ------------------------------------------------------------
    # Identify the maximum mean accuracy in this exact column.
    # ------------------------------------------------------------
    max_mean = max(
        result["mean"]
        for result in valid_results
    )

    # ------------------------------------------------------------
    # All solutions attaining the maximum mean are part of the
    # statistically highest group, regardless of CI overlap.
    # ------------------------------------------------------------
    maximum_results = [
        result
        for result in valid_results
        if np.isclose(
            result["mean"],
            max_mean,
            rtol=0.0,
            atol=1e-12,
        )
    ]

    if any(candidate is result for result in maximum_results):
        return True

    # ------------------------------------------------------------
    # A lower mean is also highlighted when its 95% CI overlaps
    # the 95% CI of at least one maximum-mean solution.
    #
    # Intervals are represented as mean ± CI margin.
    # ------------------------------------------------------------
    candidate_lower = candidate_mean - candidate_ci
    candidate_upper = candidate_mean + candidate_ci

    for maximum_result in maximum_results:
        maximum_mean = maximum_result["mean"]
        maximum_ci = maximum_result["ci"]

        maximum_lower = maximum_mean - maximum_ci
        maximum_upper = maximum_mean + maximum_ci

        intervals_overlap = (
            candidate_lower <= maximum_upper
            and candidate_upper >= maximum_lower
        )

        if intervals_overlap:
            return True

    return False


def table_accuracy_concept_drift_by_dataset(
    df_all,
    write_path,
    solutions,
    concept_alphas=(0.1, 1.0),
    gradual_windows=(5, 10),
    ci=0.95,
    accuracy_rounds=None,
):
    """
    Gera três tabelas LaTeX de acurácia para Concept Drift:

        WISDM-W
        ImageNet10
        Foursquare

    Cada tabela contém, para cada solução, as seis configurações:

        Sudden, alpha=0.1
        Sudden, alpha=1.0
        Gradual (5), alpha=0.1
        Gradual (5), alpha=1.0
        Gradual (10), alpha=0.1
        Gradual (10), alpha=1.0

    Para cada configuração, a acurácia é agregada sobre as observações
    disponíveis nos CSVs, incluindo os folds e rounds correspondentes.

    A célula é apresentada como:

        mean +/- 95% CI

    Em cada configuração, o maior valor médio de acurácia é destacado.
    Também são destacados os valores menores que possuem IC de 95% que
    se sobrepõe ao IC de pelo menos uma solução com a maior média,
    representando valores estatisticamente indistinguíveis do máximo
    segundo o critério baseado na sobreposição dos ICs.

    As tabelas são salvas como:

        accuracy_concept_drift_WISDM-W.tex
        accuracy_concept_drift_ImageNet10.tex
        accuracy_concept_drift_Foursquare.tex


    Parameters
    ----------
    accuracy_rounds : None, str, iterable or dict, optional
        Controls which rounds are used to compute the accuracy reported
        in the Concept Drift tables. ``None`` or ``"all"`` uses all
        available rounds. An iterable (e.g. ``[30, 31, 32]``) uses only
        those rounds. A tuple ``(start, end)`` uses the inclusive interval.
        A dictionary allows different selections per configuration; keys
        can be the generated column name, ``(temporal, window, alpha)``,
        or ``"default"``. This affects only accuracy tables.
    """
    if df_all is None or df_all.empty:
        print(
            "\nWARNING: empty dataframe passed to "
            "table_accuracy_concept_drift_by_dataset."
        )
        return {}

    required_columns = {
        "Solution",
        "Dataset",
        "Experiment ID",
        "Transition Window",
        "Accuracy",
    }

    missing_columns = required_columns.difference(df_all.columns)

    if missing_columns:
        raise KeyError(
            "The dataframe is missing columns required for the "
            f"accuracy tables: {sorted(missing_columns)}"
        )

    Path(write_path).mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Experimental configurations
    # ------------------------------------------------------------
    configurations = [
        {
            "temporal": "Sudden",
            "window": None,
            "alpha": float(alpha),
            "experiment_id": f"concept_drift#{alpha:.1f}_sudden",
            "column": rf"Sudden, $\alpha={alpha:g}$",
        }
        for alpha in concept_alphas
    ]

    for window in gradual_windows:
        for alpha in concept_alphas:
            configurations.append(
                {
                    "temporal": "Gradual",
                    "window": int(window),
                    "alpha": float(alpha),
                    "experiment_id": (
                        f"concept_drift#{alpha:.1f}_gradual"
                    ),
                    "column": (
                        rf"Gradual ($W={int(window)}$), "
                        rf"$\alpha={alpha:g}$"
                    ),
                }
            )

    # ------------------------------------------------------------
    # Restrict to Concept Drift and normalize transition windows.
    # ------------------------------------------------------------
    df = df_all.copy()

    df = df[
        df["Experiment ID"]
        .astype(str)
        .str.startswith("concept_drift#")
    ].copy()

    if df.empty:
        print(
            "\nWARNING: no Concept Drift data found for the "
            "accuracy tables."
        )
        return {}

    def normalize_window(value):
        if pd.isna(value):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    df["_TransitionWindow"] = df["Transition Window"].apply(
        normalize_window
    )

    # Normalize Accuracy to numeric once.
    df["_Accuracy"] = pd.to_numeric(
        df["Accuracy"],
        errors="coerce",
    )

    # ------------------------------------------------------------
    # The current experiment setup uses the transition window as an
    # explicit directory dimension. For sudden, it is None.
    # ------------------------------------------------------------
    table_results = {}

    for dataset_name in [
        "WISDM-W",
        "ImageNet10",
        "Foursquare",
    ]:

        df_dataset = df[
            df["Dataset"] == dataset_name
        ].copy()

        if df_dataset.empty:
            print(
                f"\nWARNING: no Concept Drift accuracy data for "
                f"dataset {dataset_name}."
            )
            continue

        # --------------------------------------------------------
        # Resolve which rounds are used for each configuration.
        # --------------------------------------------------------
        def resolve_accuracy_rounds(config):
            selection = accuracy_rounds

            if isinstance(accuracy_rounds, dict):
                keys = [
                    config["column"],
                    (config["temporal"], config["window"], config["alpha"]),
                    "default",
                ]
                selection = None
                for key in keys:
                    if key in accuracy_rounds:
                        selection = accuracy_rounds[key]
                        break

            if selection is None:
                return None

            if isinstance(selection, str):
                if selection.strip().lower() == "all":
                    return None
                raise ValueError(
                    "accuracy_rounds must be None/'all', an iterable of rounds, "
                    "a (start, end) tuple, or a dictionary."
                )

            if isinstance(selection, tuple) and len(selection) == 2:
                try:
                    start = int(float(selection[0]))
                    end = int(float(selection[1]))
                    if start > end:
                        raise ValueError(f"Invalid accuracy round interval: {selection}")
                    return set(range(start, end + 1))
                except (TypeError, ValueError):
                    raise ValueError(f"Invalid accuracy round interval: {selection}")

            try:
                rounds = {int(float(r)) for r in selection}
            except (TypeError, ValueError):
                raise ValueError(
                    "Invalid accuracy_rounds. Use None/'all', an iterable, "
                    "a (start, end) tuple, or a dictionary."
                )

            if any(r <= 0 for r in rounds):
                raise ValueError(f"Accuracy rounds must be positive: {sorted(rounds)}")
            return rounds

        # --------------------------------------------------------
        # Calculate mean + CI independently for every:
        #
        # Solution × temporal type × transition window × alpha
        #
        # This preserves the requested experimental configuration.
        # --------------------------------------------------------
        raw_results = {
            solution: {}
            for solution in solutions
        }

        for solution in solutions:

            df_solution = df_dataset[
                df_dataset["Solution"] == solution
            ].copy()

            for config in configurations:

                mask = (
                    df_solution["Experiment ID"]
                    .astype(str)
                    .eq(config["experiment_id"])
                )

                if config["window"] is None:
                    mask &= df_solution["_TransitionWindow"].isna()
                else:
                    mask &= (
                        df_solution["_TransitionWindow"]
                        == config["window"]
                    )

                filtered = df_solution.loc[mask].copy()

                # Optional round filter for Concept Drift accuracy only.
                selected_rounds = resolve_accuracy_rounds(config)
                if selected_rounds is not None:
                    round_numeric = pd.to_numeric(
                        filtered["Round (t)"], errors="coerce"
                    )
                    filtered = filtered.loc[round_numeric.isin(selected_rounds)]

                mean_value, ci_value, n_value = _accuracy_ci(
                    filtered["_Accuracy"],
                    ci=ci,
                )

                raw_results[solution][config["column"]] = {
                    "mean": mean_value,
                    "ci": ci_value,
                    "n": n_value,
                }

        # --------------------------------------------------------
        # Build one LaTeX row per solution.
        # --------------------------------------------------------
        output_rows = []

        for solution in solutions:

            if solution not in raw_results:
                continue

            row = {
                "Solution": _format_solution_for_accuracy_table(
                    solution
                )
            }

            for config in configurations:

                column = config["column"]
                result = raw_results[solution][column]

                if (
                    pd.isna(result["mean"])
                    or result["n"] == 0
                ):
                    row[column] = "--"
                    continue

                text = (
                    f"{result['mean']:.2f} "
                    f"$\\pm$ {result['ci']:.2f}"
                )

                # ------------------------------------------------
                # Statistical superiority:
                # compare ONLY solutions within this exact
                # dataset × temporal type × transition window
                # × alpha configuration.
                # ------------------------------------------------
                candidate = result

                all_results = [
                    raw_results[other_solution][column]
                    for other_solution in solutions
                    if (
                        other_solution in raw_results
                        and raw_results[other_solution][column]["n"] > 0
                        and not pd.isna(
                            raw_results[other_solution][column]["mean"]
                        )
                        and not pd.isna(
                            raw_results[other_solution][column]["ci"]
                        )
                    )
                ]

                if _accuracy_is_statistically_superior_candidate(
                    candidate,
                    all_results,
                ):
                    text = f"\\textbf{{{text}}}"

                row[column] = text

            output_rows.append(row)

        df_table = pd.DataFrame(
            output_rows,
            columns=(
                ["Solution"]
                + [config["column"] for config in configurations]
            ),
        )

        # --------------------------------------------------------
        # LaTeX
        # --------------------------------------------------------
        filename = os.path.join(
            write_path,
            f"accuracy_concept_drift_{dataset_name}.tex",
        )

        column_format = "l" + "c" * len(configurations)

        latex = df_table.to_latex(
            index=False,
            escape=False,
            caption=(
                f"Accuracy of the evaluated solutions under Concept "
                f"Drift for {dataset_name}. Results are reported for "
                f"sudden and gradual shifts with transition windows "
                f"$W=5$ and $W=10$, considering $\\alpha=0.1$ and "
                f"$\\alpha=1.0$. For each configuration, the mean "
                f"accuracy and its 95\\% confidence interval are "
                f"computed over the selected rounds and folds. "
                f"For each configuration, the highest mean accuracy "
                f"is highlighted in bold, together with any lower "
                f"accuracy whose 95\\% confidence interval overlaps "
                f"the interval of a solution attaining the highest "
                f"mean accuracy."
            ),
            label=(
                "tab:accuracy_concept_drift_"
                + dataset_name.lower().replace("-", "_")
            ),
            column_format=column_format,
        )

        latex = (
            "% Requires: \\usepackage{booktabs}\n"
            + latex
        )

        # Use table* so the six configurations fit comfortably in
        # a two-column paper.
        latex = latex.replace(
            "\\begin{table}",
            "\\begin{table*}",
            1,
        )
        latex = latex.replace(
            "\\end{table}",
            "\\end{table*}",
            1,
        )

        latex = latex.replace(
            "\\begin{tabular}",
            "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}",
            1,
        )
        latex = latex.replace(
            "\\end{tabular}",
            "\\end{tabular}%\n}",
            1,
        )

        latex = latex.replace(
            "MFP\\_v2\\_dh",
            "$\\textit{MFP}_{\\textit{DDH}}$",
        )
        latex = latex.replace(
            "MFP\\_v2\\_iti",
            "$\\textit{MFP}_{\\textit{ITI}}$",
        )
        latex = latex.replace(
            "MFP\\_v2",
            "$\\textit{MFP}$",
        )

        with open(
            filename,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(latex)

        table_results[dataset_name] = df_table

        print("\n" + "=" * 100)
        print(
            "ACCURACY TABLE - CONCEPT DRIFT - "
            f"{dataset_name}"
        )
        print("=" * 100)
        print(df_table.to_string(index=False))
        print(
            f"\nLaTeX table written to:\n{filename}"
        )
        print("=" * 100)

    return table_results


def table_accuracy_label_shift_by_dataset(
    df_all,
    write_path,
    solutions,
    label_config=("0.1", "1.0"),
    gradual_windows=(5, 10),
    ci=0.95,
    accuracy_rounds=None,
):
    """
    Gera três tabelas LaTeX de acurácia para Label Shift:

        WISDM-W
        ImageNet10
        Foursquare

    Como o experimento atual utiliza uma única configuração de Label Shift
    (0.1 -> 1.0), cada tabela contém as três configurações temporais:

        Sudden, 0.1 -> 1.0
        Gradual (5), 0.1 -> 1.0
        Gradual (10), 0.1 -> 1.0

    A organização e o critério estatístico seguem o mesmo padrão das
    tabelas de Concept Drift: uma linha por solução, resultados como
    mean +/- 95% CI e negrito somente quando o intervalo de confiança
    da solução é estritamente separado acima dos intervalos das demais
    soluções na mesma configuração.

    ``accuracy_rounds`` possui a mesma semântica da tabela de Concept
    Drift e afeta somente estas tabelas de acurácia.
    """
    if df_all is None or df_all.empty:
        print(
            "\nWARNING: empty dataframe passed to "
            "table_accuracy_label_shift_by_dataset."
        )
        return {}

    required_columns = {
        "Solution",
        "Dataset",
        "Experiment ID",
        "Transition Window",
        "Accuracy",
    }

    missing_columns = required_columns.difference(df_all.columns)
    if missing_columns:
        raise KeyError(
            "The dataframe is missing columns required for the "
            f"Label Shift accuracy tables: {sorted(missing_columns)}"
        )

    Path(write_path).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Current Label Shift configuration
    # ------------------------------------------------------------
    alpha_before = float(label_config[0])
    alpha_after = float(label_config[1])
    config_id = f"{alpha_before:g}-{alpha_after:g}"

    configurations = [
        {
            "temporal": "Sudden",
            "window": None,
            "experiment_id": f"label_shift#{alpha_before:.1f}-{alpha_after:.1f}_sudden",
            "column": rf"Sudden, ${alpha_before:g}\rightarrow {alpha_after:g}$",
        }
    ]

    for window in gradual_windows:
        configurations.append(
            {
                "temporal": "Gradual",
                "window": int(window),
                "experiment_id": (
                    f"label_shift#{alpha_before:.1f}-{alpha_after:.1f}_gradual"
                ),
                "column": (
                    rf"Gradual ($W={int(window)}$), "
                    rf"${alpha_before:g}\rightarrow {alpha_after:g}$"
                ),
            }
        )

    # ------------------------------------------------------------
    # Restrict to Label Shift and normalize transition windows.
    # ------------------------------------------------------------
    df = df_all.copy()
    df = df[
        df["Experiment ID"]
        .astype(str)
        .str.startswith("label_shift#")
    ].copy()

    if df.empty:
        print(
            "\nWARNING: no Label Shift data found for the "
            "accuracy tables."
        )
        return {}

    def normalize_window(value):
        if pd.isna(value):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    df["_TransitionWindow"] = df["Transition Window"].apply(
        normalize_window
    )

    df["_Accuracy"] = pd.to_numeric(
        df["Accuracy"],
        errors="coerce",
    )

    table_results = {}

    for dataset_name in [
        "WISDM-W",
        "ImageNet10",
        "Foursquare",
    ]:
        df_dataset = df[
            df["Dataset"] == dataset_name
        ].copy()

        if df_dataset.empty:
            print(
                f"\nWARNING: no Label Shift accuracy data for "
                f"dataset {dataset_name}."
            )
            continue

        # --------------------------------------------------------
        # Resolve selected rounds using the same mechanism as the
        # Concept Drift tables.
        # --------------------------------------------------------
        def resolve_accuracy_rounds(config):
            selection = accuracy_rounds

            if isinstance(accuracy_rounds, dict):
                keys = [
                    config["column"],
                    (config["temporal"], config["window"]),
                    "default",
                ]
                selection = None
                for key in keys:
                    if key in accuracy_rounds:
                        selection = accuracy_rounds[key]
                        break

            if selection is None:
                return None

            if isinstance(selection, str):
                if selection.strip().lower() == "all":
                    return None
                raise ValueError(
                    "accuracy_rounds must be None/'all', an iterable of rounds, "
                    "a (start, end) tuple, or a dictionary."
                )

            if isinstance(selection, tuple) and len(selection) == 2:
                try:
                    start = int(float(selection[0]))
                    end = int(float(selection[1]))
                    if start > end:
                        raise ValueError(
                            f"Invalid accuracy round interval: {selection}"
                        )
                    return set(range(start, end + 1))
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Invalid accuracy round interval: {selection}"
                    )

            try:
                rounds = {int(float(r)) for r in selection}
            except (TypeError, ValueError):
                raise ValueError(
                    "Invalid accuracy_rounds. Use None/'all', an iterable, "
                    "a (start, end) tuple, or a dictionary."
                )

            if any(r <= 0 for r in rounds):
                raise ValueError(
                    f"Accuracy rounds must be positive: {sorted(rounds)}"
                )
            return rounds

        # --------------------------------------------------------
        # Calculate mean + CI independently for every:
        #
        # Solution × temporal type × transition window
        #
        # There is intentionally no alpha dimension here because the
        # current Label Shift experiment has only one transition
        # configuration: 0.1 -> 1.0.
        # --------------------------------------------------------
        raw_results = {
            solution: {}
            for solution in solutions
        }

        for solution in solutions:
            df_solution = df_dataset[
                df_dataset["Solution"] == solution
            ].copy()

            for config in configurations:
                mask = (
                    df_solution["Experiment ID"]
                    .astype(str)
                    .eq(config["experiment_id"])
                )

                if config["window"] is None:
                    mask &= df_solution["_TransitionWindow"].isna()
                else:
                    mask &= (
                        df_solution["_TransitionWindow"]
                        == config["window"]
                    )

                filtered = df_solution.loc[mask].copy()

                selected_rounds = resolve_accuracy_rounds(config)
                if selected_rounds is not None:
                    round_numeric = pd.to_numeric(
                        filtered["Round (t)"],
                        errors="coerce",
                    )
                    filtered = filtered.loc[
                        round_numeric.isin(selected_rounds)
                    ]

                mean_value, ci_value, n_value = _accuracy_ci(
                    filtered["_Accuracy"],
                    ci=ci,
                )

                raw_results[solution][config["column"]] = {
                    "mean": mean_value,
                    "ci": ci_value,
                    "n": n_value,
                }

        # --------------------------------------------------------
        # Build one LaTeX row per solution.
        # --------------------------------------------------------
        output_rows = []

        for solution in solutions:
            if solution not in raw_results:
                continue

            row = {
                "Solution": _format_solution_for_accuracy_table(solution)
            }

            for config in configurations:
                column = config["column"]
                result = raw_results[solution][column]

                if (
                    pd.isna(result["mean"])
                    or result["n"] == 0
                ):
                    row[column] = "--"
                    continue

                value_text = (
                    f"{result['mean']:.2f} "
                    f"$\\pm$ {result['ci']:.2f}"
                )

                # Same statistical criterion used by the Concept Drift
                # tables: bold only for strict CI separation above all
                # other solutions in this exact configuration.
                all_results = [
                    raw_results[other_solution][column]
                    for other_solution in solutions
                    if (
                        other_solution in raw_results
                        and raw_results[other_solution][column]["n"] > 0
                        and not pd.isna(
                            raw_results[other_solution][column]["mean"]
                        )
                        and not pd.isna(
                            raw_results[other_solution][column]["ci"]
                        )
                    )
                ]

                if _accuracy_is_statistically_superior_candidate(
                    result,
                    all_results,
                ):
                    value_text = f"\\textbf{{{value_text}}}"

                row[column] = value_text

            output_rows.append(row)

        df_table = pd.DataFrame(
            output_rows,
            columns=(
                ["Solution"]
                + [config["column"] for config in configurations]
            ),
        )

        filename = os.path.join(
            write_path,
            f"accuracy_label_shift_{dataset_name}.tex",
        )

        latex = df_table.to_latex(
            index=False,
            escape=False,
            caption=(
                f"Accuracy of the evaluated solutions under Label Shift "
                f"for {dataset_name}. Results are reported for the "
                f"0.1 $\\rightarrow$ 1.0 label-distribution transition "
                f"under sudden and gradual shifts with transition windows "
                f"$W=5$ and $W=10$. For each configuration, the mean "
                f"accuracy and its 95\\% confidence interval are computed "
                f"over the selected rounds and folds. The highest mean "
                f"accuracy is highlighted in bold only when its 95\\% "
                f"confidence interval is strictly separated above the "
                f"confidence intervals of all other solutions."
            ),
            label=(
                "tab:accuracy_label_shift_"
                + dataset_name.lower().replace("-", "_")
            ),
            column_format="l" + "c" * len(configurations),
        )

        latex = (
            "% Requires: \\usepackage{booktabs}\n"
            + latex
        )

        # Same two-column-paper formatting used by Concept Drift.
        latex = latex.replace(
            "\\begin{table}",
            "\\begin{table*}",
            1,
        )
        latex = latex.replace(
            "\\end{table}",
            "\\end{table*}",
            1,
        )
        latex = latex.replace(
            "\\begin{tabular}",
            "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}",
            1,
        )
        latex = latex.replace(
            "\\end{tabular}",
            "\\end{tabular}%\n}",
            1,
        )

        latex = latex.replace(
            "MFP\\_v2\\_dh",
            "$\\textit{MFP}_{\\textit{DDH}}$",
        )
        latex = latex.replace(
            "MFP\\_v2\\_iti",
            "$\\textit{MFP}_{\\textit{ITI}}$",
        )
        latex = latex.replace(
            "MFP\\_v2",
            "$\\textit{MFP}$",
        )

        with open(filename, "w", encoding="utf-8") as f:
            f.write(latex)

        table_results[dataset_name] = df_table

        print("\n" + "=" * 100)
        print(
            "ACCURACY TABLE - LABEL SHIFT - "
            f"{dataset_name}"
        )
        print("=" * 100)
        print(df_table.to_string(index=False))
        print(f"\nLaTeX table written to:\n{filename}")
        print("=" * 100)

    return table_results

def extract_alpha_from_experiment(experiment_id):
    """
    Extrai os valores de alpha do Experiment ID.

    Concept Drift:
        concept_drift#0.1_sudden
        concept_drift#0.1_gradual
        -> 0.1

    Label Shift:
        label_shift#0.1-1.0_sudden
        label_shift#0.1-1.0_gradual
        -> (0.1, 1.0)

    Combined Shift:
        combined_shift#0.1-1.0_sudden
        combined_shift#0.1-1.0_gradual
        -> (0.1, 1.0)

        The result directory uses:
        alpha_[0.1, 0.1, 0.1]
        i.e., alpha_before for all datasets.
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

    # Remove the temporal suffix from both sudden and gradual
    # experiment identifiers.
    #
    # Supported:
    #   concept_drift#0.1_sudden
    #   concept_drift#0.1_gradual
    #   label_shift#0.1-1.0_sudden
    #   label_shift#0.1-1.0_gradual
    #   combined_shift#0.1-1.0_sudden
    #   combined_shift#0.1-1.0_gradual
    #
    # The temporal suffix must not be part of the alpha value.
    config = re.sub(
        r"_(?:sudden|gradual)$",
        "",
        config,
        flags=re.IGNORECASE,
    )

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
    # LABEL SHIFT / COMBINED SHIFT
    # ============================================================

    if shift_type in ("label_shift", "combined_shift"):

        if "-" not in config:
            raise ValueError(
                f"Configuração de {shift_type} inválida: "
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



def _accuracy_solution_display_name(solution):
    """Convert internal solution names to the names used in LaTeX tables."""
    solution = str(solution)

    replacements = {
        "MultiFedAvg+MFP_v2": "MultiFedAvg+MFP",
        "MFP_v2_dh": r"$\textit{MFP}_{\textit{DDH}}$",
        "MFP_v2_iti": r"$\textit{MFP}_{\textit{ITI}}$",
        "MFP_v2": "MFP",
    }

    if solution in replacements:
        return replacements[solution]

    return solution.replace("_", r"\_")


def _accuracy_is_statistically_superior(results, candidate_solution):
    """
    A solution is marked as superior only if:
      1. it has the highest mean accuracy; and
      2. its 95% CI is strictly separated from the 95% CIs of ALL
         other solutions in the same experimental configuration.

    Thus, overlapping confidence intervals do not produce boldface.
    """
    candidate = results.get(candidate_solution)

    if candidate is None or pd.isna(candidate["mean"]):
        return False

    candidate_mean = candidate["mean"]
    candidate_ci = candidate["ci"]

    for other_solution, other in results.items():
        if other_solution == candidate_solution:
            continue

        if other is None or pd.isna(other["mean"]):
            continue

        # If the candidate does not have a usable CI, we cannot claim
        # statistical superiority from CI separation.
        if pd.isna(candidate_ci) or pd.isna(other["ci"]):
            return False

        candidate_lower = candidate_mean - candidate_ci
        other_upper = other["mean"] + other["ci"]

        # Strict separation is required.
        if candidate_lower <= other_upper:
            return False

    return True


def table_accuracy_concept_drift(
    df,
    write_path,
    solutions_order,
    datasets_order=None,
    alphas_order=(0.1, 1.0),
    ci=0.95,
):
    """
    Generate one LaTeX accuracy table for each Concept Drift dataset.

    Each table contains:
        Sudden, alpha=0.1
        Sudden, alpha=1.0
        Gradual W=5, alpha=0.1
        Gradual W=5, alpha=1.0
        Gradual W=10, alpha=0.1
        Gradual W=10, alpha=1.0

    Accuracy is aggregated over the rounds/folds present in the CSVs.
    The mean and 95% t-confidence interval are computed over the
    available Accuracy observations.

    A solution is bolded only when its 95% CI is strictly separated
    above the CI of every other solution in that same configuration.
    """
    if df is None or df.empty:
        print("\nWARNING: empty dataframe; Concept Drift accuracy tables not generated.")
        return

    required_columns = {
        "Accuracy",
        "Dataset",
        "Solution",
        "Experiment ID",
        "Round (t)",
        "Alpha",
        "Transition Window",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise KeyError(
            "Missing columns for Concept Drift accuracy tables: "
            + ", ".join(sorted(missing))
        )

    concept = df[
        df["Experiment ID"].astype(str).str.startswith("concept_drift#")
    ].copy()

    # Keep exactly the requested alpha values.
    concept["Alpha"] = pd.to_numeric(concept["Alpha"], errors="coerce")
    concept = concept[
        concept["Alpha"].apply(
            lambda x: any(np.isclose(x, a, rtol=1e-12, atol=1e-12)
                          for a in alphas_order)
            if not pd.isna(x) else False
        )
    ].copy()

    # Derive temporal type from Experiment ID.
    exp_lower = concept["Experiment ID"].astype(str).str.strip().str.lower()
    concept["Temporal Type"] = np.select(
        [
            exp_lower.str.endswith("_sudden"),
            exp_lower.str.endswith("_gradual"),
        ],
        [
            "Sudden",
            "Gradual",
        ],
        default="N/A",
    )

    concept["Transition Window"] = pd.to_numeric(
        concept["Transition Window"],
        errors="coerce",
    )

    if datasets_order is None:
        datasets_order = sorted(concept["Dataset"].dropna().unique().tolist())

    Path(write_path).mkdir(parents=True, exist_ok=True)

    # Requested order of configurations.
    configurations = [
        ("Sudden", None, 0.1),
        ("Sudden", None, 1.0),
        ("Gradual", 5, 0.1),
        ("Gradual", 5, 1.0),
        ("Gradual", 10, 0.1),
        ("Gradual", 10, 1.0),
    ]

    # Keep only solutions that actually occur in the dataframe.
    solutions = [
        s for s in solutions_order
        if s in concept["Solution"].astype(str).values
    ]

    for dataset_name in datasets_order:
        dataset_df = concept[
            concept["Dataset"].astype(str) == str(dataset_name)
        ].copy()

        table_rows = []

        for temporal_type, transition_window, alpha in configurations:
            # Match alpha robustly rather than relying on exact floating
            # point equality.
            mask_alpha = np.isclose(
                dataset_df["Alpha"].astype(float),
                float(alpha),
                rtol=1e-12,
                atol=1e-12,
            )

            filtered = dataset_df[
                (dataset_df["Temporal Type"] == temporal_type)
                & mask_alpha
            ].copy()

            if temporal_type == "Sudden":
                filtered = filtered[filtered["Transition Window"].isna()]
            else:
                filtered = filtered[
                    np.isclose(
                        filtered["Transition Window"].astype(float),
                        float(transition_window),
                        rtol=1e-12,
                        atol=1e-12,
                    )
                ]

            results = {}

            for solution in solutions:
                values = pd.to_numeric(
                    filtered.loc[
                        filtered["Solution"].astype(str) == str(solution),
                        "Accuracy",
                    ],
                    errors="coerce",
                ).dropna()

                # Accuracy in the CSV is expected in [0, 1].
                # Convert to percentage for the table.
                values = values * 100.0

                mean_value, ci_value = mean_ci(
                    values,
                    ci=ci,
                    bounded=False,
                )

                results[solution] = {
                    "mean": mean_value,
                    "ci": ci_value,
                    "n": len(values),
                }

            for solution in solutions:
                result = results[solution]

                if pd.isna(result["mean"]):
                    value_text = "--"
                elif pd.isna(result["ci"]):
                    value_text = f"{result['mean']:.2f}"
                else:
                    value_text = (
                        f"{result['mean']:.2f} "
                        f"$\\pm$ {result['ci']:.2f}"
                    )

                if _accuracy_is_statistically_superior(results, solution):
                    value_text = f"\\textbf{{{value_text}}}"

                table_rows.append({
                    "Temporal Type": temporal_type,
                    "Transition Window": (
                        "N/A"
                        if transition_window is None
                        else str(transition_window)
                    ),
                    r"$\alpha$": f"{alpha:g}",
                    "Solution": _accuracy_solution_display_name(solution),
                    "Accuracy": value_text,
                })

        table_df = pd.DataFrame(
            table_rows,
            columns=[
                "Temporal Type",
                "Transition Window",
                r"$\alpha$",
                "Solution",
                "Accuracy",
            ],
        )

        filename = os.path.join(
            write_path,
            f"accuracy_concept_drift_{dataset_name}.tex",
        )

        latex = table_df.to_latex(
            index=False,
            escape=False,
            caption=(
                f"Accuracy of the evaluated solutions on {dataset_name} "
                "under Concept Drift. Results are reported as mean "
                "$\\pm$ 95\\% confidence interval for each temporal "
                "shift configuration and alpha. A solution is shown in "
                "\\textbf{bold} only when its confidence interval is "
                "strictly separated above the confidence intervals of "
                "all other solutions in the same configuration."
            ),
            label=(
                "tab:accuracy_concept_drift_"
                + re.sub(r"[^A-Za-z0-9]+", "_", str(dataset_name)).strip("_").lower()
            ),
            column_format="ccccc",
        )

        latex = (
            "% Requires: \\usepackage{booktabs}\n"
            + latex
        )

        with open(filename, "w", encoding="utf-8") as f:
            f.write(latex)

        print(
            f"\nAccuracy Concept Drift table generated for {dataset_name}:"
            f"\n{filename}"
        )


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

    # Rodada em que o data shift ocorre em cada dataset.
    dataset_shift_rounds = {
        "WISDM-W": 30,
        "ImageNet10": 50,
        "Foursquare": 70
    }

    model_name = [
        "gru",
        "CNN",
        "lstm"
    ]

    fraction_fit = 0.375
    number_of_rounds = 100
    local_epochs = 1

    # ============================================================
    # JANELA MÁXIMA DE DETECÇÃO
    # ============================================================
    # Tolerância máxima de detecção APÓS O FIM do episódio ground-truth.
    #
    # Sudden:
    #   shift em t=70, valor 10 -> detecções válidas de 70 até 80.
    #
    # Gradual:
    #   shift começa em t=70, Transition Window=5 -> episódio 70..74.
    #   Com valor 10 -> detecções válidas de 70 até 84.
    #
    # Assim, Transition Window representa a duração do fenômeno,
    # enquanto MAX_DETECTION_DELAY representa a tolerância do detector
    # após o término do fenômeno. A regra pós-transição é a mesma para
    # sudden e gradual.
    #
    # Use None para considerar qualquer detecção após o início do shift
    # como válida.
    MAX_DETECTION_DELAY = 10

    # ============================================================
    # COMO A TRANSITION WINDOW É USADA NAS MÉTRICAS
    # ============================================================
    # True  -> comportamento episódico para gradual:
    #          a detecção é válida desde Shift Round até
    #          Shift Round + Transition Window - 1, mais
    #          MAX_DETECTION_DELAY após o fim do episódio.
    #
    # False -> gradual é avaliado como sudden para as métricas:
    #          a detecção é válida desde Shift Round até
    #          Shift Round + MAX_DETECTION_DELAY.
    #
    # IMPORTANTE: isso NÃO remove a Transition Window da tabela.
    # Ela continua sendo mantida como configuração experimental
    # e as janelas 5, 10 etc. continuam sendo grupos separados.
    USE_TRANSITION_WINDOW_FOR_GRADUAL_METRICS = False

    train_test = "test"

    # Soluções utilizadas nas tabelas de detecção e nos demais shifts.
    # Ordem canônica das soluções nas tabelas:
    # MFP -> FPD (quando aplicável) -> demais soluções/variantes -> baselines.
    # A mesma ordem é reutilizada nas tabelas de Concept Drift e nas demais
    # tabelas que usam esta lista.
    solutions = [
        "MultiFedAvg+MFP_v2",
        "MultiFedAvg+FPD",
        "MultiFedAvg+MFP_v2_iti",
        "MultiFedAvg+MFP_v2_dh",
        "MultiFedAvg",
        "MultiFedAvgRR",
        "JS-Drift",
        "FedConD",
        "FedDCA",
        "CDA-FedAvg",
    ]

    # Lista específica para as tabelas de Accuracy de Concept Drift.
    # Mantém exatamente a mesma ordem canônica:
    # MFP -> FPD -> demais variantes/soluções -> baselines.
    concept_solutions = [
        "MultiFedAvg+MFP_v2",
        "MultiFedAvg+MFP_v2_dh",
        "MultiFedAvg+MFP_v2_iti",
        "MultiFedAvg+FPD",
        "MultiFedAvg",
        "MultiFedAvgRR",
        "FedConD",
        "FedDCA",
        "CDA-FedAvg",
    ]

    concept_experiments = [
        "concept_drift#0.1_sudden",
        "concept_drift#1.0_sudden",
        "concept_drift#10.0_sudden",
        "concept_drift#0.1_gradual",
        "concept_drift#1.0_gradual",
        "concept_drift#10.0_gradual"
    ]

    # Lista específica para as tabelas de Accuracy de Label Shift.
    # Mantém a mesma ordem canônica usada em Concept Drift.
    label_solutions = [
        "MultiFedAvg+MFP_v2",
        "MultiFedAvg+MFP_v2_dh",
        "MultiFedAvg+MFP_v2_iti",
        "MultiFedAvg+FPD",
        "MultiFedAvg",
        "MultiFedAvgRR",
        "FedConD",
        "FedDCA",
        "CDA-FedAvg",
    ]

    label_experiments = [
        "label_shift#0.1-1.0_sudden",
        "label_shift#0.1-1.0_gradual",
        # "label_shift#0.1-10.0_sudden",
        # "label_shift#1.0-0.1_sudden",
        # "label_shift#1.0-10.0_sudden",
        # "label_shift#10.0-0.1_sudden",
        # "label_shift#10.0-1.0_sudden"
    ]

    combined_experiments = [
        # "combined_shift#0.1-1.0_sudden",
        # "combined_shift#0.1-10.0_sudden"
    ]

    experiment_ids = (
            concept_experiments
            + label_experiments
            + combined_experiments
    )

    # Gradual shifts have an explicit transition-window directory level.
    # Read every configured gradual window independently.
    GRADUAL_TRANSITION_WINDOWS = [5, 10]

    experiment_configurations = []
    for _experiment_id in experiment_ids:
        if _experiment_id.lower().endswith("_gradual"):
            for _transition_window in GRADUAL_TRANSITION_WINDOWS:
                experiment_configurations.append(
                    (_experiment_id, _transition_window)
                )
        else:
            experiment_configurations.append(
                (_experiment_id, None)
            )

    df_all = None

    # ============================================================
    # LEITURA DOS RESULTADOS
    # ============================================================

    for experiment_id, transition_window in experiment_configurations:

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

        # ============================================================
        # COMBINED SHIFT
        # ============================================================
        elif experiment_id.startswith("combined_shift#"):
            alpha_before, alpha_after = alpha_config

            # IMPORTANT:
            # The result directories generated for Combined Shift use
            # the same alpha value for all datasets. For example:
            #
            # combined_shift#0.1-1.0_sudden
            #
            # is stored under:
            #
            # alpha_[0.1, 0.1, 0.1]
            #
            # The alpha_after value is part of the Experiment ID and
            # identifies the shift configuration, but it is NOT used
            # in the directory name.
            alphas = [
                alpha_before,
                alpha_before,
                alpha_before
            ]

            alpha_value = alpha_config

        else:

            raise ValueError(
                f"Experiment ID não suportado: {experiment_id}"
            )

        # Concept Drift usa a lista expandida; Label/Combined Shift
        # continuam usando somente as soluções atuais.
        active_solutions = (
            concept_solutions
            if experiment_id.startswith("concept_drift#")
            else solutions
        )

        read_solutions = {
            solution: []
            for solution in active_solutions
        }

        read_dataset_order = []

        for solution in active_solutions:

            # A diretoria continua sendo construída com a LISTA
            # completa de datasets, exatamente como no código original.
            base_read_path = (
                "../system/results/"
                "experiment_id_{}/"
                "clients_{}/"
                "alpha_{}/"
            ).format(
                experiment_id,
                total_clients,
                alphas,
            )

            if transition_window is not None:
                base_read_path += (
                    f"transition_window_{transition_window}/"
                )

            read_path = (
                base_read_path
                + "{}/"
                + "{}/"
                + "fc_{}/"
                + "rounds_{}/"
                + "epochs_{}/"
                + "{}/"
            ).format(
                dataset,
                model_name,
                fraction_fit,
                number_of_rounds,
                local_epochs,
                train_test
            )

            # Existe um CSV por dataset para cada solução.
            # Portanto, o nome do dataset entra SOMENTE no nome do CSV,
            # e não substitui a lista 'dataset' utilizada na diretoria.
            for dataset_name in dataset:

                detection_file = os.path.join(
                    read_path,
                    f"{dataset_name}_{solution}.csv"
                )

                read_solutions[solution].append(
                    detection_file
                )

                print(
                    f"\nLendo resultados para {solution} / {dataset_name}:"
                    f"\n{detection_file}"
                )

        print("\n" + "=" * 80)
        print("DEBUG - DIRETÓRIO DE LEITURA")
        print("=" * 80)

        print(f"Experiment ID     : {experiment_id}")
        print(f"Shift mode        : {'Gradual' if transition_window is not None else 'Sudden'}")
        print(f"Transition Window : {transition_window if transition_window is not None else 'N/A'}")
        print(f"Alphas            : {alphas}")
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
            alpha_value=alpha_value,
            dataset_shift_rounds=dataset_shift_rounds,
            transition_window=transition_window
        )

        # Preserve the experiment identifier in the final dataframe.
        # This is required so that detection metrics are first computed
        # independently for each Solution × Dataset × Experiment × Fold ID.
        if df is not None and not df.empty:
            df["Experiment ID"] = experiment_id
            df["Shift Type"] = (
                "Concept drift"
                if experiment_id.startswith("concept_drift#")
                else "Label shift"
                if experiment_id.startswith("label_shift#")
                else "Combined shift"
                if experiment_id.startswith("combined_shift#")
                else "N/A"
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
            "Nenhum CSV de resultados foi encontrado."
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

    # A leitura atual utiliza somente as colunas do novo CSV:
    # Fold ID, Round (t), Data shift e Solution.
    print(
        df_all[
            [
                "Solution",
                "Dataset",
                "Fold ID",
                "Round (t)",
                "Data shift",
                "Accuracy",
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
    # TABELAS ADICIONAIS: ACCURACY -- CONCEPT DRIFT
    # ============================================================
    #
    # IMPORTANT:
    # Alpha is taken from the Experiment ID and stored explicitly in
    # df_all["Alpha"]. The requested values 0.1 and 1.0 are therefore
    # kept separate, including for gradual windows 5 and 10.
    # ============================================================

    table_accuracy_concept_drift(
        df=df_all,
        write_path=write_path,
        solutions_order=concept_solutions,
        datasets_order=dataset,
        alphas_order=(0.1, 1.0),
        ci=0.95,
    )

    # ============================================================
    # TABELAS ADICIONAIS: ACCURACY -- LABEL SHIFT
    # ============================================================
    #
    # O experimento atual possui uma única configuração de Label Shift:
    # 0.1 -> 1.0. Portanto, a tabela mantém essa configuração fixa e
    # separa apenas o tipo temporal:
    #
    #   Sudden
    #   Gradual, W=5
    #   Gradual, W=10
    #
    # A estrutura, IC de 95% e critério de negrito são os mesmos
    # utilizados nas tabelas de Concept Drift.
    # ============================================================
    ACCURACY_LABEL_ROUNDS = None

    accuracy_label_tables = (
        table_accuracy_label_shift_by_dataset(
            df_all=df_all,
            write_path=write_path,
            solutions=label_solutions,
            label_config=("0.1", "1.0"),
            gradual_windows=(5, 10),
            ci=0.95,
            accuracy_rounds=ACCURACY_LABEL_ROUNDS,
        )
    )

    print(
        "\nTabelas adicionais de acurácia de Label Shift geradas: "
        f"{list(accuracy_label_tables.keys())}"
    )

    # ============================================================
    # TABELA PRINCIPAL
    # ============================================================

    # Detection metrics are computed per:
    # Solution × Dataset × Experiment × Fold ID
    #
    # "Data shift == DATA_SHIFT" is interpreted as a detector alarm.
    metrics = [
        "Episode F1",
        "Detection Rate",
        "False Alarm Rate",
        "Average Detection Delay",
    ]

    higher_is_better_metrics = {
        "Detection Rate",
        "Episode F1",
    }

    print("\n" + "=" * 100)
    print("VALIDAÇÃO DOS CSVs UTILIZADOS NA TABELA")
    print("=" * 100)

    for solution in concept_solutions:

        solution_df = df_all[
            df_all["Solution"] == solution
            ].copy()

        print(f"\nSolution: {solution}")

        if solution_df.empty:
            print("  *** NENHUM DADO CARREGADO ***")
            continue

        print(f"  Número de linhas: {len(solution_df)}")
        print(f"  Datasets: {solution_df['Dataset'].unique()}")
        print(f"  Fold IDs: {solution_df['Fold ID'].unique()}")
        print(f"  Data shift: {solution_df['Data shift'].unique()}")
        print(
            f"  Shift Rounds: "
            f"{solution_df[['Dataset', 'Shift Round']].drop_duplicates().to_dict('records')}"
        )
        print(
            f"  Rounds: {solution_df['Round (t)'].min()} "
            f"-> {solution_df['Round (t)'].max()}"
        )

    print(df_all)

    # ============================================================
    # HELPERS PARA IDENTIFICAR A CONFIGURAÇÃO EXPERIMENTAL
    # ============================================================

    def _validation_shift_type(experiment_id):
        """
        Extrai o tipo de data shift do Experiment ID.
        """
        exp = str(experiment_id or "").strip().lower()

        if exp.startswith("concept_drift#"):
            return "Concept drift"
        if exp.startswith("label_shift#"):
            return "Label shift"
        if exp.startswith("combined_shift#"):
            return "Combined shift"

        return "N/A"


    def _validation_temporal_type(experiment_id):
        """
        Extrai se o shift é sudden ou gradual.
        """
        exp = str(experiment_id or "").strip().lower()

        if exp.endswith("_sudden"):
            return "Sudden"
        if exp.endswith("_gradual"):
            return "Gradual"

        return "N/A"


    def _validation_alpha(experiment_id):
        """
        Extrai a configuração/alpha do Experiment ID.

        Exemplos:
            concept_drift#0.1_sudden
                -> 0.1

            label_shift#0.1-1.0_sudden
                -> 0.1 -> 1.0

            combined_shift#0.1-1.0_sudden
                -> 0.1 -> 1.0
        """
        exp = str(experiment_id or "").strip()

        if "#" not in exp:
            return "N/A"

        config = exp.split("#", 1)[1]

        if "_" in config:
            config = config.split("_", 1)[0]

        if not config:
            return "N/A"

        # Mantém a configuração de label/combined shift legível.
        if "-" in config:
            parts = config.split("-", 1)

            try:
                first = float(parts[0])
                second = float(parts[1])
                return f"{first:g} -> {second:g}"
            except (ValueError, TypeError):
                return config

        try:
            return f"{float(config):g}"
        except (ValueError, TypeError):
            return config


    # ============================================================
    # VERIFICAÇÃO FINAL: ARQUIVOS AUSENTES OU INCOMPLETOS
    # ============================================================
    #
    # Critério solicitado:
    #   * 1 fold distinto  -> 100 rodadas completas;
    #   * >1 fold distinto -> 500 rodadas completas.
    #
    # A lista abaixo inclui também configurações sem arquivo, porque
    # read_data() registra TODOS os caminhos esperados antes de continuar.
    # ============================================================

    print("\n" + "=" * 100)
    print("CONFIGURAÇÕES SEM ARQUIVO OU INCOMPLETAS")
    print("=" * 100)

    incomplete_records = [
        record
        for record in EXPERIMENT_VALIDATION
        if not record.get("complete", False)
    ]

    if incomplete_records:

        for record in sorted(
            incomplete_records,
            key=lambda r: (
                str(r.get("experiment_id")),
                str(r.get("transition_window")),
                str(r.get("solution")),
                str(r.get("dataset")),
                r.get("path", ""),
            ),
        ):

            status = record.get("status", "incomplete")
            solution = record.get("solution", "N/A")
            dataset_name = record.get("dataset", "N/A")
            experiment = record.get("experiment_id", "N/A")
            transition_window = record.get(
                "transition_window",
                None
            )

            shift_type = _validation_shift_type(experiment)
            temporal_type = _validation_temporal_type(experiment)
            alpha = _validation_alpha(experiment)

            fold_count = record.get("fold_count", 0)
            expected_rounds = record.get(
                "expected_rounds",
                None
            )

            print("\n------------------------------------------------------------")
            print(f"Experiment ID      : {experiment}")
            print(f"Solução             : {solution}")
            print(f"Dataset             : {dataset_name}")
            print(f"Data shift          : {shift_type}")
            print(f"Temporal type       : {temporal_type}")
            print(f"Alpha/configuração  : {alpha}")
            print(
                f"Transition Window   : "
                f"{transition_window if transition_window is not None else 'N/A'}"
            )
            print(f"Status              : {status}")
            print(f"Arquivo            : {record.get('path', 'N/A')}")
            print(f"Folds encontrados  : {fold_count}")

            if expected_rounds is not None:
                print(
                    f"Rodadas esperadas  : "
                    f"{expected_rounds} por fold"
                )

            fold_details = record.get(
                "fold_details",
                {}
            )

            if fold_details:

                for fold_id in sorted(
                    fold_details.keys()
                ):
                    details = fold_details[fold_id]

                    print(
                        f"  Fold {fold_id}: "
                        f"{details['round_count']} rodadas | "
                        f"intervalo "
                        f"{details['min_round']} -> "
                        f"{details['max_round']} | "
                        f"completo={details['complete']}"
                    )

                    missing = details.get(
                        "missing_rounds",
                        []
                    )

                    extra = details.get(
                        "extra_rounds",
                        []
                    )

                    if missing:
                        # Evita imprimir uma lista gigantesca.
                        preview = missing[:20]
                        suffix = (
                            " ..."
                            if len(missing) > 20
                            else ""
                        )
                        print(
                            f"    Rodadas ausentes: "
                            f"{preview}{suffix}"
                        )

                    if extra:
                        preview = extra[:20]
                        suffix = (
                            " ..."
                            if len(extra) > 20
                            else ""
                        )
                        print(
                            f"    Rodadas extras: "
                            f"{preview}{suffix}"
                        )

            if record.get("error"):
                print(
                    f"Erro: {record['error']}"
                )

    else:
        print(
            "Todos os arquivos esperados estão completos segundo "
            "o critério de 100 rodadas (1 fold) ou 500 rodadas (>1 fold)."
        )

    print("=" * 100 + "\n")

    print(
        f"Resumo da validação: "
        f"{len(incomplete_records)} configuração(ões) sem arquivo ou incompleta(s) "
        f"de {len(EXPERIMENT_VALIDATION)} arquivo(s)/configuração(ões) esperados."
    )

    print(
        f"\nTolerância máxima de detecção: {MAX_DETECTION_DELAY} rodada(s) após o fim do episódio"
        if MAX_DETECTION_DELAY is not None
        else "\nTolerância máxima de detecção: ilimitada"
    )
    print(
        "Gradual usa Transition Window nas métricas: "
        f"{USE_TRANSITION_WINDOW_FOR_GRADUAL_METRICS}"
    )

    # ============================================================
    # TABELA CONSOLIDADA DE QUALIDADE DA DETECÇÃO
    # ============================================================

    df_experimental, df_detection_table, df_overall_detection_table = (
        table_detection_quality_by_shift_type(
            df_final=df_all,
            write_path=write_path,
            solutions=solutions,
            metrics=metrics,
            higher_is_better_metrics=higher_is_better_metrics,
            ci=0.95,
            max_detection_delay=MAX_DETECTION_DELAY,
            use_transition_window_for_gradual=(
                USE_TRANSITION_WINDOW_FOR_GRADUAL_METRICS
            ),
        )
    )

    # ============================================================
    # TABELAS ADICIONAIS DE ACURÁCIA - CONCEPT DRIFT
    # ============================================================
    #
    # Uma tabela independente é gerada para cada dataset:
    #   * WISDM-W
    #   * ImageNet10
    #   * Foursquare
    #
    # Cada tabela contém as seis configurações:
    #   * Sudden, alpha=0.1
    #   * Sudden, alpha=1.0
    #   * Gradual W=5, alpha=0.1
    #   * Gradual W=5, alpha=1.0
    #   * Gradual W=10, alpha=0.1
    #   * Gradual W=10, alpha=1.0
    #
    # A acurácia é apresentada como média +/- IC 95%.
    # A solução só é marcada em negrito quando é estritamente
    # superior a todas as demais e seu IC não se sobrepõe ao IC
    # de nenhuma outra solução na mesma configuração.
    # ============================================================

    # ======================================================================
    # DEBUG - VERIFICAR RESULTADOS LIDOS PARA ALPHA = 1.0
    # ======================================================================
    print("\n" + "=" * 100)
    print("DEBUG - RESULTADOS LIDOS PARA ALPHA = 1.0")
    print("=" * 100)

    if "Alpha" not in df_all.columns:
        print("ERRO: a coluna 'Alpha' NÃO existe em df_all.")
    else:
        alpha_numeric = pd.to_numeric(df_all["Alpha"], errors="coerce")

        print(f"Total de linhas lidas: {len(df_all)}")
        print(f"Linhas com Alpha válido: {alpha_numeric.notna().sum()}")
        print(
            "Linhas com Alpha = 1.0: "
            f"{np.isclose(alpha_numeric, 1.0, rtol=0, atol=1e-8).sum()}"
        )

        print("\nDistribuição dos valores de Alpha:")
        print(
            df_all.assign(_AlphaNumeric=alpha_numeric)["_AlphaNumeric"]
            .value_counts(dropna=False)
            .sort_index()
            .to_string()
        )

        alpha1 = df_all[
            np.isclose(alpha_numeric, 1.0, rtol=0, atol=1e-8)
        ].copy()

        if alpha1.empty:
            print("\nNENHUM RESULTADO FOI LIDO PARA ALPHA = 1.0.")
        else:
            print("\nRESULTADOS ENCONTRADOS PARA ALPHA = 1.0")
            print(f"Total de linhas: {len(alpha1)}")

            for col in [
                "Experiment ID",
                "Dataset",
                "Solution",
                "Temporal Shift Type",
                "Transition Window",
                "Fold ID",
                "Round (t)",
            ]:
                if col in alpha1.columns:
                    print(f"\n--- {col} ---")
                    print(
                        alpha1[col]
                        .value_counts(dropna=False)
                        .sort_index()
                        .to_string()
                    )

            group_cols = [
                c for c in
                ["Experiment ID", "Dataset", "Solution", "Transition Window"]
                if c in alpha1.columns
            ]

            if group_cols:
                print("\nResumo por experimento / dataset / solução / janela:")
                print(
                    alpha1.groupby(group_cols, dropna=False)
                    .size()
                    .reset_index(name="Rows")
                    .to_string(index=False)
                )

    print("=" * 100)
    # ======================================================================
    # FIM DEBUG ALPHA = 1.0
    # ======================================================================

    # ============================================================
    # RODADAS USADAS NAS TABELAS DE ACCURACY - CONCEPT DRIFT
    # ============================================================
    # None / "all"       -> todas as rodadas disponíveis (padrão).
    # [30, 31, 32]       -> somente essas rodadas.
    # (30, 50)           -> intervalo inclusivo 30..50.
    # {"default": ...}  -> seleção por configuração; também aceita
    #                        chaves (temporal, window, alpha).
    # Exemplo:
    # ACCURACY_CONCEPT_ROUNDS = {
    #     "default": (30, 50),
    #     ("Gradual", 5, 0.1): (30, 34),
    # }
    # ============================================================
    ACCURACY_CONCEPT_ROUNDS = None

    accuracy_concept_tables = (
        table_accuracy_concept_drift_by_dataset(
            df_all=df_all,
            write_path=write_path,
            solutions=concept_solutions,
            concept_alphas=(0.1, 1.0),
            gradual_windows=(5, 10),
            ci=0.95,
            accuracy_rounds=ACCURACY_CONCEPT_ROUNDS,
        )
    )

    print(
        "\nTabelas adicionais de acurácia geradas: "
        f"{list(accuracy_concept_tables.keys())}"
    )