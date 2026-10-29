from pathlib import Path
import numpy as np
import pandas as pd
import scipy.stats as st
import os
import re
import copy

from base_plots import bar_plot, line_plot, ecdf_plot
import matplotlib.pyplot as plt
import seaborn as sns

# Registro de validação dos experimentos
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
    Lê os CSVs específicos de cada dataset e solução e valida
    se possuem exatamente 5 folds com 100 rodadas completas (1 a 100).
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
                    EXPERIMENT_VALIDATION.append({
                        "path": os.path.abspath(path),
                        "solution": solution_names.get(solution, solution),
                        "dataset": None,
                        "experiment_id": experiment_id,
                        "transition_window": transition_window,
                        "fold_count": 0,
                        "fold_ids": [],
                        "expected_folds": 5,
                        "expected_rounds_per_fold": 100,
                        "fold_details": {},
                        "complete_5folds_100rounds": False,
                        "status": "missing",
                    })
                    continue

                if os.path.basename(path).endswith("_metrics.csv"):
                    raise ValueError(
                        f"Arquivo _metrics.csv não pode ser utilizado: {path}"
                    )

                header = pd.read_csv(path, nrows=0).columns.tolist()
                required_columns = ["Fold ID", "Round (t)", "Accuracy"]
                missing_required = [c for c in required_columns if c not in header]

                if missing_required:
                    raise ValueError(
                        f"Colunas obrigatórias ausentes: {', '.join(missing_required)}"
                    )

                columns_to_read = required_columns.copy()
                has_data_shift = "Data shift" in header
                if has_data_shift:
                    columns_to_read.append("Data shift")

                df = pd.read_csv(path, usecols=columns_to_read)

                if not has_data_shift:
                    df["Data shift"] = "NO_SHIFT"

                filename = os.path.basename(path)
                dataset_name = None
                for candidate_dataset in dataset_shift_rounds.keys():
                    if filename.startswith(f"{candidate_dataset}_"):
                        dataset_name = candidate_dataset
                        break

                if df.empty:
                    EXPERIMENT_VALIDATION.append({
                        "path": os.path.abspath(path),
                        "solution": solution_names.get(solution, solution),
                        "dataset": dataset_name,
                        "experiment_id": experiment_id,
                        "transition_window": transition_window,
                        "fold_count": 0,
                        "fold_ids": [],
                        "expected_folds": 5,
                        "expected_rounds_per_fold": 100,
                        "fold_details": {},
                        "complete_5folds_100rounds": False,
                        "status": "empty",
                    })
                    continue

                df = df[["Fold ID", "Round (t)", "Data shift", "Accuracy"]].copy()

                if dataset_name is None:
                    raise ValueError(
                        f"Não foi possível identificar o dataset a partir do arquivo: {filename}"
                    )

                df["Dataset"] = dataset_name

                if isinstance(alpha_value, (tuple, list)):
                    df["Alpha"] = float(alpha_value[0])
                else:
                    df["Alpha"] = float(alpha_value) if alpha_value is not None else np.nan

                df["Solution"] = solution_names.get(solution, solution)
                df["Shift Round"] = dataset_shift_rounds[dataset_name]
                df["Transition Window"] = transition_window

                # --------------------------------------------------------
                # VALIDAÇÃO: 5 FOLDS E 100 RODADAS COMPLETAS (1..100)
                # --------------------------------------------------------
                fold_numeric = pd.to_numeric(df["Fold ID"], errors="coerce")
                round_numeric = pd.to_numeric(df["Round (t)"], errors="coerce")
                valid_mask = fold_numeric.notna() & round_numeric.notna()

                fold_round_df = pd.DataFrame({
                    "Fold ID": fold_numeric[valid_mask].astype(int),
                    "Round (t)": round_numeric[valid_mask].astype(int),
                })

                fold_ids = sorted(fold_round_df["Fold ID"].unique().tolist())
                fold_count = len(fold_ids)
                expected_rounds = 100
                expected_round_set = set(range(1, expected_rounds + 1))

                fold_details = {}
                for fold_id in fold_ids:
                    rounds = sorted(
                        set(
                            fold_round_df.loc[
                                fold_round_df["Fold ID"] == fold_id, "Round (t)"
                            ].tolist()
                        )
                    )
                    actual_round_set = set(rounds)
                    missing_rounds = sorted(expected_round_set - actual_round_set)
                    extra_rounds = sorted(actual_round_set - expected_round_set)

                    fold_details[fold_id] = {
                        "round_count": len(rounds),
                        "min_round": min(rounds) if rounds else None,
                        "max_round": max(rounds) if rounds else None,
                        "missing_rounds": missing_rounds,
                        "extra_rounds": extra_rounds,
                        "complete": (actual_round_set == expected_round_set),
                    }

                # Critério estrito: exatamente 5 folds e cada fold contendo 1..100
                has_exact_5_folds = (fold_count == 5)
                all_folds_complete = (
                        has_exact_5_folds and
                        all(details["complete"] for details in fold_details.values())
                )

                EXPERIMENT_VALIDATION.append({
                    "path": os.path.abspath(path),
                    "solution": df["Solution"].iloc[0],
                    "dataset": dataset_name,
                    "experiment_id": experiment_id,
                    "transition_window": transition_window,
                    "fold_count": fold_count,
                    "fold_ids": fold_ids,
                    "expected_folds": 5,
                    "expected_rounds_per_fold": 100,
                    "fold_details": fold_details,
                    "complete_5folds_100rounds": all_folds_complete,
                    "status": "complete" if all_folds_complete else "incomplete",
                })

                df_list.append(df)

            except ValueError as e:
                EXPERIMENT_VALIDATION.append({
                    "path": os.path.abspath(path),
                    "solution": solution_names.get(solution, solution),
                    "dataset": None,
                    "experiment_id": experiment_id,
                    "transition_window": transition_window,
                    "fold_count": 0,
                    "fold_ids": [],
                    "expected_folds": 5,
                    "expected_rounds_per_fold": 100,
                    "fold_details": {},
                    "complete_5folds_100rounds": False,
                    "status": "invalid",
                    "error": str(e),
                })
                continue

            except Exception as e:
                EXPERIMENT_VALIDATION.append({
                    "path": os.path.abspath(path),
                    "solution": solution_names.get(solution, solution),
                    "dataset": None,
                    "experiment_id": experiment_id,
                    "transition_window": transition_window,
                    "fold_count": 0,
                    "fold_ids": [],
                    "expected_folds": 5,
                    "expected_rounds_per_fold": 100,
                    "fold_details": {},
                    "complete_5folds_100rounds": False,
                    "status": "error",
                    "error": str(e),
                })
                continue

    if not df_list:
        return pd.DataFrame(
            columns=[
                "Fold ID", "Round (t)", "Data shift", "Accuracy",
                "Solution", "Dataset", "Shift Round"
            ]
        )

    return pd.concat(df_list, ignore_index=True)


def format_shift_configuration(shift_type, experiment_id):
    if pd.isna(experiment_id):
        return "--"
    experiment_id = str(experiment_id).strip()
    if "#" not in experiment_id:
        return experiment_id.replace("_", r"\_")

    config = experiment_id.split("#", 1)[1].split("_sudden", 1)[0]

    if str(shift_type).lower() == "concept":
        try:
            alpha = float(config)
            return rf"$\alpha={alpha:g}$"
        except ValueError:
            return config.replace("_", r"\_")

    if str(shift_type).lower() == "label":
        if "-" in config:
            alpha_before, alpha_after = config.split("-", 1)
            try:
                alpha_before = float(alpha_before)
                alpha_after = float(alpha_after)
                return rf"${alpha_before:g}\rightarrow {alpha_after:g}$"
            except ValueError:
                pass

    return config.replace("_", r"\_")


def _get_shift_detection_bounds(
        row,
        max_detection_delay=None,
        use_transition_window_for_gradual=True,
):
    shift_round = row.get("Shift Round", np.nan)
    if pd.isna(shift_round):
        return np.nan, None

    shift_round = float(shift_round)
    temporal_shift_type = str(row.get("Temporal Shift Type", "")).strip().lower()
    transition_window = row.get("Transition Window", np.nan)

    if (
            temporal_shift_type == "gradual"
            and use_transition_window_for_gradual
            and not pd.isna(transition_window)
    ):
        try:
            transition_window = int(float(transition_window))
        except (TypeError, ValueError):
            transition_window = None

        if transition_window is not None and transition_window > 0:
            shift_end = shift_round + transition_window - 1
        else:
            shift_end = shift_round
    else:
        shift_end = shift_round

    if max_detection_delay is None:
        detection_window_end = None
    else:
        detection_window_end = shift_end + float(max_detection_delay)

    return shift_round, detection_window_end


def _get_valid_detection_rounds(
        row,
        max_detection_delay=None,
        use_transition_window_for_gradual=True,
):
    shift_round, detection_window_end = _get_shift_detection_bounds(
        row,
        max_detection_delay=max_detection_delay,
        use_transition_window_for_gradual=use_transition_window_for_gradual,
    )

    if pd.isna(shift_round):
        return []

    detection_rounds = row.get("Detection Rounds", [])
    if not isinstance(detection_rounds, (list, tuple)):
        return []

    valid = []
    for detection_round in detection_rounds:
        if pd.isna(detection_round):
            continue
        detection_round = float(detection_round)
        if detection_round < shift_round:
            continue
        if detection_window_end is not None and detection_round > detection_window_end:
            continue
        valid.append(detection_round)

    return sorted(set(valid))


def calculate_detection_rate(df, max_detection_delay=None):
    if df.empty:
        return np.nan
    detected = 0
    total = 0
    for _, row in df.iterrows():
        shift_round = row["Shift Round"]
        if pd.isna(shift_round):
            continue
        total += 1
        valid_detections = _get_valid_detection_rounds(row, max_detection_delay)
        if len(valid_detections) > 0:
            detected += 1
    return (detected / total) if total > 0 else np.nan


def calculate_average_detection_delay(df, max_detection_delay=None):
    if df.empty:
        return np.nan
    delays = []
    for _, row in df.iterrows():
        shift_round = row["Shift Round"]
        if pd.isna(shift_round):
            continue
        valid_detections = _get_valid_detection_rounds(row, max_detection_delay)
        if not valid_detections:
            continue
        delay = valid_detections[0] - float(shift_round)
        if delay >= 0:
            delays.append(delay)
    return float(np.mean(delays)) if delays else np.nan


def calculate_episode_f1(df, max_detection_delay=None):
    if df.empty:
        return np.nan
    tp, fn, fp = 0, 0, 0
    for _, row in df.iterrows():
        shift_round = row["Shift Round"]
        if pd.isna(shift_round):
            continue
        detection_rounds = row.get("Detection Rounds", [])
        if not isinstance(detection_rounds, (list, tuple)):
            detection_rounds = []
        detection_rounds = sorted(set(float(x) for x in detection_rounds if not pd.isna(x)))
        valid_detections = _get_valid_detection_rounds(row, max_detection_delay)

        if valid_detections:
            tp += 1
        else:
            fn += 1

        valid_set = set(valid_detections)
        for detection_round in detection_rounds:
            if detection_round < shift_round:
                fp += 1
                continue
            if max_detection_delay is not None and detection_round > shift_round + max_detection_delay:
                fp += 1
                continue
            if detection_round in valid_set:
                continue

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def calculate_alarm_rate(df):
    if df.empty:
        return np.nan
    total_alarms = 0
    total_rounds = 0
    for _, row in df.iterrows():
        detection_rounds = row.get("Detection Rounds", [])
        if isinstance(detection_rounds, (list, tuple)):
            total_alarms += len(set(detection_rounds))
        last_round = row.get("Last Round", np.nan)
        if not pd.isna(last_round):
            total_rounds += int(last_round)
    return (total_alarms / total_rounds) if total_rounds > 0 else np.nan


def calculate_detection_metric_values(
        df,
        metric,
        max_detection_delay=None,
        use_transition_window_for_gradual=True,
):
    if df is None or df.empty:
        return pd.Series(dtype=float)

    values = []
    for _, row in df.iterrows():
        shift_round = row.get("Shift Round", np.nan)
        if pd.isna(shift_round):
            continue
        shift_round = float(shift_round)

        detection_rounds = row.get("Detection Rounds", [])
        if not isinstance(detection_rounds, (list, tuple, np.ndarray)):
            detection_rounds = []

        normalized = []
        for d in detection_rounds:
            try:
                val = float(d)
                if val >= 0:
                    normalized.append(val)
            except (ValueError, TypeError):
                continue
        detection_rounds = sorted(set(normalized))

        valid_detections = _get_valid_detection_rounds(
            row,
            max_detection_delay=max_detection_delay,
            use_transition_window_for_gradual=use_transition_window_for_gradual,
        )

        if metric == "Detection Rate":
            values.append(1.0 if valid_detections else 0.0)

        elif metric == "Average Detection Delay":
            if not valid_detections:
                continue
            delay = float(valid_detections[0]) - shift_round
            if delay >= 0:
                values.append(delay)

        elif metric == "Episode F1":
            tp = 1 if valid_detections else 0
            fn = 0 if valid_detections else 1
            fp = 0
            valid_set = set(valid_detections)

            for d in detection_rounds:
                if d < shift_round:
                    fp += 1
                    continue
                _, bound_end = _get_shift_detection_bounds(
                    row,
                    max_detection_delay=max_detection_delay,
                    use_transition_window_for_gradual=use_transition_window_for_gradual,
                )
                if max_detection_delay is not None and bound_end is not None and d > bound_end:
                    fp += 1
                    continue
                if d in valid_set:
                    continue

            prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            values.append((2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0)

        elif metric == "Alarm Rate":
            last_round = row.get("Last Round", np.nan)
            if pd.isna(last_round) or float(last_round) <= 0:
                continue
            values.append(len(detection_rounds) / float(last_round))

        else:
            raise ValueError(f"Unsupported detection metric: '{metric}'")

    return pd.Series(values, dtype=float)


def mean_ci(values, ci=0.95, bounded=False, metric=None):
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    n = len(values)
    if n == 0:
        return np.nan, np.nan
    mean = float(np.mean(values))
    if n == 1:
        return round(mean, 2), np.nan

    alpha = 1.0 - ci

    if metric == "Detection Rate":
        successes = int(np.sum(np.isclose(values, 1.0)))
        p_hat = successes / n
        z = float(st.norm.ppf(1.0 - alpha / 2.0))
        denom = 1.0 + z ** 2 / n
        center = (p_hat + z ** 2 / (2.0 * n)) / denom
        half = (z * np.sqrt((p_hat * (1.0 - p_hat) / n) + z ** 2 / (4.0 * n ** 2)) / denom)
        lower, upper = max(0.0, center - half), min(1.0, center + half)
        margin = max(mean - lower, upper - mean)

    elif metric == "Episode F1":
        rng = np.random.default_rng(20261017)
        bootstrap_means = np.mean(rng.choice(values, size=(10000, n), replace=True), axis=1)
        lower, upper = np.quantile(bootstrap_means, [alpha / 2.0, 1.0 - alpha / 2.0])
        margin = max(mean - float(lower), float(upper) - mean)

    else:
        sem = float(st.sem(values))
        if not np.isfinite(sem):
            return round(mean, 2), np.nan
        lower, upper = st.t.interval(confidence=ci, df=n - 1, loc=mean, scale=sem)
        margin = max(mean - float(lower), float(upper) - mean)

    return round(mean, 2), round(float(margin), 2)


def normalize_configuration_for_table(configuration):
    if pd.isna(configuration):
        return configuration
    cfg = str(configuration)
    if "#" in cfg:
        cfg = cfg.split("#", 1)[1]
    if cfg.endswith("_sudden"):
        cfg = cfg[:-len("_sudden")]
    return cfg


def normalize_shift_type_for_table(shift_type):
    if pd.isna(shift_type):
        return shift_type
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
    return normalized.get(str(shift_type).strip(), shift_type)


def format_configuration(shift_type, configuration):
    if pd.isna(configuration):
        return "N/A"
    cfg = str(configuration).strip()
    if "#" in cfg:
        cfg = cfg.split("#", 1)[1]
    if "_" in cfg:
        cfg = cfg.split("_", 1)[0]
    cfg = cfg.strip()

    if "-" in cfg:
        parts = cfg.split("-", 1)
        try:
            return f"{float(parts[0]):g}-{float(parts[1]):g}"
        except (ValueError, TypeError):
            pass
    else:
        try:
            return f"{float(cfg):g}"
        except (ValueError, TypeError):
            pass
    return cfg.replace("_", r"\_")


def format_shift_type(shift_type):
    if pd.isna(shift_type):
        return "N/A"
    normalized = str(shift_type).strip().lower().replace("_", " ").replace("-", " ")
    if "combined" in normalized:
        return "Combined shift"
    if "concept" in normalized:
        return "Concept drift"
    if "label" in normalized:
        return "Label shift"
    return shift_type


def generate_latex_table(df_table, filename, caption, label, column_format):
    latex = df_table.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        column_format=column_format,
    )
    latex = (
        latex.replace("MFP\\_v2\\_dh", "$\\textit{MFP}_{\\textit{DDH}}$")
        .replace("MFP\\_v2\\_iti", "$\\textit{MFP}_{\\textit{ITI}}$")
        .replace("MFP\\_v2", "$\\textit{MFP}$")
        .replace("CONCEPT_DRIFT", "Concept drift")
        .replace("concept_drift", "Concept drift")
        .replace("CONCEPT", "Concept drift")
        .replace("LABEL_SHIFT", "Label shift")
        .replace("label_shift", "Label shift")
        .replace("LABEL", "Label shift")
        .replace("\\toprule\n\\toprule", "\\toprule")
        .replace(r"\&", "&")
    )
    latex = (
            "% Requires: \\usepackage{booktabs}\n"
            "% Requires: \\usepackage{multirow}\n"
            "% Requires: \\usepackage{graphicx}\n"
            + latex
    )
    with open(filename, "w", encoding="utf-8") as f:
        f.write(latex)


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
    return mean_ci(values, ci=ci, metric=metric)


def _extract_detection_rounds_from_unit(df_experiment):
    if "Data shift" not in df_experiment.columns:
        return []
    shift_values = df_experiment["Data shift"].astype(str).str.strip().str.upper()
    rounds = pd.to_numeric(df_experiment["Round (t)"], errors="coerce")
    detection_rounds = rounds[shift_values == "DATA_SHIFT"].dropna().astype(float).tolist()
    return sorted(set(detection_rounds))


def _calculate_experiment_detection_metrics(
        df_experiment,
        max_detection_delay=None,
        use_transition_window_for_gradual=True,
):
    if df_experiment.empty:
        return {
            "Detection Rate": np.nan,
            "Episode F1": np.nan,
            "Average Detection Delay": np.nan,
            "False Alarm Rate": np.nan,
        }

    row = df_experiment.iloc[0].copy()
    shift_round = row.get("Shift Round", np.nan)
    if pd.isna(shift_round):
        return {
            "Detection Rate": np.nan,
            "Episode F1": np.nan,
            "Average Detection Delay": np.nan,
            "False Alarm Rate": np.nan,
        }

    shift_round = float(shift_round)
    detection_rounds = _extract_detection_rounds_from_unit(df_experiment)

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

    detection_rate = 1.0 if valid_detections else 0.0
    average_detection_delay = (float(valid_detections[0]) - shift_round) if valid_detections else np.nan

    false_alarms = []
    pre_shift_false_alarms = []

    for d in detection_rounds:
        if d < shift_round:
            false_alarms.append(d)
            pre_shift_false_alarms.append(d)
            continue

        _, detection_window_end = _get_shift_detection_bounds(
            detection_row,
            max_detection_delay=max_detection_delay,
            use_transition_window_for_gradual=use_transition_window_for_gradual,
        )

        if detection_window_end is not None and d > detection_window_end:
            false_alarms.append(d)

    tp = 1 if valid_detections else 0
    fn = 0 if valid_detections else 1
    fp = len(false_alarms)

    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    episode_f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    stable_rounds = max(shift_round - 1.0, 0.0)
    false_alarm_rate = (len(pre_shift_false_alarms) / stable_rounds) if stable_rounds > 0 else np.nan

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
    if df_final is None or df_final.empty:
        return None, None, None

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

    df_eval["Shift Type"] = df_eval["Shift Type"].apply(format_shift_type)

    experiment_column = next(
        (c for c in ["Experiment ID", "experiment_id", "Experiment"] if c in df_eval.columns),
        None,
    )
    df_eval["_Experiment"] = df_eval[experiment_column].astype(str).str.strip()
    exp_lower = df_eval[experiment_column].astype(str).str.strip().str.lower()
    df_eval["Temporal Shift Type"] = np.select(
        [exp_lower.str.endswith("_sudden"), exp_lower.str.endswith("_gradual")],
        ["sudden", "gradual"],
        default="N/A",
    )

    if "Transition Window" not in df_eval.columns:
        df_eval["Transition Window"] = np.nan

    def normalize_transition_window(value):
        if pd.isna(value):
            return np.nan
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return np.nan

    df_eval["Transition Window"] = df_eval["Transition Window"].apply(normalize_transition_window)

    is_sudden = df_eval["Temporal Shift Type"].eq("sudden")
    is_gradual_w10 = (
            df_eval["Temporal Shift Type"].eq("gradual")
            & df_eval["Transition Window"].eq(10)
    )
    df_eval = df_eval[is_sudden | is_gradual_w10].copy()

    if df_eval.empty:
        return None, None, None

    valid_shift_types = ["Concept drift", "Label shift", "Combined shift"]
    valid_temporal_types = ["sudden", "gradual"]

    df_eval = df_eval[
        df_eval["Shift Type"].isin(valid_shift_types)
        & df_eval["Temporal Shift Type"].isin(valid_temporal_types)
        ].copy()

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
        return None, None, None

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
                values = pd.to_numeric(df_solution[metric], errors="coerce").dropna()
                mean_value, ci_value = mean_ci(values, ci=ci, metric=metric)
                aggregated[(shift_type, temporal_type, window)][solution][metric] = {
                    "mean": mean_value,
                    "ci": ci_value,
                    "n": len(values),
                }

            if (
                    str(shift_type).strip().lower() == "concept drift"
                    and str(solution).strip().lower() == "js-drift"
                    and "Detection Rate" in aggregated[(shift_type, temporal_type, window)][solution]
                    and np.isclose(
                aggregated[(shift_type, temporal_type, window)][solution]["Detection Rate"]["mean"],
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

    shift_labels = {
        "Concept drift": "Concept Drift",
        "Label shift": "Label Shift",
        "Combined shift": "Combined Shift",
    }
    temporal_labels = {"sudden": "Sudden", "gradual": "Gradual"}

    table_rows = []
    for shift_type in valid_shift_types:
        keys = [key for key in aggregated if key[0] == shift_type]
        if not keys:
            continue
        keys.sort(key=lambda k: (0 if k[1] == "sudden" else 1, -1 if k[2] is None else k[2]))

        shift_row_count = sum(len(aggregated[k]) for k in keys)
        shift_first = True

        temporal_row_counts = {}
        for temporal_type in ["sudden", "gradual"]:
            temporal_keys = [k for k in keys if k[1] == temporal_type]
            temporal_row_counts[temporal_type] = sum(len(aggregated[k]) for k in temporal_keys)

        temporal_first = {"sudden": True, "gradual": True}

        for key in keys:
            _, temporal_type, transition_window = key
            temporal_rows = aggregated[key]
            solutions_present = [s for s in solutions if s in temporal_rows]
            if not solutions_present:
                continue

            best_by_metric = {}
            for metric in metrics:
                candidates = [sol for sol in solutions_present if not pd.isna(temporal_rows[sol][metric]["mean"])]
                if not candidates:
                    continue
                if metric in higher_is_better_metrics:
                    best_by_metric[metric] = max(candidates, key=lambda sol: temporal_rows[sol][metric]["mean"])
                else:
                    best_by_metric[metric] = min(candidates, key=lambda sol: temporal_rows[sol][metric]["mean"])

            for solution in solutions_present:
                row = {}
                if shift_first:
                    row["Shift"] = f"\\multirow{{{shift_row_count}}}{{*}}{{{shift_labels[shift_type]}}}"
                    shift_first = False
                else:
                    row["Shift"] = ""

                if temporal_first[temporal_type]:
                    t_count = temporal_row_counts[temporal_type]
                    row["Shift Type"] = f"\\multirow{{{t_count}}}{{*}}{{{temporal_labels[temporal_type]}}}"
                    temporal_first[temporal_type] = False
                else:
                    row["Shift Type"] = ""

                solution_display = str(solution)
                if solution_display == "MFP_v2_dh":
                    solution_display = "$\\textit{MFP}_{\\textit{DDH}}$"
                elif solution_display == "MFP_v2_iti":
                    solution_display = "$\\textit{MFP}_{\\textit{ITI}}$"
                elif solution_display == "MFP_v2":
                    solution_display = "MFP"
                elif solution_display == "MultiFedAvg+MFP_v2":
                    solution_display = "MultiFedAvg+MFP"
                else:
                    solution_display = solution_display.replace("_", r"\_")

                row["Solution"] = solution_display

                for metric in metrics:
                    result = temporal_rows[solution].get(metric)
                    if result is None or pd.isna(result["mean"]):
                        row[metric] = "--"
                        continue

                    mean_val = result["mean"]
                    ci_val = result["ci"]
                    text = f"{mean_val:.2f}" if pd.isna(ci_val) else f"{mean_val:.2f} $\\pm$ {ci_val:.2f}"

                    best_solution = best_by_metric.get(metric)
                    is_bold = False
                    if best_solution is not None:
                        best = temporal_rows[best_solution][metric]
                        if not pd.isna(best["mean"]):
                            if pd.isna(best["ci"]):
                                is_bold = np.isclose(mean_val, best["mean"], rtol=1e-12, atol=1e-12)
                            else:
                                is_bold = (best["mean"] - best["ci"] <= mean_val <= best["mean"] + best["ci"])
                    if is_bold:
                        text = f"\\textbf{{{text}}}"
                    row[metric] = text

                table_rows.append(row)

    df_table = pd.DataFrame(table_rows, columns=["Shift", "Shift Type", "Solution"] + metrics)
    df_table = df_table.rename(
        columns={
            "Detection Rate": "DR $\\uparrow$",
            "Episode F1": "Episode F1 $\\uparrow$",
            "Average Detection Delay": "MTD $\\downarrow$",
            "False Alarm Rate": "FAR $\\downarrow$",
        }
    )

    filename = os.path.join(write_path, "overall_detection_quality.tex")
    latex = df_table.to_latex(
        index=False,
        escape=False,
        caption=(
            "Performance of data-shift detection methods. Results are separated by shift family "
            "(Concept Drift, Label Shift, and Combined Shift), and temporal shift type (Sudden or Gradual)."
        ),
        label="tab:overall_detection_quality",
        column_format="lll" + "c" * len(metrics),
    )

    lines = latex.splitlines()
    processed_lines = []
    previous_shift = None
    previous_temporal = None

    for line in lines:
        current_shift = None
        if "\\multirow" in line:
            if "Concept Drift" in line:
                current_shift = "Concept Drift"
            elif "Label Shift" in line:
                current_shift = "Label Shift"
            elif "Combined Shift" in line:
                current_shift = "Combined Shift"

        current_temporal = None
        if "\\multirow" in line:
            if "{*}{Sudden}" in line:
                current_temporal = "Sudden"
            elif "{*}{Gradual}" in line:
                current_temporal = "Gradual"

        active_shift = current_shift if current_shift is not None else previous_shift
        if current_shift is not None and previous_shift is not None and current_shift != previous_shift:
            processed_lines.append("\\midrule")
        if current_temporal == "Gradual" and previous_temporal == "Sudden" and active_shift == previous_shift:
            processed_lines.append("\\cline{2-7}")

        processed_lines.append(line)
        if current_shift is not None:
            previous_shift = current_shift
        if current_temporal is not None:
            previous_temporal = current_temporal

    latex = "\n".join(processed_lines) + "\n"
    latex = latex.replace("\\begin{table}", "\\begin{table*}", 1).replace("\\end{table}", "\\end{table*}", 1)
    latex = latex.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", 1).replace(
        "\\end{tabular}", "\\end{tabular}%\n}", 1
    )
    latex = "% Requires: \\usepackage{booktabs}\n% Requires: \\usepackage{multirow}\n% Requires: \\usepackage{graphicx}\n" + latex

    Path(write_path).mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(latex)

    # ------------------------------------------------------------
    # Tabela 2: Overall Average
    # ------------------------------------------------------------
    overall_rows = []
    for solution in solutions:
        df_solution = df_experimental[df_experimental["Solution"] == solution].copy()
        if str(solution).strip().lower() == "js-drift":
            concept_invalid = (
                                      df_solution["Shift Type"].astype(str).str.strip().str.lower() == "concept drift"
                              ) & (pd.to_numeric(df_solution["Detection Rate"], errors="coerce").fillna(0.0) <= 0.0)
            df_solution = df_solution.loc[~concept_invalid].copy()

        if df_solution.empty:
            continue

        row = {"Solution": solution}
        for metric in metrics:
            values = pd.to_numeric(df_solution[metric], errors="coerce").dropna()
            mean_val, ci_val = mean_ci(values, ci=ci, metric=metric)
            row[metric] = {"mean": mean_val, "ci": ci_val, "n": len(values)}
        overall_rows.append(row)

    overall_best_by_metric = {}
    for metric in metrics:
        candidates = [(r["Solution"], r[metric]["mean"]) for r in overall_rows if
                      r.get(metric) and not pd.isna(r[metric]["mean"])]
        if not candidates:
            continue
        if metric in higher_is_better_metrics:
            overall_best_by_metric[metric] = max(candidates, key=lambda x: x[1])[0]
        else:
            overall_best_by_metric[metric] = min(candidates, key=lambda x: x[1])[0]

    overall_table_rows = []
    for row_data in overall_rows:
        solution = row_data["Solution"]
        s_display = str(solution)
        if s_display == "MFP_v2_dh":
            s_display = "$\\textit{MFP}_{\\textit{DDH}}$"
        elif s_display == "MFP_v2_iti":
            s_display = "$\\textit{MFP}_{\\textit{ITI}}$"
        elif s_display == "MFP_v2":
            s_display = "MFP"
        elif s_display == "MultiFedAvg+MFP_v2":
            s_display = "MultiFedAvg+MFP"
        else:
            s_display = s_display.replace("_", r"\_")

        output_row = {"Solution": s_display}
        for metric in metrics:
            result = row_data.get(metric)
            if result is None or pd.isna(result["mean"]):
                output_row[metric] = "--"
                continue
            mean_val = result["mean"]
            ci_val = result["ci"]
            text = f"{mean_val:.2f}" if pd.isna(ci_val) else f"{mean_val:.2f} $\\pm$ {ci_val:.2f}"

            best_solution = overall_best_by_metric.get(metric)
            is_bold = False
            if best_solution == solution:
                best_res = next((r[metric] for r in overall_rows if r["Solution"] == best_solution), None)
                if best_res and not pd.isna(best_res["mean"]):
                    if pd.isna(best_res["ci"]):
                        is_bold = np.isclose(mean_val, best_res["mean"], rtol=1e-12, atol=1e-12)
                    else:
                        is_bold = (best_res["mean"] - best_res["ci"] <= mean_val <= best_res["mean"] + best_res["ci"])
            if is_bold:
                text = f"\\textbf{{{text}}}"
            output_row[metric] = text
        overall_table_rows.append(output_row)

    df_overall_table = pd.DataFrame(overall_table_rows, columns=["Solution"] + metrics)
    df_overall_table = df_overall_table.rename(
        columns={
            "Detection Rate": "DR $\\uparrow$",
            "Episode F1": "Episode F1 $\\uparrow$",
            "Average Detection Delay": "MTD $\\downarrow$",
            "False Alarm Rate": "FAR $\\downarrow$",
        }
    )

    overall_filename = os.path.join(write_path, "overall_detection_quality_average.tex")
    overall_latex = df_overall_table.to_latex(
        index=False,
        escape=False,
        caption="Average performance of data-shift detection methods across all evaluated shifts.",
        label="tab:overall_detection_quality_average",
        column_format="l" + "c" * len(metrics),
    )
    overall_latex = overall_latex.replace("\\begin{table}", "\\begin{table*}", 1).replace("\\end{table}",
                                                                                          "\\end{table*}", 1)
    overall_latex = overall_latex.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}",
                                          1).replace(
        "\\end{tabular}", "\\end{tabular}%\n}", 1
    )
    overall_latex = "% Requires: \\usepackage{booktabs}\n% Requires: \\usepackage{graphicx}\n" + overall_latex

    with open(overall_filename, "w", encoding="utf-8") as f:
        f.write(overall_latex)

    return df_experimental, df_table, df_overall_table


def _format_solution_for_accuracy_table(solution):
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
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.nan, np.nan, 0
    values = values.to_numpy(dtype=float) * 100.0
    mean_value = float(np.mean(values))
    if len(values) == 1 or np.allclose(values, values[0]):
        return round(mean_value, 2), 0.00, len(values)

    sem = st.sem(values)
    lower, upper = st.t.interval(confidence=ci, df=len(values) - 1, loc=mean_value, scale=sem)
    margin = max(mean_value - lower, upper - mean_value)
    return round(mean_value, 2), round(margin, 2), len(values)


def _accuracy_is_statistically_superior_candidate(candidate, all_results):
    if candidate is None or not all_results:
        return False
    candidate_mean = candidate.get("mean", np.nan)
    candidate_ci = candidate.get("ci", np.nan)
    if pd.isna(candidate_mean) or pd.isna(candidate_ci):
        return False

    valid_results = [r for r in all_results if r and not pd.isna(r.get("mean")) and not pd.isna(r.get("ci"))]
    if not valid_results:
        return False

    max_mean = max(r["mean"] for r in valid_results)
    maximum_results = [r for r in valid_results if np.isclose(r["mean"], max_mean, rtol=0.0, atol=1e-12)]

    if any(candidate is r for r in maximum_results):
        return True

    candidate_lower = candidate_mean - candidate_ci
    candidate_upper = candidate_mean + candidate_ci

    for max_res in maximum_results:
        m_lower = max_res["mean"] - max_res["ci"]
        m_upper = max_res["mean"] + max_res["ci"]
        if candidate_lower <= m_upper and candidate_upper >= m_lower:
            return True
    return False


def table_accuracy_concept_drift_by_dataset(
        df_all,
        write_path,
        solutions,
        concept_alphas=(0.1, 1.0),
        gradual_windows=(10,),
        ci=0.95,
        accuracy_rounds=None,
):
    if df_all is None or df_all.empty:
        return {}

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
            configurations.append({
                "temporal": "Gradual",
                "window": int(window),
                "alpha": float(alpha),
                "experiment_id": f"concept_drift#{alpha:.1f}_gradual",
                "column": rf"Gradual ($W={int(window)}$), $\alpha={alpha:g}$",
            })

    df = df_all[df_all["Experiment ID"].astype(str).str.startswith("concept_drift#")].copy()
    if df.empty:
        return {}

    def normalize_window(value):
        if pd.isna(value):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    df["_TransitionWindow"] = df["Transition Window"].apply(normalize_window)
    df["_Accuracy"] = pd.to_numeric(df["Accuracy"], errors="coerce")

    table_results = {}
    for dataset_name in ["WISDM-W", "ImageNet10", "Foursquare"]:
        df_dataset = df[df["Dataset"] == dataset_name].copy()
        if df_dataset.empty:
            continue

        raw_results = {solution: {} for solution in solutions}
        for solution in solutions:
            df_solution = df_dataset[df_dataset["Solution"] == solution].copy()
            for config in configurations:
                mask = df_solution["Experiment ID"].astype(str).eq(config["experiment_id"])
                if config["window"] is None:
                    mask &= df_solution["_TransitionWindow"].isna()
                else:
                    mask &= (df_solution["_TransitionWindow"] == config["window"])

                filtered = df_solution.loc[mask].copy()
                mean_value, ci_value, n_value = _accuracy_ci(filtered["_Accuracy"], ci=ci)
                raw_results[solution][config["column"]] = {
                    "mean": mean_value,
                    "ci": ci_value,
                    "n": n_value,
                }

        output_rows = []
        for solution in solutions:
            if solution not in raw_results:
                continue
            row = {"Solution": _format_solution_for_accuracy_table(solution)}
            for config in configurations:
                column = config["column"]
                result = raw_results[solution][column]
                if pd.isna(result["mean"]) or result["n"] == 0:
                    row[column] = "--"
                    continue
                text = f"{result['mean']:.2f} $\\pm$ {result['ci']:.2f}"
                all_results = [
                    raw_results[other][column]
                    for other in solutions
                    if other in raw_results and raw_results[other][column]["n"] > 0
                ]
                if _accuracy_is_statistically_superior_candidate(result, all_results):
                    text = f"\\textbf{{{text}}}"
                row[column] = text
            output_rows.append(row)

        df_table = pd.DataFrame(output_rows, columns=["Solution"] + [c["column"] for c in configurations])
        filename = os.path.join(write_path, f"accuracy_concept_drift_{dataset_name}.tex")
        latex = df_table.to_latex(
            index=False,
            escape=False,
            caption=f"Accuracy under Concept Drift for {dataset_name}.",
            label=f"tab:accuracy_concept_drift_{dataset_name.lower().replace('-', '_')}",
            column_format="l" + "c" * len(configurations),
        )
        latex = "% Requires: \\usepackage{booktabs}\n" + latex
        latex = latex.replace("\\begin{table}", "\\begin{table*}", 1).replace("\\end{table}", "\\end{table*}", 1)
        latex = latex.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", 1).replace(
            "\\end{tabular}", "\\end{tabular}%\n}", 1
        )
        latex = (
            latex.replace("MFP\\_v2\\_dh", "$\\textit{MFP}_{\\textit{DDH}}$")
            .replace("MFP\\_v2\\_iti", "$\\textit{MFP}_{\\textit{ITI}}$")
            .replace("MFP\\_v2", "$\\textit{MFP}$")
        )

        with open(filename, "w", encoding="utf-8") as f:
            f.write(latex)
        table_results[dataset_name] = df_table

    return table_results


def table_accuracy_label_shift_by_dataset(
        df_all,
        write_path,
        solutions,
        gradual_windows=(10,),
        ci=0.95,
        accuracy_rounds=None,
):
    if df_all is None or df_all.empty:
        return {}

    df = df_all[df_all["Experiment ID"].astype(str).str.startswith("label_shift#")].copy()
    if df.empty:
        return {}

    def normalize_window(value):
        if pd.isna(value):
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    df["_TransitionWindow"] = df["Transition Window"].apply(normalize_window)
    df["_Accuracy"] = pd.to_numeric(df["Accuracy"], errors="coerce")
    exp_lower = df["Experiment ID"].astype(str).str.lower()
    df["_Temporal"] = np.select(
        [exp_lower.str.endswith("_sudden"), exp_lower.str.endswith("_gradual")],
        ["Sudden", "Gradual"],
        default="Unknown",
    )

    table_results = {}
    configurations = [{"temporal": "Sudden", "window": None, "column": "Sudden"}]
    for window in gradual_windows:
        configurations.append({"temporal": "Gradual", "window": int(window), "column": rf"Gradual ($W={int(window)}$)"})

    for dataset_name in ["WISDM-W", "ImageNet10", "Foursquare"]:
        df_dataset = df[df["Dataset"] == dataset_name].copy()
        if df_dataset.empty:
            continue

        raw_results = {solution: {} for solution in solutions}
        for solution in solutions:
            df_solution = df_dataset[df_dataset["Solution"] == solution].copy()
            for config in configurations:
                mask = df_solution["_Temporal"] == config["temporal"]
                if config["temporal"] == "Gradual":
                    mask &= (df_solution["_TransitionWindow"] == config["window"])
                else:
                    mask &= df_solution["_TransitionWindow"].isna()

                filtered = df_solution.loc[mask].copy()
                mean_value, ci_value, n_value = _accuracy_ci(filtered["_Accuracy"], ci=ci)
                raw_results[solution][config["column"]] = {
                    "mean": mean_value,
                    "ci": ci_value,
                    "n": n_value,
                }

        output_rows = []
        for solution in solutions:
            if solution not in raw_results:
                continue
            row = {"Solution": _format_solution_for_accuracy_table(solution)}
            for config in configurations:
                column = config["column"]
                result = raw_results[solution][column]
                if pd.isna(result["mean"]) or result["n"] == 0:
                    row[column] = "--"
                    continue
                text = f"{result['mean']:.2f} $\\pm$ {result['ci']:.2f}"
                all_results = [
                    raw_results[other][column]
                    for other in solutions
                    if other in raw_results and raw_results[other][column]["n"] > 0
                ]
                if _accuracy_is_statistically_superior_candidate(result, all_results):
                    text = f"\\textbf{{{text}}}"
                row[column] = text
            output_rows.append(row)

        df_table = pd.DataFrame(output_rows, columns=["Solution"] + [c["column"] for c in configurations])
        filename = os.path.join(write_path, f"accuracy_label_shift_{dataset_name}.tex")
        latex = df_table.to_latex(
            index=False,
            escape=False,
            caption=f"Accuracy under Label Shift for {dataset_name}.",
            label=f"tab:accuracy_label_shift_{dataset_name.lower().replace('-', '_')}",
            column_format="l" + "c" * len(configurations),
        )
        latex = "% Requires: \\usepackage{booktabs}\n" + latex
        latex = latex.replace("\\begin{table}", "\\begin{table*}", 1).replace("\\end{table}", "\\end{table*}", 1)
        latex = latex.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", 1).replace(
            "\\end{tabular}", "\\end{tabular}%\n}", 1
        )
        latex = (
            latex.replace("MFP\\_v2\\_dh", "$\\textit{MFP}_{\\textit{DDH}}$")
            .replace("MFP\\_v2\\_iti", "$\\textit{MFP}_{\\textit{ITI}}$")
            .replace("MFP\\_v2", "$\\textit{MFP}$")
        )

        with open(filename, "w", encoding="utf-8") as f:
            f.write(latex)
        table_results[dataset_name] = df_table

    return table_results


def extract_alpha_from_experiment(experiment_id):
    experiment_id = str(experiment_id).strip()
    if "#" not in experiment_id:
        raise ValueError(f"Experiment ID inválido: {experiment_id}")

    shift_type, config = experiment_id.split("#", 1)
    config = re.sub(r"_(?:sudden|gradual)$", "", config, flags=re.IGNORECASE)

    if shift_type == "concept_drift":
        try:
            return float(config)
        except ValueError as e:
            raise ValueError(f"Alpha inválido: {experiment_id}") from e

    if shift_type in ("label_shift", "combined_shift"):
        if "-" not in config:
            raise ValueError(f"Configuração inválida: {experiment_id}")
        alpha_before, alpha_after = config.split("-", 1)
        try:
            return (float(alpha_before), float(alpha_after))
        except ValueError as e:
            raise ValueError(f"Alpha inválido: {experiment_id}") from e

    raise ValueError(f"Tipo de shift desconhecido: {shift_type}")


def _accuracy_solution_display_name(solution):
    solution = str(solution)
    replacements = {
        "MultiFedAvg+MFP_v2": "MultiFedAvg+MFP",
        "MFP_v2_dh": r"$\textit{MFP}_{\textit{DDH}}$",
        "MFP_v2_iti": r"$\textit{MFP}_{\textit{ITI}}$",
        "MFP_v2": "MFP",
    }
    return replacements.get(solution, solution.replace("_", r"\_"))


def _accuracy_is_statistically_superior(results, candidate_solution):
    candidate = results.get(candidate_solution)
    if candidate is None or pd.isna(candidate["mean"]):
        return False
    candidate_mean = candidate["mean"]
    candidate_ci = candidate["ci"]

    for other_solution, other in results.items():
        if other_solution == candidate_solution or other is None or pd.isna(other["mean"]):
            continue
        if pd.isna(candidate_ci) or pd.isna(other["ci"]):
            return False
        if (candidate_mean - candidate_ci) <= (other["mean"] + other["ci"]):
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
    if df is None or df.empty:
        return

    concept = df[df["Experiment ID"].astype(str).str.startswith("concept_drift#")].copy()
    concept["Alpha"] = pd.to_numeric(concept["Alpha"], errors="coerce")
    concept = concept[
        concept["Alpha"].apply(
            lambda x: any(np.isclose(x, a, rtol=1e-12, atol=1e-12) for a in alphas_order)
            if not pd.isna(x) else False
        )
    ].copy()

    exp_lower = concept["Experiment ID"].astype(str).str.strip().str.lower()
    concept["Temporal Type"] = np.select(
        [exp_lower.str.endswith("_sudden"), exp_lower.str.endswith("_gradual")],
        ["Sudden", "Gradual"],
        default="N/A",
    )
    concept["Transition Window"] = pd.to_numeric(concept["Transition Window"], errors="coerce")

    if datasets_order is None:
        datasets_order = sorted(concept["Dataset"].dropna().unique().tolist())

    Path(write_path).mkdir(parents=True, exist_ok=True)
    configurations = [
        ("Sudden", None, 0.1),
        ("Sudden", None, 1.0),
        ("Gradual", 10, 0.1),
        ("Gradual", 10, 1.0),
    ]

    solutions = [s for s in solutions_order if s in concept["Solution"].astype(str).values]

    for dataset_name in datasets_order:
        dataset_df = concept[concept["Dataset"].astype(str) == str(dataset_name)].copy()
        table_rows = []

        for temporal_type, transition_window, alpha in configurations:
            mask_alpha = np.isclose(dataset_df["Alpha"].astype(float), float(alpha), rtol=1e-12, atol=1e-12)
            filtered = dataset_df[(dataset_df["Temporal Type"] == temporal_type) & mask_alpha].copy()

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
                    filtered.loc[filtered["Solution"].astype(str) == str(solution), "Accuracy"],
                    errors="coerce",
                ).dropna() * 100.0
                mean_value, ci_value = mean_ci(values, ci=ci, bounded=False)
                results[solution] = {"mean": mean_value, "ci": ci_value, "n": len(values)}

            for solution in solutions:
                result = results[solution]
                if pd.isna(result["mean"]):
                    value_text = "--"
                elif pd.isna(result["ci"]):
                    value_text = f"{result['mean']:.2f}"
                else:
                    value_text = f"{result['mean']:.2f} $\\pm$ {result['ci']:.2f}"

                if _accuracy_is_statistically_superior(results, solution):
                    value_text = f"\\textbf{{{value_text}}}"

                table_rows.append({
                    "Temporal Type": temporal_type,
                    "Transition Window": "N/A" if transition_window is None else str(transition_window),
                    r"$\alpha$": f"{alpha:g}",
                    "Solution": _accuracy_solution_display_name(solution),
                    "Accuracy": value_text,
                })

        table_df = pd.DataFrame(
            table_rows,
            columns=["Temporal Type", "Transition Window", r"$\alpha$", "Solution", "Accuracy"],
        )
        filename = os.path.join(write_path, f"accuracy_concept_drift_{dataset_name}.tex")
        latex = table_df.to_latex(
            index=False,
            escape=False,
            caption=f"Accuracy on {dataset_name} under Concept Drift.",
            label="tab:accuracy_concept_drift_" + re.sub(r"[^A-Za-z0-9]+", "_", str(dataset_name)).strip("_").lower(),
            column_format="ccccc",
        )
        latex = "% Requires: \\usepackage{booktabs}\n" + latex
        with open(filename, "w", encoding="utf-8") as f:
            f.write(latex)


def _validation_shift_type(experiment_id):
    exp = str(experiment_id or "").strip().lower()
    if exp.startswith("concept_drift#"):
        return "Concept drift"
    if exp.startswith("label_shift#"):
        return "Label shift"
    if exp.startswith("combined_shift#"):
        return "Combined shift"
    return "N/A"


def _validation_temporal_type(experiment_id):
    exp = str(experiment_id or "").strip().lower()
    if exp.endswith("_sudden"):
        return "Sudden"
    if exp.endswith("_gradual"):
        return "Gradual"
    return "N/A"


def _validation_alpha(experiment_id):
    exp = str(experiment_id or "").strip()
    if "#" not in exp:
        return "N/A"
    config = exp.split("#", 1)[1]
    if "_" in config:
        config = config.split("_", 1)[0]
    if not config:
        return "N/A"
    if "-" in config:
        parts = config.split("-", 1)
        try:
            return f"{float(parts[0]):g} -> {float(parts[1]):g}"
        except (ValueError, TypeError):
            return config
    try:
        return f"{float(config):g}"
    except (ValueError, TypeError):
        return config


if __name__ == "__main__":

    total_clients = 40
    dataset = ["WISDM-W", "ImageNet10", "Foursquare"]
    dataset_shift_rounds = {
        "WISDM-W": 30,
        "ImageNet10": 50,
        "Foursquare": 70
    }
    model_name = ["gru", "CNN", "lstm"]
    fraction_fit = 0.375
    number_of_rounds = 100
    local_epochs = 1
    MAX_DETECTION_DELAY = 10
    USE_TRANSITION_WINDOW_FOR_GRADUAL_METRICS = False
    train_test = "test"

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
        "concept_drift#0.1_gradual",
        "concept_drift#1.0_gradual",
    ]

    label_solutions = [
        "MultiFedAvg+MFP_v2",
        "MultiFedAvg+MFP_v2_dh",
        "MultiFedAvg+MFP_v2_iti",
        "MultiFedAvg+FPD",
        "MultiFedAvg",
        "MultiFedAvgRR",
        "JS-Drift",
        "FedConD",
        "FedDCA",
        "CDA-FedAvg",
    ]

    label_experiments = [
        "label_shift#0.1-1.0_sudden",
        "label_shift#0.1-1.0_gradual",
        "label_shift#1.0-0.1_sudden",
        "label_shift#1.0-0.1_gradual",
    ]

    combined_experiments = []

    experiment_ids = (
            concept_experiments
            + label_experiments
            + combined_experiments
    )

    GRADUAL_TRANSITION_WINDOWS = [10]

    experiment_configurations = []
    for _experiment_id in experiment_ids:
        if _experiment_id.lower().endswith("_gradual"):
            for _transition_window in GRADUAL_TRANSITION_WINDOWS:
                experiment_configurations.append((_experiment_id, _transition_window))
        else:
            experiment_configurations.append((_experiment_id, None))

    df_all = None

    for experiment_id, transition_window in experiment_configurations:
        alpha_config = extract_alpha_from_experiment(experiment_id)

        if experiment_id.startswith("concept_drift#"):
            alpha_value = alpha_config
            alphas = [alpha_value] * len(dataset)
        elif experiment_id.startswith("label_shift#"):
            alpha_before, alpha_after = alpha_config
            alphas = [alpha_before, alpha_before, alpha_before]
            alpha_value = alpha_config
        elif experiment_id.startswith("combined_shift#"):
            alpha_before, alpha_after = alpha_config
            alphas = [alpha_before, alpha_before, alpha_before]
            alpha_value = alpha_config
        else:
            raise ValueError(f"Experiment ID não suportado: {experiment_id}")

        active_solutions = (
            concept_solutions if experiment_id.startswith("concept_drift#") else solutions
        )

        read_solutions = {solution: [] for solution in active_solutions}

        for solution in active_solutions:
            base_read_path = (
                "../system/results/"
                "experiment_id_{}/"
                "clients_{}/"
                "alpha_{}/"
            ).format(experiment_id, total_clients, alphas)

            if transition_window is not None:
                base_read_path += f"transition_window_{transition_window}/"

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

            for dataset_name in dataset:
                detection_file = os.path.join(read_path, f"{dataset_name}_{solution}.csv")
                read_solutions[solution].append(detection_file)

        df = read_data(
            read_solutions,
            experiment_id=experiment_id,
            alpha_value=alpha_value,
            dataset_shift_rounds=dataset_shift_rounds,
            transition_window=transition_window,
        )

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

            if df_all is None:
                df_all = df.copy()
            else:
                df_all = pd.concat([df_all, df], ignore_index=True)

    if df_all is None or df_all.empty:
        print("\n[AVISO] Nenhum dado carregado para gerar tabelas LaTeX.")
    else:
        write_path = "plots/MEFL/multi_experiments/"
        Path(write_path).mkdir(parents=True, exist_ok=True)

        table_accuracy_concept_drift(
            df=df_all,
            write_path=write_path,
            solutions_order=concept_solutions,
            datasets_order=dataset,
            alphas_order=(0.1, 1.0),
            ci=0.95,
        )

        table_accuracy_label_shift_by_dataset(
            df_all=df_all,
            write_path=write_path,
            solutions=label_solutions,
            gradual_windows=(10,),
            ci=0.95,
            accuracy_rounds=None,
        )

        detection_solutions = [
            "MultiFedAvg+MFP_v2",
            "JS-Drift",
            "FedConD",
            "FedDCA",
            "CDA-FedAvg",
        ]

        table_detection_quality_by_shift_type(
            df_final=df_all,
            write_path=write_path,
            solutions=detection_solutions,
            metrics=[
                "Episode F1",
                "Detection Rate",
                "False Alarm Rate",
                "Average Detection Delay",
            ],
            higher_is_better_metrics={"Detection Rate", "Episode F1"},
            ci=0.95,
            max_detection_delay=MAX_DETECTION_DELAY,
            use_transition_window_for_gradual=USE_TRANSITION_WINDOW_FOR_GRADUAL_METRICS,
        )

        table_accuracy_concept_drift_by_dataset(
            df_all=df_all,
            write_path=write_path,
            solutions=concept_solutions,
            concept_alphas=(0.1, 1.0),
            gradual_windows=(10,),
            ci=0.95,
            accuracy_rounds=None,
        )

    # =========================================================================
    # RELATÓRIO FINAL CONSOLIDADO: EXPERIMENTOS SEM 100 RODADAS EM 5 FOLDS
    # =========================================================================
    print("\n" + "#" * 100)
    print("RELATÓRIO FINAL: EXPERIMENTOS QUE NÃO POSSUEM 100 RODADAS EM 5 FOLDS COMPLETOS")
    print("#" * 100)

    failed_experiments = [
        rec for rec in EXPERIMENT_VALIDATION
        if not rec.get("complete_5folds_100rounds", False)
    ]

    total_esperados = len(EXPERIMENT_VALIDATION)
    total_incompletos = len(failed_experiments)
    total_completos = total_esperados - total_incompletos

    print(f"Total avaliado : {total_esperados}")
    print(f"Completos (5 folds x 100 rodadas) : {total_completos}")
    print(f"Incompletos / Ausentes            : {total_incompletos}")

    if not failed_experiments:
        print("\nTodos os experimentos possuem exatamente 5 folds com as 100 rodadas (1 a 100) completas!")
    else:
        # Agrupamento para visão sintetizada por Experimento / Solução / Dataset
        resumo_incompletos = []
        for r in failed_experiments:
            exp_id = r.get("experiment_id", "N/A")
            sol = r.get("solution", "N/A")
            dset = r.get("dataset") or (os.path.basename(r.get("path", "")).split("_")[0] if r.get("path") else "N/A")
            win = r.get("transition_window")
            status = r.get("status", "N/A")
            f_count = r.get("fold_count", 0)
            f_ids = r.get("fold_ids", [])
            details = r.get("fold_details", {})

            # Motivos da incompletude
            motivos = []
            if status == "missing":
                motivos.append("Arquivo CSV inexistente no disco")
            elif status == "empty":
                motivos.append("Arquivo CSV vazio")
            elif status in ("invalid", "error"):
                motivos.append(f"Erro de leitura: {r.get('error', 'desconhecido')}")
            else:
                if f_count != 5:
                    motivos.append(f"Quantidade de folds incorreta: {f_count} (esperado 5: {f_ids})")

                folds_com_faltas = []
                for fid, f_info in details.items():
                    if not f_info.get("complete", False):
                        missing_len = len(f_info.get("missing_rounds", []))
                        r_count = f_info.get("round_count", 0)
                        folds_com_faltas.append(f"Fold {fid} ({r_count}/100 rounds, faltam {missing_len})")

                if folds_com_faltas:
                    motivos.append("Folds com rodadas incompletas: " + "; ".join(folds_com_faltas))

            resumo_incompletos.append({
                "Experimento": exp_id,
                "Janela": f"W={win}" if win is not None else "Sudden",
                "Solução": sol,
                "Dataset": dset,
                "Status": status,
                "Folds": f"{f_count}/5",
                "Motivos": " | ".join(motivos),
                "Arquivo": r.get("path", "N/A")
            })

        print("\n" + "=" * 100)
        print("LISTA DETALHADA DE CONFIGURAÇÕES INCOMPLETAS")
        print("=" * 100)
        for idx, item in enumerate(resumo_incompletos, start=1):
            print(f"\n[{idx}] Experimento: {item['Experimento']} ({item['Janela']})")
            print(f"    Solução : {item['Solução']} | Dataset: {item['Dataset']}")
            print(f"    Status  : {item['Status']} | Folds encontrados: {item['Folds']}")
            print(f"    Motivo  : {item['Motivos']}")
            print(f"    Caminho : {item['Arquivo']}")

    print("\n" + "#" * 100 + "\n")