import copy
import csv
import os
import random
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from flwr.server.strategy.aggregate import weighted_loss_avg

from flcore.servers.server_multifedavg import MultiFedAvg
from flcore.clients.client_feddca import ClientFedDCA
from flcore.clients.utils.models_utils import get_weights


# ---------------------------------------------------------------------------
# Basic metric aggregation: intentionally compatible with MultiFedAvg.
# ---------------------------------------------------------------------------
def weighted_average(metrics):
    if not metrics:
        return {}
    examples = [n for n, _ in metrics]
    total = max(sum(examples), 1)

    def wmean(key):
        vals = [n * float(m.get(key, 0.0)) for n, m in metrics]
        return sum(vals) / total

    first = metrics[0][1]
    return {
        "Accuracy": wmean("Accuracy"),
        "Balanced accuracy": wmean("Balanced accuracy"),
        "Loss": wmean("Loss"),
        "Round (t)": first.get("Round (t)"),
        "Model size": first.get("Model size"),
        "Alpha": first.get("Alpha", first.get("alpha", 0.0)),
    }


def weighted_average_fit(metrics):
    if not metrics:
        return {}
    examples = [n for n, _ in metrics]
    total = max(sum(examples), 1)

    def wmean(key):
        vals = [n * float(m.get(key, 0.0)) for n, m in metrics]
        return sum(vals) / total

    first = metrics[0][1]
    return {
        "Accuracy": wmean("train_accuracy"),
        "Balanced accuracy": wmean("train_balanced_accuracy"),
        "Loss": wmean("train_loss"),
        "Round (t)": first.get("Round (t)"),
        "Model size": first.get("Model size"),
    }


class FedDCA(MultiFedAvg):
    """FedDCA adapted to the existing MultiFedAvg/MEFL framework.

    Important architectural adaptation
    -----------------------------------
    The original FedDCA paper assumes one shared feature extractor and
    cluster-specific classifiers. MEFL, however, contains independent
    models. Here each ``me`` is treated as an independent FedDCA instance.
    Within each ``me`` we preserve the paper's hybrid aggregation as far as
    the existing model API permits:

      * feature/extractor parameters: aggregated globally across clients;
      * final classifier parameters: aggregated separately by FedDCA cluster.

    The cluster assignment is driven exclusively by Label Profiles, and
    drifting clients are excluded from VWC and subsequently assigned to the
    nearest stable anchor.
    """

    def __init__(self, args, models, fold_id):
        super().__init__(args, models, fold_id)

        # FedDCA hyperparameters. The paper reports sensitivity ranges for
        # k, lambda and alpha_ewma, but does not provide a single canonical
        # implementation-level default for every quantity. We expose them
        # explicitly instead of silently hard-coding undocumented values.
        self.feddca_num_clusters = int(getattr(args, "feddca_num_clusters", 3))
        self.feddca_num_prototypes = int(
            getattr(args, "feddca_num_prototypes", 5)
        )
        self.feddca_ewma_alpha = float(
            getattr(args, "feddca_ewma_alpha", 0.5)
        )
        self.feddca_sinkhorn_reg = float(
            getattr(args, "feddca_sinkhorn_reg", 1.0)
        )
        self.feddca_sinkhorn_iters = int(
            getattr(args, "feddca_sinkhorn_iters", 100)
        )
        self.feddca_vwc_iters = int(
            getattr(args, "feddca_vwc_iters", 10)
        )
        self.feddca_tol = float(getattr(args, "feddca_tol", 1e-4))
        self.feddca_eps = float(getattr(args, "feddca_eps", 1e-8))

        # The paper states that drift is compared with a dynamic threshold
        # computed from EWMA. Since the manuscript does not specify a unique
        # extra margin, the default uses the EWMA itself; users may provide a
        # multiplicative margin when reproducing a particular implementation.
        self.feddca_drift_threshold_factor = float(
            getattr(args, "feddca_drift_threshold_factor", 1.0)
        )
        # The manuscript specifies an EWMA-based dynamic threshold but does
        # not publish the complete numerical threshold equation.  We therefore
        # use the EWMA as the adaptive baseline and estimate normal temporal
        # variation from the client's own distance history.
        self.feddca_threshold_sigma = float(
            getattr(args, "feddca_threshold_sigma", 3.0)
        )
        self.feddca_min_history = int(
            getattr(args, "feddca_min_history", 3)
        )

        # IMPORTANT: these schemas intentionally match FedConD exactly.
        # Do not add FedDCA-specific columns here because the downstream
        # analysis scripts expect the same CSV structure as FedConD.
        self.train_metrics_names = [
            "Accuracy",
            "Balanced accuracy",
            "Loss",
            "Round (t)",
            "Fraction fit",
            "# training clients",
            "training clients and models",
            "Model size",
            "Alpha",
            "Drift clients",
            "Drift rate",
            "Data shift"
        ]
        self.test_metrics_names = [
            "Accuracy",
            "Balanced accuracy",
            "Loss",
            "Round (t)",
            "Fraction fit",
            "# training clients",
            "training clients and models",
            "Model size",
            "Fold ID",
            "Alpha",
            "Drift clients",
            "Drift rate",
            "Data shift",
            "Ground truth shift"
        ]
        # MultiFedAvg creates these dictionaries, but recreate them here
        # because the FedDCA schemas above replace the inherited schemas.
        self.results_train_metrics = {
            me: {metric: [] for metric in self.train_metrics_names}
            for me in range(self.ME)
        }
        self.results_test_metrics = {
            me: {metric: [] for metric in self.test_metrics_names}
            for me in range(self.ME)
        }

        self.detector = "FedDCA"
        self.shift_type = (
            "Label" if "label_shift" in args.experiment_id else "Concept"
        )
        self.shift_configuration = (
            args.experiment_id
            .replace("label_shift#", "")
            .replace("concept_drift#", "")
            .replace("_sudden", "")
        )

        self._initialize_feddca_state()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def _initialize_feddca_state(self):
        self.previous_lp = {me: {} for me in range(self.ME)}
        self.current_lp = {me: {} for me in range(self.ME)}
        self.drift_scores = {me: {} for me in range(self.ME)}
        self.ewma_scores = {me: {} for me in range(self.ME)}
        self.drift_flags = {me: {} for me in range(self.ME)}

        # Per-client temporal history of W-distances.  This is kept separate
        # for each ME because the MEFL models represent independent tasks.
        self.distance_history = {me: {} for me in range(self.ME)}

        self.stable_clients = {me: [] for me in range(self.ME)}
        self.drift_clients_ids = {me: [] for me in range(self.ME)}
        self.client_clusters = {
            me: {cid: 0 for cid in range(self.total_clients)}
            for me in range(self.ME)
        }
        self.cluster_members = {me: {} for me in range(self.ME)}
        self.anchor_centroids = {me: {} for me in range(self.ME)}

        # Per-cluster complete model parameters. These are rebuilt after each
        # aggregation and used by the next round's assigned clients.
        self.cluster_models = {me: {} for me in range(self.ME)}
        self.parameters_aggregated_mefl = {me: [] for me in range(self.ME)}

        # Parameter indices belonging to the final classifier.
        self.classifier_param_indices = {
            me: self._find_classifier_parameter_indices(self.global_model[me])
            for me in range(self.ME)
        }

        # Ground-truth/detection bookkeeping compatible with the previous
        # FedConD analysis scripts.
        self.data_shift_type = {me: "NO_SHIFT" for me in range(self.ME)}
        self.drift_clients = {me: 0 for me in range(self.ME)}
        self.drift_rate = {me: 0.0 for me in range(self.ME)}
        self.drift_rate_history = {me: [] for me in range(self.ME)}
        self.shift_rounds = {me: [] for me in range(self.ME)}
        self.shift_ground_truth_state = {me: [] for me in range(self.ME)}
        self.shift_ground_truth_event = {me: [] for me in range(self.ME)}
        self.shift_detected = {me: [] for me in range(self.ME)}
        self.previous_detector_state = {
            me: "NO_SHIFT" for me in range(self.ME)
        }
        self.detection_event = {me: 0 for me in range(self.ME)}
        self.false_alarm_rounds = {me: [] for me in range(self.ME)}
        self.true_detection_round = {me: None for me in range(self.ME)}
        self.detection_delay = {me: -1 for me in range(self.ME)}

        # FedDCA is model-agnostic but requires a configured number of
        # clusters. If no experiment-specific ground truth is available, the
        # shift-round list remains empty.
        if self.total_clients > 0:
            try:
                client0 = self.clients[0]
                for me in range(self.ME):
                    if me in client0.data_shift_config:
                        self.shift_rounds[me] = client0.data_shift_config[me].get(
                            "data_shift_rounds", []
                        )
            except Exception:
                pass

    def set_clients(self):
        self.clients = []
        for cid in range(self.total_clients):
            self.clients.append(
                ClientFedDCA(
                    self.args,
                    id=cid,
                    model=copy.deepcopy(self.global_model),
                    fold_id=self.fold_id,
                )
            )

        # Same auxiliary CSV initialization used by FedConD.
        self._init_shift_detection_files()

    # ------------------------------------------------------------------
    # Wasserstein geometry
    # ------------------------------------------------------------------
    def _sinkhorn_w2(self, x, y):
        """Entropic OT approximation of W2 for two uniform empirical measures."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if y.ndim == 1:
            y = y[None, :]
        if len(x) == 0 or len(y) == 0:
            return 0.0

        # Squared Euclidean ground cost.
        xx = np.sum(x * x, axis=1, keepdims=True)
        yy = np.sum(y * y, axis=1, keepdims=True).T
        cost = np.maximum(xx + yy - 2.0 * x @ y.T, 0.0)

        n, m = cost.shape
        a = np.full(n, 1.0 / n)
        b = np.full(m, 1.0 / m)
        eps = max(self.feddca_sinkhorn_reg, 1e-6)

        # Stable scaling iteration. The small epsilon floor prevents division
        # by zero without changing the intended OT formulation materially.
        K = np.exp(-np.clip(cost / eps, 0.0, 700.0))
        u = np.ones(n)
        v = np.ones(m)
        tiny = 1e-300

        for _ in range(self.feddca_sinkhorn_iters):
            u_prev = u.copy()
            Kv = K @ v + tiny
            u = a / Kv
            KTu = K.T @ u + tiny
            v = b / KTu
            if np.max(np.abs(u - u_prev)) < self.feddca_tol:
                break

        transport = (u[:, None] * K) * v[None, :]
        value = float(np.sum(transport * cost))
        return float(np.sqrt(max(value, 0.0)))

    def _lp_distance(self, lp_a, lp_b):
        """Label-conditional W2 distance between two Label Profiles."""
        labels = sorted(set(lp_a.keys()).intersection(lp_b.keys()))
        if not labels:
            return float("inf")

        distances = []
        for label in labels:
            distances.append(self._sinkhorn_w2(lp_a[label], lp_b[label]))
        return float(np.mean(distances))

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------
    def _update_drift_state(self, me, client_ids):
        """Detect temporal distribution changes from LP Wasserstein distances.

        The paper specifies an EWMA-based dynamic threshold, but does not give
        the complete numerical threshold equation.  The previous implementation
        used ``threshold = previous_EWMA``, which makes any small upward
        fluctuation look like drift and was producing the false alarms observed
        in the experiments.

        Here the EWMA remains the adaptive baseline, while a client's own
        historical W-distance variability provides the detection margin.  The
        current observation is classified BEFORE it is appended to the history,
        so a possible drift cannot inflate its own threshold.
        """
        stable = []
        drift = []

        for cid in client_ids:
            current = self.current_lp[me].get(cid)
            previous = self.previous_lp[me].get(cid)

            history = self.distance_history[me].setdefault(cid, [])

            # First profile establishes the temporal reference; there is no
            # previous LP against which a drift can be measured.
            if previous is None:
                distance = 0.0
                ewma = 0.0
                is_drift = False
            else:
                distance = float(self._lp_distance(current, previous))

                if not np.isfinite(distance):
                    distance = 0.0

                old_ewma = self.ewma_scores[me].get(cid)
                if old_ewma is None or old_ewma <= self.feddca_eps:
                    old_ewma = distance

                # Detection uses only distances from previous rounds.
                # This prevents the current observation from raising its own
                # threshold.
                if len(history) >= self.feddca_min_history:
                    hist = np.asarray(history, dtype=np.float64)
                    hist_mean = float(np.mean(hist))
                    hist_std = float(np.std(hist, ddof=0))

                    # EWMA is the adaptive baseline required by FedDCA.
                    # The historical standard deviation supplies a noise
                    # margin so normal LP fluctuations are not flagged.
                    threshold = max(
                        self.feddca_drift_threshold_factor
                        * max(old_ewma, self.feddca_eps),
                        old_ewma
                        + self.feddca_threshold_sigma * max(
                            hist_std, self.feddca_eps
                        ),
                        hist_mean
                        + self.feddca_threshold_sigma * max(
                            hist_std, self.feddca_eps
                        ),
                    )
                    is_drift = distance > threshold
                else:
                    # Warm-up: establish the client's normal temporal
                    # variability before activating the detector.
                    is_drift = False

                # Update EWMA only after the current observation has been
                # classified.
                ewma = (
                    self.feddca_ewma_alpha * distance
                    + (1.0 - self.feddca_ewma_alpha) * old_ewma
                )

            self.drift_scores[me][cid] = distance
            self.ewma_scores[me][cid] = ewma
            self.drift_flags[me][cid] = bool(is_drift)

            # Keep the history bounded.  A fixed recent window makes the
            # threshold adaptive to non-stationary normal variability.
            history.append(distance)
            max_history = max(20, self.feddca_min_history + 1)
            if len(history) > max_history:
                del history[:-max_history]

            if is_drift:
                drift.append(cid)
            else:
                stable.append(cid)

        self.stable_clients[me] = stable
        self.drift_clients_ids[me] = drift
        return stable, drift

    # ------------------------------------------------------------------
    # VWC-compatible stable-first clustering
    # ------------------------------------------------------------------
    def _init_anchors(self, profiles, client_ids, k):
        """Farthest-first initialization in the Wasserstein geometry."""
        if not client_ids:
            return {}
        k = max(1, min(k, len(client_ids)))

        anchors = {0: copy.deepcopy(profiles[client_ids[0]])}
        selected = [client_ids[0]]

        while len(selected) < k:
            best_cid = None
            best_distance = -1.0
            for cid in client_ids:
                if cid in selected:
                    continue
                d = min(
                    self._lp_distance(profiles[cid], anchors[a])
                    for a in anchors
                )
                if d > best_distance:
                    best_distance = d
                    best_cid = cid
            if best_cid is None:
                break
            idx = len(anchors)
            anchors[idx] = copy.deepcopy(profiles[best_cid])
            selected.append(best_cid)
        return anchors

    def _barycenter_label(self, members, current_support):
        """Approximate the W2 barycenter of equal-weight empirical supports.

        Exact VWC requires solving the power-diagram/Newton subproblem. The
        paper does not provide enough implementation detail to reproduce that
        numerical solver exactly. This routine therefore implements the
        standard equal-mass Wasserstein barycenter update by optimal matching
        to the current support, which is the closest self-contained
        approximation for the present framework.
        """
        if not members:
            return current_support

        support = np.asarray(current_support, dtype=np.float32)
        if support.ndim == 1:
            support = support[None, :]

        accum = np.zeros_like(support, dtype=np.float64)
        weight_sum = 0.0

        for weight, points in members:
            points = np.asarray(points, dtype=np.float32)
            if points.ndim == 1:
                points = points[None, :]
            if len(points) == 0:
                continue

            cost = np.sum(
                (support[:, None, :] - points[None, :, :]) ** 2,
                axis=2,
            )
            rows, cols = linear_sum_assignment(cost)

            matched = np.zeros_like(support, dtype=np.float64)
            counts = np.zeros(len(support), dtype=np.float64)
            for r, c in zip(rows, cols):
                matched[r] += points[c]
                counts[r] += 1.0

            # If support cardinalities differ, fill unmatched support points
            # with the empirical mean of the client's support.
            mean_point = np.mean(points, axis=0)
            for r in range(len(support)):
                if counts[r] == 0:
                    matched[r] = mean_point
                    counts[r] = 1.0
                else:
                    matched[r] /= counts[r]

            accum += float(weight) * matched
            weight_sum += float(weight)

        if weight_sum <= 0:
            return support
        return (accum / weight_sum).astype(np.float32)

    def _barycenter(self, profiles, members, fallback):
        labels = set(fallback.keys())
        for cid in members:
            labels.update(profiles[cid].keys())

        centroid = {}
        for label in sorted(labels):
            data = []
            for cid in members:
                if label in profiles[cid]:
                    # Equal client weights here; the outer clustering uses
                    # dataset-size weights in the objective through assignment
                    # and barycenter updates.
                    n = max(1, len(profiles[cid][label]))
                    data.append((1.0, profiles[cid][label]))
            if not data:
                continue

            if label in fallback:
                init = fallback[label]
            else:
                init = data[0][1]
            centroid[label] = self._barycenter_label(data, init)
        return centroid

    def _vwc_stable(self, me, stable_ids):
        if not stable_ids:
            return {}, {}

        k = max(1, min(self.feddca_num_clusters, len(stable_ids)))
        profiles = self.current_lp[me]
        anchors = self._init_anchors(profiles, stable_ids, k)
        assignments = {}

        for _ in range(self.feddca_vwc_iters):
            changed = False

            # Partition update: nearest Wasserstein anchor.
            for cid in stable_ids:
                distances = {
                    a: self._lp_distance(profiles[cid], anchors[a])
                    for a in anchors
                }
                new_cluster = min(distances, key=distances.get)
                if assignments.get(cid) != new_cluster:
                    changed = True
                assignments[cid] = new_cluster

            # Centroid/barycenter update.
            new_anchors = {}
            for a in sorted(anchors):
                members = [cid for cid in stable_ids if assignments[cid] == a]
                if not members:
                    new_anchors[a] = anchors[a]
                    continue
                new_anchors[a] = self._barycenter(
                    profiles, members, anchors[a]
                )

            # Convergence check in Wasserstein space.
            max_shift = 0.0
            for a in new_anchors:
                if a in anchors:
                    max_shift = max(
                        max_shift,
                        self._lp_distance(new_anchors[a], anchors[a]),
                    )
            anchors = new_anchors
            if not changed or max_shift < self.feddca_tol:
                break

        return assignments, anchors

    def _assign_drift_clients(self, me, assignments, anchors):
        if not anchors:
            return assignments
        profiles = self.current_lp[me]
        for cid in self.drift_clients_ids[me]:
            distances = {
                a: self._lp_distance(profiles[cid], anchors[a])
                for a in anchors
            }
            assignments[cid] = min(distances, key=distances.get)
        return assignments

    # ------------------------------------------------------------------
    # Model aggregation
    # ------------------------------------------------------------------
    @staticmethod
    def _weighted_average_arrays(items):
        """FedAvg-style weighted average of parameter lists."""
        if not items:
            return None
        total = float(sum(max(1, n) for _, n in items))
        out = []
        for pidx in range(len(items[0][0])):
            value = None
            for params, n in items:
                arr = np.asarray(params[pidx])
                contribution = arr * (float(max(1, n)) / total)
                value = contribution if value is None else value + contribution
            out.append(value.astype(np.asarray(items[0][0][pidx]).dtype))
        return out

    def _find_classifier_parameter_indices(self, model):
        """Find parameters of the final classifier layer.

        The original FedDCA has an explicit classifier head. MEFL models in
        this codebase are heterogeneous, so we infer the last Linear module.
        If no Linear layer exists, the last two parameter tensors are used as
        a conservative fallback (typically weight+bias of the classifier).
        """
        linear_modules = [
            (name, module)
            for name, module in model.named_modules()
            if name and module.__class__.__name__.lower() == "linear"
        ]
        if linear_modules:
            classifier_name, _ = linear_modules[-1]
            prefix = classifier_name + "."
            names = [name for name, _ in model.named_parameters()]
            idx = [i for i, name in enumerate(names) if name.startswith(prefix)]
            if idx:
                return idx

        n = len(list(model.parameters()))
        return list(range(max(0, n - 2), n))

    def _aggregate_hybrid(self, me, results_by_me, assignments):
        """Global feature aggregation + cluster classifier aggregation."""
        if not results_by_me:
            return None, {}

        classifier_idx = set(self.classifier_param_indices[me])
        all_items = [
            (params, n) for params, n, _ in results_by_me
        ]
        global_params = self._weighted_average_arrays(all_items)

        cluster_params = {}
        for cluster in sorted(set(assignments.values())):
            cluster_items = []
            for params, n, meta in results_by_me:
                cid = int(meta["client_id"])
                if assignments.get(cid) == cluster:
                    cluster_items.append((params, n))

            if not cluster_items:
                continue

            cluster_model = [np.array(p, copy=True) for p in global_params]
            classifier_avg = self._weighted_average_arrays(cluster_items)
            for idx in classifier_idx:
                cluster_model[idx] = classifier_avg[idx]
            cluster_params[cluster] = cluster_model

        # Ensure at least one complete model exists even if clustering has
        # degenerated to an empty partition.
        if not cluster_params:
            cluster_params[0] = global_params

        return global_params, cluster_params

    # ------------------------------------------------------------------
    # Training loop: same MultiFedAvg semantics, but cluster-aware models.
    # ------------------------------------------------------------------
    def _initial_parameters(self):
        for me in range(self.ME):
            self.parameters_aggregated_mefl[me] = get_weights(self.global_model[me])
            self.cluster_models[me] = {0: self.parameters_aggregated_mefl[me]}
            for cid in range(self.total_clients):
                self.client_clusters[me][cid] = 0

    def train(self):
        try:
            self._get_models_size()
            self._initial_parameters()

            for t in range(1, self.number_of_rounds + 1):
                start = time.time()
                self.selected_clients = self.select_clients(t + self.fold_id)
                print("selected clients:", self.selected_clients)

                fit_results = []
                for me in range(self.ME):
                    for cid in self.selected_clients[me]:
                        cluster = self.client_clusters[me].get(int(cid), 0)
                        if cluster not in self.cluster_models[me]:
                            cluster = 0
                        model_parameters = self.cluster_models[me].get(
                            cluster,
                            self.parameters_aggregated_mefl[me],
                        )
                        fit_results.append(
                            self.clients[int(cid)].fit(
                                me, t, model_parameters
                            )
                        )

                self.parameters_aggregated_mefl, _ = self.aggregate_fit(
                    server_round=t, results=fit_results, failures=[]
                )
                self.evaluate(t, self.parameters_aggregated_mefl)
                print(
                    f"FedDCA round {t} completed in {time.time() - start:.2f}s"
                )

        except Exception as e:
            print("FedDCA train error")
            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno, type(e).__name__, e
                )
            )
            raise

    # ------------------------------------------------------------------
    # Aggregation entry point
    # ------------------------------------------------------------------
    def aggregate_fit(self, server_round: int, results, failures):
        try:
            self.selected_clients_m = [[] for _ in range(self.ME)]
            results_mefl = {me: [] for me in range(self.ME)}

            for params, num_examples, metrics in results:
                me = int(metrics["me"])
                cid = int(metrics["client_id"])
                self.selected_clients_m[me].append(cid)
                results_mefl[me].append((params, num_examples, metrics))
                self.current_lp[me][cid] = metrics["Label Profile"]

            metrics_aggregated = {me: {} for me in range(self.ME)}
            aggregated = {me: self.parameters_aggregated_mefl[me] for me in range(self.ME)}

            for me in range(self.ME):
                if not results_mefl[me]:
                    continue

                client_ids = [m["client_id"] for _, _, m in results_mefl[me]]

                # 1. Temporal drift analysis.
                stable, drift = self._update_drift_state(me, client_ids)

                # 2. VWC on stable clients only.
                stable_assignments, anchors = self._vwc_stable(me, stable)

                # 3. Drifting clients -> nearest stable anchor.
                assignments = self._assign_drift_clients(
                    me, stable_assignments, anchors
                )

                # If the first round contains only profiles with no temporal
                # reference, every client is stable and clustering is entirely
                # driven by current distributions.
                self.client_clusters[me].update(assignments)
                self.cluster_members[me] = {}
                for cid, cluster in assignments.items():
                    self.cluster_members[me].setdefault(cluster, []).append(cid)
                self.anchor_centroids[me] = anchors

                # 4. Hybrid aggregation: global feature extractor + cluster
                # classifier. This is the MEFL-compatible implementation of
                # the paper's two aggregation components.
                global_params, cluster_params = self._aggregate_hybrid(
                    me, results_mefl[me], assignments
                )
                if global_params is not None:
                    aggregated[me] = global_params
                self.cluster_models[me] = cluster_params

                # 5. Keep the global model synchronized for any inherited
                # MultiFedAvg functionality.
                if global_params is not None:
                    self.parameters_aggregated_mefl[me] = global_params

                # 6. Metrics.
                n_drift = len(drift)
                rate = n_drift / max(1, len(client_ids))
                self.drift_clients[me] = n_drift
                self.drift_rate[me] = rate
                self.drift_rate_history[me].append(rate)
                self.data_shift_type[me] = (
                    "DATA_SHIFT" if n_drift > 0 else "NO_SHIFT"
                )

                if self.fit_metrics_aggregation_fn:
                    fit_metrics = [
                        (n, metrics) for _, n, metrics in results_mefl[me]
                    ]
                    metrics_aggregated[me] = self.fit_metrics_aggregation_fn(
                        fit_metrics
                    )
                else:
                    metrics_aggregated[me] = {}

                metrics_aggregated[me].update(
                    {
                        "Drift clients": n_drift,
                        "Drift rate": rate,
                        "Data shift": self.data_shift_type[me],
                        "Round (t)": server_round,
                        "Fraction fit": self.fraction_fit,
                        "# training clients": self.n_trained_clients,
                        "training clients and models": self.selected_clients_m[me],
                        "Model size": self.models_size[me],
                        "Alpha": metrics_aggregated[me].get("Alpha", self.alpha[me]),
                    }
                )

                self._update_detection_metrics(me, server_round)

                # Keep the training CSV structure identical to FedConD.
                for metric in self.train_metrics_names:
                    self.results_train_metrics[me][metric].append(
                        metrics_aggregated[me].get(metric, 0)
                    )

                # Current LP becomes the previous temporal reference only
                # after all current-round drift decisions have been made.
                self.previous_lp[me] = {
                    cid: copy.deepcopy(lp)
                    for cid, lp in self.current_lp[me].items()
                }

                print(
                    f"FedDCA me={me} round={server_round}: "
                    f"stable={stable}, drift={drift}, "
                    f"clusters={self.cluster_members[me]}"
                )

            # FedConD writes the auxiliary detection CSVs once per round,
            # after all ME models have updated their detection state.
            self._save_shift_detection_metrics(server_round)
            self._save_shift_detection_curve(server_round)

            self.parameters_aggregated_mefl = aggregated
            self.metrics_aggregated_mefl = metrics_aggregated

            if server_round > 10:
                try:
                    self._save_data_metrics()
                except Exception:
                    pass

            return self.parameters_aggregated_mefl, metrics_aggregated

        except Exception as e:
            print("FedDCA aggregate_fit error")
            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno, type(e).__name__, e
                )
            )
            raise

    # ------------------------------------------------------------------
    # Detection evaluation compatible with the previous analysis scripts.
    # ------------------------------------------------------------------
    def _update_detection_metrics(self, me, server_round):
        shift_rounds = self.shift_rounds.get(me, [])
        ground_truth_state = int(
            any(server_round >= r for r in shift_rounds)
        )
        ground_truth_event = int(server_round in shift_rounds)
        current_state = self.data_shift_type[me]

        self.detection_event[me] = int(
            self.previous_detector_state[me] == "NO_SHIFT"
            and current_state == "DATA_SHIFT"
        )
        self.previous_detector_state[me] = current_state

        self.shift_ground_truth_state[me].append(ground_truth_state)
        self.shift_ground_truth_event[me].append(ground_truth_event)
        self.shift_detected[me].append(self.detection_event[me])

        if self.detection_event[me]:
            if shift_rounds:
                shift_round = min(shift_rounds)
                if server_round < shift_round:
                    self.false_alarm_rounds[me].append(server_round)
                elif self.true_detection_round[me] is None:
                    self.true_detection_round[me] = server_round
                    self.detection_delay[me] = server_round - shift_round

    # ------------------------------------------------------------------
    # CSV persistence: intentionally copied in structure from FedConD.
    # ------------------------------------------------------------------
    def add_metrics(self, server_round, metrics_aggregated, me):
        try:
            metrics_aggregated[me]["Fraction fit"] = self.fraction_fit
            metrics_aggregated[me]["# training clients"] = self.n_trained_clients
            metrics_aggregated[me]["training clients and models"] = self.selected_clients_m[me]
            metrics_aggregated[me]["Fold ID"] = self.fold_id
            metrics_aggregated[me]["Data shift"] = self.data_shift_type[me]
            metrics_aggregated[me]["Drift clients"] = self.drift_clients[me]
            metrics_aggregated[me]["Drift rate"] = self.drift_rate[me]
            metrics_aggregated[me]["Ground truth shift"] = (
                self.shift_ground_truth_state[me][-1]
                if len(self.shift_ground_truth_state[me]) > 0
                else 0
            )

            for metric in self.test_metrics_names:
                self.results_test_metrics[me][metric].append(
                    metrics_aggregated[me].get(metric, 0)
                )
        except Exception as e:
            print("add_metrics error")
            print("Error on line {} {} {}".format(
                sys.exc_info()[-1].tb_lineno, type(e).__name__, e
            ))

    def _get_results(self, train_test, mode, me):
        try:
            algo = self.dataset[me] + "_" + self.strategy_name
            result_path = self.get_result_path(train_test)
            if not os.path.exists(result_path):
                os.makedirs(result_path)

            file_path = result_path + "{}.csv".format(algo)

            if train_test == "test":
                header = self.test_metrics_names
                metric_dict = self.results_test_metrics[me]
            else:
                header = self.train_metrics_names
                metric_dict = self.results_train_metrics[me]

            keys = list(metric_dict.keys())
            length = max((len(metric_dict[k]) for k in keys), default=0)
            data = []
            for i in range(length):
                row = []
                for k in keys:
                    values = metric_dict[k]
                    row.append(values[i] if i < len(values) else 0)
                data.append(row)

            return file_path, header, data
        except Exception as e:
            print("get_results error")
            print("Error on line {} {} {}".format(
                sys.exc_info()[-1].tb_lineno, type(e).__name__, e
            ))
            raise

    def _save_results(self, server_round, me):
        try:
            # Train CSV
            file_path, header, data = self._get_results("train", "", me)
            if self.fold_id == 1 and server_round == 1:
                self._write_header(file_path, header=header, mode="w")
            self._write_outputs(file_path, data=data)

            # Test CSV
            file_path, header, data = self._get_results("test", "", me)
            if self.fold_id == 1 and server_round == 1:
                self._write_header(file_path, header=header, mode="w")
            self._write_outputs(file_path, data=data)
        except Exception as e:
            print("save_results error")
            print("Error on line {} {} {}".format(
                sys.exc_info()[-1].tb_lineno, type(e).__name__, e
            ))

    def _save_shift_detection_metrics(self, server_round):
        try:
            result_path = self.get_result_path("test")
            file_path = result_path + f"shift_detection_metrics_{self.strategy_name}.csv"

            for me in range(self.ME):
                y_true = self.shift_ground_truth_event[me]
                if len(y_true) == 0:
                    precision = recall = f1 = 0.0
                else:
                    tp = 1 if self.true_detection_round[me] is not None else 0
                    fp = len(self.false_alarm_rounds[me])
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = float(tp)
                    f1 = (2 * precision * recall / (precision + recall)
                          if precision + recall > 0 else 0.0)

                shift_round = self.shift_rounds[me][0] if self.shift_rounds[me] else -1
                self._write_rows(file_path, [[
                    self.detector,
                    self.dataset[me],
                    self.fold_id,
                    server_round,
                    me,
                    self.shift_type,
                    self.shift_configuration,
                    precision,
                    recall,
                    f1,
                    self.detection_delay[me],
                    len(self.false_alarm_rounds[me]),
                    self.true_detection_round[me] if self.true_detection_round[me] is not None else -1,
                    shift_round,
                ]])
        except Exception as e:
            print("_save_shift_detection_metrics error")
            print("Error on line {} {} {}".format(
                sys.exc_info()[-1].tb_lineno, type(e).__name__, e
            ))

    def _save_shift_detection_curve(self, server_round):
        try:
            result_path = self.get_result_path("test")
            file_path = result_path + f"shift_detection_curve_{self.strategy_name}.csv"
            for me in range(self.ME):
                if not self.shift_ground_truth_state[me]:
                    continue
                self._write_rows(file_path, [[
                    self.detector,
                    self.dataset[me],
                    self.fold_id,
                    server_round,
                    me,
                    self.shift_ground_truth_state[me][-1],
                    self.detection_event[me],
                    self.data_shift_type[me],
                    self.drift_clients[me],
                    self.drift_rate[me],
                ]])
        except Exception as e:
            print("_save_shift_detection_curve error")
            print("Error on line {} {} {}".format(
                sys.exc_info()[-1].tb_lineno, type(e).__name__, e
            ))

    def _init_shift_detection_files(self):
        result_path = self.get_result_path("test")
        self._write_header(
            result_path + f"shift_detection_metrics_{self.strategy_name}.csv",
            ["Detector", "Dataset", "Fold ID", "Round", "Model", "Shift Type",
             "Shift Configuration", "Precision", "Recall", "F1", "Detection Delay",
             "False Alarms", "First Detection Round", "Shift Round"],
            mode="w",
        )
        self._write_header(
            result_path + f"shift_detection_curve_{self.strategy_name}.csv",
            ["Detector", "Dataset", "Fold ID", "Round", "Model", "Ground Truth",
             "Detection Event", "Detector State", "Drift Clients", "Drift Rate"],
            mode="w",
        )

    # ------------------------------------------------------------------
    # Evaluation: every client is evaluated with its current cluster model.
    # ------------------------------------------------------------------
    def evaluate(self, t, parameters_aggregated_mefl):
        try:
            evaluate_results = []
            for me in range(self.ME):
                for client in self.clients:
                    cid = int(client.client_id)
                    cluster = self.client_clusters[me].get(cid, 0)
                    model_parameters = self.cluster_models[me].get(
                        cluster,
                        parameters_aggregated_mefl[me],
                    )
                    evaluate_results.append(
                        client.evaluate(me, t, model_parameters)
                    )

            self.aggregate_evaluate(
                server_round=t,
                results=evaluate_results,
                failures=[],
            )
        except Exception as e:
            print("FedDCA evaluate error")
            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno, type(e).__name__, e
                )
            )
            raise

    def aggregate_evaluate(self, server_round, results, failures):
        try:
            results_mefl = {me: [] for me in range(self.ME)}
            for result in results:
                loss, num_examples, metrics = result
                me = int(metrics["me"])
                results_mefl[me].append(result)

            loss_aggregated_mefl = {me: 0.0 for me in range(self.ME)}
            metrics_aggregated_mefl = {me: {} for me in range(self.ME)}

            for me in range(self.ME):
                if not results_mefl[me]:
                    continue
                loss_aggregated_mefl[me] = weighted_loss_avg(
                    [
                        (num_examples, loss)
                        for loss, num_examples, _ in results_mefl[me]
                    ]
                )

                eval_metrics = [
                    (num_examples, metrics)
                    for _, num_examples, metrics in results_mefl[me]
                ]
                if self.evaluate_metrics_aggregation_fn:
                    metrics_aggregated_mefl[me] = self.evaluate_metrics_aggregation_fn(
                        eval_metrics
                    )

                metrics_aggregated_mefl[me].update(
                    {
                        "Fraction fit": self.fraction_fit,
                        "# training clients": len(self.selected_clients_m[me]),
                        "training clients and models": self.selected_clients_m[me],
                        "Fold ID": self.fold_id,
                        "Drift clients": self.drift_clients[me],
                        "Drift rate": self.drift_rate[me],
                        "Data shift": self.data_shift_type[me],
                        "Ground truth shift": int(
                            any(
                                server_round >= r
                                for r in self.shift_rounds[me]
                            )
                        ),
                    }
                )

                self.add_metrics(server_round, metrics_aggregated_mefl, me)
                self._save_results(server_round, me)

            return loss_aggregated_mefl, metrics_aggregated_mefl

        except Exception as e:
            print("FedDCA aggregate_evaluate error")
            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno, type(e).__name__, e
                )
            )
            raise