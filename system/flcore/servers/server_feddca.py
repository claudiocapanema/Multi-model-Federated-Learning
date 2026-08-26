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

        self.feddca_newton_iters = int(
            getattr(
                args,
                "feddca_newton_iters",
                10
            )
        )

        self.feddca_boundary_tolerance = float(
            getattr(
                args,
                "feddca_boundary_tolerance",
                0.05
            )
        )

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

    def _power_distance(self, lp, centroid, h):
        """
        FedDCA power distance.

        According to Eq. (4) of the paper:

            d_power(c, j)
                = W2^2(LP_c, nu_j) - h_j

        where:
            LP_c  = client Label Profile
            nu_j  = Wasserstein barycenter of cluster j
            h_j   = Power Diagram weight.
        """

        w2 = self._lp_distance(lp, centroid)

        if not np.isfinite(w2):
            return float("inf")

        return float(w2 ** 2 - h)

    def _initialize_power_weights(self, me, anchors):
        """
        Initialize the Power Diagram weight vector h.

        The Power Diagram weights are defined only up to an additive constant.
        We initialize them at zero, which fixes a valid reference gauge.

        If previous weights are available and the number of anchors is
        unchanged, they are reused to provide temporal continuity.
        """

        k = len(anchors)

        if k == 0:
            return {}

        previous = getattr(self, "power_weights", {}).get(me, {})

        h = {}

        for cluster in sorted(anchors):
            if cluster in previous:
                h[cluster] = float(previous[cluster])
            else:
                h[cluster] = 0.0

        # Fix the additive degree of freedom of the Power Diagram.
        # Subtracting a constant from all h_j does not change the partition.
        if h:
            reference = min(h.values())
            for cluster in h:
                h[cluster] -= reference

        return h

    def _get_vwc_target_weights(self, me, stable_ids, assignments, k):
        """
        Obtain the target cluster weights v_j used by the VWC Power Diagram.

        The paper describes two possibilities:
          1. uniform target weights;
          2. target weights equal to the cluster weights obtained in the
             previous round.

        We use the previous-round cluster weights whenever available.
        Otherwise, uniform weights are used.
        """

        client_weights = self.client_weights.get(me, {})

        total_weight = sum(
            client_weights.get(cid, 1.0)
            for cid in stable_ids
        )

        if total_weight <= 0:
            total_weight = float(max(len(stable_ids), 1))

        # Previous cluster target weights.
        previous_targets = getattr(
            self,
            "vwc_target_weights",
            {}
        ).get(me, {})

        target = {}

        # --------------------------------------------------------------
        # If previous target weights are available, preserve them.
        # --------------------------------------------------------------
        if previous_targets:
            previous_total = sum(
                max(float(v), 0.0)
                for v in previous_targets.values()
            )

            if previous_total > 0:
                for cluster_idx in range(k):
                    target[cluster_idx] = (
                            previous_targets.get(cluster_idx, 0.0)
                            / previous_total
                            * total_weight
                    )

                # Numerical normalization.
                target_sum = sum(target.values())

                if target_sum > 0:
                    scale = total_weight / target_sum
                    target = {
                        cluster: weight * scale
                        for cluster, weight in target.items()
                    }

                    return target

        # --------------------------------------------------------------
        # First round: uniform target masses.
        #
        # v_j = total_weight / K
        # --------------------------------------------------------------
        uniform = total_weight / float(k)

        return {
            cluster_idx: uniform
            for cluster_idx in range(k)
        }

    def _assign_power_clusters(
            self,
            me,
            stable_ids,
            anchors,
            h
    ):
        """
        Assign stable clients using the FedDCA Power Diagram.

        Eq. (4):

            k_c = argmin_j [
                W2^2(LP_c, nu_j) - h_j
            ]

        Returns:
            assignments:
                {client_id: cluster_id}

            power_distances:
                {client_id: {cluster_id: power_distance}}
        """

        profiles = self.current_lp[me]

        assignments = {}
        power_distances = {}

        for cid in stable_ids:

            lp = profiles[cid]

            distances = {}

            for cluster, centroid in anchors.items():
                distances[cluster] = self._power_distance(
                    lp=lp,
                    centroid=centroid,
                    h=float(h.get(cluster, 0.0))
                )

            if not distances:
                continue

            cluster = min(
                distances,
                key=distances.get
            )

            assignments[cid] = cluster
            power_distances[cid] = distances

        return assignments, power_distances

    def _approximate_vwc_hessian(
            self,
            me,
            stable_ids,
            power_distances,
            client_weights,
            boundary_tolerance
    ):
        """
        Approximate the Hessian of the VWC energy function.

        FedDCA uses a communication-efficient Hessian approximation:
        clients identify their two closest clusters in power-distance space,
        and clients close to the corresponding boundary contribute their
        weights to the off-diagonal Hessian entries.

        The diagonal entries are then defined as:

            H_jj = - sum_{k != j} H_jk

        so that the rows sum to zero.

        This implementation approximates the boundary condition using the
        relative gap between the two smallest power distances.
        """

        clusters = sorted({
            cluster
            for distances in power_distances.values()
            for cluster in distances
        })

        k = len(clusters)

        if k <= 1:
            return np.zeros((k, k), dtype=np.float64)

        cluster_to_idx = {
            cluster: idx
            for idx, cluster in enumerate(clusters)
        }

        H = np.zeros(
            (k, k),
            dtype=np.float64
        )

        for cid in stable_ids:

            distances = power_distances.get(cid)

            if distances is None or len(distances) < 2:
                continue

            ordered = sorted(
                distances.items(),
                key=lambda item: item[1]
            )

            cluster_a, distance_a = ordered[0]
            cluster_b, distance_b = ordered[1]

            # ----------------------------------------------------------
            # Boundary criterion.
            #
            # The paper says that clients near the boundary report their
            # weight and the two closest clusters. It does not provide a
            # numerical tolerance. Therefore we expose this tolerance as
            # a configurable approximation.
            # ----------------------------------------------------------
            scale = max(
                abs(float(distance_a)),
                abs(float(distance_b)),
                self.feddca_eps
            )

            relative_gap = (
                    abs(float(distance_b) - float(distance_a))
                    / scale
            )

            if relative_gap > boundary_tolerance:
                continue

            ia = cluster_to_idx[cluster_a]
            ib = cluster_to_idx[cluster_b]

            weight = max(
                float(client_weights.get(cid, 1.0)),
                0.0
            )

            if weight <= 0:
                continue

            # Off-diagonal Hessian entries.
            H[ia, ib] -= weight
            H[ib, ia] -= weight

        # --------------------------------------------------------------
        # Diagonal entries.
        #
        # H_jj = - sum_{k != j} H_jk
        # --------------------------------------------------------------
        for j in range(k):
            H[j, j] = -np.sum(
                H[j, np.arange(k) != j]
            )

        return H

    def _solve_vwc_power_weights(
            self,
            me,
            stable_ids,
            anchors,
            h,
            target_weights,
            max_iters,
            tol,
            boundary_tolerance
    ):
        """
        Solve the VWC Power Diagram weight vector h using Newton's method.

        The energy gradient is:

            grad_j E(h) = m_j(h) - v_j

        where:
            m_j(h) = total weight assigned to cluster j
            v_j    = target weight of cluster j.

        Newton's update:

            H * delta_h = -grad E(h)

            h <- h + delta_h

        The additive constant ambiguity of h is removed by fixing the last
        cluster weight to zero during the linear solve.
        """

        if not anchors:
            return h, {}, {}

        clusters = sorted(anchors)

        if len(clusters) == 1:
            h[clusters[0]] = 0.0

            assignments = {
                cid: clusters[0]
                for cid in stable_ids
            }

            return h, assignments, {}

        client_weights = self.client_weights.get(me, {})

        for _ in range(max_iters):

            # ----------------------------------------------------------
            # Current Power Diagram partition.
            # ----------------------------------------------------------
            assignments, power_distances = (
                self._assign_power_clusters(
                    me=me,
                    stable_ids=stable_ids,
                    anchors=anchors,
                    h=h
                )
            )

            # ----------------------------------------------------------
            # Compute m_j(h):
            #
            # m_j(h) = sum_{c in C_j(h)} w_c
            # ----------------------------------------------------------
            cluster_weights = {
                cluster: 0.0
                for cluster in clusters
            }

            for cid, cluster in assignments.items():
                cluster_weights[cluster] += max(
                    float(client_weights.get(cid, 1.0)),
                    0.0
                )

            # ----------------------------------------------------------
            # Gradient:
            #
            # grad_j = m_j(h) - v_j
            # ----------------------------------------------------------
            gradient = np.array(
                [
                    cluster_weights.get(cluster, 0.0)
                    - target_weights.get(cluster, 0.0)
                    for cluster in clusters
                ],
                dtype=np.float64
            )

            # Because the Power Diagram is invariant to adding a constant
            # to every h_j, the gradient has one redundant dimension.
            #
            # Stop when the partition satisfies the target masses.
            # ----------------------------------------------------------
            if np.max(np.abs(gradient)) < tol:
                break

            # ----------------------------------------------------------
            # Approximate Hessian.
            # ----------------------------------------------------------
            H = self._approximate_vwc_hessian(
                me=me,
                stable_ids=stable_ids,
                power_distances=power_distances,
                client_weights=client_weights,
                boundary_tolerance=boundary_tolerance
            )

            # ----------------------------------------------------------
            # Fix the gauge:
            #
            # h_last = 0
            #
            # Solve only for the first K-1 dimensions.
            # ----------------------------------------------------------
            reduced_H = H[:-1, :-1]
            reduced_gradient = gradient[:-1]

            if reduced_H.size == 0:
                break

            try:
                delta_reduced = np.linalg.solve(
                    reduced_H + self.feddca_eps * np.eye(
                        reduced_H.shape[0]
                    ),
                    -reduced_gradient
                )

            except np.linalg.LinAlgError:

                # ------------------------------------------------------
                # The Hessian can be singular when there are no or too few
                # clients close to cluster boundaries.
                #
                # A pseudo-inverse is a stable fallback.
                # ------------------------------------------------------
                delta_reduced = (
                        -np.linalg.pinv(
                            reduced_H
                        )
                        @ reduced_gradient
                )

            delta = np.zeros(
                len(clusters),
                dtype=np.float64
            )

            delta[:-1] = delta_reduced

            # Last h remains the gauge reference.
            delta[-1] = 0.0

            # ----------------------------------------------------------
            # Damped Newton step.
            #
            # This prevents a single poorly conditioned Hessian from
            # producing an excessively large power-weight update.
            # ----------------------------------------------------------
            max_step = 10.0

            step_norm = np.max(
                np.abs(delta)
            )

            if step_norm > max_step:
                delta *= (
                        max_step / step_norm
                )

            for idx, cluster in enumerate(clusters):
                h[cluster] += float(delta[idx])

            # ----------------------------------------------------------
            # Re-center h to avoid numerical drift.
            # ----------------------------------------------------------
            reference = h[clusters[-1]]

            for cluster in clusters:
                h[cluster] -= reference

        # Final assignment after Newton convergence.
        assignments, power_distances = (
            self._assign_power_clusters(
                me=me,
                stable_ids=stable_ids,
                anchors=anchors,
                h=h
            )
        )

        return h, assignments, power_distances

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def _initialize_feddca_state(self):
        """
        Initialize all persistent FedDCA state.

        The state is maintained independently for each MEFL model.
        """

        # ==============================================================
        # Temporal Label Profile state
        # ==============================================================

        self.previous_lp = {
            me: {}
            for me in range(self.ME)
        }

        self.current_lp = {
            me: {}
            for me in range(self.ME)
        }

        # ==============================================================
        # Drift detection state
        # ==============================================================

        self.drift_scores = {
            me: {}
            for me in range(self.ME)
        }

        self.ewma_scores = {
            me: {}
            for me in range(self.ME)
        }

        self.drift_flags = {
            me: {}
            for me in range(self.ME)
        }

        # --------------------------------------------------------------
        # IMPORTANT:
        # _update_drift_state() stores the temporal W2 distance for each
        # client here. This history is required to compute the EWMA and
        # detect persistent changes rather than reacting to a single
        # anomalous round.
        # --------------------------------------------------------------

        self.distance_history = {
            me: {}
            for me in range(self.ME)
        }

        # ==============================================================
        # Client stability / drift
        # ==============================================================

        self.stable_clients = {
            me: []
            for me in range(self.ME)
        }

        self.drift_clients_ids = {
            me: []
            for me in range(self.ME)
        }

        self.drift_clients = {
            me: 0
            for me in range(self.ME)
        }

        self.drift_rate = {
            me: 0.0
            for me in range(self.ME)
        }

        self.drift_rate_history = {
            me: []
            for me in range(self.ME)
        }

        # ==============================================================
        # Persistent cluster state
        # ==============================================================

        self.client_clusters = {
            me: {
                cid: 0
                for cid in range(self.total_clients)
            }
            for me in range(self.ME)
        }

        self.cluster_members = {
            me: {}
            for me in range(self.ME)
        }

        self.cluster_models = {
            me: {}
            for me in range(self.ME)
        }

        # ==============================================================
        # VWC / anchor state
        # ==============================================================

        self.anchor_centroids = {
            me: {}
            for me in range(self.ME)
        }

        self.anchor_models = {
            me: {}
            for me in range(self.ME)
        }

        self.power_weights = {
            me: {}
            for me in range(self.ME)
        }

        # ==============================================================
        # Ground-truth / detection bookkeeping
        # ==============================================================

        self.data_shift_type = {
            me: "NO_SHIFT"
            for me in range(self.ME)
        }

        self.shift_ground_truth_state = {
            me: []
            for me in range(self.ME)
        }

        self.shift_ground_truth_event = {
            me: []
            for me in range(self.ME)
        }

        self.shift_detected = {
            me: []
            for me in range(self.ME)
        }

        self.previous_detector_state = {
            me: "NO_SHIFT"
            for me in range(self.ME)
        }

        self.detection_event = {
            me: 0
            for me in range(self.ME)
        }

        # ==============================================================
        # Detection delay / false alarms
        # ==============================================================

        self.detection_delay = {
            me: None
            for me in range(self.ME)
        }

        self.true_detection_round = {
            me: None
            for me in range(self.ME)
        }

        self.false_alarm_rounds = {
            me: []
            for me in range(self.ME)
        }

        # ==============================================================
        # Detection metric histories
        # ==============================================================

        self.detection_precision = {
            me: []
            for me in range(self.ME)
        }

        self.detection_recall = {
            me: []
            for me in range(self.ME)
        }

        self.detection_f1 = {
            me: []
            for me in range(self.ME)
        }

        # ==============================================================
        # Aggregation state
        # ==============================================================

        self.parameters_aggregated_mefl = {}

        self.client_weights = {
            me: {}
            for me in range(self.ME)
        }

        # ==============================================================
        # Classifier parameter indices
        # ==============================================================

        self.classifier_param_indices = {
            me: []
            for me in range(self.ME)
        }

    def set_clients(self):
        """
        Create all FedDCA clients and obtain the ground-truth shift
        configuration from the clients.

        The first client's data_shift_config is used as the reference,
        following the same mechanism used by FedConD.
        """

        self.shift_rounds = {
            me: []
            for me in range(self.ME)
        }

        self.clients = []

        # ==============================================================
        # Create clients
        # ==============================================================

        for cid in range(self.total_clients):
            client = ClientFedDCA(
                self.args,
                id=cid,
                model=copy.deepcopy(self.global_model),
                fold_id=self.fold_id
            )

            self.clients.append(client)

        # ==============================================================
        # Obtain ground-truth shift rounds
        #
        # This must happen AFTER the clients have been created.
        #
        # FedConD uses:
        #
        # self.clients[0].data_shift_config[me]["data_shift_rounds"]
        #
        # We use exactly the same source here.
        # ==============================================================

        if len(self.clients) > 0:

            for me in range(self.ME):

                if me in self.clients[0].data_shift_config:

                    self.shift_rounds[me] = list(
                        self.clients[0]
                        .data_shift_config[me]
                        ["data_shift_rounds"]
                    )

                else:

                    self.shift_rounds[me] = []

        # ==============================================================
        # Diagnostic information
        # ==============================================================

        for me in range(self.ME):
            print(
                f"FedDCA ground-truth shift rounds "
                f"model={me}: {self.shift_rounds[me]}"
            )

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
        """
        Detect temporal concept drift from changes in the client's Label Profile.

        For each client:

            d_t = W2(LP_t, LP_{t-1})

        The detector uses the client's own historical temporal distances to
        estimate the normal variation of its Label Profile.

        Before enough historical observations are available, the client is
        conservatively considered stable.

        Once enough history exists, the adaptive threshold is:

            threshold_t =
                EWMA_{t-1} + sigma * std(d_{history})

        and drift is detected when:

            d_t > threshold_t

        IMPORTANT:
            The current distance d_t is NEVER included in the history before
            making the current drift decision.

        This prevents the current observation from influencing its own threshold.
        """

        stable = []
        drift = []

        alpha = float(self.feddca_ewma_alpha)
        sigma = float(self.feddca_threshold_sigma)
        min_history = max(1, int(self.feddca_min_history))
        eps = float(self.feddca_eps)

        # Defensive bounds.
        alpha = min(max(alpha, 0.0), 1.0)
        sigma = max(0.0, sigma)

        for cid in client_ids:

            cid = int(cid)

            current = self.current_lp[me].get(cid)
            previous = self.previous_lp[me].get(cid)

            # --------------------------------------------------------------
            # No current Label Profile.
            # --------------------------------------------------------------
            if current is None:
                continue

            # Ensure the client's temporal-distance history exists.
            history = self.distance_history[me].setdefault(cid, [])

            # --------------------------------------------------------------
            # First temporal observation.
            #
            # There is no previous Label Profile, therefore no temporal
            # distance can be computed and no drift can be detected.
            # --------------------------------------------------------------
            if previous is None:

                distance = 0.0
                ewma = 0.0
                threshold = float("inf")
                is_drift = False

            else:

                # ----------------------------------------------------------
                # Temporal Label Profile distance.
                #
                # d_t = W2(LP_t, LP_{t-1})
                # ----------------------------------------------------------
                distance = float(
                    self._lp_distance(
                        current,
                        previous
                    )
                )

                if not np.isfinite(distance):
                    distance = 0.0

                # ----------------------------------------------------------
                # No sufficient history yet.
                #
                # We intentionally do NOT detect drift during this warm-up
                # period. This avoids interpreting normal early training
                # fluctuations as concept drift.
                # ----------------------------------------------------------
                if len(history) < min_history:

                    is_drift = False

                    if history:
                        old_ewma = float(
                            self.ewma_scores[me].get(
                                cid,
                                np.mean(history)
                            )
                        )
                    else:
                        old_ewma = distance

                    if not np.isfinite(old_ewma):
                        old_ewma = distance

                    ewma = (
                            alpha * distance
                            + (1.0 - alpha) * old_ewma
                    )

                    threshold = float("inf")

                else:

                    # ------------------------------------------------------
                    # Historical temporal variation.
                    #
                    # IMPORTANT:
                    # history contains ONLY distances from previous rounds.
                    # The current distance is not included.
                    # ------------------------------------------------------
                    history_array = np.asarray(
                        history,
                        dtype=np.float64
                    )

                    history_array = history_array[
                        np.isfinite(history_array)
                    ]

                    if history_array.size == 0:
                        old_ewma = distance
                        historical_std = 0.0
                    else:
                        old_ewma = float(
                            self.ewma_scores[me].get(
                                cid,
                                np.mean(history_array)
                            )
                        )

                        if not np.isfinite(old_ewma):
                            old_ewma = float(
                                np.mean(history_array)
                            )

                        historical_std = float(
                            np.std(
                                history_array,
                                ddof=0
                            )
                        )

                    # ------------------------------------------------------
                    # Adaptive threshold.
                    #
                    # threshold_t =
                    #     EWMA_{t-1} + sigma * std(history)
                    #
                    # This makes the detector substantially less sensitive
                    # to ordinary temporal fluctuations.
                    # ------------------------------------------------------
                    threshold = (
                            old_ewma
                            + sigma * historical_std
                    )

                    # Numerical safety.
                    threshold = max(
                        threshold,
                        eps
                    )

                    # ------------------------------------------------------
                    # Drift decision.
                    # ------------------------------------------------------
                    is_drift = bool(
                        distance > threshold
                    )

                    # ------------------------------------------------------
                    # Update EWMA AFTER the decision.
                    # ------------------------------------------------------
                    ewma = (
                            alpha * distance
                            + (1.0 - alpha) * old_ewma
                    )

                # ----------------------------------------------------------
                # Append the CURRENT distance only after making the decision.
                #
                # This guarantees that d_t cannot inflate its own threshold.
                # ----------------------------------------------------------
                history.append(float(distance))

            # --------------------------------------------------------------
            # Store temporal statistics.
            # --------------------------------------------------------------
            self.drift_scores[me][cid] = float(distance)

            self.ewma_scores[me][cid] = float(
                ewma
            )

            self.drift_flags[me][cid] = bool(
                is_drift
            )

            # Store threshold if the server has this auxiliary dictionary.
            if not hasattr(self, "drift_thresholds"):
                self.drift_thresholds = {
                    model_idx: {}
                    for model_idx in range(self.ME)
                }

            self.drift_thresholds[me][cid] = float(
                threshold
            )

            # --------------------------------------------------------------
            # Stable / drifting partition.
            # --------------------------------------------------------------
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

    def _barycenter(
            self,
            profiles,
            members,
            fallback,
            me=None
    ):
        """
        Compute a weighted Wasserstein barycenter for a cluster.

        Implements the centroid objective from Eq. (7):

            nu_new =
                argmin_nu
                sum_c w_c W2^2(LP_c, nu)

        The barycenter is computed independently for each label because
        each Label Profile is a collection of label-conditional empirical
        measures.
        """

        labels = set(fallback.keys())

        for cid in members:
            labels.update(
                profiles[cid].keys()
            )

        centroid = {}

        client_weights = (
            self.client_weights.get(me, {})
            if me is not None
            else {}
        )

        for label in sorted(labels):

            data = []

            for cid in members:

                if label not in profiles[cid]:
                    continue

                weight = max(
                    float(
                        client_weights.get(cid, 1.0)
                    ),
                    0.0
                )

                if weight <= 0:
                    continue

                data.append(
                    (
                        weight,
                        profiles[cid][label]
                    )
                )

            if not data:
                continue

            if label in fallback:
                init = fallback[label]
            else:
                init = data[0][1]

            centroid[label] = self._barycenter_label(
                data,
                init
            )

        return centroid

    def _barycenter_label(
            self,
            members,
            current_support
    ):
        """
        Approximate the Wasserstein barycenter of equal-cardinality empirical
        supports using weighted optimal matching.

        members:
            list of (client_weight, empirical_support)

        current_support:
            current barycenter support used as initialization.
        """

        if not members:
            return current_support

        support = np.asarray(
            current_support,
            dtype=np.float32
        )

        if support.ndim == 1:
            support = support[None, :]

        accum = np.zeros_like(
            support,
            dtype=np.float64
        )

        weight_sum = 0.0

        for weight, points in members:

            points = np.asarray(
                points,
                dtype=np.float32
            )

            if points.ndim == 1:
                points = points[None, :]

            if len(points) == 0:
                continue

            cost = np.sum(
                (
                        support[:, None, :]
                        - points[None, :, :]
                ) ** 2,
                axis=2
            )

            rows, cols = linear_sum_assignment(
                cost
            )

            matched = np.zeros_like(
                support,
                dtype=np.float64
            )

            counts = np.zeros(
                len(support),
                dtype=np.float64
            )

            for r, c in zip(rows, cols):
                matched[r] += points[c]
                counts[r] += 1.0

            mean_point = np.mean(
                points,
                axis=0
            )

            for r in range(len(support)):

                if counts[r] == 0:
                    matched[r] = mean_point
                    counts[r] = 1.0

                else:
                    matched[r] /= counts[r]

            weight = max(
                float(weight),
                0.0
            )

            accum += (
                    weight * matched
            )

            weight_sum += weight

        if weight_sum <= 0:
            return support

        return (
                accum / weight_sum
        ).astype(np.float32)

    def _vwc_stable(self, me, stable_ids):
        """
        FedDCA Variational Wasserstein Clustering (VWC).

        This implementation follows the formulation described in Section 3.4
        of the paper:

            1. initialize K Wasserstein anchor centroids;
            2. construct a Power Diagram using weight vector h;
            3. assign stable clients using:

                   argmin_j [
                       W2^2(LP_c, nu_j) - h_j
                   ];

            4. compute the cluster masses m_j(h);
            5. update h using Newton's method;
            6. recompute the Wasserstein barycenter of each cluster;
            7. alternate partition and centroid updates until convergence.

        Only stable clients participate in VWC. Drifting clients are NOT
        allowed to modify the anchor structure.
        """

        if not stable_ids:
            return {}, {}

        profiles = self.current_lp[me]

        # --------------------------------------------------------------
        # Number of clusters.
        # --------------------------------------------------------------
        k = max(
            1,
            min(
                int(self.feddca_num_clusters),
                len(stable_ids)
            )
        )

        # --------------------------------------------------------------
        # Initialize anchors.
        #
        # Prefer previous anchors when possible because FedDCA aims to
        # preserve a stable collaborative structure over time.
        # --------------------------------------------------------------
        previous_anchors = (
            self.anchor_centroids
            .get(me, {})
        )

        if previous_anchors:

            valid_previous = {
                cluster: copy.deepcopy(
                    centroid
                )
                for cluster, centroid
                in previous_anchors.items()
                if centroid
            }

            if len(valid_previous) >= k:

                anchors = {
                    cluster: valid_previous[cluster]
                    for cluster in sorted(
                        valid_previous
                    )[:k]
                }

            else:

                anchors = self._init_anchors(
                    profiles,
                    stable_ids,
                    k
                )

        else:

            anchors = self._init_anchors(
                profiles,
                stable_ids,
                k
            )

        if not anchors:
            return {}, {}

        # --------------------------------------------------------------
        # Initialize Power Diagram weights.
        # --------------------------------------------------------------
        h = self._initialize_power_weights(
            me,
            anchors
        )

        # --------------------------------------------------------------
        # Boundary tolerance.
        #
        # The paper does not provide an exact numerical value for this
        # approximation. It is therefore exposed as a hyperparameter.
        # --------------------------------------------------------------
        boundary_tolerance = float(
            getattr(
                self,
                "feddca_boundary_tolerance",
                0.05
            )
        )

        # --------------------------------------------------------------
        # Newton iterations per partition update.
        # --------------------------------------------------------------
        newton_iters = int(
            getattr(
                self,
                "feddca_newton_iters",
                10
            )
        )

        # --------------------------------------------------------------
        # Target cluster masses.
        #
        # Initial assignments are used only to determine previous cluster
        # masses when they are available.
        # --------------------------------------------------------------
        initial_assignments, _ = (
            self._assign_power_clusters(
                me=me,
                stable_ids=stable_ids,
                anchors=anchors,
                h=h
            )
        )

        target_weights = (
            self._get_vwc_target_weights(
                me=me,
                stable_ids=stable_ids,
                assignments=initial_assignments,
                k=len(anchors)
            )
        )

        assignments = {}
        power_distances = {}

        # --------------------------------------------------------------
        # Alternating VWC optimization.
        # --------------------------------------------------------------
        for iteration in range(
                int(self.feddca_vwc_iters)
        ):

            old_assignments = dict(
                assignments
            )

            old_anchors = copy.deepcopy(
                anchors
            )

            # ----------------------------------------------------------
            # 1. Power Diagram partition + Newton optimization of h.
            # ----------------------------------------------------------
            h, assignments, power_distances = (
                self._solve_vwc_power_weights(
                    me=me,
                    stable_ids=stable_ids,
                    anchors=anchors,
                    h=h,
                    target_weights=target_weights,
                    max_iters=newton_iters,
                    tol=self.feddca_tol,
                    boundary_tolerance=boundary_tolerance
                )
            )

            # ----------------------------------------------------------
            # 2. Wasserstein barycenter update.
            # ----------------------------------------------------------
            new_anchors = {}

            for cluster in sorted(anchors):

                members = [
                    cid
                    for cid in stable_ids
                    if assignments.get(cid) == cluster
                ]

                if not members:
                    # Preserve the existing anchor if no client is
                    # assigned to it.
                    new_anchors[cluster] = (
                        anchors[cluster]
                    )

                    continue

                new_anchors[cluster] = (
                    self._barycenter(
                        profiles=profiles,
                        members=members,
                        fallback=anchors[cluster],
                        me=me
                    )
                )

            # ----------------------------------------------------------
            # 3. Convergence check.
            # ----------------------------------------------------------
            assignment_changed = (
                    old_assignments != assignments
            )

            max_anchor_shift = 0.0

            for cluster in new_anchors:

                if cluster not in old_anchors:
                    continue

                shift = self._lp_distance(
                    new_anchors[cluster],
                    old_anchors[cluster]
                )

                if np.isfinite(shift):
                    max_anchor_shift = max(
                        max_anchor_shift,
                        float(shift)
                    )

            anchors = new_anchors

            if (
                    not assignment_changed
                    and max_anchor_shift
                    < self.feddca_tol
            ):
                break

        # --------------------------------------------------------------
        # Save Power Diagram state for the next communication round.
        # --------------------------------------------------------------
        if not hasattr(
                self,
                "power_weights"
        ):
            self.power_weights = {}

        self.power_weights[me] = {
            cluster: float(weight)
            for cluster, weight
            in h.items()
        }

        # --------------------------------------------------------------
        # Store target cluster masses for the next round.
        # --------------------------------------------------------------
        if not hasattr(
                self,
                "vwc_target_weights"
        ):
            self.vwc_target_weights = {}

        client_weights = self.client_weights.get(
            me,
            {}
        )

        current_cluster_weights = {}

        for cid, cluster in assignments.items():
            current_cluster_weights[cluster] = (
                    current_cluster_weights.get(
                        cluster,
                        0.0
                    )
                    + max(
                float(
                    client_weights.get(
                        cid,
                        1.0
                    )
                ),
                0.0
            )
            )

        self.vwc_target_weights[me] = (
            current_cluster_weights
        )

        return assignments, anchors

    def _assign_drift_clients(self, me, assignments, anchors):
        """
        Assign drifting clients to the nearest VWC anchor.

        Stable clients are assigned by VWC.

        Drifting clients do not modify the VWC structure. They are assigned
        to the closest persistent anchor.

        If the current round does not produce new anchors, previously learned
        anchors are reused.
        """

        # --------------------------------------------------------------
        # If no anchors were generated in the current round, reuse the
        # persistent anchors from previous rounds.
        # --------------------------------------------------------------
        if not anchors:
            anchors = copy.deepcopy(
                self.anchor_centroids.get(
                    me,
                    {}
                )
            )

        # --------------------------------------------------------------
        # If no anchors exist at all, there is nothing to which drifting
        # clients can be assigned.
        # --------------------------------------------------------------
        if not anchors:
            return assignments

        profiles = self.current_lp.get(
            me,
            {}
        )

        drift_ids = self.drift_clients_ids.get(
            me,
            []
        )

        # --------------------------------------------------------------
        # Assign each drifting client to the nearest anchor.
        # --------------------------------------------------------------
        for cid in drift_ids:

            cid = int(cid)

            if cid not in profiles:
                continue

            distances = {}

            for cluster_id, anchor in anchors.items():

                try:

                    distance = float(
                        self._lp_distance(
                            profiles[cid],
                            anchor
                        )
                    )

                    if np.isfinite(distance):
                        distances[cluster_id] = distance

                except Exception:
                    continue

            # No valid distance was obtained.
            if not distances:
                continue

            # Nearest Wasserstein anchor.
            best_cluster = min(
                distances,
                key=distances.get
            )

            assignments[cid] = best_cluster

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
        """
        Aggregate the FedDCA hybrid model.

        The model is composed of:

            - globally aggregated feature-extractor parameters;
            - cluster-specific classifier parameters.

        The global feature extractor is updated every round.

        Cluster-specific classifiers are updated only for clusters that have
        participating clients in the current round.

        IMPORTANT:
            A cluster that receives no clients in the current round MUST NOT
            disappear.

        For inactive clusters:
            - keep their previous classifier;
            - synchronize their shared feature extractor with the new global
              parameters.

        This preserves cluster models across rounds under partial client
        participation.
        """

        if not results_by_me:
            return None, copy.deepcopy(
                self.cluster_models.get(me, {})
            )

        classifier_idx = set(
            self.classifier_param_indices[me]
        )

        # ------------------------------------------------------------------
        # 1. Global aggregation.
        #
        # All model parameters participate in the global aggregation.
        # This produces the new shared feature extractor.
        # ------------------------------------------------------------------
        all_items = [
            (params, n)
            for params, n, _ in results_by_me
        ]

        global_params = self._weighted_average_arrays(
            all_items
        )

        # ------------------------------------------------------------------
        # 2. Start from the PREVIOUS cluster models.
        #
        # This is the key correction.
        #
        # Existing clusters are preserved instead of being replaced by only
        # the clusters represented in the current round.
        # ------------------------------------------------------------------
        previous_cluster_models = getattr(
            self,
            "cluster_models",
            {}
        ).get(me, {})

        cluster_params = {}

        for cluster_id, old_model in previous_cluster_models.items():

            cluster_id = int(cluster_id)

            # Start with the NEW global parameters.
            #
            # Therefore the shared feature extractor is synchronized with
            # the current global model.
            new_cluster_model = [
                np.array(
                    p,
                    copy=True
                )
                for p in global_params
            ]

            # --------------------------------------------------------------
            # Preserve the old cluster-specific classifier.
            # --------------------------------------------------------------
            if old_model is not None:

                for idx in classifier_idx:

                    if (
                            idx < len(old_model)
                            and idx < len(new_cluster_model)
                    ):
                        new_cluster_model[idx] = np.array(
                            old_model[idx],
                            copy=True
                        )

            cluster_params[cluster_id] = new_cluster_model

        # ------------------------------------------------------------------
        # 3. Update classifiers of clusters represented in this round.
        # ------------------------------------------------------------------
        active_clusters = sorted(
            set(
                int(cluster)
                for cluster in assignments.values()
            )
        )

        for cluster in active_clusters:

            cluster_items = []

            for params, n, meta in results_by_me:

                cid = int(
                    meta["client_id"]
                )

                if int(
                        assignments.get(cid, -1)
                ) == cluster:
                    cluster_items.append(
                        (params, n)
                    )

            if not cluster_items:
                continue

            # --------------------------------------------------------------
            # Start from the NEW global model.
            # --------------------------------------------------------------
            cluster_model = [
                np.array(
                    p,
                    copy=True
                )
                for p in global_params
            ]

            # --------------------------------------------------------------
            # Aggregate only the classifier parameters for this cluster.
            # --------------------------------------------------------------
            classifier_avg = self._weighted_average_arrays(
                cluster_items
            )

            for idx in classifier_idx:

                if (
                        idx < len(cluster_model)
                        and idx < len(classifier_avg)
                ):
                    cluster_model[idx] = np.array(
                        classifier_avg[idx],
                        copy=True
                    )

            cluster_params[cluster] = cluster_model

        # ------------------------------------------------------------------
        # 4. Guarantee cluster 0 exists.
        #
        # This is only a safety fallback for initialization/degenerated
        # situations.
        # ------------------------------------------------------------------
        if not cluster_params:
            cluster_params[0] = [
                np.array(
                    p,
                    copy=True
                )
                for p in global_params
            ]

        # ------------------------------------------------------------------
        # 5. Guarantee the global model itself remains represented.
        # ------------------------------------------------------------------
        if 0 not in cluster_params:
            cluster_params[0] = [
                np.array(
                    p,
                    copy=True
                )
                for p in global_params
            ]

        return global_params, cluster_params

    # ------------------------------------------------------------------
    # Training loop: same MultiFedAvg semantics, but cluster-aware models.
    # ------------------------------------------------------------------
    def _initial_parameters(self):
        """
        Initialize the aggregated global parameters and the persistent
        FedDCA cluster state for all MEFL models.

        The initialization includes:

            - global aggregated parameters;
            - classifier parameter indices;
            - persistent cluster models;
            - persistent client -> cluster assignments;
            - persistent cluster membership;
            - persistent VWC anchors;
            - persistent VWC auxiliary state.

        The classifier parameter indices are inferred once for each model
        and then reused by _aggregate_hybrid().
        """

        # ------------------------------------------------------------------
        # Global aggregated parameters.
        # ------------------------------------------------------------------
        self.parameters_aggregated_mefl = {}

        # ------------------------------------------------------------------
        # Persistent cluster models.
        #
        # cluster_models[me][cluster_id] =
        #     complete model parameters for that cluster.
        # ------------------------------------------------------------------
        if not hasattr(self, "cluster_models"):
            self.cluster_models = {}

        # ------------------------------------------------------------------
        # Persistent client -> cluster assignment.
        #
        # client_clusters[me][client_id] = cluster_id
        # ------------------------------------------------------------------
        if not hasattr(self, "client_clusters"):
            self.client_clusters = {}

        # ------------------------------------------------------------------
        # Persistent cluster membership.
        #
        # cluster_members[me][cluster_id] = list of client IDs.
        # ------------------------------------------------------------------
        if not hasattr(self, "cluster_members"):
            self.cluster_members = {}

        # ------------------------------------------------------------------
        # Persistent VWC anchor centroids.
        #
        # anchor_centroids[me][cluster_id] = Label Profile anchor.
        # ------------------------------------------------------------------
        if not hasattr(self, "anchor_centroids"):
            self.anchor_centroids = {}

        # ------------------------------------------------------------------
        # Persistent VWC power weights.
        # ------------------------------------------------------------------
        if not hasattr(self, "power_weights"):
            self.power_weights = {}

        # ------------------------------------------------------------------
        # Persistent VWC target weights.
        # ------------------------------------------------------------------
        if not hasattr(self, "vwc_target_weights"):
            self.vwc_target_weights = {}

        # ------------------------------------------------------------------
        # Classifier parameter indices.
        #
        # This dictionary is REQUIRED by _aggregate_hybrid().
        #
        # classifier_param_indices[me] contains the parameter tensor
        # indices corresponding to the final classifier layer.
        # ------------------------------------------------------------------
        if not hasattr(self, "classifier_param_indices"):
            self.classifier_param_indices = {}

        # ------------------------------------------------------------------
        # Initialize every MEFL model independently.
        # ------------------------------------------------------------------
        for me in range(self.ME):

            # ==============================================================
            # 1. Global model parameters
            # ==============================================================

            self.parameters_aggregated_mefl[me] = get_weights(
                self.global_model[me]
            )

            # ==============================================================
            # 2. Classifier parameter indices
            # ==============================================================

            self.classifier_param_indices[me] = (
                self._find_classifier_parameter_indices(
                    self.global_model[me]
                )
            )

            # --------------------------------------------------------------
            # Defensive validation.
            # --------------------------------------------------------------

            n_parameters = len(
                list(
                    self.global_model[me].parameters()
                )
            )

            self.classifier_param_indices[me] = [
                int(idx)
                for idx in self.classifier_param_indices[me]
                if 0 <= int(idx) < n_parameters
            ]

            # --------------------------------------------------------------
            # There should normally be at least one classifier parameter.
            #
            # _find_classifier_parameter_indices() already has a fallback
            # to the last two tensors, so this is only a final safeguard.
            # --------------------------------------------------------------
            if not self.classifier_param_indices[me]:
                if n_parameters > 0:
                    self.classifier_param_indices[me] = [
                        n_parameters - 1
                    ]

            # ==============================================================
            # 3. Cluster model storage
            # ==============================================================

            if me not in self.cluster_models:
                self.cluster_models[me] = {}

            # ==============================================================
            # 4. Client -> cluster mapping
            # ==============================================================

            if me not in self.client_clusters:
                self.client_clusters[me] = {}

            # ==============================================================
            # 5. Cluster membership
            # ==============================================================

            if me not in self.cluster_members:
                self.cluster_members[me] = {}

            # ==============================================================
            # 6. VWC anchors
            # ==============================================================

            if me not in self.anchor_centroids:
                self.anchor_centroids[me] = {}

            # ==============================================================
            # 7. VWC power weights
            # ==============================================================

            if me not in self.power_weights:
                self.power_weights[me] = {}

            # ==============================================================
            # 8. VWC target weights
            # ==============================================================

            if me not in self.vwc_target_weights:
                self.vwc_target_weights[me] = {}

            # ==============================================================
            # 9. Initial cluster 0
            #
            # Cluster 0 starts from the initial global model.
            #
            # Additional clusters will be created later by VWC.
            # ==============================================================

            if 0 not in self.cluster_models[me]:
                self.cluster_models[me][0] = [
                    np.array(
                        p,
                        copy=True
                    )
                    for p in self.parameters_aggregated_mefl[me]
                ]

            # ==============================================================
            # 10. Diagnostic information
            # ==============================================================

            print(
                f"FedDCA initialization - model {me}: "
                f"classifier parameter indices = "
                f"{self.classifier_param_indices[me]}"
            )

    def train(self):
        """
        Main FedDCA training loop.

        Each client receives the model associated with its current cluster.

        Cluster identities are preserved across rounds even when a cluster has
        no participating clients in a particular round.
        """

        try:

            self._get_models_size()

            self._initial_parameters()

            for t in range(
                    1,
                    self.number_of_rounds + 1
            ):

                start = time.time()

                # ----------------------------------------------------------
                # Client selection.
                # ----------------------------------------------------------
                self.selected_clients = self.select_clients(
                    t + self.fold_id
                )

                print(
                    "selected clients:",
                    self.selected_clients
                )

                fit_results = []

                # ----------------------------------------------------------
                # Local training.
                # ----------------------------------------------------------
                for me in range(self.ME):

                    for cid in self.selected_clients[me]:

                        cid = int(cid)

                        # --------------------------------------------------
                        # Keep the client's current cluster assignment.
                        #
                        # DO NOT silently change the client to cluster 0
                        # merely because the cluster was not active in the
                        # previous round.
                        # --------------------------------------------------
                        cluster = self.client_clusters[me].get(
                            cid,
                            0
                        )

                        model_parameters = (
                            self.cluster_models[me].get(
                                cluster
                            )
                        )

                        # --------------------------------------------------
                        # Defensive fallback.
                        #
                        # This should only occur if the cluster state was
                        # externally corrupted or this is an unexpected
                        # initialization case.
                        # --------------------------------------------------
                        if model_parameters is None:
                            model_parameters = (
                                self.parameters_aggregated_mefl[me]
                            )

                            print(
                                f"Warning: missing cluster model "
                                f"me={me}, cid={cid}, "
                                f"cluster={cluster}. "
                                f"Using global model as fallback."
                            )

                        fit_results.append(
                            self.clients[cid].fit(
                                me,
                                t,
                                model_parameters
                            )
                        )

                # ----------------------------------------------------------
                # Server aggregation.
                #
                # aggregate_fit() performs:
                #
                #   1. Label Profile update
                #   2. temporal drift detection
                #   3. VWC clustering
                #   4. hybrid model aggregation
                #   5. cluster-model persistence
                # ----------------------------------------------------------
                (
                    self.parameters_aggregated_mefl,
                    _
                ) = self.aggregate_fit(
                    server_round=t,
                    results=fit_results,
                    failures=[]
                )

                # ----------------------------------------------------------
                # Evaluation.
                # ----------------------------------------------------------
                self.evaluate(
                    t,
                    self.parameters_aggregated_mefl
                )

                print(
                    f"FedDCA round {t} completed in "
                    f"{time.time() - start:.2f}s"
                )

        except Exception as e:

            print(
                "FedDCA train error"
            )

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
                )
            )

            raise

    # ------------------------------------------------------------------
    # Aggregation entry point
    # ------------------------------------------------------------------
    def aggregate_fit(self, server_round: int, results, failures):
        """
        FedDCA aggregation for MEFL.

        Processing order:

            1. collect client updates and Label Profiles;
            2. detect temporal drift;
            3. perform VWC using stable clients only;
            4. preserve/reuse persistent VWC anchors;
            5. assign drifting clients to the nearest anchor;
            6. update persistent client-cluster assignments;
            7. update persistent cluster membership;
            8. perform hybrid global/cluster aggregation;
            9. preserve inactive cluster models;
           10. update detection and training metrics.

        The cluster structure is persistent across rounds.
        """

        try:

            # ==========================================================
            # 1. Round-specific containers
            # ==========================================================

            self.selected_clients_m = [
                []
                for _ in range(self.ME)
            ]

            results_mefl = {
                me: []
                for me in range(self.ME)
            }

            # ----------------------------------------------------------
            # Defensive initialization.
            #
            # This also protects against old objects/state where these
            # dictionaries were not created during initialization.
            # ----------------------------------------------------------

            if not hasattr(self, "cluster_members"):
                self.cluster_members = {
                    me: {}
                    for me in range(self.ME)
                }

            if not hasattr(self, "client_clusters"):
                self.client_clusters = {
                    me: {}
                    for me in range(self.ME)
                }

            if not hasattr(self, "anchor_centroids"):
                self.anchor_centroids = {
                    me: {}
                    for me in range(self.ME)
                }

            if not hasattr(self, "cluster_models"):
                self.cluster_models = {
                    me: {}
                    for me in range(self.ME)
                }

            if not hasattr(self, "client_weights"):
                self.client_weights = {
                    me: {}
                    for me in range(self.ME)
                }

            # ==========================================================
            # 2. Collect client results
            # ==========================================================

            for params, num_examples, metrics in results:
                me = int(
                    metrics["me"]
                )

                cid = int(
                    metrics["client_id"]
                )

                self.selected_clients_m[me].append(
                    cid
                )

                results_mefl[me].append(
                    (
                        params,
                        num_examples,
                        metrics
                    )
                )

                # ------------------------------------------------------
                # Current Label Profile.
                # ------------------------------------------------------

                self.current_lp[me][cid] = (
                    metrics["Label Profile"]
                )

                # ------------------------------------------------------
                # FedDCA client weight.
                # ------------------------------------------------------

                self.client_weights[me][cid] = max(
                    1.0,
                    float(num_examples)
                )

            # ==========================================================
            # 3. Initialize aggregation containers
            # ==========================================================

            metrics_aggregated = {
                me: {}
                for me in range(self.ME)
            }

            aggregated = {
                me: self.parameters_aggregated_mefl[me]
                for me in range(self.ME)
            }

            # ==========================================================
            # 4. Process each MEFL model independently
            # ==========================================================

            for me in range(self.ME):

                if not results_mefl[me]:
                    continue

                client_ids = [
                    int(metrics["client_id"])
                    for _, _, metrics
                    in results_mefl[me]
                ]

                # ======================================================
                # 4.1 Temporal drift detection
                # ======================================================

                stable, drift = (
                    self._update_drift_state(
                        me,
                        client_ids
                    )
                )

                # ======================================================
                # 4.2 VWC using stable clients only
                # ======================================================

                stable_assignments, new_anchors = (
                    self._vwc_stable(
                        me,
                        stable
                    )
                )

                # ======================================================
                # 4.3 Persistent anchor handling
                #
                # IMPORTANT:
                #
                # If VWC successfully generated anchors, update the
                # persistent structure.
                #
                # If VWC returned {}, {}, DO NOT erase previous anchors.
                # ======================================================

                previous_anchors = copy.deepcopy(
                    self.anchor_centroids.get(
                        me,
                        {}
                    )
                )

                if new_anchors:

                    anchors = copy.deepcopy(
                        new_anchors
                    )

                    self.anchor_centroids[me] = (
                        copy.deepcopy(
                            anchors
                        )
                    )

                else:

                    anchors = previous_anchors

                # ======================================================
                # 4.4 Assign drifting clients to nearest anchor
                # ======================================================

                assignments = self._assign_drift_clients(
                    me,
                    dict(stable_assignments),
                    anchors
                )

                # ======================================================
                # 4.5 Preserve previous assignment when necessary
                #
                # This is important if:
                #
                #   - there are no stable clients;
                #   - anchors temporarily cannot be recomputed;
                #   - a participating client has no new assignment.
                #
                # Existing cluster assignment is preferable to silently
                # moving the client to cluster 0.
                # ======================================================

                previous_client_clusters = (
                    self.client_clusters.get(
                        me,
                        {}
                    )
                )

                for cid in client_ids:

                    if cid in assignments:
                        continue

                    if cid in previous_client_clusters:

                        previous_cluster = (
                            previous_client_clusters[cid]
                        )

                        # Preserve the old assignment if its cluster
                        # model or anchor still exists.
                        if (
                                previous_cluster in anchors
                                or
                                previous_cluster in self.cluster_models.get(
                            me,
                            {}
                        )
                        ):
                            assignments[cid] = (
                                previous_cluster
                            )

                # ======================================================
                # 4.6 Persist client -> cluster assignments
                # ======================================================

                self.client_clusters[me].update(
                    assignments
                )

                # ======================================================
                # 4.7 Persistent cluster membership
                #
                # DO NOT do:
                #
                #     self.cluster_members[me] = {}
                #
                # because only a fraction of clients participates in
                # each round.
                #
                # Instead, preserve previous membership and update only
                # the clients that participated in this round.
                # ======================================================

                previous_members = copy.deepcopy(
                    self.cluster_members.get(
                        me,
                        {}
                    )
                )

                # Normalize previous membership to sets.
                persistent_members = {}

                for cluster_id, members in previous_members.items():
                    cluster_id = int(
                        cluster_id
                    )

                    persistent_members[cluster_id] = set(
                        int(cid)
                        for cid in members
                    )

                current_client_set = set(
                    client_ids
                )

                # ------------------------------------------------------
                # Remove current-round clients from their old clusters.
                # They will be inserted again according to their new
                # assignments below.
                # ------------------------------------------------------

                for cluster_id in list(
                        persistent_members.keys()
                ):
                    persistent_members[cluster_id] = {
                        cid
                        for cid in persistent_members[cluster_id]
                        if cid not in current_client_set
                    }

                # ------------------------------------------------------
                # Insert current assignments.
                # ------------------------------------------------------

                for cid, cluster_id in assignments.items():
                    cid = int(cid)
                    cluster_id = int(cluster_id)

                    persistent_members.setdefault(
                        cluster_id,
                        set()
                    )

                    persistent_members[
                        cluster_id
                    ].add(
                        cid
                    )

                # ------------------------------------------------------
                # Store membership as sorted lists.
                #
                # Empty membership metadata is removed, but the cluster
                # model itself is NOT removed.
                # ------------------------------------------------------

                self.cluster_members[me] = {
                    cluster_id: sorted(
                        members
                    )
                    for cluster_id, members
                    in persistent_members.items()
                    if members
                }

                # ======================================================
                # 4.8 Hybrid model aggregation
                #
                # _aggregate_hybrid() starts from previous cluster models
                # and therefore preserves clusters with no participating
                # clients in this round.
                # ======================================================

                global_params, cluster_params = (
                    self._aggregate_hybrid(
                        me,
                        results_mefl[me],
                        assignments
                    )
                )

                # ------------------------------------------------------
                # Update global parameters.
                # ------------------------------------------------------

                if global_params is not None:
                    aggregated[me] = (
                        global_params
                    )

                    self.parameters_aggregated_mefl[me] = (
                        global_params
                    )

                # ------------------------------------------------------
                # Update persistent cluster models.
                # ------------------------------------------------------

                if cluster_params is not None:
                    self.cluster_models[me] = (
                        cluster_params
                    )

                # ======================================================
                # 4.9 Drift metrics
                # ======================================================

                n_drift = len(
                    drift
                )

                rate = (
                        n_drift
                        /
                        max(
                            1,
                            len(client_ids)
                        )
                )

                self.drift_clients[me] = (
                    n_drift
                )

                self.drift_rate[me] = (
                    rate
                )

                self.drift_rate_history[me].append(
                    rate
                )

                self.data_shift_type[me] = (
                    "DATA_SHIFT"
                    if n_drift > 0
                    else "NO_SHIFT"
                )

                # ======================================================
                # 4.10 Aggregate fit metrics
                # ======================================================

                if self.fit_metrics_aggregation_fn:

                    fit_metrics = [
                        (
                            n,
                            metrics
                        )
                        for _, n, metrics
                        in results_mefl[me]
                    ]

                    metrics_aggregated[me] = (
                        self.fit_metrics_aggregation_fn(
                            fit_metrics
                        )
                    )

                else:

                    metrics_aggregated[me] = {}

                # ======================================================
                # 4.11 Add FedDCA metrics
                # ======================================================

                metrics_aggregated[me].update(
                    {
                        "Drift clients": n_drift,

                        "Drift rate": rate,

                        "Data shift": (
                            self.data_shift_type[me]
                        ),

                        "Round (t)": server_round,

                        "Fraction fit": (
                            self.fraction_fit
                        ),

                        "# training clients": (
                            self.n_trained_clients
                        ),

                        "training clients and models": (
                            self.selected_clients_m[me]
                        ),

                        "Model size": (
                            self.models_size[me]
                        ),

                        "Alpha": (
                            metrics_aggregated[me].get(
                                "Alpha",
                                self.alpha[me]
                            )
                        ),
                    }
                )

                # ======================================================
                # 4.12 Detection evaluation
                # ======================================================

                self._update_detection_metrics(
                    me,
                    server_round
                )

                # ======================================================
                # 4.13 Training metrics
                # ======================================================

                for metric in self.train_metrics_names:
                    self.results_train_metrics[me][
                        metric
                    ].append(
                        metrics_aggregated[me].get(
                            metric,
                            0
                        )
                    )

                # ======================================================
                # 4.14 Update temporal reference
                #
                # This happens AFTER drift detection.
                # ======================================================

                self.previous_lp[me] = {
                    cid: copy.deepcopy(lp)
                    for cid, lp
                    in self.current_lp[me].items()
                }

                # ======================================================
                # Diagnostic output
                # ======================================================

                print(
                    f"FedDCA me={me} round={server_round}: "
                    f"stable={stable}, "
                    f"drift={drift}, "
                    f"anchors={list(anchors.keys())}, "
                    f"clusters={self.cluster_members[me]}"
                )

            # ==========================================================
            # 5. Save detection information
            # ==========================================================

            self._save_shift_detection_metrics(
                server_round
            )

            self._save_shift_detection_curve(
                server_round
            )

            # ==========================================================
            # 6. Update server state
            # ==========================================================

            self.parameters_aggregated_mefl = (
                aggregated
            )

            self.metrics_aggregated_mefl = (
                metrics_aggregated
            )

            # ==========================================================
            # 7. Auxiliary data persistence
            # ==========================================================

            if server_round > 10:

                try:

                    self._save_data_metrics()

                except Exception:

                    pass

            # ==========================================================
            # 8. Return aggregated parameters and metrics
            # ==========================================================

            return (
                self.parameters_aggregated_mefl,
                metrics_aggregated
            )

        except Exception as e:

            print(
                "FedDCA aggregate_fit error"
            )

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e
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
        """
        Save cumulative shift-detection metrics for every MEFL model.

        The format follows FedConD:
          - Detection Delay = -1 until a true detection occurs.
          - First Detection Round = -1 until a true detection occurs.
          - Shift Round comes from the configured ground-truth
            data_shift_rounds of the client.
        """

        try:

            print("save shift detection metrics")

            result_path = self.get_result_path("test")

            file_path = (
                    result_path
                    + f"shift_detection_metrics_{self.strategy_name}.csv"
            )

            for me in range(self.ME):

                # ==========================================================
                # Detection history
                # ==========================================================

                y_true = self.shift_ground_truth_event.get(
                    me,
                    []
                )

                y_pred = self.shift_detected.get(
                    me,
                    []
                )

                # ==========================================================
                # Precision / Recall / F1
                #
                # Keep the same logic used by FedConD.
                # ==========================================================

                if len(y_true) == 0:

                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0

                else:

                    tp = (
                        1
                        if self.true_detection_round.get(me)
                           is not None
                        else 0
                    )

                    fp = len(
                        self.false_alarm_rounds.get(
                            me,
                            []
                        )
                    )

                    precision = (
                        tp / (tp + fp)
                        if (tp + fp) > 0
                        else 0.0
                    )

                    recall = float(tp)

                    f1 = (
                        2 * precision * recall
                        / (precision + recall)
                        if precision + recall > 0
                        else 0.0
                    )

                # ==========================================================
                # Detection delay
                #
                # Initialized to -1 and remains -1 until a true detection
                # occurs.
                # ==========================================================

                detection_delay = self.detection_delay.get(
                    me,
                    -1
                )

                if detection_delay is None:
                    detection_delay = -1

                # ==========================================================
                # First detection round
                #
                # None is converted to -1, exactly as in FedConD.
                # ==========================================================

                true_detection_round = (
                    self.true_detection_round.get(me)
                )

                if true_detection_round is None:

                    first_detection_round = -1

                else:

                    first_detection_round = (
                        true_detection_round
                    )

                # ==========================================================
                # Number of false alarms
                # ==========================================================

                false_alarms = len(
                    self.false_alarm_rounds.get(
                        me,
                        []
                    )
                )

                # ==========================================================
                # CSV row
                # ==========================================================

                row = [[

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

                    detection_delay,

                    false_alarms,

                    first_detection_round,

                    self.shift_rounds[me][0],

                ]]

                self._write_rows(
                    file_path,
                    row
                )

        except Exception as e:

            print(
                "_save_shift_detection_metrics error"
            )

            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno,
                    type(e).__name__,
                    e,
                )
            )

            raise

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