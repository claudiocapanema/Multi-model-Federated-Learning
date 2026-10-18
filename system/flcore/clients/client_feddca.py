import copy
import random
import sys
import numpy as np
import torch
import torch.nn.functional as F

from flcore.clients.client_multifedavg import MultiFedAvgClient
from flcore.clients.utils.models_utils import (
    get_weights,
    set_weights,
    test,
    train,
)


class ClientFedDCA(MultiFedAvgClient):
    """FedDCA client adapted to the existing MEFL/MultiFedAvg client.

    The client keeps the original MultiFedAvg data/MEFL machinery and adds
    only the label-conditional profiling step required by FedDCA. The
    profiling loader uses the exact dataset input keys used by PFLlib.

    Returned fit result:
        (model_weights, n_examples, metrics)
    where metrics additionally contains ``Label Profile``.
    """

    def __init__(self, args, id, model, fold_id):
        super().__init__(args, id, model, fold_id)

        self.feddca_num_prototypes = int(getattr(args, "feddca_num_prototypes", 5))
        self.feddca_profile_batch_size = int(
            getattr(args, "feddca_profile_batch_size", 0)
        )
        self.feddca_feature_module = getattr(
            args, "feddca_feature_module", None
        )
        # Gradient-stability controls for local FedDCA training.
        # The clipping is intentionally conservative and only affects the
        # optimizer step; the FedDCA profiling/detection logic is unchanged.
        self.feddca_grad_clip_norm = float(
            getattr(args, "feddca_grad_clip_norm", 1.0)
        )

    # ------------------------------------------------------------------
    # Input / output helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _unpack_batch(batch, dataset_name=None):
        """Extract the input tensor and labels using the same keys as PFLlib.

        ``load_data`` defines DATASET_INPUT_MAP as:
            CIFAR10 -> ``img``
            image datasets -> ``image``
            sequence datasets -> ``sequence``
            wikitext -> ``text``

        This method deliberately follows that mapping instead of guessing a
        generic ``data`` key, because the existing MultiFedAvg/FedConD
        training code uses the dataset-specific key.
        """
        if isinstance(batch, dict):
            if "label" not in batch:
                raise KeyError(
                    f"Could not find 'label' in batch for dataset "
                    f"{dataset_name!r}. Available keys: {list(batch.keys())}"
                )

            dataset_key = {
                "CIFAR10": "img",
                "MNIST": "image",
                "EMNIST": "image",
                "GTSRB": "image",
                "ImageNet": "image",
                "ImageNet10": "image",
                "Gowalla": "sequence",
                "WISDM-W": "sequence",
                "WISDM-P": "sequence",
                "Foursquare": "sequence",
                "wikitext": "text",
            }.get(dataset_name)

            # Prefer the exact PFLlib key. The fallbacks are only for custom
            # datasets and do not change the normal code path.
            candidate_keys = []
            if dataset_key is not None:
                candidate_keys.append(dataset_key)
            candidate_keys.extend(
                ["img", "image", "sequence", "text", "data", "x", "input", "features"]
            )

            x = None
            selected_key = None
            for key in candidate_keys:
                if key in batch:
                    x = batch[key]
                    selected_key = key
                    break

            if x is None:
                raise KeyError(
                    f"Could not find input tensor for dataset {dataset_name!r}. "
                    f"Available keys: {list(batch.keys())}"
                )

            y = batch["label"]
            return x, y

        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]

        raise TypeError(
            f"Unsupported dataloader batch format for dataset "
            f"{dataset_name!r}: {type(batch).__name__}"
        )

    @staticmethod
    def _as_logits(output):
        if isinstance(output, dict):
            for key in ("logits", "output", "out", "prediction"):
                if key in output:
                    return output[key]
            raise TypeError("Model returned a dict without a logits-like key.")
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def _resolve_feature_module(self, model):
        """Find a feature layer for generic PFLlib models.

        Priority:
          1. explicit ``args.feddca_feature_module`` path;
          2. ``forward_features`` / ``extract_features`` methods;
          3. common classifier/head modules, using their input as features;
          4. penultimate leaf module as a last-resort hook target.

        For a paper-faithful experiment, an explicit feature module is
        recommended whenever the backbone exposes one.
        """
        explicit = self.feddca_feature_module
        if explicit:
            module = model
            for name in str(explicit).split("."):
                if not hasattr(module, name):
                    raise AttributeError(
                        f"feddca_feature_module='{explicit}' not found in model."
                    )
                module = getattr(module, name)
            return module

        for method_name in ("forward_features", "extract_features"):
            if callable(getattr(model, method_name, None)):
                return method_name

        named = list(model.named_modules())
        candidates = []
        keywords = ("classifier", "fc", "head", "linear", "output")
        for name, module in named:
            if name and isinstance(module, torch.nn.Linear):
                if any(k in name.lower().split(".")[-1] for k in keywords):
                    candidates.append((name, module))

        if candidates:
            return candidates[-1][1]

        leaves = [m for n, m in named if n and len(list(m.children())) == 0]
        if len(leaves) >= 2:
            return leaves[-2]
        if leaves:
            return leaves[-1]
        raise RuntimeError("Unable to infer a feature module for FedDCA profiling.")

    def _forward_with_features(self, model, x):
        """Return (logits, features) without changing the model API."""
        feature_target = self._resolve_feature_module(model)

        if isinstance(feature_target, str):
            output = getattr(model, feature_target)(x)
            # forward_features normally returns the representation. The
            # classification logits still have to be obtained from forward.
            features = output[0] if isinstance(output, (tuple, list)) else output
            full_output = model(x)
            return self._as_logits(full_output), features

        captured = {}

        def hook(_module, inputs, output):
            if inputs:
                captured["features"] = inputs[0]
            else:
                captured["features"] = output

        handle = feature_target.register_forward_hook(hook)
        try:
            output = model(x)
        finally:
            handle.remove()

        logits = self._as_logits(output)
        features = captured.get("features")
        if features is None:
            raise RuntimeError("Feature hook did not capture a representation.")
        if isinstance(features, (tuple, list)):
            features = features[0]
        return logits, features

    def _check_finite_model_parameters(self, model, stage, me, t):
        """Raise with the exact parameter name if model weights are non-finite."""
        for name, param in model.named_parameters():
            if not torch.isfinite(param).all():
                finite = param.detach()[torch.isfinite(param.detach())]
                if finite.numel():
                    min_value = float(finite.min().cpu())
                    max_value = float(finite.max().cpu())
                else:
                    min_value = float("nan")
                    max_value = float("nan")
                raise RuntimeError(
                    f"Client {self.client_id}, model {me}, round {t}: "
                    f"non-finite parameter at {stage}: {name} "
                    f"shape={tuple(param.shape)} min_finite={min_value} "
                    f"max_finite={max_value}"
                )

    def _check_finite_tensor(self, tensor, name, me, t):
        """Raise with tensor statistics when an input/output is non-finite."""
        if not torch.isfinite(tensor).all():
            finite = tensor.detach()[torch.isfinite(tensor.detach())]
            if finite.numel():
                min_value = float(finite.min().cpu())
                max_value = float(finite.max().cpu())
            else:
                min_value = float("nan")
                max_value = float("nan")
            raise RuntimeError(
                f"Client {self.client_id}, model {me}, round {t}: "
                f"non-finite {name} detected; shape={tuple(tensor.shape)} "
                f"dtype={tensor.dtype} min_finite={min_value} "
                f"max_finite={max_value}"
            )

    # ------------------------------------------------------------------
    # FedDCA Label Profile
    # ------------------------------------------------------------------
    def _normalize_features(self, features):
        """Flatten non-vector features while preserving batch dimension."""
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        return features

    @torch.no_grad()
    def build_label_profile(self, me):
        """Build LP_{c,t} from lowest-loss core samples per class.

        For each class, the lowest-loss samples are selected and their
        feature vectors form the empirical measure for that label, exactly
        following Eq. (2) of the paper.
        """
        model = self.model[me]
        model.eval()

        n_classes = self.n_classes[me]
        max_prototypes = max(1, self.feddca_num_prototypes)

        # class -> list[(loss, feature)]
        candidates = {label: [] for label in range(n_classes)}
        loader = self.trainloader[me]

        # The profile is built after local training. Check the model state
        # before the first forward pass so that divergence is attributed to
        # the weights rather than being reported later by SciPy.
        self._check_finite_model_parameters(model, "before_label_profile", me, self.lt[me])

        for batch in loader:
            x, y = self._unpack_batch(batch, self.args.dataset[me])
            x = x.to(self.device)
            y = y.to(self.device).long()

            self._check_finite_tensor(x, "input", me, self.lt[me])
            self._check_finite_tensor(y, "labels", me, self.lt[me])

            logits, features = self._forward_with_features(model, x)
            self._check_finite_tensor(logits, "logits", me, self.lt[me])
            features = self._normalize_features(features)
            self._check_finite_tensor(features, "features", me, self.lt[me])
            per_sample_loss = F.cross_entropy(logits, y, reduction="none")
            self._check_finite_tensor(per_sample_loss, "per-sample loss", me, self.lt[me])

            for label in torch.unique(y).tolist():
                mask = y == int(label)
                idx = torch.nonzero(mask, as_tuple=False).flatten()
                if idx.numel() == 0:
                    continue
                for j in idx.tolist():
                    candidates[int(label)].append(
                        (
                            float(per_sample_loss[j].detach().cpu()),
                            features[j].detach().cpu().float().numpy(),
                        )
                    )

        # Keep the k lowest-loss/core samples for each label.
        profile = {}
        for label in range(n_classes):
            values = candidates[label]
            if not values:
                continue
            values.sort(key=lambda item: item[0])
            selected = values[:max_prototypes]
            profile[label] = np.stack([v[1] for v in selected], axis=0).astype(
                np.float32
            )

        if not profile:
            raise RuntimeError(
                f"Client {self.client_id}, model {me}: empty Label Profile."
            )

        return profile


    def evaluate(self, me, t, global_model):
        """Evaluate one MEFL model and return a flat Flower-style result.

        IMPORTANT:
        MultiFedAvg's server-side aggregate_evaluate expects:
            (loss, num_examples, metrics_dict)

        The inherited client implementation can return a nested tuple
        (loss, num_examples, (loss, num_examples, metrics_dict)), which makes
        metrics a tuple on the server and causes:
            TypeError: tuple indices must be integers or slices, not str
        """
        try:
            g = torch.Generator()
            g.manual_seed(t + self.fold_id)
            random.seed(t + self.fold_id)
            np.random.seed(t + self.fold_id)
            torch.manual_seed(t + self.fold_id)

            self.update_local_test_data(t, me)
            set_weights(self.model[me], global_model)

            loss, metrics = test(
                self.model[me],
                self.valloader[me],
                self.device,
                self.client_id,
                t,
                self.args.dataset[me],
                self.n_classes[me]
            )

            metrics["Model size"] = self.models_size[me]
            metrics["Dataset size"] = len(self.valloader[me].dataset)
            metrics["me"] = me
            metrics["Alpha"] = self.alpha_test[me]

            # CRITICAL: return the metrics dict directly as the third item.
            return (
                loss,
                len(self.valloader[me].dataset),
                metrics,
            )

        except Exception as e:
            print("FedDCA evaluate error")
            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno, type(e).__name__, e
                )
            )
            raise

    # ------------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------------
    def _train_with_diagnostics(self, me, t):
        """Train locally while identifying the first source of NaNs/Infs.

        This replaces the generic ``models_utils.train`` call only for the
        FedDCA client so that we can inspect every batch before backward,
        every gradient after backward, and every parameter immediately after
        ``optimizer.step()``. Gradient clipping is applied before the step
        to prevent exploding recurrent-model gradients from corrupting the
        GRU parameters.
        """
        model = self.model[me]
        trainloader = self.trainloader[me]
        valloader = self.valloader[me]
        optimizer = self.optimizer[me]
        dataset_name = self.args.dataset[me]
        n_classes = self.n_classes[me]

        dataset_key = {
            "CIFAR10": "img", "MNIST": "image", "EMNIST": "image",
            "GTSRB": "image", "ImageNet": "image", "ImageNet10": "image",
            "Gowalla": "sequence", "WISDM-W": "sequence",
            "WISDM-P": "sequence", "Foursquare": "sequence",
            "wikitext": "text",
        }.get(dataset_name)
        if dataset_key is None:
            raise KeyError(f"Unknown dataset input key for {dataset_name!r}")

        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        model.to(self.device)
        model.train()

        loss_total = 0.0
        correct = 0
        total = 0
        y_true = []
        y_prob = []

        for epoch in range(self.local_epochs):
            for batch_idx, batch in enumerate(trainloader):
                x = batch[dataset_key].to(self.device)
                labels = batch["label"].to(self.device).long()
                self._check_finite_tensor(x, f"input (epoch={epoch}, batch={batch_idx})", me, t)
                self._check_finite_tensor(labels, f"labels (epoch={epoch}, batch={batch_idx})", me, t)

                optimizer.zero_grad(set_to_none=True)

                outputs = self._as_logits(model(x))
                self._check_finite_tensor(
                    outputs, f"logits before loss (epoch={epoch}, batch={batch_idx})", me, t
                )

                loss = criterion(outputs, labels)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"Client {self.client_id}, model {me}, round {t}: "
                        f"non-finite loss before backward at epoch={epoch}, batch={batch_idx}: {loss.item()}"
                    )

                loss.backward()

                # Inspect gradients before clipping. Compute the global
                # gradient norm in float64 to avoid float32 overflow when
                # combining many large but individually finite gradients.
                # We then apply clipping manually using the same global norm,
                # which avoids a second potentially overflowing norm
                # calculation inside torch.nn.utils.clip_grad_norm_.
                grad_norms = []
                grad_params = []
                for name, param in model.named_parameters():
                    if param.grad is None:
                        continue
                    if not torch.isfinite(param.grad).all():
                        raise RuntimeError(
                            f"Client {self.client_id}, model {me}, round {t}: "
                            f"non-finite gradient at epoch={epoch}, batch={batch_idx}: {name}"
                        )
                    grad_params.append(param)
                    grad_norms.append(param.grad.detach().double().norm(2))

                if grad_norms:
                    total_grad_norm_tensor = torch.linalg.vector_norm(
                        torch.stack(grad_norms), ord=2
                    )
                    total_grad_norm = float(total_grad_norm_tensor.cpu())

                    if self.feddca_grad_clip_norm > 0:
                        clip_norm = float(self.feddca_grad_clip_norm)
                        if np.isfinite(total_grad_norm) and total_grad_norm > clip_norm:
                            scale = clip_norm / (total_grad_norm + 1e-12)
                        elif not np.isfinite(total_grad_norm):
                            # This is only possible for an extreme norm beyond
                            # float64 range. All individual gradients were
                            # already verified finite, so zeroing this update
                            # is safer than allowing an unstable optimizer step.
                            scale = 0.0
                        else:
                            scale = 1.0

                        if scale < 1.0:
                            for param in grad_params:
                                param.grad.mul_(scale)
                else:
                    total_grad_norm = 0.0

                optimizer.step()

                # This pinpoints the exact batch whose optimizer step
                # corrupted the parameters.
                try:
                    self._check_finite_model_parameters(
                        model,
                        f"after_optimizer_step epoch={epoch} batch={batch_idx} "
                        f"grad_norm={total_grad_norm:.6g}",
                        me,
                        t,
                    )
                except RuntimeError:
                    raise

                batch_size = labels.shape[0]
                loss_total += float(loss.detach().cpu()) * batch_size
                total += batch_size
                y_true.append(labels.detach().cpu().numpy())
                y_prob.append(outputs.detach().cpu().numpy())
                correct += int((torch.argmax(outputs.detach(), dim=1) == labels).sum().item())

        if total == 0:
            raise RuntimeError(
                f"Client {self.client_id}, model {me}, round {t}: empty training loader."
            )

        y_true_np = np.concatenate(y_true, axis=0)
        y_prob_np = np.concatenate(y_prob, axis=0)
        accuracy = correct / total

        # Keep the same metric shape expected by the existing server.
        from sklearn import metrics as sk_metrics
        try:
            balanced_accuracy = float(
                sk_metrics.balanced_accuracy_score(y_true_np, y_prob_np.argmax(axis=1))
            )
        except Exception:
            balanced_accuracy = float("nan")

        model.eval()
        val_loss, test_metrics = test(
            model, valloader, self.device, self.client_id, t, dataset_name, n_classes
        )

        return {
            "val_loss": val_loss,
            "val_accuracy": test_metrics["Accuracy"],
            "val_balanced_accuracy": test_metrics["Balanced accuracy"],
            "train_loss": loss_total / total,
            "train_accuracy": accuracy,
            "train_balanced_accuracy": balanced_accuracy,
            "Round (t)": t,
        }

    def fit(self, me, t, global_model):
        try:
            self.lt[me] = t
            set_weights(self.model[me], global_model)
            self._check_finite_model_parameters(
                self.model[me], "after_set_weights", me, t
            )

            if t > 1:
                self.update_local_train_data(t, me)

            self.optimizer[me] = self._get_optimizer(
                dataset_name=self.args.dataset[me], me=me
            )

            results = self._train_with_diagnostics(me, t)

            # Detect divergence immediately after local training.
            self._check_finite_model_parameters(
                self.model[me], "after_local_training", me, t
            )

            # FedDCA profiling happens AFTER local training.
            label_profile = self.build_label_profile(me)

            results["me"] = me
            results["client_id"] = self.client_id
            results["Model size"] = self.models_size[me]
            results["alpha"] = self.alpha_train[me]
            results["Label Profile"] = label_profile
            results["FedDCA prototypes"] = self.feddca_num_prototypes
            results["Data shift"] = "UNKNOWN"
            self.loss_ME[me] = results["train_loss"]

            return (
                get_weights(self.model[me]),
                len(self.trainloader[me].dataset),
                results,
            )

        except Exception as e:
            print("FedDCA fit error")
            print(
                "Error on line {} {} {}".format(
                    sys.exc_info()[-1].tb_lineno, type(e).__name__, e
                )
            )
            raise