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

        for batch in loader:
            x, y = self._unpack_batch(batch, self.args.dataset[me])
            x = x.to(self.device)
            y = y.to(self.device).long()

            logits, features = self._forward_with_features(model, x)
            features = self._normalize_features(features)
            per_sample_loss = F.cross_entropy(logits, y, reduction="none")

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
    def fit(self, me, t, global_model):
        try:
            self.lt[me] = t
            set_weights(self.model[me], global_model)

            if t > 1:
                self.update_local_train_data(t, me)

            self.optimizer[me] = self._get_optimizer(
                dataset_name=self.args.dataset[me], me=me
            )

            results = train(
                model=self.model[me],
                trainloader=self.trainloader[me],
                valloader=self.valloader[me],
                optimizer=self.optimizer[me],
                epochs=self.local_epochs,
                learning_rate=self.lr,
                device=self.device,
                client_id=self.client_id,
                t=t,
                dataset_name=self.args.dataset[me],
                n_classes=self.n_classes[me],
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