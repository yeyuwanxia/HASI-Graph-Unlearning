from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from baselines import BaselineRunResult, GIFBaseline, GIFConfig, GraphEraserBaseline
from baselines.baselines import apply_edge_deletion, apply_feature_deletion, apply_node_deletion


ADAPTED_BASELINES = ("grapheraser-bekm", "grapheraser-blpa", "gif", "gnndelete", "megu")


class OpenGUAdaptedGraphEraser(GraphEraserBaseline):
    """OpenGU-sourced GraphEraser adapter for this repository's protocol.

    The original OpenGU source snapshot is kept under
    `opengu_adapted_baselines/vendor/opengu_selected/`. This class keeps the
    same public experiment protocol as this repository: PyG data objects,
    shared-base splits, JSON forget sets, and the standard metrics pipeline.
    """

    _OPEN_GU_METHODS = {
        "bekm": {
            "opengu_partition_method": "sage_km_base",
            "opengu_source": "GULib-master/unlearning/unlearning_methods/GraphEraser/partition/partition_kmeans.py",
            "note": "OpenGU BEKM-style balanced KMeans path adapted to the local PyG/shared-base protocol.",
        },
        "blpa": {
            "opengu_partition_method": "lpa_base",
            "opengu_source": "GULib-master/unlearning/unlearning_methods/GraphEraser/partition/partition_lpa.py",
            "note": "OpenGU BLPA-style constrained LPA path adapted to the local PyG/shared-base protocol.",
        },
    }

    def __init__(
        self,
        public_name: str,
        partition_method: str,
        *,
        num_shards: int = 10,
        epochs: int = 100,
        lr: float = 0.005,
        shard_size_delta: float = 0.05,
        terminate_delta: int = 1,
        seed: int = 42,
    ):
        normalized = partition_method.lower()
        if normalized not in self._OPEN_GU_METHODS:
            raise ValueError(f"Unsupported OpenGU GraphEraser partition method: {partition_method}")
        super().__init__(
            name=f"opengu-{public_name}",
            partition_method=normalized,
            num_shards=num_shards,
            epochs=epochs,
            lr=lr,
            shard_size_delta=shard_size_delta,
            terminate_delta=terminate_delta,
            seed=seed,
        )
        self.public_name = public_name

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "source_family": "OpenGU",
            "source_zip": "OpenGU-main.zip",
            "vendor_snapshot": "opengu_adapted_baselines/vendor/opengu_selected",
            "baseline": self.public_name,
            "adapter_method": self.name,
            "runtime_adapter": "opengu_adapted_baselines.OpenGUAdaptedGraphEraser",
            **self._OPEN_GU_METHODS[self.partition_method],
            "protocol": "local_shared_base_json_forget_sets_standard_metrics",
        }

    def prepare_artifact(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        metadata = super().prepare_artifact(*args, **kwargs)
        metadata["opengu_provenance"] = self.provenance
        metadata["baseline"] = self.name
        metadata["public_baseline"] = self.public_name
        return metadata

    def run_node_unlearning(self, *args: Any, **kwargs: Any) -> BaselineRunResult:
        return self._with_provenance(super().run_node_unlearning(*args, **kwargs))

    def run_edge_unlearning(self, *args: Any, **kwargs: Any) -> BaselineRunResult:
        return self._with_provenance(super().run_edge_unlearning(*args, **kwargs))

    def run_feature_unlearning(self, *args: Any, **kwargs: Any) -> BaselineRunResult:
        result = super().run_feature_unlearning(*args, **kwargs)
        annotated = self._with_provenance(result)
        if isinstance(annotated.training, dict):
            annotated.training["feature_forgetting_note"] = (
                "Feature-dimension forgetting is a local unified-protocol extension; "
                "OpenGU's original feature branch uses a different node-feature masking semantics."
            )
        return annotated

    def _with_provenance(self, result: BaselineRunResult) -> BaselineRunResult:
        training = dict(result.training or {})
        training["opengu_provenance"] = self.provenance
        training["public_baseline"] = self.public_name
        return BaselineRunResult(
            method=self.name,
            unlearning_type=result.unlearning_type,
            forget_count=result.forget_count,
            training=training,
            data=result.data,
            model=result.model,
            trainer=result.trainer,
            logits=result.logits,
            embeddings=result.embeddings,
        )


class OpenGUAdaptedGIF(GIFBaseline):
    """OpenGU-sourced GIF adapter for local edge-forgetting experiments."""

    def __init__(self, config: GIFConfig | None = None):
        super().__init__("opengu-gif", config=config or GIFConfig())

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "source_family": "OpenGU",
            "source_zip": "OpenGU-main.zip",
            "vendor_snapshot": "opengu_adapted_baselines/vendor/opengu_selected",
            "baseline": "gif",
            "adapter_method": self.name,
            "runtime_adapter": "opengu_adapted_baselines.OpenGUAdaptedGIF",
            "opengu_source": "GULib-master/unlearning/unlearning_methods/GIF/gif.py",
            "opengu_trainer_source": "GULib-master/task/GIFTrainer.py",
            "gif_config": asdict(self.config),
            "protocol": "local_shared_base_json_forget_sets_standard_metrics",
            "note": "OpenGU GIF influence-function path adapted to this repository's edge-forgetting protocol.",
        }

    def run_edge_unlearning(self, *args: Any, **kwargs: Any) -> BaselineRunResult:
        result = super().run_edge_unlearning(*args, **kwargs)
        training = dict(result.training or {})
        training["opengu_provenance"] = self.provenance
        training["public_baseline"] = "gif"
        return BaselineRunResult(
            method=self.name,
            unlearning_type=result.unlearning_type,
            forget_count=result.forget_count,
            training=training,
            data=result.data,
            model=result.model,
            trainer=result.trainer,
            logits=result.logits,
            embeddings=result.embeddings,
        )


class OpenGUAdaptedGNNDelete:
    """OpenGU-sourced GNNDelete adapter for this repository's protocol.

    OpenGU's original GNNDelete implementation depends on its model_zoo,
    deletion-layer GNN variants, trainer stack, and checkpoint layout. This
    adapter keeps the local HASI protocol fixed and ports the core GNNDelete
    training objective shape: preserve retained-node embeddings while applying
    deletion pressure to the forgotten nodes or edges.
    """

    def __init__(
        self,
        *,
        epochs: int = 50,
        lr: float = 0.01,
        alpha: float = 0.5,
        preserve_weight: float = 1.0,
        delete_weight: float = 1.0,
        max_forget_edges_for_loss: int = 4096,
    ):
        self.name = "opengu-gnndelete"
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.alpha = float(alpha)
        self.preserve_weight = float(preserve_weight)
        self.delete_weight = float(delete_weight)
        self.max_forget_edges_for_loss = int(max_forget_edges_for_loss)

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "source_family": "OpenGU",
            "source_zip": "OpenGU-main.zip",
            "vendor_snapshot": "opengu_adapted_baselines/vendor/opengu_selected",
            "baseline": "gnndelete",
            "adapter_method": self.name,
            "runtime_adapter": "opengu_adapted_baselines.OpenGUAdaptedGNNDelete",
            "opengu_source": "GULib-master/unlearning/unlearning_methods/GNNDelete/gnndelete.py",
            "opengu_trainer_source": "GULib-master/task/GNNDeleteTrainer.py",
            "opengu_mia_source": "GULib-master/attack/Attack_methods/GNNDelete_MIA.py",
            "protocol": "local_shared_base_json_forget_sets_standard_metrics",
            "note": (
                "OpenGU GNNDelete adapted to the local shared-base protocol. "
                "The original OpenGU code requires its own model_zoo and deletion-layer trainer stack; "
                "this adapter ports the retention/randomness objective shape onto the local GNNTrainer."
            ),
        }

    @property
    def config(self) -> dict[str, Any]:
        return {
            "unlearning_epochs": self.epochs,
            "unlearn_lr": self.lr,
            "alpha": self.alpha,
            "preserve_weight": self.preserve_weight,
            "delete_weight": self.delete_weight,
            "max_forget_edges_for_loss": self.max_forget_edges_for_loss,
            "loss_fct": "mse_mean",
            "loss_type": "local_protocol_embedding_preservation_plus_deletion_pressure",
        }

    def run_node_unlearning(self, data, forget_nodes, trainer, **_: Any) -> BaselineRunResult:
        forget_nodes = self._valid_nodes(data, forget_nodes)
        reference_embeddings = self._reference_embeddings(trainer, data)
        data_after = apply_node_deletion(data, forget_nodes)
        preserve_nodes = self._retain_nodes(data_after.num_nodes, forget_nodes)
        extra_loss = self._extra_loss_fn(
            reference_embeddings=reference_embeddings,
            preserve_nodes=preserve_nodes,
            uniform_nodes=forget_nodes,
        )
        result = trainer.fine_tune(data_after, epochs=self.epochs, lr=self.lr, extra_loss_fn=extra_loss)
        training = self._training_metadata(result.as_dict(), "node", len(forget_nodes))
        return BaselineRunResult(self.name, "node", len(forget_nodes), training, data_after, trainer.model, trainer)

    def run_edge_unlearning(self, data, forget_edges, trainer, **_: Any) -> BaselineRunResult:
        forget_edges = self._valid_edges(data, forget_edges)
        reference_embeddings = self._reference_embeddings(trainer, data)
        data_after = apply_edge_deletion(data, forget_edges)
        affected_nodes = sorted({node for edge in forget_edges for node in edge})
        preserve_nodes = self._retain_nodes(data_after.num_nodes, affected_nodes)
        extra_loss = self._extra_loss_fn(
            reference_embeddings=reference_embeddings,
            preserve_nodes=preserve_nodes,
            uniform_nodes=affected_nodes,
            forget_edges=forget_edges,
        )
        result = trainer.fine_tune(data_after, epochs=self.epochs, lr=self.lr, extra_loss_fn=extra_loss)
        training = self._training_metadata(result.as_dict(), "edge", len(forget_edges))
        return BaselineRunResult(self.name, "edge", len(forget_edges), training, data_after, trainer.model, trainer)

    def run_feature_unlearning(self, data, forget_features, trainer, **_: Any) -> BaselineRunResult:
        forget_features = sorted(
            {int(item) for item in forget_features if 0 <= int(item) < int(data.x.shape[1])}
        )
        reference_embeddings = self._reference_embeddings(trainer, data)
        data_after = apply_feature_deletion(data, forget_features)
        preserve_nodes = list(range(int(data_after.num_nodes)))
        extra_loss = self._extra_loss_fn(
            reference_embeddings=reference_embeddings,
            preserve_nodes=preserve_nodes,
        )
        result = trainer.fine_tune(data_after, epochs=self.epochs, lr=self.lr, extra_loss_fn=extra_loss)
        training = self._training_metadata(result.as_dict(), "feature", len(forget_features))
        training["feature_forgetting_note"] = (
            "Feature-dimension forgetting is a local unified-protocol extension; "
            "OpenGU GNNDelete's original feature branch masks selected nodes' features."
        )
        return BaselineRunResult(self.name, "feature", len(forget_features), training, data_after, trainer.model, trainer)

    def _training_metadata(self, result: dict[str, Any], unlearning_type: str, forget_count: int) -> dict[str, Any]:
        return {
            **result,
            "opengu_provenance": self.provenance,
            "public_baseline": "gnndelete",
            "gnndelete_config": self.config,
            "unlearning_type": unlearning_type,
            "forget_count": forget_count,
        }

    @staticmethod
    def _reference_embeddings(trainer, data) -> torch.Tensor:
        _, embeddings = trainer.predict_with_embeddings(data)
        return embeddings.detach()

    def _extra_loss_fn(
        self,
        *,
        reference_embeddings: torch.Tensor,
        preserve_nodes: list[int],
        uniform_nodes: list[int] | None = None,
        forget_edges: list[tuple[int, int]] | None = None,
    ):
        preserve_template = torch.tensor(preserve_nodes, dtype=torch.long)
        uniform_template = torch.tensor(uniform_nodes or [], dtype=torch.long)
        edge_template = self._edge_tensor(forget_edges or [])

        def extra_loss(logits: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
            device = embeddings.device
            ref = reference_embeddings.to(device=device, dtype=embeddings.dtype)
            total = embeddings.new_tensor(0.0)
            preserve_idx = preserve_template.to(device=device)
            if preserve_idx.numel() > 0:
                preserve_loss = F.mse_loss(embeddings[preserve_idx], ref[preserve_idx])
                total = total + self.preserve_weight * (1.0 - self.alpha) * preserve_loss

            delete_loss = embeddings.new_tensor(0.0)
            uniform_idx = uniform_template.to(device=device)
            if uniform_idx.numel() > 0:
                delete_loss = delete_loss - F.log_softmax(logits[uniform_idx], dim=-1).mean()
            edges = edge_template.to(device=device)
            if edges.numel() > 0:
                source, target = edges[:, 0], edges[:, 1]
                scores = (embeddings[source] * embeddings[target]).sum(dim=-1)
                delete_loss = delete_loss + F.softplus(scores).mean()
            if uniform_idx.numel() > 0 or edges.numel() > 0:
                total = total + self.delete_weight * self.alpha * delete_loss
            return total

        return extra_loss

    def _edge_tensor(self, edges: list[tuple[int, int]]) -> torch.Tensor:
        if not edges:
            return torch.empty((0, 2), dtype=torch.long)
        if len(edges) > self.max_forget_edges_for_loss:
            edges = edges[: self.max_forget_edges_for_loss]
        return torch.tensor(edges, dtype=torch.long)

    @staticmethod
    def _valid_nodes(data, nodes) -> list[int]:
        num_nodes = int(data.num_nodes)
        return sorted({int(node) for node in nodes if 0 <= int(node) < num_nodes})

    @staticmethod
    def _valid_edges(data, edges) -> list[tuple[int, int]]:
        num_nodes = int(data.num_nodes)
        valid = []
        seen = set()
        for source, target in edges:
            edge = (int(source), int(target))
            if edge in seen:
                continue
            if 0 <= edge[0] < num_nodes and 0 <= edge[1] < num_nodes:
                seen.add(edge)
                valid.append(edge)
        return valid

    @staticmethod
    def _retain_nodes(num_nodes: int, excluded_nodes: list[int]) -> list[int]:
        excluded = set(excluded_nodes)
        return [node for node in range(int(num_nodes)) if node not in excluded]


class OpenGUAdaptedMEGU:
    """OpenGU-sourced MEGU adapter for this repository's protocol.

    OpenGU MEGU uses a mutual-evolution objective: preserve the retained
    neighborhood while pushing the forgotten targets away from the original
    predictions. This adapter keeps that objective shape but runs it through the
    local shared-base trainer, JSON forget sets, and standard metric pipeline.
    """

    def __init__(
        self,
        *,
        epochs: int = 50,
        lr: float = 0.01,
        kappa: float = 0.01,
        num_hops: int = 2,
        retain_weight: float = 1.0,
        embedding_retain_weight: float = 0.1,
        edge_forget_weight: float = 1.0,
        max_forget_edges_for_loss: int = 4096,
    ):
        self.name = "opengu-megu"
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.kappa = float(kappa)
        self.num_hops = int(num_hops)
        self.retain_weight = float(retain_weight)
        self.embedding_retain_weight = float(embedding_retain_weight)
        self.edge_forget_weight = float(edge_forget_weight)
        self.max_forget_edges_for_loss = int(max_forget_edges_for_loss)

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "source_family": "OpenGU",
            "source_zip": "OpenGU-main.zip",
            "vendor_snapshot": "opengu_adapted_baselines/vendor/opengu_selected",
            "baseline": "megu",
            "adapter_method": self.name,
            "runtime_adapter": "opengu_adapted_baselines.OpenGUAdaptedMEGU",
            "opengu_source": "GULib-master/unlearning/unlearning_methods/MEGU/megu.py",
            "opengu_trainer_source": "GULib-master/task/MEGUTrainer.py",
            "protocol": "local_shared_base_json_forget_sets_standard_metrics",
            "note": (
                "OpenGU MEGU adapted to the local shared-base protocol. "
                "The original OpenGU implementation depends on its Learning_based_pipeline, "
                "trainer stack, text forget-set paths, and CorrectAndSmooth post-processing; "
                "this adapter ports the retain/forget objective shape onto the local GNNTrainer."
            ),
        }

    @property
    def config(self) -> dict[str, Any]:
        return {
            "unlearning_epochs": self.epochs,
            "unlearn_lr": self.lr,
            "kappa": self.kappa,
            "num_hops": self.num_hops,
            "retain_weight": self.retain_weight,
            "embedding_retain_weight": self.embedding_retain_weight,
            "edge_forget_weight": self.edge_forget_weight,
            "max_forget_edges_for_loss": self.max_forget_edges_for_loss,
            "loss_type": "local_protocol_megu_retain_plus_uniform_forget_pressure",
            "opengu_defaults": {
                "unlearning_epochs": 50,
                "kappa": 0.01,
                "alpha1": 0.8,
                "alpha2": 0.5,
            },
        }

    def run_node_unlearning(self, data, forget_nodes, trainer, **_: Any) -> BaselineRunResult:
        forget_nodes = self._valid_nodes(data, forget_nodes)
        reference_logits, reference_embeddings = self._reference_outputs(trainer, data)
        data_after = apply_node_deletion(data, forget_nodes)
        retain_nodes = self._khop_neighbors(data, forget_nodes)
        if not retain_nodes:
            retain_nodes = self._retain_nodes(data_after.num_nodes, forget_nodes)
        extra_loss = self._extra_loss_fn(
            reference_logits=reference_logits,
            reference_embeddings=reference_embeddings,
            retain_nodes=retain_nodes,
            forget_nodes=forget_nodes,
        )
        result = trainer.fine_tune(data_after, epochs=self.epochs, lr=self.lr, extra_loss_fn=extra_loss)
        training = self._training_metadata(
            result.as_dict(),
            "node",
            len(forget_nodes),
            retain_node_count=len(retain_nodes),
            forget_node_count=len(forget_nodes),
        )
        return BaselineRunResult(self.name, "node", len(forget_nodes), training, data_after, trainer.model, trainer)

    def run_edge_unlearning(self, data, forget_edges, trainer, **_: Any) -> BaselineRunResult:
        forget_edges = self._valid_edges(data, forget_edges)
        reference_logits, reference_embeddings = self._reference_outputs(trainer, data)
        data_after = apply_edge_deletion(data, forget_edges)
        affected_nodes = sorted({node for edge in forget_edges for node in edge})
        retain_nodes = self._khop_neighbors(data, affected_nodes)
        if not retain_nodes:
            retain_nodes = self._retain_nodes(data_after.num_nodes, affected_nodes)
        extra_loss = self._extra_loss_fn(
            reference_logits=reference_logits,
            reference_embeddings=reference_embeddings,
            retain_nodes=retain_nodes,
            forget_nodes=affected_nodes,
            forget_edges=forget_edges,
        )
        result = trainer.fine_tune(data_after, epochs=self.epochs, lr=self.lr, extra_loss_fn=extra_loss)
        training = self._training_metadata(
            result.as_dict(),
            "edge",
            len(forget_edges),
            retain_node_count=len(retain_nodes),
            forget_node_count=len(affected_nodes),
        )
        return BaselineRunResult(self.name, "edge", len(forget_edges), training, data_after, trainer.model, trainer)

    def run_feature_unlearning(self, data, forget_features, trainer, **_: Any) -> BaselineRunResult:
        forget_features = self._valid_features(data, forget_features)
        reference_logits, reference_embeddings = self._reference_outputs(trainer, data)
        data_after = apply_feature_deletion(data, forget_features)
        retain_nodes = list(range(int(data_after.num_nodes)))
        extra_loss = self._extra_loss_fn(
            reference_logits=reference_logits,
            reference_embeddings=reference_embeddings,
            retain_nodes=retain_nodes,
        )
        result = trainer.fine_tune(data_after, epochs=self.epochs, lr=self.lr, extra_loss_fn=extra_loss)
        training = self._training_metadata(
            result.as_dict(),
            "feature",
            len(forget_features),
            retain_node_count=len(retain_nodes),
            forget_node_count=0,
        )
        training["feature_forgetting_note"] = (
            "Feature-dimension forgetting is a local unified-protocol extension; "
            "OpenGU MEGU's original feature branch masks selected nodes' full feature vectors."
        )
        return BaselineRunResult(self.name, "feature", len(forget_features), training, data_after, trainer.model, trainer)

    def _training_metadata(
        self,
        result: dict[str, Any],
        unlearning_type: str,
        forget_count: int,
        *,
        retain_node_count: int,
        forget_node_count: int,
    ) -> dict[str, Any]:
        return {
            **result,
            "opengu_provenance": self.provenance,
            "public_baseline": "megu",
            "megu_config": self.config,
            "unlearning_type": unlearning_type,
            "forget_count": forget_count,
            "retain_node_count": retain_node_count,
            "forget_node_count_for_objective": forget_node_count,
        }

    @staticmethod
    def _reference_outputs(trainer, data) -> tuple[torch.Tensor, torch.Tensor]:
        logits, embeddings = trainer.predict_with_embeddings(data)
        return logits.detach(), embeddings.detach()

    def _extra_loss_fn(
        self,
        *,
        reference_logits: torch.Tensor,
        reference_embeddings: torch.Tensor,
        retain_nodes: list[int],
        forget_nodes: list[int] | None = None,
        forget_edges: list[tuple[int, int]] | None = None,
    ):
        retain_template = torch.tensor(retain_nodes, dtype=torch.long)
        forget_template = torch.tensor(forget_nodes or [], dtype=torch.long)
        edge_template = self._edge_tensor(forget_edges or [])
        reference_probs = F.softmax(reference_logits, dim=-1).detach()

        def extra_loss(logits: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
            device = embeddings.device
            total = embeddings.new_tensor(0.0)
            ref_probs = reference_probs.to(device=device, dtype=logits.dtype)
            ref_embeddings = reference_embeddings.to(device=device, dtype=embeddings.dtype)

            retain_idx = retain_template.to(device=device)
            if retain_idx.numel() > 0:
                retain_log_probs = F.log_softmax(logits[retain_idx], dim=-1)
                retain_kl = F.kl_div(retain_log_probs, ref_probs[retain_idx], reduction="batchmean")
                retain_embedding = F.mse_loss(embeddings[retain_idx], ref_embeddings[retain_idx])
                total = total + self.retain_weight * (
                    retain_kl + self.embedding_retain_weight * retain_embedding
                )

            forget_idx = forget_template.to(device=device)
            if forget_idx.numel() > 0:
                log_probs = F.log_softmax(logits[forget_idx], dim=-1)
                uniform = torch.full_like(log_probs, 1.0 / float(log_probs.shape[-1]))
                total = total + self.kappa * F.kl_div(log_probs, uniform, reduction="batchmean")

            edges = edge_template.to(device=device)
            if edges.numel() > 0:
                source, target = edges[:, 0], edges[:, 1]
                scores = (embeddings[source] * embeddings[target]).sum(dim=-1)
                total = total + self.kappa * self.edge_forget_weight * F.softplus(scores).mean()

            return total

        return extra_loss

    def _khop_neighbors(self, data, seed_nodes: list[int]) -> list[int]:
        if not seed_nodes:
            return []
        try:
            from torch_geometric.utils import k_hop_subgraph
        except ImportError:
            return []
        node_idx = torch.tensor(seed_nodes, dtype=torch.long)
        subset, _, _, _ = k_hop_subgraph(
            node_idx,
            self.num_hops,
            data.edge_index.detach().cpu(),
            relabel_nodes=False,
            num_nodes=int(data.num_nodes),
        )
        seed_set = set(seed_nodes)
        return sorted({int(node) for node in subset.detach().cpu().tolist() if int(node) not in seed_set})

    def _edge_tensor(self, edges: list[tuple[int, int]]) -> torch.Tensor:
        if not edges:
            return torch.empty((0, 2), dtype=torch.long)
        if len(edges) > self.max_forget_edges_for_loss:
            edges = edges[: self.max_forget_edges_for_loss]
        return torch.tensor(edges, dtype=torch.long)

    @staticmethod
    def _valid_nodes(data, nodes) -> list[int]:
        num_nodes = int(data.num_nodes)
        return sorted({int(node) for node in nodes if 0 <= int(node) < num_nodes})

    @staticmethod
    def _valid_edges(data, edges) -> list[tuple[int, int]]:
        num_nodes = int(data.num_nodes)
        valid = []
        seen = set()
        for source, target in edges:
            edge = (int(source), int(target))
            if edge in seen:
                continue
            if 0 <= edge[0] < num_nodes and 0 <= edge[1] < num_nodes:
                seen.add(edge)
                valid.append(edge)
        return valid

    @staticmethod
    def _valid_features(data, features) -> list[int]:
        if not hasattr(data, "x") or data.x is None:
            return []
        num_features = int(data.x.shape[1])
        return sorted({int(item) for item in features if 0 <= int(item) < num_features})

    @staticmethod
    def _retain_nodes(num_nodes: int, excluded_nodes: list[int]) -> list[int]:
        excluded = set(excluded_nodes)
        return [node for node in range(int(num_nodes)) if node not in excluded]


def get_adapted_baseline(name: str):
    key = str(name).lower().replace("_", "-")
    if key == "grapheraser-bekm":
        return OpenGUAdaptedGraphEraser("grapheraser-bekm", "bekm")
    if key == "grapheraser-blpa":
        return OpenGUAdaptedGraphEraser("grapheraser-blpa", "blpa")
    if key == "gif":
        return OpenGUAdaptedGIF()
    if key == "gnndelete":
        return OpenGUAdaptedGNNDelete()
    if key == "megu":
        return OpenGUAdaptedMEGU()
    supported = ", ".join(ADAPTED_BASELINES)
    raise ValueError(f"Unsupported OpenGU-adapted baseline {name!r}. Supported: {supported}")


def method_key(method: str) -> str:
    return str(method).lower().replace("-", "_")


def default_artifact_dir(
    root: str | Path,
    *,
    dataset: str,
    experiment_name: str,
    method: str,
    unlearning_type: str,
    seed: int,
) -> Path:
    base = Path(root) / f"{dataset}_eval"
    if experiment_name and experiment_name != "__root__":
        base = base / experiment_name
    return base / "baselines" / method_key(method) / unlearning_type / "artifacts" / f"seed{seed}"
