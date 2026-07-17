from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

import torch
import torch.nn.functional as F
from torch.autograd import grad

from .baselines import BaselineRunResult, apply_edge_deletion
from .official import OfficialBaselineUnavailable


@dataclass(frozen=True)
class GIFConfig:
    iteration: int = 100
    scale: float = 500.0
    damp: float = 0.0
    hops: int = 2


class GIFBaseline:
    """Native edge-unlearning adapter for the official GIF baseline.

    This ports the core Graph-oriented Influence Function update from
    GIF-torch while keeping this repository's PyG data object, trained model,
    fixed forget-set protocols, and metric pipeline.
    """

    def __init__(self, name: str = "gif", config: Optional[GIFConfig] = None):
        self.name = name
        self.config = config or GIFConfig()

    def run_node_unlearning(self, data, forget_nodes: Iterable[int], **_: Any) -> BaselineRunResult:
        raise OfficialBaselineUnavailable("The native GIF adapter currently supports only edge unlearning.")

    def run_feature_unlearning(self, data, forget_features: Iterable[int], **_: Any) -> BaselineRunResult:
        raise OfficialBaselineUnavailable("The native GIF adapter currently supports only edge unlearning.")

    def run_edge_unlearning(
        self,
        data,
        forget_edges: Iterable[tuple[int, int]],
        trainer,
        **_: Any,
    ) -> BaselineRunResult:
        forget_edges = [(int(source), int(target)) for source, target in forget_edges]
        data_after = apply_edge_deletion(data, forget_edges)
        model = trainer.model
        device = trainer.device
        data = trainer._to_device(data)
        data_after = trainer._to_device(data_after)
        y = _labels(data)
        train_mask = trainer._resolve_mask(data, None, "train_mask")
        affected_nodes = _affected_nodes(data.edge_index, forget_edges, self.config.hops, int(data.num_nodes))
        affected_mask = _node_mask(affected_nodes, int(data.num_nodes), device)

        if affected_mask.sum().item() == 0:
            logits_after, embeddings_after = trainer.predict_with_embeddings(data_after)
            return BaselineRunResult(
                self.name,
                "edge",
                len(forget_edges),
                {
                    "status": "skipped",
                    "reason": "no affected nodes resolved from forget_edges",
                    "iteration": self.config.iteration,
                    "scale": self.config.scale,
                    "damp": self.config.damp,
                    "hops": self.config.hops,
                    "num_affected_nodes": 0,
                },
                data_after,
                model,
                trainer,
                logits_after,
                embeddings_after,
            )

        was_training = model.training
        model.eval()
        model.zero_grad(set_to_none=True)

        params = [param for param in model.parameters() if param.requires_grad]
        logits_original = model(data.x, data.edge_index)
        loss_all = F.cross_entropy(logits_original[train_mask], y[train_mask], reduction="sum")
        loss_original_region = F.cross_entropy(logits_original[affected_mask], y[affected_mask], reduction="sum")
        grad_all = _grad(loss_all, params, retain_graph=True, create_graph=True)
        grad_original_region = _grad(loss_original_region, params, retain_graph=True, create_graph=False)

        logits_deleted = model(data_after.x, data_after.edge_index)
        y_after = _labels(data_after)
        loss_deleted_region = F.cross_entropy(logits_deleted[affected_mask], y_after[affected_mask], reduction="sum")
        grad_deleted_region = _grad(loss_deleted_region, params, retain_graph=False, create_graph=False)

        # The region-gradient difference is the fixed right-hand side of the
        # inverse-Hessian estimate; retaining its autograd graphs only wastes memory.
        vector = tuple((before - after).detach() for before, after in zip(grad_original_region, grad_deleted_region))
        del logits_deleted, loss_deleted_region, grad_deleted_region, grad_original_region, loss_original_region
        h_estimate = tuple(item.detach().clone() for item in vector)
        for _ in range(max(0, int(self.config.iteration))):
            hvp = _hvp(grad_all, params, h_estimate)
            with torch.no_grad():
                h_estimate = tuple(
                    v + (1.0 - float(self.config.damp)) * h - hv / float(self.config.scale)
                    for v, h, hv in zip(vector, h_estimate, hvp)
                )

        with torch.no_grad():
            for param, delta in zip(params, h_estimate):
                param.add_(delta / float(self.config.scale))

        model.train(was_training)
        logits_after, embeddings_after = trainer.predict_with_embeddings(data_after)
        training_summary = {
            "status": "ok",
            "source": "native_adapter_ported_from_GIF_torch",
            "iteration": int(self.config.iteration),
            "scale": float(self.config.scale),
            "damp": float(self.config.damp),
            "hops": int(self.config.hops),
            "num_affected_nodes": len(affected_nodes),
            "affected_node_sample": affected_nodes[:20],
        }
        return BaselineRunResult(
            self.name,
            "edge",
            len(forget_edges),
            training_summary,
            data_after,
            model,
            trainer,
            logits_after,
            embeddings_after,
        )


def _grad(loss: torch.Tensor, params: list[torch.nn.Parameter], retain_graph: bool, create_graph: bool) -> tuple[torch.Tensor, ...]:
    grads = grad(loss, params, retain_graph=retain_graph, create_graph=create_graph, allow_unused=True)
    return tuple(torch.zeros_like(param) if item is None else item for item, param in zip(grads, params))


def _hvp(
    grad_all: tuple[torch.Tensor, ...],
    params: list[torch.nn.Parameter],
    vector: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    product = sum(torch.sum(grad_item * vector_item) for grad_item, vector_item in zip(grad_all, vector))
    return _grad(product, params, retain_graph=True, create_graph=True)


def _affected_nodes(edge_index: torch.Tensor, forget_edges: list[tuple[int, int]], hops: int, num_nodes: int) -> list[int]:
    frontier = {node for edge in forget_edges for node in edge if 0 <= int(node) < num_nodes}
    affected = set(frontier)
    if not frontier:
        return []
    source = edge_index[0].detach().cpu()
    target = edge_index[1].detach().cpu()
    for _ in range(max(0, int(hops))):
        if not frontier:
            break
        frontier_tensor = torch.tensor(sorted(frontier), dtype=source.dtype)
        source_hit = torch.isin(source, frontier_tensor)
        target_hit = torch.isin(target, frontier_tensor)
        neighbors = set(target[source_hit].tolist()) | set(source[target_hit].tolist())
        frontier = {int(node) for node in neighbors if 0 <= int(node) < num_nodes and int(node) not in affected}
        affected.update(frontier)
    return sorted(affected)


def _node_mask(nodes: list[int], num_nodes: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    if nodes:
        mask[torch.tensor(nodes, dtype=torch.long, device=device)] = True
    return mask


def _labels(data) -> torch.Tensor:
    y = data.y
    if y.dim() > 1:
        y = y.squeeze(-1)
    return y.to(device=data.x.device, dtype=torch.long)
