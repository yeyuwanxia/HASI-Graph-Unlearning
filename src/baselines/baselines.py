from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import torch

from data import clone_data
from models import GNNTrainer, TrainingConfig


ModelFactory = Callable[[Any], torch.nn.Module]


@dataclass(frozen=True)
class BaselineRunResult:
    method: str
    unlearning_type: str
    forget_count: int
    training: Optional[dict[str, Any]]
    data: Any
    model: Optional[torch.nn.Module] = None
    trainer: Optional[GNNTrainer] = None
    logits: Optional[torch.Tensor] = None
    embeddings: Optional[torch.Tensor] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "unlearning_type": self.unlearning_type,
            "forget_count": self.forget_count,
            "training": self.training,
        }


class ZeroShotDeleteBaseline:
    """Delete the requested graph data without additional optimization."""

    def __init__(self, name: str = "zero_shot_delete"):
        self.name = name

    def run_node_unlearning(self, data, forget_nodes: Iterable[int], **_: Any) -> BaselineRunResult:
        forget_nodes = list(forget_nodes)
        data_after = apply_node_deletion(data, forget_nodes)
        return BaselineRunResult(self.name, "node", len(forget_nodes), None, data_after)

    def run_edge_unlearning(self, data, forget_edges: Iterable[tuple[int, int]], **_: Any) -> BaselineRunResult:
        forget_edges = list(forget_edges)
        data_after = apply_edge_deletion(data, forget_edges)
        return BaselineRunResult(self.name, "edge", len(forget_edges), None, data_after)

    def run_feature_unlearning(self, data, forget_features: Iterable[int], **_: Any) -> BaselineRunResult:
        forget_features = list(forget_features)
        data_after = apply_feature_deletion(data, forget_features)
        return BaselineRunResult(self.name, "feature", len(forget_features), None, data_after)


class FineTuneDeleteBaseline:
    """Delete then fine-tune an existing trained model."""

    def __init__(self, name: str = "finetune_delete", epochs: int = 50, lr: float = 0.005):
        self.name = name
        self.epochs = int(epochs)
        self.lr = float(lr)

    def run_node_unlearning(self, data, forget_nodes: Iterable[int], trainer: GNNTrainer, **_: Any) -> BaselineRunResult:
        forget_nodes = list(forget_nodes)
        data_after = apply_node_deletion(data, forget_nodes)
        result = trainer.fine_tune(data_after, epochs=self.epochs, lr=self.lr)
        return BaselineRunResult(self.name, "node", len(forget_nodes), result.as_dict(), data_after, trainer.model, trainer)

    def run_edge_unlearning(self, data, forget_edges: Iterable[tuple[int, int]], trainer: GNNTrainer, **_: Any) -> BaselineRunResult:
        forget_edges = list(forget_edges)
        data_after = apply_edge_deletion(data, forget_edges)
        result = trainer.fine_tune(data_after, epochs=self.epochs, lr=self.lr)
        return BaselineRunResult(self.name, "edge", len(forget_edges), result.as_dict(), data_after, trainer.model, trainer)

    def run_feature_unlearning(self, data, forget_features: Iterable[int], trainer: GNNTrainer, **_: Any) -> BaselineRunResult:
        forget_features = list(forget_features)
        data_after = apply_feature_deletion(data, forget_features)
        result = trainer.fine_tune(data_after, epochs=self.epochs, lr=self.lr)
        return BaselineRunResult(self.name, "feature", len(forget_features), result.as_dict(), data_after, trainer.model, trainer)


class RetrainBaseline:
    """Golden baseline: delete requested data and train a fresh model."""

    def __init__(self, name: str = "retrain", config: Optional[TrainingConfig] = None):
        self.name = name
        self.config = config or TrainingConfig()

    def run_node_unlearning(
        self,
        data,
        forget_nodes: Iterable[int],
        model_factory: ModelFactory,
        epochs: Optional[int] = None,
        **_: Any,
    ) -> BaselineRunResult:
        forget_nodes = list(forget_nodes)
        data_after = apply_node_deletion(data, forget_nodes)
        return self._train(data_after, model_factory, "node", len(forget_nodes), epochs)

    def run_edge_unlearning(
        self,
        data,
        forget_edges: Iterable[tuple[int, int]],
        model_factory: ModelFactory,
        epochs: Optional[int] = None,
        **_: Any,
    ) -> BaselineRunResult:
        forget_edges = list(forget_edges)
        data_after = apply_edge_deletion(data, forget_edges)
        return self._train(data_after, model_factory, "edge", len(forget_edges), epochs)

    def run_feature_unlearning(
        self,
        data,
        forget_features: Iterable[int],
        model_factory: ModelFactory,
        epochs: Optional[int] = None,
        **_: Any,
    ) -> BaselineRunResult:
        forget_features = list(forget_features)
        data_after = apply_feature_deletion(data, forget_features)
        return self._train(data_after, model_factory, "feature", len(forget_features), epochs)

    def _train(self, data_after, model_factory: ModelFactory, kind: str, forget_count: int, epochs: Optional[int]) -> BaselineRunResult:
        model = model_factory(data_after)
        trainer = GNNTrainer(model, self.config)
        result = trainer.train_full_batch(data_after, epochs=epochs)
        return BaselineRunResult(self.name, kind, forget_count, result.as_dict(), data_after, model, trainer)


def baseline_registry(root: str | Path | None = None) -> dict[str, object]:
    """Return runnable baseline adapters.

    GraphEraser entries use the stitched adapter in `grapheraser.py`. The
    `*-surrogate` entries are non-official fine-tune sanity checks. Official
    baselines without native adapters fail loudly instead of silently reporting
    surrogate results as paper baselines.
    """

    from .grapheraser import GraphEraserBaseline
    from .gif import GIFBaseline
    from .official import ExternalOfficialBaseline
    from .official_sources import get_official_spec

    project_root = Path(root) if root is not None else Path.cwd()
    return {
        "zero_shot": ZeroShotDeleteBaseline(),
        "finetune": FineTuneDeleteBaseline(),
        "retrain": RetrainBaseline(),
        "grapheraser-bekm": GraphEraserBaseline("grapheraser-bekm", "bekm", epochs=80, lr=0.005),
        "grapheraser-blpa": GraphEraserBaseline("grapheraser-blpa", "blpa", epochs=80, lr=0.005),
        "gnndelete": ExternalOfficialBaseline(get_official_spec("gnndelete"), project_root),
        "gif": GIFBaseline("gif"),
        "gnndelete-surrogate": FineTuneDeleteBaseline("gnndelete-surrogate", epochs=60, lr=0.003),
        "gif-surrogate": FineTuneDeleteBaseline("gif-surrogate", epochs=40, lr=0.003),
        "sgu-surrogate": FineTuneDeleteBaseline("sgu-surrogate", epochs=40, lr=0.005),
        "agu-surrogate": FineTuneDeleteBaseline("agu-surrogate", epochs=40, lr=0.005),
    }


def get_baseline(name: str, root: str | Path | None = None):
    registry = baseline_registry(root)
    key = name.lower()
    if key not in registry:
        supported = ", ".join(sorted(registry))
        raise ValueError(f"Unsupported baseline {name!r}. Supported baselines: {supported}")
    return registry[key]


def apply_node_deletion(data, forget_nodes: Iterable[int]):
    data_after = clone_data(data)
    node_set = {int(node) for node in forget_nodes}
    if hasattr(data_after, "x") and data_after.x is not None:
        data_after.x = data_after.x.clone()
        valid_nodes = [node for node in node_set if 0 <= node < data_after.x.shape[0]]
        if valid_nodes:
            data_after.x[valid_nodes] = 0
    if hasattr(data_after, "edge_index"):
        keep = _edge_keep_indices(data_after.edge_index, lambda u, v: u not in node_set and v not in node_set)
        _apply_edge_keep(data_after, keep)
    _disable_masks(data_after, node_set)
    return data_after


def apply_edge_deletion(data, forget_edges: Iterable[tuple[int, int]]):
    data_after = clone_data(data)
    edge_set = {(int(u), int(v)) for u, v in forget_edges}
    edge_set |= {(v, u) for u, v in edge_set}
    if hasattr(data_after, "edge_index"):
        keep = _edge_keep_indices(data_after.edge_index, lambda u, v: (u, v) not in edge_set)
        _apply_edge_keep(data_after, keep)
    return data_after


def apply_feature_deletion(data, forget_features: Iterable[int]):
    data_after = clone_data(data)
    if not hasattr(data_after, "x") or data_after.x is None:
        return data_after
    data_after.x = data_after.x.clone()
    feature_ids = [idx for idx in {int(item) for item in forget_features} if 0 <= idx < data_after.x.shape[1]]
    if feature_ids:
        data_after.x[:, feature_ids] = 0
    return data_after


def _edge_keep_indices(edge_index: torch.Tensor, keep_fn) -> torch.Tensor:
    pairs = edge_index.detach().cpu().t().tolist()
    keep = [idx for idx, (source, target) in enumerate(pairs) if keep_fn(int(source), int(target))]
    return torch.tensor(keep, dtype=torch.long, device=edge_index.device)


def _apply_edge_keep(data: Any, keep: torch.Tensor) -> None:
    num_edges = int(data.edge_index.shape[1])
    data.edge_index = data.edge_index[:, keep]
    if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.shape[0] == num_edges:
        data.edge_attr = data.edge_attr[keep]


def _disable_masks(data: Any, nodes: set[int]) -> None:
    for name in ("train_mask", "val_mask", "test_mask"):
        if not hasattr(data, name):
            continue
        mask = getattr(data, name)
        if mask is None:
            continue
        mask = mask.clone()
        valid_nodes = [node for node in nodes if 0 <= node < mask.shape[0]]
        if valid_nodes:
            mask[valid_nodes] = False
        setattr(data, name, mask)
