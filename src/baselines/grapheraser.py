from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import networkx as nx
import torch

from data import clone_data
from models import GNNTrainer, TrainingConfig

from .baselines import (
    BaselineRunResult,
    ModelFactory,
    apply_edge_deletion,
    apply_feature_deletion,
    apply_node_deletion,
)


METADATA_FILE = "metadata.json"
PARTITION_FILE = "partition.json"
SHARDS_DIR = "shards"


@dataclass(frozen=True)
class ShardRun:
    shard_id: int
    num_nodes: int
    num_edges: int
    training: dict[str, Any]
    state_file: str | None = None
    status: str = "trained"


class GraphEraserBaseline:
    """GraphEraser-style partition, shard training, and posterior aggregation.

    Without an artifact directory this adapter keeps the historical behavior:
    partition the given graph and train all shard models during the baseline run.
    With an artifact directory it treats partitioning and initial shard-model
    training as offline preprocessing, then online unlearning retrains only the
    shards affected by the forget request.
    """

    def __init__(
        self,
        name: str,
        partition_method: str,
        num_shards: int = 10,
        epochs: int = 100,
        lr: float = 0.01,
        shard_size_delta: float = 0.05,
        terminate_delta: int = 1,
        seed: int = 42,
    ):
        self.name = name
        self.partition_method = partition_method
        self.num_shards = int(num_shards)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.shard_size_delta = float(shard_size_delta)
        self.terminate_delta = int(terminate_delta)
        self.seed = int(seed)

    def run_node_unlearning(
        self,
        data,
        forget_nodes: Iterable[int],
        model_factory: ModelFactory,
        epochs: Optional[int] = None,
        artifact_dir: str | Path | None = None,
        device: str | None = None,
        **_: Any,
    ) -> BaselineRunResult:
        forget_nodes = list(forget_nodes)
        if artifact_dir:
            return self._run_from_artifact(data, forget_nodes, model_factory, "node", epochs, artifact_dir, device)
        data_after = apply_node_deletion(data, forget_nodes)
        return self._train_partitioned(data_after, model_factory, "node", len(forget_nodes), epochs, device=device)

    def run_edge_unlearning(
        self,
        data,
        forget_edges: Iterable[tuple[int, int]],
        model_factory: ModelFactory,
        epochs: Optional[int] = None,
        artifact_dir: str | Path | None = None,
        device: str | None = None,
        **_: Any,
    ) -> BaselineRunResult:
        forget_edges = list(forget_edges)
        if artifact_dir:
            return self._run_from_artifact(data, forget_edges, model_factory, "edge", epochs, artifact_dir, device)
        data_after = apply_edge_deletion(data, forget_edges)
        return self._train_partitioned(data_after, model_factory, "edge", len(forget_edges), epochs, device=device)

    def run_feature_unlearning(
        self,
        data,
        forget_features: Iterable[int],
        model_factory: ModelFactory,
        epochs: Optional[int] = None,
        artifact_dir: str | Path | None = None,
        device: str | None = None,
        **_: Any,
    ) -> BaselineRunResult:
        forget_features = list(forget_features)
        if artifact_dir:
            return self._run_from_artifact(data, forget_features, model_factory, "feature", epochs, artifact_dir, device)
        data_after = apply_feature_deletion(data, forget_features)
        return self._train_partitioned(data_after, model_factory, "feature", len(forget_features), epochs, device=device)

    def prepare_artifact(
        self,
        data,
        model_factory: ModelFactory,
        artifact_dir: str | Path,
        *,
        dataset_name: str,
        seed: int,
        unlearning_type: str,
        model_config: dict[str, Any] | None = None,
        base_artifact: dict[str, Any] | None = None,
        epochs: Optional[int] = None,
        device: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        artifact_path = Path(artifact_dir)
        metadata_path = artifact_path / METADATA_FILE
        if metadata_path.exists() and not overwrite:
            return json.loads(metadata_path.read_text(encoding="utf-8"))

        artifact_path.mkdir(parents=True, exist_ok=True)
        shard_dir = artifact_path / SHARDS_DIR
        shard_dir.mkdir(parents=True, exist_ok=True)

        graph = _graph_from_data(data)
        communities = self._partition(graph, data)
        _save_partition(artifact_path / PARTITION_FILE, communities)

        run_epochs = int(epochs or self.epochs)
        shard_runs: list[ShardRun] = []
        logits_sum = None
        embeddings_sum = None
        for shard_id, nodes in sorted(communities.items()):
            shard_data = _build_shard_data(data, nodes)
            model = model_factory(shard_data)
            trainer = GNNTrainer(model, TrainingConfig(lr=self.lr, epochs=run_epochs, device=device))
            training = trainer.train_full_batch(shard_data, epochs=run_epochs)
            state_file = f"{SHARDS_DIR}/shard_{int(shard_id)}_model_state.pt"
            torch.save(trainer.model.state_dict(), artifact_path / state_file)
            logits, embeddings = trainer.predict_with_embeddings(data)
            logits_sum = logits if logits_sum is None else logits_sum + logits
            embeddings_sum = embeddings if embeddings_sum is None else embeddings_sum + embeddings
            shard_runs.append(
                ShardRun(
                    shard_id=int(shard_id),
                    num_nodes=len(nodes),
                    num_edges=int(shard_data.edge_index.shape[1]) if hasattr(shard_data, "edge_index") else 0,
                    training=training.as_dict(),
                    state_file=state_file,
                )
            )

        aggregate_logits = logits_sum / max(1, len(shard_runs)) if logits_sum is not None else None
        aggregate_accuracy = _masked_accuracy(aggregate_logits, data, "test_mask") if aggregate_logits is not None else None
        metadata = {
            "dataset": dataset_name,
            "seed": int(seed),
            "baseline": self.name,
            "unlearning_type": unlearning_type,
            "partition_method": self.partition_method,
            "num_shards": len(communities),
            "requested_num_shards": self.num_shards,
            "shard_size_delta": self.shard_size_delta,
            "terminate_delta": self.terminate_delta,
            "training": {"epochs": run_epochs, "lr": self.lr, "device": device},
            "model": model_config or {},
            "base_artifact": base_artifact or {},
            "aggregator": "mean",
            "aggregate_test_accuracy": aggregate_accuracy,
            "num_nodes": int(getattr(data, "num_nodes", data.x.shape[0] if hasattr(data, "x") else 0)),
            "num_edges": int(data.edge_index.shape[1]) if hasattr(data, "edge_index") else None,
            "files": {"partition": PARTITION_FILE, "shards_dir": SHARDS_DIR},
            "shards": [run.__dict__ for run in shard_runs],
        }
        metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2) + "\n", encoding="utf-8")
        return metadata

    def _run_from_artifact(
        self,
        data,
        forget_targets: Iterable[Any],
        model_factory: ModelFactory,
        kind: str,
        epochs: Optional[int],
        artifact_dir: str | Path,
        device: str | None,
    ) -> BaselineRunResult:
        forget_targets = list(forget_targets)
        artifact_path = Path(artifact_dir)
        metadata, communities = self._load_artifact(artifact_path)
        if kind == "node":
            data_after = apply_node_deletion(data, forget_targets)
        elif kind == "edge":
            data_after = apply_edge_deletion(data, forget_targets)
        else:
            data_after = apply_feature_deletion(data, forget_targets)

        affected_shards = _affected_shards(kind, forget_targets, communities)
        run_epochs = int(epochs or metadata.get("training", {}).get("epochs") or self.epochs)
        shard_state_files = {
            int(item["shard_id"]): str(item.get("state_file") or f"{SHARDS_DIR}/shard_{int(item['shard_id'])}_model_state.pt")
            for item in metadata.get("shards", [])
        }

        logits_sum = None
        embeddings_sum = None
        shard_runs: list[ShardRun] = []
        trainers: list[GNNTrainer] = []
        for shard_id, nodes in sorted(communities.items()):
            shard_data = _build_shard_data(data_after, nodes)
            model = model_factory(shard_data)
            trainer = GNNTrainer(model, TrainingConfig(lr=self.lr, epochs=run_epochs, device=device))
            state_file = shard_state_files.get(int(shard_id), f"{SHARDS_DIR}/shard_{int(shard_id)}_model_state.pt")
            if int(shard_id) in affected_shards:
                training = trainer.train_full_batch(shard_data, epochs=run_epochs).as_dict()
                status = "retrained_online"
            else:
                state = torch.load(artifact_path / state_file, map_location=trainer.device)
                trainer.model.load_state_dict(state)
                trainer.model.to(trainer.device)
                trainer.model.eval()
                training = {"status": "reused_offline_model"}
                status = "reused_offline_model"
            logits, embeddings = trainer.predict_with_embeddings(data_after)
            logits_sum = logits if logits_sum is None else logits_sum + logits
            embeddings_sum = embeddings if embeddings_sum is None else embeddings_sum + embeddings
            trainers.append(trainer)
            shard_runs.append(
                ShardRun(
                    shard_id=int(shard_id),
                    num_nodes=len(nodes),
                    num_edges=int(shard_data.edge_index.shape[1]) if hasattr(shard_data, "edge_index") else 0,
                    training=training,
                    state_file=state_file,
                    status=status,
                )
            )

        aggregate_logits = logits_sum / max(1, len(trainers)) if logits_sum is not None else None
        aggregate_embeddings = embeddings_sum / max(1, len(trainers)) if embeddings_sum is not None else None
        aggregate_accuracy = _masked_accuracy(aggregate_logits, data_after, "test_mask") if aggregate_logits is not None else None
        training_summary = {
            "mode": "online_from_artifact",
            "partition_method": self.partition_method,
            "num_shards": len(communities),
            "aggregator": metadata.get("aggregator", "mean"),
            "aggregate_test_accuracy": aggregate_accuracy,
            "artifact": {
                "path": str(artifact_path),
                "loaded": True,
                "offline_preprocessing_seconds": metadata.get("offline_preprocessing_seconds"),
                "partition_method": metadata.get("partition_method"),
                "num_shards": metadata.get("num_shards"),
            },
            "affected_shards": sorted(affected_shards),
            "num_affected_shards": len(affected_shards),
            "num_reused_shards": max(0, len(communities) - len(affected_shards)),
            "shards": [run.__dict__ for run in shard_runs],
        }
        return BaselineRunResult(
            self.name,
            kind,
            len(forget_targets),
            training_summary,
            data_after,
            logits=aggregate_logits,
            embeddings=aggregate_embeddings,
        )

    def _train_partitioned(
        self,
        data_after,
        model_factory: ModelFactory,
        kind: str,
        forget_count: int,
        epochs: Optional[int],
        device: str | None = None,
    ) -> BaselineRunResult:
        graph = _graph_from_data(data_after)
        communities = self._partition(graph, data_after)
        trainers: list[GNNTrainer] = []
        shard_runs: list[ShardRun] = []
        logits_sum = None
        embeddings_sum = None
        run_epochs = int(epochs or self.epochs)

        for shard_id, nodes in sorted(communities.items()):
            shard_data = _build_shard_data(data_after, nodes)
            model = model_factory(shard_data)
            trainer = GNNTrainer(model, TrainingConfig(lr=self.lr, epochs=run_epochs, device=device))
            training = trainer.train_full_batch(shard_data, epochs=run_epochs)
            logits, embeddings = trainer.predict_with_embeddings(data_after)
            logits_sum = logits if logits_sum is None else logits_sum + logits
            embeddings_sum = embeddings if embeddings_sum is None else embeddings_sum + embeddings
            trainers.append(trainer)
            shard_runs.append(
                ShardRun(
                    shard_id=int(shard_id),
                    num_nodes=len(nodes),
                    num_edges=int(shard_data.edge_index.shape[1]) if hasattr(shard_data, "edge_index") else 0,
                    training=training.as_dict(),
                )
            )

        aggregate_logits = logits_sum / max(1, len(trainers)) if logits_sum is not None else None
        aggregate_embeddings = embeddings_sum / max(1, len(trainers)) if embeddings_sum is not None else None
        aggregate_accuracy = _masked_accuracy(aggregate_logits, data_after, "test_mask") if aggregate_logits is not None else None
        training_summary = {
            "mode": "online_rebuild_all_shards",
            "partition_method": self.partition_method,
            "num_shards": len(communities),
            "aggregator": "mean",
            "aggregate_test_accuracy": aggregate_accuracy,
            "shards": [run.__dict__ for run in shard_runs],
        }
        return BaselineRunResult(
            self.name,
            kind,
            forget_count,
            training_summary,
            data_after,
            logits=aggregate_logits,
            embeddings=aggregate_embeddings,
        )

    def _load_artifact(self, artifact_path: Path) -> tuple[dict[str, Any], dict[int, list[int]]]:
        metadata_path = artifact_path / METADATA_FILE
        partition_path = artifact_path / PARTITION_FILE
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing GraphEraser artifact metadata: {metadata_path}")
        if not partition_path.exists():
            raise FileNotFoundError(f"Missing GraphEraser partition: {partition_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("partition_method") != self.partition_method:
            raise ValueError(
                f"GraphEraser partition mismatch: expected {self.partition_method}, got {metadata.get('partition_method')}"
            )
        payload = json.loads(partition_path.read_text(encoding="utf-8"))
        communities = {int(shard): [int(node) for node in nodes] for shard, nodes in payload.get("communities", {}).items()}
        return metadata, communities

    def _partition(self, graph: nx.Graph, data) -> dict[int, list[int]]:
        if graph.number_of_nodes() == 0:
            return {0: []}
        if self.partition_method == "bekm":
            return _balanced_feature_kmeans(data, graph, self.num_shards, self.seed)
        if self.partition_method == "blpa":
            return _constrained_lpa(
                graph,
                self.num_shards,
                self.shard_size_delta,
                self.terminate_delta,
                self.seed,
            )
        raise ValueError(f"Unsupported GraphEraser partition method: {self.partition_method}")


def _balanced_feature_kmeans(data, graph: nx.Graph, num_shards: int, seed: int) -> dict[int, list[int]]:
    nodes = sorted(int(node) for node in graph.nodes)
    if not nodes:
        return {0: []}
    k = max(1, min(int(num_shards), len(nodes)))
    features = _node_partition_features(data, graph, nodes)
    labels = _kmeans_labels(features, k, seed)
    raw = defaultdict(list)
    for node, label in zip(nodes, labels):
        raw[int(label)].append(int(node))
    return _rebalance(raw, k, math.ceil(len(nodes) / k))


def _constrained_lpa(
    graph: nx.Graph,
    num_shards: int,
    shard_size_delta: float,
    terminate_delta: int,
    seed: int,
    iterations: int = 100,
) -> dict[int, list[int]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    nodes = sorted(int(node) for node in graph.nodes)
    k = max(1, min(int(num_shards), len(nodes)))
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    adjacency = [set() for _ in nodes]
    for source, target in graph.edges:
        if source in node_to_idx and target in node_to_idx:
            adjacency[node_to_idx[source]].add(node_to_idx[target])
            adjacency[node_to_idx[target]].add(node_to_idx[source])

    random_indices = np.arange(len(nodes))
    rng.shuffle(random_indices)
    communities = {community: set(part.tolist()) for community, part in enumerate(np.array_split(random_indices, k))}
    node_community = np.zeros(len(nodes), dtype=int)
    for community, members in communities.items():
        for node_idx in members:
            node_community[node_idx] = community

    threshold = math.ceil(len(nodes) / k + shard_size_delta * (len(nodes) - len(nodes) / k))
    previous = {community: set(members) for community, members in communities.items()}
    for _ in range(iterations):
        desire_move = []
        for node_idx, neighbors in enumerate(adjacency):
            if not neighbors:
                desire_move.append((node_idx, node_community[node_idx], node_community[node_idx], 0))
                continue
            counts = Counter(int(node_community[neighbor]) for neighbor in neighbors)
            best_count = max(counts.values())
            best = [community for community, count in counts.items() if count == best_count]
            dst = int(rng.choice(best))
            desire_move.append((node_idx, int(node_community[node_idx]), dst, best_count))

        moved = 0
        for node_idx, src, dst, score in sorted(desire_move, key=lambda item: item[3], reverse=True):
            if src == dst or len(communities[dst]) >= threshold:
                continue
            communities[src].discard(node_idx)
            communities[dst].add(node_idx)
            node_community[node_idx] = dst
            moved += 1

        delta = sum(len((communities[i] | previous[i]) - (communities[i] & previous[i])) for i in range(k))
        previous = {community: set(members) for community, members in communities.items()}
        if delta <= terminate_delta or moved == 0:
            break

    return {community: [nodes[idx] for idx in sorted(indices)] for community, indices in communities.items()}


def _node_partition_features(data, graph: nx.Graph, nodes: list[int]):
    import numpy as np

    feature_parts = []
    if hasattr(data, "x") and data.x is not None:
        x = data.x.detach().cpu()
        feature_parts.append(x[nodes].numpy())

    degree = np.array([graph.degree(node) for node in nodes], dtype=float).reshape(-1, 1)
    degree = degree / max(1.0, float(degree.max()))
    feature_parts.append(degree)
    return np.concatenate(feature_parts, axis=1)


def _kmeans_labels(features, k: int, seed: int):
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return [idx % k for idx in range(features.shape[0])]

    cluster = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return cluster.fit_predict(features).tolist()


def _rebalance(raw: dict[int, list[int]], k: int, threshold: int) -> dict[int, list[int]]:
    communities = {idx: list(raw.get(idx, [])) for idx in range(k)}
    overfull = [idx for idx, nodes in communities.items() if len(nodes) > threshold]
    underfull = [idx for idx, nodes in communities.items() if len(nodes) < threshold]
    for source in overfull:
        while len(communities[source]) > threshold and underfull:
            target = min(underfull, key=lambda idx: len(communities[idx]))
            communities[target].append(communities[source].pop())
            if len(communities[target]) >= threshold:
                underfull.remove(target)
    return {idx: sorted(nodes) for idx, nodes in communities.items()}


def _build_shard_data(data, nodes: Iterable[int]):
    shard = clone_data(data)
    node_set = {int(node) for node in nodes}
    if hasattr(shard, "edge_index"):
        keep = _edge_keep_indices(shard.edge_index, lambda u, v: u in node_set and v in node_set)
        _apply_edge_keep(shard, keep)
    for name in ("train_mask", "val_mask", "test_mask"):
        if hasattr(shard, name):
            mask = getattr(shard, name)
            if mask is not None:
                shard_mask = torch.zeros_like(mask, dtype=torch.bool)
                valid = [node for node in node_set if 0 <= node < mask.shape[0]]
                if valid:
                    shard_mask[valid] = mask[valid].bool()
                setattr(shard, name, shard_mask)
    return shard


def _graph_from_data(data) -> nx.Graph:
    graph = nx.Graph()
    num_nodes = int(getattr(data, "num_nodes", data.x.shape[0]))
    graph.add_nodes_from(range(num_nodes))
    edge_index = data.edge_index.detach().cpu().tolist()
    for source, target in zip(edge_index[0], edge_index[1]):
        graph.add_edge(int(source), int(target))
    return graph


def _masked_accuracy(logits: Optional[torch.Tensor], data, mask_name: str) -> Optional[float]:
    if logits is None or not hasattr(data, mask_name):
        return None
    mask = getattr(data, mask_name)
    if mask is None or mask.sum().item() == 0:
        return None
    y = data.y
    if y.dim() > 1:
        y = y.squeeze(-1)
    pred = logits.argmax(dim=-1).to(y.device)
    return float((pred[mask] == y[mask]).float().mean().item())


def _edge_keep_indices(edge_index: torch.Tensor, keep_fn) -> torch.Tensor:
    pairs = edge_index.detach().cpu().t().tolist()
    keep = [idx for idx, (source, target) in enumerate(pairs) if keep_fn(int(source), int(target))]
    return torch.tensor(keep, dtype=torch.long, device=edge_index.device)


def _apply_edge_keep(data: Any, keep: torch.Tensor) -> None:
    num_edges = int(data.edge_index.shape[1])
    data.edge_index = data.edge_index[:, keep]
    if hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.shape[0] == num_edges:
        data.edge_attr = data.edge_attr[keep]


def _save_partition(path: Path, communities: dict[int, list[int]]) -> None:
    node_to_shard = {}
    for shard_id, nodes in communities.items():
        for node in nodes:
            node_to_shard[str(int(node))] = int(shard_id)
    payload = {
        "communities": {str(int(shard)): [int(node) for node in nodes] for shard, nodes in communities.items()},
        "node_to_shard": node_to_shard,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _affected_shards(kind: str, forget_targets: Iterable[Any], communities: dict[int, list[int]]) -> set[int]:
    if kind == "feature":
        return {int(shard_id) for shard_id in communities}
    node_to_shard = {}
    for shard_id, nodes in communities.items():
        for node in nodes:
            node_to_shard[int(node)] = int(shard_id)
    affected: set[int] = set()
    if kind == "node":
        for node in forget_targets:
            shard = node_to_shard.get(int(node))
            if shard is not None:
                affected.add(shard)
    elif kind == "edge":
        for source, target in forget_targets:
            for node in (source, target):
                shard = node_to_shard.get(int(node))
                if shard is not None:
                    affected.add(shard)
    return affected


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
