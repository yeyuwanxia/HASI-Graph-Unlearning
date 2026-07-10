from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import networkx as nx
import numpy as np
import torch

from evaluation.mia import PrivacyEvaluator


def build_experiment_metrics(
    *,
    method: str,
    dataset: str,
    unlearning_type: str,
    data_before: Any,
    data_after: Any,
    logits_before: Optional[torch.Tensor],
    logits_after: Optional[torch.Tensor],
    embeddings_before: Optional[torch.Tensor],
    embeddings_after: Optional[torch.Tensor],
    forget_targets: Any,
    unlearn_time_seconds: Optional[float] = None,
    retrain_time_seconds: Optional[float] = None,
    online_wall_clock_seconds: Optional[float] = None,
    time_breakdown: Optional[dict[str, float]] = None,
    offline_preprocessing_seconds: Optional[float] = None,
    primary_anchor_nodes: Optional[Sequence[int]] = None,
    secondary_anchor_nodes: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    graph_before = graph_from_data(data_before)
    graph_after = graph_from_data(data_after)
    member_nodes = infer_member_nodes(unlearning_type, forget_targets, data_before, embeddings_before, embeddings_after)
    non_member_nodes = infer_non_member_nodes(data_before, member_nodes, len(member_nodes))

    metrics = {
        "method": method,
        "dataset": dataset,
        "unlearning_type": unlearning_type,
        "forget_count": count_forget_targets(forget_targets),
        "utility": utility_metrics(data_before, data_after, logits_before, logits_after, "test_mask"),
        "validation_utility": utility_metrics(data_before, data_after, logits_before, logits_after, "val_mask"),
        "representation": representation_metrics(
            graph_before,
            embeddings_before,
            embeddings_after,
            member_nodes,
            primary_anchor_nodes=primary_anchor_nodes,
            secondary_anchor_nodes=secondary_anchor_nodes,
        ),
        "structure": structural_metrics(graph_before, graph_after),
        "privacy": privacy_metrics(
            logits_before,
            logits_after,
            embeddings_before,
            embeddings_after,
            graph_before,
            graph_after,
            member_nodes,
            non_member_nodes,
            unlearning_type,
            labels=getattr(data_before, "y", None) if data_before is not None else None,
        ),
        "efficiency": efficiency_metrics(
            unlearn_time_seconds,
            retrain_time_seconds,
            online_wall_clock_seconds=online_wall_clock_seconds,
            time_breakdown=time_breakdown,
            offline_preprocessing_seconds=offline_preprocessing_seconds,
        ),
    }
    metrics["rq_summary"] = rq_summary(metrics)
    return metrics


def save_metrics(result: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(result), indent=2) + "\n", encoding="utf-8")
    return path


def default_metrics_path(
    root: str | Path,
    method: str,
    dataset: str,
    unlearning_type: str,
    ratio: float,
    *,
    selection: str | None = None,
    seed: int | str | None = None,
) -> Path:
    ratio_label = str(ratio).replace(".", "p")
    parts = [str(method), str(dataset), str(unlearning_type), f"r{ratio_label}"]
    if selection:
        parts.append(_path_label(selection))
    if seed is not None:
        parts.append(f"seed{seed}")
    name = "_".join(parts) + ".json"
    return default_results_dir(root, method) / name


def default_results_dir(root: str | Path, method: str) -> Path:
    normalized_method = _path_label(method)
    if normalized_method == "hasi":
        return Path(root) / "results" / "hasi"
    return Path(root) / "results" / "baselines" / normalized_method


def _path_label(value: str) -> str:
    label = str(value).strip().lower().replace(" ", "_")
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in label)


def utility_metrics(data_before, data_after, logits_before, logits_after, mask_name: str = "test_mask") -> dict[str, Any]:
    before = _classification_metrics(data_before, logits_before, mask_name)
    after = _classification_metrics(data_after, logits_after, mask_name)
    return {
        "accuracy_before": before["accuracy"],
        "accuracy_after": after["accuracy"],
        "accuracy_drop": _drop(before["accuracy"], after["accuracy"]),
        "f1_micro_before": before["f1_micro"],
        "f1_micro_after": after["f1_micro"],
        "f1_micro_drop": _drop(before["f1_micro"], after["f1_micro"]),
        "f1_macro_before": before["f1_macro"],
        "f1_macro_after": after["f1_macro"],
        "f1_macro_drop": _drop(before["f1_macro"], after["f1_macro"]),
        "num_eval_before": before["num_eval"],
        "num_eval_after": after["num_eval"],
    }


def representation_metrics(
    graph_before: nx.Graph,
    embeddings_before,
    embeddings_after,
    member_nodes: Sequence[int],
    *,
    primary_anchor_nodes: Optional[Sequence[int]] = None,
    secondary_anchor_nodes: Optional[Sequence[int]] = None,
) -> dict[str, Any]:
    if embeddings_before is None or embeddings_after is None:
        return {
            "embedding_l2_mean": None,
            "embedding_l2_median": None,
            "embedding_l2_max": None,
            "member_embedding_l2_mean": None,
            "neighbor_drift_mean": None,
            "neighbor_drift_max": None,
            "primary_anchor_drift_mean": None,
            "primary_anchor_drift_max": None,
            "secondary_anchor_drift_mean": None,
            "secondary_anchor_drift_max": None,
            "num_primary_anchor_nodes": 0,
            "num_secondary_anchor_nodes": 0,
            "status": "missing_embeddings",
        }

    before = _to_numpy(embeddings_before)
    after = _to_numpy(embeddings_after)
    n_rows = min(before.shape[0], after.shape[0])
    drift = np.linalg.norm(before[:n_rows] - after[:n_rows], axis=1)
    valid_members = [node for node in member_nodes if 0 <= node < n_rows]
    neighbor_nodes = sorted(
        {
            int(neighbor)
            for node in valid_members
            if node in graph_before
            for neighbor in graph_before.neighbors(node)
            if 0 <= int(neighbor) < n_rows
        }
    )
    primary_anchor_nodes = _valid_node_list(primary_anchor_nodes or [], n_rows)
    secondary_anchor_nodes = _valid_node_list(secondary_anchor_nodes or [], n_rows)
    return {
        "embedding_l2_mean": _safe_float(np.mean(drift)),
        "embedding_l2_median": _safe_float(np.median(drift)),
        "embedding_l2_max": _safe_float(np.max(drift)),
        "member_embedding_l2_mean": _safe_float(np.mean(drift[valid_members])) if valid_members else None,
        "neighbor_drift_mean": _safe_float(np.mean(drift[neighbor_nodes])) if neighbor_nodes else None,
        "neighbor_drift_max": _safe_float(np.max(drift[neighbor_nodes])) if neighbor_nodes else None,
        "primary_anchor_drift_mean": _safe_float(np.mean(drift[primary_anchor_nodes])) if primary_anchor_nodes else None,
        "primary_anchor_drift_max": _safe_float(np.max(drift[primary_anchor_nodes])) if primary_anchor_nodes else None,
        "secondary_anchor_drift_mean": _safe_float(np.mean(drift[secondary_anchor_nodes])) if secondary_anchor_nodes else None,
        "secondary_anchor_drift_max": _safe_float(np.max(drift[secondary_anchor_nodes])) if secondary_anchor_nodes else None,
        "num_member_nodes": len(valid_members),
        "num_neighbor_nodes": len(neighbor_nodes),
        "num_primary_anchor_nodes": len(primary_anchor_nodes),
        "num_secondary_anchor_nodes": len(secondary_anchor_nodes),
        "status": "ok",
    }


def structural_metrics(graph_before: nx.Graph, graph_after: nx.Graph) -> dict[str, Any]:
    before_cc = nx.average_clustering(graph_before) if graph_before.number_of_nodes() > 1 else 0.0
    after_cc = nx.average_clustering(graph_after) if graph_after.number_of_nodes() > 1 else 0.0
    before_components = nx.number_connected_components(graph_before) if graph_before.number_of_nodes() else 0
    after_components = nx.number_connected_components(graph_after) if graph_after.number_of_nodes() else 0
    return {
        "degree_kl_divergence": degree_kl_divergence(graph_before, graph_after),
        "clustering_coefficient_before": _safe_float(before_cc),
        "clustering_coefficient_after": _safe_float(after_cc),
        "clustering_coefficient_change": _safe_float(after_cc - before_cc),
        "component_count_before": int(before_components),
        "component_count_after": int(after_components),
        "component_count_change": int(after_components - before_components),
        "num_edges_before": graph_before.number_of_edges(),
        "num_edges_after": graph_after.number_of_edges(),
        "num_nodes_before": graph_before.number_of_nodes(),
        "num_nodes_after": graph_after.number_of_nodes(),
    }


def privacy_metrics(
    logits_before,
    logits_after,
    embeddings_before,
    embeddings_after,
    graph_before: nx.Graph,
    graph_after: nx.Graph,
    member_nodes: Sequence[int],
    non_member_nodes: Sequence[int],
    unlearning_type: str,
    labels=None,
) -> dict[str, Any]:
    base = {
        "weak_auc": None,
        "medium_auc": None,
        "strong_auc": None,
        "overall_mia_auc": None,
        "privacy_score": None,
        "num_members": len(member_nodes),
        "num_non_members": len(non_member_nodes),
        "feature_proxy": unlearning_type == "feature",
        "status": "ok",
    }
    if logits_before is None or logits_after is None:
        base["status"] = "missing_logits"
        return base
    if not member_nodes or not non_member_nodes:
        base["status"] = "missing_member_or_non_member_nodes"
        return base
    row_limit = min(_to_numpy(logits_before).shape[0], _to_numpy(logits_after).shape[0])
    member_nodes = [node for node in member_nodes if 0 <= int(node) < row_limit]
    non_member_nodes = [node for node in non_member_nodes if 0 <= int(node) < row_limit]
    if not member_nodes or not non_member_nodes:
        base["status"] = "member_or_non_member_nodes_out_of_logit_range"
        base["num_members"] = len(member_nodes)
        base["num_non_members"] = len(non_member_nodes)
        return base

    evaluator = PrivacyEvaluator()
    degree_before = dict(graph_before.degree())
    degree_after = dict(graph_after.degree())
    result = evaluator.evaluate_from_logits(
        logits_before,
        logits_after,
        member_nodes,
        non_member_nodes,
        embeddings_before=embeddings_before,
        embeddings_after=embeddings_after,
        degree_before=degree_before,
        degree_after=degree_after,
        graph_before=graph_before,
        graph_after=graph_after,
        labels=labels,
    ).as_dict()
    base.update(result)
    return base


def efficiency_metrics(
    unlearn_time_seconds: Optional[float],
    retrain_time_seconds: Optional[float],
    *,
    online_wall_clock_seconds: Optional[float] = None,
    time_breakdown: Optional[dict[str, float]] = None,
    offline_preprocessing_seconds: Optional[float] = None,
) -> dict[str, Any]:
    speedup = None
    if unlearn_time_seconds and retrain_time_seconds:
        speedup = retrain_time_seconds / unlearn_time_seconds if unlearn_time_seconds > 0 else None
    return {
        "unlearn_time_seconds": _safe_float(unlearn_time_seconds),
        "retrain_time_seconds": _safe_float(retrain_time_seconds),
        "speedup_vs_retrain": _safe_float(speedup),
        "online_wall_clock_seconds": _safe_float(online_wall_clock_seconds),
        "time_breakdown": {str(k): _safe_float(v) for k, v in (time_breakdown or {}).items()},
        "offline_preprocessing_seconds": _safe_float(offline_preprocessing_seconds),
    }


def rq_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    utility = metrics["utility"]
    structure = metrics["structure"]
    privacy = metrics["privacy"]
    efficiency = metrics["efficiency"]
    representation = metrics["representation"]
    return {
        "RQ1_overall_utility": {
            "accuracy_after": utility["accuracy_after"],
            "accuracy_drop": utility["accuracy_drop"],
            "f1_macro_after": utility["f1_macro_after"],
            "f1_macro_drop": utility["f1_macro_drop"],
            "speedup_vs_retrain": efficiency["speedup_vs_retrain"],
        },
        "RQ2_anchor_stabilization": {
            "embedding_l2_mean": representation["embedding_l2_mean"],
            "member_embedding_l2_mean": representation["member_embedding_l2_mean"],
            "neighbor_drift_mean": representation["neighbor_drift_mean"],
        },
        "RQ3_partition_structure": {
            "degree_kl_divergence": structure["degree_kl_divergence"],
            "clustering_coefficient_change": structure["clustering_coefficient_change"],
            "component_count_change": structure["component_count_change"],
        },
        "RQ4_privacy_dar": {
            "weak_auc": privacy["weak_auc"],
            "medium_auc": privacy["medium_auc"],
            "strong_auc": privacy["strong_auc"],
            "overall_mia_auc": privacy["overall_mia_auc"],
            "privacy_score": privacy["privacy_score"],
        },
        "RQ5_inpainting_privacy_structure": {
            "overall_mia_auc": privacy["overall_mia_auc"],
            "degree_kl_divergence": structure["degree_kl_divergence"],
            "clustering_coefficient_change": structure["clustering_coefficient_change"],
        },
        "RQ6_unlearning_type_scope": {
            "unlearning_type": metrics["unlearning_type"],
            "accuracy_after": utility["accuracy_after"],
            "f1_macro_after": utility["f1_macro_after"],
            "overall_mia_auc": privacy["overall_mia_auc"],
        },
    }


def graph_from_data(data) -> nx.Graph:
    graph = nx.Graph()
    if data is None:
        return graph
    num_nodes = int(getattr(data, "num_nodes", data.x.shape[0] if hasattr(data, "x") else 0))
    graph.add_nodes_from(range(num_nodes))
    if not hasattr(data, "edge_index"):
        return graph
    edge_index = data.edge_index.detach().cpu().tolist()
    for source, target in zip(edge_index[0], edge_index[1]):
        graph.add_edge(int(source), int(target))
    return graph


def _valid_node_list(nodes: Iterable[int], limit: int) -> list[int]:
    return sorted({int(node) for node in nodes if 0 <= int(node) < limit})


def infer_member_nodes(unlearning_type: str, forget_targets: Any, data, embeddings_before=None, embeddings_after=None) -> list[int]:
    num_nodes = int(getattr(data, "num_nodes", data.x.shape[0] if hasattr(data, "x") else 0))
    if unlearning_type == "node":
        return _valid_nodes([int(node) for node in forget_targets], num_nodes)
    if unlearning_type == "edge":
        return _valid_nodes([node for edge in forget_targets for node in edge], num_nodes)
    if embeddings_before is not None and embeddings_after is not None:
        before = _to_numpy(embeddings_before)
        after = _to_numpy(embeddings_after)
        n_rows = min(before.shape[0], after.shape[0], num_nodes)
        drift = np.linalg.norm(before[:n_rows] - after[:n_rows], axis=1)
        count = max(1, min(n_rows // 4, 256))
        return [int(idx) for idx in np.argsort(drift)[-count:]]
    return list(range(max(0, min(num_nodes, 1))))


def infer_non_member_nodes(data, member_nodes: Sequence[int], count: int) -> list[int]:
    num_nodes = int(getattr(data, "num_nodes", data.x.shape[0] if hasattr(data, "x") else 0))
    members = set(int(node) for node in member_nodes)
    candidates = [node for node in range(num_nodes) if node not in members]
    if count <= 0:
        return []
    return candidates[: min(len(candidates), count)]


def count_forget_targets(forget_targets: Any) -> int:
    if forget_targets is None:
        return 0
    if isinstance(forget_targets, (list, tuple, set)):
        return len(forget_targets)
    return 1


def degree_kl_divergence(graph_before: nx.Graph, graph_after: nx.Graph) -> Optional[float]:
    if graph_before.number_of_nodes() == 0 and graph_after.number_of_nodes() == 0:
        return 0.0
    before = _degree_distribution(graph_before)
    after = _degree_distribution(graph_after)
    max_len = max(len(before), len(after), 1)
    before = np.pad(before, (0, max_len - len(before)), constant_values=0.0)
    after = np.pad(after, (0, max_len - len(after)), constant_values=0.0)
    eps = 1e-12
    before = (before + eps) / np.sum(before + eps)
    after = (after + eps) / np.sum(after + eps)
    return _safe_float(np.sum(before * np.log(before / after)))


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(json_safe(item) for item in value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _safe_float(float(value))
    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())
    if hasattr(value, "as_dict"):
        return json_safe(value.as_dict())
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return json_safe(value.__dict__)
    if isinstance(value, float):
        return _safe_float(value)
    return value


def _classification_metrics(data, logits, mask_name: str) -> dict[str, Any]:
    if data is None or logits is None or not hasattr(data, "y"):
        return {"accuracy": None, "f1_micro": None, "f1_macro": None, "num_eval": 0}
    y = data.y
    if y.dim() > 1:
        y = y.squeeze(-1)
    y_np = y.detach().cpu().numpy().astype(int)
    pred_np = logits.detach().cpu().argmax(dim=-1).numpy().astype(int)
    n_rows = min(len(y_np), len(pred_np))
    mask = getattr(data, mask_name, None)
    if mask is None:
        mask_np = np.ones(n_rows, dtype=bool)
    else:
        if mask.dim() > 1:
            mask = mask[:, 0]
        mask_np = mask.detach().cpu().numpy().astype(bool)[:n_rows]
    y_np = y_np[:n_rows][mask_np]
    pred_np = pred_np[:n_rows][mask_np]
    if len(y_np) == 0:
        return {"accuracy": None, "f1_micro": None, "f1_macro": None, "num_eval": 0}
    accuracy = float(np.mean(pred_np == y_np))
    return {
        "accuracy": _safe_float(accuracy),
        "f1_micro": _safe_float(_f1_micro(y_np, pred_np)),
        "f1_macro": _safe_float(_f1_macro(y_np, pred_np)),
        "num_eval": int(len(y_np)),
    }


def _f1_micro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def _f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    scores = []
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(scores)) if scores else 0.0


def _degree_distribution(graph: nx.Graph) -> np.ndarray:
    degrees = [degree for _, degree in graph.degree()]
    if not degrees:
        return np.array([1.0])
    counts = np.bincount(np.array(degrees, dtype=int))
    return counts.astype(float)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def _valid_nodes(nodes: Iterable[int], num_nodes: int) -> list[int]:
    return sorted({int(node) for node in nodes if 0 <= int(node) < num_nodes})


def _drop(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if before is None or after is None:
        return None
    return _safe_float(before - after)


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value
