from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import (
    apply_stratified_split,
    default_forget_set_path,
    load_dataset,
    save_forget_set,
    select_forget_edges,
    select_forget_features,
    select_forget_nodes,
)
from evaluation.metrics import graph_from_data, json_safe
from hasi.hub_identification import HubScorer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate fixed forget-set protocol files.")
    parser.add_argument("--dataset_name", default="cora", choices=["cora", "citeseer", "pubmed", "primekg", "primekg-homo", "primekg-disease-gene-small", "primekg-disease-gene-small-nosource", "hetionet-small-nosource", "ppi-homo-sl-filtered", "ppi-inductive-sl-filtered", "ppi-inductive-sl-mostfreq-filtered", "ppi-inductive-sl-balanced20-filtered", "ppi-inductive-sl-balanced10-filtered", "reddit"])
    parser.add_argument("--unlearning_type", default="node", choices=["node", "edge", "feature"])
    parser.add_argument("--forget_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection",
        default="random_train",
        choices=[
            "random_train",
            "random_all",
            "hub",
            "low_degree",
            "hub_train",
            "low_degree_train",
            "hub_neighbor_train",
        ],
        help="Node selection policy. Edge/feature generation uses random sampling with a recorded scope.",
    )
    parser.add_argument(
        "--split_source",
        default="shared_base",
        choices=["shared_base", "dataset"],
        help="Use shared_base metadata to reconstruct the base-model split, or keep dataset-provided masks.",
    )
    parser.add_argument("--base_artifact_root", default=str(ROOT / "results" / "shared_base"))
    parser.add_argument("--base_artifact_dir", default="", help="Explicit shared base artifact directory for this dataset/seed.")
    parser.add_argument(
        "--edge_scope",
        default="train_subgraph",
        choices=["train_subgraph", "full"],
        help="For edge unlearning, sample from train-train edges or the full edge_index.",
    )
    parser.add_argument("--data_root", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--allow_download", action="store_true", help="Allow this command to download missing datasets.")
    parser.add_argument("--output", default=None, help="Defaults to experiments/forget_sets/<dataset>/<dataset>_<type>_*.json.")
    return parser.parse_args()


def main():
    args = parse_args()
    bundle = load_dataset(args.dataset_name, args.data_root, download=args.allow_download)
    data, protocol = _prepare_protocol_data(bundle.data, args, bundle.name)
    targets, target_protocol = _select_targets(data, args)
    protocol.update(target_protocol)
    output_path = Path(args.output) if args.output else _default_output_path(
        ROOT,
        ROOT / "experiments" / "forget_sets",
        dataset=bundle.name,
        unlearning_type=args.unlearning_type,
        ratio=args.forget_ratio,
        seed=args.seed,
        selection=args.selection,
    )
    saved = save_forget_set(
        output_path,
        dataset=bundle.name,
        unlearning_type=args.unlearning_type,
        ratio=args.forget_ratio,
        seed=args.seed,
        selection=args.selection,
        targets=targets,
        protocol_metadata=protocol,
    )
    result = {
        "path": str(saved),
        "dataset": bundle.name,
        "unlearning_type": args.unlearning_type,
        "forget_ratio": args.forget_ratio,
        "seed": args.seed,
        "selection": args.selection,
        "forget_count": len(targets),
        "targets_preview": targets[:10],
        "protocol": protocol,
    }
    print(json.dumps(json_safe(result), indent=2))


def _prepare_protocol_data(data, args, dataset_name: str):
    if args.split_source == "dataset":
        return data, {
            "split_source": "dataset",
            "mask_counts": _mask_counts(data),
        }

    artifact_dir = _resolve_base_artifact_dir(args, dataset_name)
    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Shared base metadata not found: {metadata_path}. "
            "Run experiments/prepare_base_models.py first, or pass --split_source dataset."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    training = metadata.get("training", {})
    if training.get("split") == "stratified_random":
        data = apply_stratified_split(
            data,
            train_ratio=float(training.get("train_ratio", 0.6)),
            val_ratio=float(training.get("val_ratio", 0.2)),
            test_ratio=float(training.get("test_ratio", 0.2)),
            seed=int(training.get("seed", args.seed)),
        )

    actual_counts = _mask_counts(data)
    expected_counts = metadata.get("mask_counts") or {}
    mismatches = {
        name: {"expected": int(expected), "actual": int(actual_counts.get(name, -1))}
        for name, expected in expected_counts.items()
        if name in actual_counts and int(expected) != int(actual_counts[name])
    }
    if mismatches:
        raise ValueError(f"Reconstructed split does not match shared_base metadata: {mismatches}")

    return data, {
        "split_source": "shared_base",
        "base_artifact_dir": str(artifact_dir),
        "base_metadata_path": str(metadata_path),
        "base_training_graph": training.get("training_graph"),
        "split": training.get("split"),
        "split_seed": int(training.get("seed", args.seed)),
        "train_ratio": float(training.get("train_ratio", 0.6)),
        "val_ratio": float(training.get("val_ratio", 0.2)),
        "test_ratio": float(training.get("test_ratio", 0.2)),
        "mask_counts": actual_counts,
        "expected_mask_counts": expected_counts,
    }


def _select_targets(data, args):
    if args.unlearning_type == "node":
        if args.selection == "random_train":
            targets = select_forget_nodes(data, args.forget_ratio, seed=args.seed, mask_name="train_mask")
            protocol = _target_protocol("train_mask_nodes", _mask_count(data, "train_mask"))
            protocol.update(
                {
                    "candidate_scope": "train_mask",
                    "score_graph": "train_subgraph",
                    "ranking_metric": "random_uniform",
                    "train_subgraph_edge_count": _train_subgraph_edge_count(data),
                }
            )
            return targets, protocol
        if args.selection == "random_all":
            targets = select_forget_nodes(data, args.forget_ratio, seed=args.seed, mask_name="")
            return targets, _target_protocol("all_nodes", int(data.num_nodes))
        if args.selection == "hub_train":
            return _ranked_train_nodes(data, args.forget_ratio, ranking="hub_score")
        if args.selection == "low_degree_train":
            return _ranked_train_nodes(data, args.forget_ratio, ranking="low_degree")
        if args.selection == "hub_neighbor_train":
            return _hub_neighbor_train_nodes(data, args.forget_ratio)
        if args.selection == "hub":
            targets = _ranked_nodes(data, args.forget_ratio, reverse=True)
            return targets, _target_protocol("ranked_all_nodes_hub", int(data.num_nodes))
        if args.selection == "low_degree":
            targets = _ranked_nodes(data, args.forget_ratio, reverse=False)
            return targets, _target_protocol("ranked_all_nodes_low_degree", int(data.num_nodes))

    if args.selection not in {"random_train", "random_all"}:
        raise ValueError(f"Selection {args.selection!r} is only supported for node forget sets.")
    if args.unlearning_type == "edge":
        if args.edge_scope == "train_subgraph":
            targets, candidate_count = _select_train_subgraph_edges(data, args.forget_ratio, seed=args.seed)
            return targets, _target_protocol("train_subgraph_edges", candidate_count)
        targets = select_forget_edges(data, args.forget_ratio, seed=args.seed)
        return targets, _target_protocol("full_edge_index", int(data.edge_index.shape[1]))

    targets = select_forget_features(data, args.forget_ratio, seed=args.seed)
    return targets, _target_protocol("feature_dimensions", int(data.x.shape[1]))


def _select_train_subgraph_edges(data, ratio: float, *, seed: int) -> tuple[list[tuple[int, int]], int]:
    import torch

    train_mask = getattr(data, "train_mask", None)
    if train_mask is None:
        raise ValueError("train_subgraph edge selection requires data.train_mask.")
    if train_mask.dim() > 1:
        train_mask = train_mask[:, 0]
    train_mask = train_mask.detach().cpu().to(dtype=torch.bool)
    edge_index = data.edge_index.detach().cpu()
    keep = train_mask[edge_index[0]] & train_mask[edge_index[1]]
    candidates = torch.nonzero(keep, as_tuple=False).view(-1)
    candidate_count = int(candidates.numel())
    count = _sample_count(candidate_count, ratio)
    if count <= 0:
        return [], candidate_count
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    order = torch.randperm(candidate_count, generator=generator)[:count]
    selected = candidates[order].tolist()
    return [(int(edge_index[0, idx]), int(edge_index[1, idx])) for idx in selected], candidate_count


def _ranked_train_nodes(data, ratio: float, *, ranking: str) -> tuple[list[int], dict[str, Any]]:
    graph, train_nodes, train_edge_count = _train_subgraph_graph(data)
    candidate_count = len(train_nodes)
    count = _sample_count(candidate_count, ratio)
    if count <= 0:
        return [], _target_protocol("train_mask_nodes", candidate_count)

    if ranking == "hub_score":
        scores = HubScorer().compute_hub_scores(graph, candidate_nodes=train_nodes)
        ranked = sorted(
            train_nodes,
            key=lambda node: (-float(scores.get(node, 0.0)), -int(graph.degree(node)), int(node)),
        )
        metric = "hasi_hub_score_desc"
    elif ranking == "low_degree":
        ranked = sorted(train_nodes, key=lambda node: (int(graph.degree(node)), int(node)))
        metric = "degree_asc"
    else:
        raise ValueError(f"Unsupported train ranking: {ranking}")

    protocol = _target_protocol("train_mask_nodes", candidate_count)
    protocol.update(
        {
            "candidate_scope": "train_mask",
            "score_graph": "train_subgraph",
            "ranking_metric": metric,
            "train_subgraph_edge_count": int(train_edge_count),
        }
    )
    return [int(node) for node in ranked[:count]], protocol


def _hub_neighbor_train_nodes(data, ratio: float) -> tuple[list[int], dict[str, Any]]:
    graph, train_nodes, train_edge_count = _train_subgraph_graph(data)
    train_count = len(train_nodes)
    count = _sample_count(train_count, ratio)
    if count <= 0:
        return [], _target_protocol("train_mask_hub_neighbors", 0)

    scorer = HubScorer()
    scores = scorer.compute_hub_scores(graph, candidate_nodes=train_nodes)
    ranked_hubs = sorted(
        train_nodes,
        key=lambda node: (-float(scores.get(node, 0.0)), -int(graph.degree(node)), int(node)),
    )
    protected_hub_count = max(1, int(round(train_count * scorer.config.secondary_ratio)))
    protected_hubs = set(ranked_hubs[:protected_hub_count])

    candidate_info: dict[int, tuple[float, int, int, int]] = {}
    hub_center_count = 0
    for center_rank, center in enumerate(ranked_hubs):
        if center not in graph:
            continue
        hub_center_count += 1
        center_score = float(scores.get(center, 0.0))
        center_degree = int(graph.degree(center))
        for neighbor in graph.neighbors(center):
            node = int(neighbor)
            if node in protected_hubs:
                continue
            candidate = (center_score, center_degree, -center_rank, -int(center))
            previous = candidate_info.get(node)
            if previous is None or candidate > previous:
                candidate_info[node] = candidate
        if hub_center_count >= protected_hub_count and len(candidate_info) >= count:
            break

    if len(candidate_info) < count:
        raise ValueError(
            "Not enough train-subgraph hub-neighbor candidates: "
            f"requested {count}, found {len(candidate_info)}."
        )

    ranked_neighbors = sorted(
        candidate_info,
        key=lambda node: (
            -candidate_info[node][0],
            -candidate_info[node][1],
            candidate_info[node][2],
            -int(graph.degree(node)),
            int(node),
        ),
    )
    protocol = _target_protocol("train_mask_hub_neighbors", len(candidate_info))
    protocol.update(
        {
            "candidate_scope": "train_mask_hub_neighbors",
            "score_graph": "train_subgraph",
            "ranking_metric": "neighbor_of_hasi_hub_score_desc",
            "train_candidate_count": train_count,
            "requested_count": count,
            "protected_hub_count": protected_hub_count,
            "hub_center_count": hub_center_count,
            "train_subgraph_edge_count": int(train_edge_count),
        }
    )
    return [int(node) for node in ranked_neighbors[:count]], protocol


def _train_subgraph_graph(data):
    import networkx as nx
    import torch

    train_mask = getattr(data, "train_mask", None)
    if train_mask is None:
        raise ValueError("train-ranked node selection requires data.train_mask.")
    if train_mask.dim() > 1:
        train_mask = train_mask[:, 0]
    train_mask = train_mask.detach().cpu().to(dtype=torch.bool)
    train_nodes = torch.nonzero(train_mask, as_tuple=False).view(-1).tolist()
    train_nodes = [int(node) for node in train_nodes]

    graph = nx.Graph()
    graph.add_nodes_from(train_nodes)
    edge_index = data.edge_index.detach().cpu()
    keep = train_mask[edge_index[0]] & train_mask[edge_index[1]]
    kept_edges = torch.nonzero(keep, as_tuple=False).view(-1).tolist()
    for idx in kept_edges:
        source = int(edge_index[0, idx])
        target = int(edge_index[1, idx])
        graph.add_edge(source, target)
    return graph, train_nodes, len(kept_edges)


def _train_subgraph_edge_count(data) -> int:
    import torch

    train_mask = getattr(data, "train_mask", None)
    if train_mask is None:
        raise ValueError("train-subgraph metadata requires data.train_mask.")
    if train_mask.dim() > 1:
        train_mask = train_mask[:, 0]
    train_mask = train_mask.detach().cpu().to(dtype=torch.bool)
    edge_index = data.edge_index.detach().cpu()
    keep = train_mask[edge_index[0]] & train_mask[edge_index[1]]
    return int(torch.count_nonzero(keep).item())


def _ranked_nodes(data, ratio: float, *, reverse: bool) -> list[int]:
    graph = graph_from_data(data)
    count = _sample_count(graph.number_of_nodes(), ratio)
    if count <= 0:
        return []
    if reverse:
        scores = HubScorer().compute_hub_scores(graph)
        ranked = sorted(graph.nodes, key=lambda node: scores.get(node, 0.0), reverse=True)
    else:
        ranked = sorted(graph.nodes, key=lambda node: (graph.degree(node), int(node)))
    return [int(node) for node in ranked[:count]]


def _default_output_path(
    root: Path,
    output_dir: Path,
    *,
    dataset: str,
    unlearning_type: str,
    ratio: float,
    seed: int,
    selection: str,
) -> Path:
    filename = default_forget_set_path(
        root,
        dataset=dataset,
        unlearning_type=unlearning_type,
        ratio=ratio,
        seed=seed,
        selection=selection,
    ).name
    return Path(output_dir) / dataset / filename


def _resolve_base_artifact_dir(args, dataset_name: str) -> Path:
    if getattr(args, "base_artifact_dir", ""):
        return Path(args.base_artifact_dir)
    return Path(args.base_artifact_root) / dataset_name / f"seed{int(args.seed)}"


def _target_protocol(selection_scope: str, candidate_count: int) -> dict[str, Any]:
    return {
        "selection_scope": selection_scope,
        "candidate_count": int(candidate_count),
    }


def _mask_counts(data) -> dict[str, int]:
    return {name: _mask_count(data, name) for name in ("train_mask", "val_mask", "test_mask") if hasattr(data, name)}


def _mask_count(data, name: str) -> int:
    mask = getattr(data, name, None)
    if mask is None:
        return 0
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu()
    if getattr(mask, "dim", lambda: 1)() > 1:
        mask = mask[:, 0]
    return int(mask.to(dtype=__import__("torch").bool).sum().item())


def _sample_count(total: int, ratio: float) -> int:
    if total <= 0 or ratio <= 0:
        return 0
    if ratio >= 1 and float(ratio).is_integer():
        return min(total, int(ratio))
    return max(1, min(total, int(round(total * ratio))))


if __name__ == "__main__":
    main()
