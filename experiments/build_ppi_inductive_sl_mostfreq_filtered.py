from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


SPLITS = ("train", "valid", "test")
CANONICAL_NAME = "ppi-inductive-sl-mostfreq-filtered"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PPI-Inductive-SL-MostFreq-MostFreq-Filtered from PyG PPI raw files.")
    parser.add_argument("--raw_dir", default=str(ROOT / "data" / "raw" / "Planetoid" / "PPI" / "raw"))
    parser.add_argument(
        "--output_dir",
        default=str(ROOT / "data" / "raw" / "Planetoid" / "PPI" / "processed_ppi_inductive_sl_mostfreq_filtered"),
    )
    parser.add_argument("--min_train_projected_class_count", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    data_path = output_dir / "data.pt"
    if data_path.exists() and not args.overwrite:
        raise SystemExit(f"{data_path} already exists. Pass --overwrite to rebuild.")

    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise SystemExit("torch-geometric is required to build PPI-Inductive-SL-MostFreq-MostFreq-Filtered.") from exc

    xs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    graph_ids: list[np.ndarray] = []
    split_names: list[np.ndarray] = []
    split_node_ids: list[np.ndarray] = []
    edge_parts: list[np.ndarray] = []
    source_stats: dict[str, Any] = {}
    node_offset = 0

    for split in SPLITS:
        x = np.load(raw_dir / f"{split}_feats.npy")
        y_multi = np.load(raw_dir / f"{split}_labels.npy")
        graph_id = np.load(raw_dir / f"{split}_graph_id.npy")
        graph = _read_json(raw_dir / f"{split}_graph.json")
        edges = np.array(
            [[int(link["source"]) + node_offset, int(link["target"]) + node_offset] for link in graph["links"]],
            dtype=np.int64,
        )
        if edges.size == 0:
            edges = np.empty((0, 2), dtype=np.int64)

        num_nodes = int(x.shape[0])
        xs.append(x.astype(np.float32, copy=False))
        labels.append(y_multi.astype(np.float32, copy=False))
        graph_ids.append(graph_id.astype(np.int64, copy=False))
        split_names.append(np.full(num_nodes, split, dtype=object))
        split_node_ids.append(np.arange(num_nodes, dtype=np.int64))
        edge_parts.append(edges)
        source_stats[split] = {
            "num_nodes": num_nodes,
            "num_edges": int(edges.shape[0]),
            "num_graphs": int(len(set(graph_id.tolist()))),
            "num_features": int(x.shape[1]),
            "label_dim": int(y_multi.shape[1]),
            "zero_label_nodes": int((y_multi.sum(axis=1) == 0).sum()),
        }
        node_offset += num_nodes

    x_all = np.concatenate(xs, axis=0)
    y_multi_all = np.concatenate(labels, axis=0)
    graph_id_all = np.concatenate(graph_ids, axis=0)
    split_name_all = np.concatenate(split_names, axis=0)
    split_node_id_all = np.concatenate(split_node_ids, axis=0)
    edge_index_all = np.concatenate(edge_parts, axis=0) if edge_parts else np.empty((0, 2), dtype=np.int64)

    train_source_mask = split_name_all == "train"
    positive_counts = y_multi_all.sum(axis=1)
    nonzero_label_mask = positive_counts > 0
    projection_fit_mask = train_source_mask & nonzero_label_mask
    original_label_freq_train = y_multi_all[projection_fit_mask].sum(axis=0)

    projected_original = np.full(y_multi_all.shape[0], -1, dtype=np.int64)
    nonzero_indices = np.flatnonzero(nonzero_label_mask)
    for node in nonzero_indices:
        positives = np.flatnonzero(y_multi_all[node] > 0)
        chosen_pos = positives[np.argmax(original_label_freq_train[positives])]
        projected_original[node] = int(chosen_pos)

    train_projected = projected_original[train_source_mask & (projected_original >= 0)]
    projected_train_counts = np.bincount(train_projected, minlength=y_multi_all.shape[1])
    retained_original_labels = np.flatnonzero(
        projected_train_counts >= int(args.min_train_projected_class_count)
    )
    retained_label_set = set(int(label) for label in retained_original_labels.tolist())
    final_node_mask = np.array([int(label) in retained_label_set for label in projected_original], dtype=bool)

    original_to_new = {int(label): idx for idx, label in enumerate(retained_original_labels.tolist())}
    final_indices = np.flatnonzero(final_node_mask)
    final_y = np.array([original_to_new[int(projected_original[node])] for node in final_indices], dtype=np.int64)
    final_x = x_all[final_indices]
    final_splits = split_name_all[final_indices]

    old_to_new = np.full(x_all.shape[0], -1, dtype=np.int64)
    old_to_new[final_indices] = np.arange(final_indices.shape[0], dtype=np.int64)
    edge_keep = final_node_mask[edge_index_all[:, 0]] & final_node_mask[edge_index_all[:, 1]]
    final_edges = old_to_new[edge_index_all[edge_keep]]
    if final_edges.size == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.from_numpy(final_edges.T).to(dtype=torch.long).contiguous()

    data = Data(
        x=torch.from_numpy(final_x).to(dtype=torch.float32),
        y=torch.from_numpy(final_y).to(dtype=torch.long),
        edge_index=edge_index,
    )
    data.num_nodes = int(final_x.shape[0])
    data.train_mask = torch.from_numpy(final_splits == "train").to(dtype=torch.bool)
    data.val_mask = torch.from_numpy(final_splits == "valid").to(dtype=torch.bool)
    data.test_mask = torch.from_numpy(final_splits == "test").to(dtype=torch.bool)

    final_counts = np.bincount(final_y, minlength=len(retained_original_labels))
    train_counts = np.bincount(final_y[data.train_mask.numpy()], minlength=len(retained_original_labels))
    val_counts = np.bincount(final_y[data.val_mask.numpy()], minlength=len(retained_original_labels))
    test_counts = np.bincount(final_y[data.test_mask.numpy()], minlength=len(retained_original_labels))
    isolated_count = _isolated_node_count(data.num_nodes, edge_index)
    mask_counts = {
        "train_mask": int(data.train_mask.sum().item()),
        "val_mask": int(data.val_mask.sum().item()),
        "test_mask": int(data.test_mask.sum().item()),
    }
    split_edge_counts = {
        "train_train": _edge_count_for_mask(edge_index, data.train_mask),
        "val_val": _edge_count_for_mask(edge_index, data.val_mask),
        "test_test": _edge_count_for_mask(edge_index, data.test_mask),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, data_path)

    node_mapping = [
        {
            "new_node_id": int(new_id),
            "source_split": str(split_name_all[old_id]),
            "source_node_id": int(split_node_id_all[old_id]),
            "source_graph_id": int(graph_id_all[old_id]),
            "projected_original_label": int(projected_original[old_id]),
            "new_label": int(final_y[new_id]),
        }
        for new_id, old_id in enumerate(final_indices.tolist())
    ]
    label_projection = {
        "rule": "most_frequent_positive",
        "fit_scope": "official_train_graphs",
        "tie_break": "lowest_original_label_id",
        "original_label_dim": int(y_multi_all.shape[1]),
        "original_label_frequencies_train": [int(value) for value in original_label_freq_train.tolist()],
        "projected_class_counts_train_before_filter": {
            str(idx): int(count) for idx, count in enumerate(projected_train_counts.tolist()) if int(count) > 0
        },
        "retained_original_labels": [int(label) for label in retained_original_labels.tolist()],
        "original_to_new_label": {str(label): int(new) for label, new in original_to_new.items()},
    }
    class_counts = {
        "min_train_projected_class_count": int(args.min_train_projected_class_count),
        "num_classes": int(len(retained_original_labels)),
        "classes": [
            {
                "new_label": int(new_label),
                "original_label": int(original_label),
                "count": int(final_counts[new_label]),
                "train_count": int(train_counts[new_label]),
                "val_count": int(val_counts[new_label]),
                "test_count": int(test_counts[new_label]),
            }
            for new_label, original_label in enumerate(retained_original_labels.tolist())
        ],
    }
    metadata = {
        "dataset": CANONICAL_NAME,
        "display_name": "PPI-Inductive-SL-MostFreq",
        "source": "PyG PPI raw files",
        "construction": "official_split_disjoint_union_graphs",
        "task": "single-label node classification",
        "split": "official_ppi_train_valid_test_graphs",
        "label_projection": "most_frequent_positive",
        "label_projection_fit_scope": "official_train_graphs",
        "min_train_projected_class_count": int(args.min_train_projected_class_count),
        "num_source_nodes": int(x_all.shape[0]),
        "num_source_edges": int(edge_index_all.shape[0]),
        "num_source_graphs": int(sum(stats["num_graphs"] for stats in source_stats.values())),
        "removed_zero_label_nodes": int((~nonzero_label_mask).sum()),
        "removed_low_count_projected_label_nodes": int(nonzero_label_mask.sum() - final_node_mask.sum()),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(data.edge_index.shape[1]),
        "num_features": int(data.x.shape[1]),
        "num_classes": int(len(retained_original_labels)),
        "isolated_nodes": int(isolated_count),
        "mask_counts": mask_counts,
        "split_edge_counts": split_edge_counts,
        "source_splits": source_stats,
        "files": {
            "data": "data.pt",
            "metadata": "metadata.json",
            "class_counts": "class_counts.json",
            "label_projection": "label_projection.json",
            "node_mapping": "node_mapping.json",
        },
    }
    _write_json(output_dir / "metadata.json", metadata)
    _write_json(output_dir / "class_counts.json", class_counts)
    _write_json(output_dir / "label_projection.json", label_projection)
    _write_json(output_dir / "node_mapping.json", node_mapping)

    print(json.dumps(metadata, indent=2))


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _isolated_node_count(num_nodes: int, edge_index: torch.Tensor) -> int:
    if num_nodes <= 0:
        return 0
    if edge_index.numel() == 0:
        return int(num_nodes)
    degree = torch.zeros(num_nodes, dtype=torch.long)
    degree.scatter_add_(0, edge_index[0].cpu(), torch.ones(edge_index.shape[1], dtype=torch.long))
    degree.scatter_add_(0, edge_index[1].cpu(), torch.ones(edge_index.shape[1], dtype=torch.long))
    return int((degree == 0).sum().item())


def _edge_count_for_mask(edge_index: torch.Tensor, mask: torch.Tensor) -> int:
    if edge_index.numel() == 0:
        return 0
    mask_cpu = mask.detach().cpu().to(dtype=torch.bool)
    edge_index_cpu = edge_index.detach().cpu()
    return int((mask_cpu[edge_index_cpu[0]] & mask_cpu[edge_index_cpu[1]]).sum().item())


if __name__ == "__main__":
    main()
