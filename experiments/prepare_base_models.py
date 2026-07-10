from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import apply_stratified_split, induced_subgraph_from_mask, load_dataset
from evaluation.metrics import json_safe
from models import GNNTrainer, TrainingConfig, build_gnn_model, default_base_artifact_dir, save_base_artifact


def parse_args():
    parser = argparse.ArgumentParser(description="Train one shared base model per dataset + seed.")
    parser.add_argument("--datasets", default="cora,citeseer,pubmed", help="Comma-separated dataset names.")
    parser.add_argument("--seeds", default="42,123,2024", help="Comma-separated training seeds.")
    parser.add_argument("--data_root", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--allow_download", action="store_true", help="Allow this command to download missing datasets.")
    parser.add_argument("--output_root", default=str(ROOT / "results" / "shared_base"))
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing artifact directories.")
    parser.add_argument("--model_type", default="GCN", choices=["GCN", "GAT", "GraphSAGE"])
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--train_epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--split", default="stratified_random", choices=["planetoid", "dataset", "stratified_random"])
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--training_graph", default="train_subgraph", choices=["full", "train_subgraph"])
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    summaries = []

    for dataset_name in datasets:
        bundle = load_dataset(dataset_name, args.data_root, download=args.allow_download)
        for seed in seeds:
            data = bundle.data
            if args.split == "stratified_random":
                data = apply_stratified_split(
                    data,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio,
                    seed=seed,
                )
            artifact_dir = default_base_artifact_dir(args.output_root, bundle.name, seed)
            metadata_path = artifact_dir / "metadata.json"
            if metadata_path.exists() and not args.overwrite:
                summaries.append(
                    {
                        "dataset": bundle.name,
                        "seed": seed,
                        "artifact_dir": str(artifact_dir),
                        "status": "skipped_existing",
                    }
                )
                continue

            _set_training_seed(seed)
            model = build_gnn_model(
                args.model_type,
                in_channels=bundle.num_features,
                hidden_channels=args.hidden_channels,
                out_channels=bundle.num_classes,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
            trainer = GNNTrainer(
                model,
                TrainingConfig(lr=args.lr, weight_decay=args.weight_decay, epochs=args.train_epochs, device=args.device),
            )
            train_data = induced_subgraph_from_mask(data, "train_mask") if args.training_graph == "train_subgraph" else data
            training_result = trainer.train_full_batch(train_data, epochs=args.train_epochs)
            model_config = {
                "type": args.model_type,
                "hidden_channels": args.hidden_channels,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "in_channels": bundle.num_features,
                "out_channels": bundle.num_classes,
            }
            training_config = {
                "epochs": args.train_epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": seed,
                "device": args.device,
                "split": args.split,
                "train_ratio": args.train_ratio if args.split == "stratified_random" else None,
                "val_ratio": args.val_ratio if args.split == "stratified_random" else None,
                "test_ratio": args.test_ratio if args.split == "stratified_random" else None,
                "training_graph": args.training_graph,
                "training_graph_num_nodes": int(train_data.num_nodes),
                "training_graph_num_edges": int(train_data.edge_index.shape[1]) if hasattr(train_data, "edge_index") else None,
            }
            metadata = save_base_artifact(
                artifact_dir,
                trainer=trainer,
                data=data,
                dataset_name=bundle.name,
                seed=seed,
                model_config=model_config,
                training_config=training_config,
                training_result=training_result,
            )
            summaries.append(
                {
                    "dataset": bundle.name,
                    "seed": seed,
                    "artifact_dir": str(artifact_dir),
                    "status": "trained",
                    "training_graph": args.training_graph,
                    "training_graph_num_nodes": int(train_data.num_nodes),
                    "training_graph_num_edges": int(train_data.edge_index.shape[1]) if hasattr(train_data, "edge_index") else None,
                    "base_training": metadata["base_training"],
                }
            )

    print(json.dumps(json_safe({"artifacts": summaries}), indent=2))


def _set_training_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
