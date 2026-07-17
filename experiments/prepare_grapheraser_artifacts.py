from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines import get_baseline
from data import apply_stratified_split, load_dataset
from evaluation.metrics import json_safe
from models import build_gnn_model, default_base_artifact_dir
from utils import RuntimeTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare offline GraphEraser partition and shard-model artifacts.")
    parser.add_argument("--dataset_name", default="pubmed", choices=["cora", "citeseer", "pubmed", "primekg", "primekg-homo", "primekg-full-nosource", "primekg-disease-gene-small", "primekg-disease-gene-small-nosource", "hetionet-small-nosource", "hetionet-full-nosource", "ppi-homo-sl-filtered", "ppi-inductive-sl-filtered", "ppi-inductive-sl-mostfreq-filtered", "ppi-inductive-sl-balanced20-filtered", "ppi-inductive-sl-balanced10-filtered", "reddit"])
    parser.add_argument("--unlearning_type", default="node", choices=["node", "edge", "feature"])
    parser.add_argument("--baselines", default="grapheraser-bekm,grapheraser-blpa")
    parser.add_argument("--seeds", default="42,123,2024")
    parser.add_argument("--data_root", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--output_root", default=str(ROOT / "results"))
    parser.add_argument("--output_dataset_name", default="", help="Override the dataset directory name under --output_root, e.g. pubmed_eval.")
    parser.add_argument(
        "--experiment_name",
        default="default_main",
        help="Experiment directory between dataset and baselines. Use an empty string or __root__ to put baselines directly under the dataset directory.",
    )
    parser.add_argument("--base_artifact_root", default=str(ROOT / "results" / "shared_base"))
    parser.add_argument("--model_type", default="GCN", choices=["GCN", "GAT", "GraphSAGE"])
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--train_epochs", type=int, default=200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baselines = _parse_strings(args.baselines)
    seeds = [int(item) for item in _parse_strings(args.seeds)]
    summaries = []

    bundle = load_dataset(args.dataset_name, args.data_root, download=args.allow_download)
    for seed in seeds:
        _set_training_seed(seed)
        base_artifact_dir = default_base_artifact_dir(args.base_artifact_root, bundle.name, seed)
        base_metadata = _load_base_metadata(base_artifact_dir)
        data = _apply_base_artifact_split(bundle.data, base_metadata, seed)
        model_config = _model_config(args, bundle)

        def model_factory(_data):
            return build_gnn_model(
                args.model_type,
                in_channels=bundle.num_features,
                hidden_channels=args.hidden_channels,
                out_channels=bundle.num_classes,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )

        for baseline_name in baselines:
            baseline = get_baseline(baseline_name, ROOT)
            if not hasattr(baseline, "prepare_artifact"):
                raise ValueError(f"Baseline does not support GraphEraser artifacts: {baseline_name}")
            if hasattr(baseline, "seed"):
                baseline.seed = int(seed)
            output_dataset = args.output_dataset_name or bundle.name
            artifact_dir = _artifact_dir(
                args.output_root,
                output_dataset,
                args.experiment_name,
                baseline_name,
                args.unlearning_type,
                seed,
            )
            metadata_path = artifact_dir / "metadata.json"
            if metadata_path.exists() and not args.overwrite:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                summaries.append(
                    {
                        "dataset": bundle.name,
                        "baseline": baseline_name,
                        "unlearning_type": args.unlearning_type,
                        "seed": seed,
                        "artifact_dir": str(artifact_dir),
                        "status": "skipped_existing",
                        "offline_preprocessing_seconds": metadata.get("offline_preprocessing_seconds"),
                    }
                )
                continue

            timer = RuntimeTracker()
            timer.start_total()
            with timer.track("offline_partition_and_train_shards"):
                metadata = baseline.prepare_artifact(
                    data,
                    model_factory,
                    artifact_dir,
                    dataset_name=bundle.name,
                    seed=seed,
                    unlearning_type=args.unlearning_type,
                    model_config=model_config,
                    base_artifact={"path": str(base_artifact_dir), "training": base_metadata.get("training")},
                    epochs=args.train_epochs,
                    device=args.device,
                    overwrite=True,
                )
            timer.end_total()
            metadata["offline_preprocessing_seconds"] = timer.total
            metadata["offline_time_breakdown"] = timer.times
            (artifact_dir / "metadata.json").write_text(json.dumps(json_safe(metadata), indent=2) + "\n", encoding="utf-8")
            summaries.append(
                {
                    "dataset": bundle.name,
                    "baseline": baseline_name,
                    "unlearning_type": args.unlearning_type,
                    "seed": seed,
                    "artifact_dir": str(artifact_dir),
                    "status": "prepared",
                    "offline_preprocessing_seconds": timer.total,
                    "num_shards": metadata.get("num_shards"),
                }
            )
            print(json.dumps(json_safe(summaries[-1])), flush=True)

    print(json.dumps(json_safe({"artifacts": summaries}), indent=2))


def _artifact_dir(
    output_root: str | Path,
    dataset: str,
    experiment_name: str,
    baseline: str,
    unlearning_type: str,
    seed: int,
) -> Path:
    key = baseline.replace("-", "_")
    root = Path(output_root) / dataset
    if experiment_name and experiment_name != "__root__":
        root = root / experiment_name
    return root / "baselines" / key / unlearning_type / "artifacts" / f"seed{seed}"


def _model_config(args, bundle) -> dict[str, Any]:
    return {
        "type": args.model_type,
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "in_channels": bundle.num_features,
        "out_channels": bundle.num_classes,
    }


def _load_base_metadata(base_artifact_dir: Path) -> dict[str, Any]:
    metadata_path = base_artifact_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing shared base metadata: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _apply_base_artifact_split(data, metadata: dict[str, Any] | None, seed: int):
    training = (metadata or {}).get("training", {})
    if training.get("split") != "stratified_random":
        return data
    return apply_stratified_split(
        data,
        train_ratio=float(training.get("train_ratio", 0.6)),
        val_ratio=float(training.get("val_ratio", 0.2)),
        test_ratio=float(training.get("test_ratio", 0.2)),
        seed=int(training.get("seed", seed)),
    )


def _parse_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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
