from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from data import apply_stratified_split, load_dataset, load_forget_set, parse_forget_targets
from evaluation import build_experiment_metrics, save_metrics
from evaluation.metrics import json_safe
from models import GNNTrainer, TrainingConfig, build_gnn_model, default_base_artifact_dir, load_base_artifact
from utils import RuntimeTracker

from .baseline_adapters import ADAPTED_BASELINES, default_artifact_dir, get_adapted_baseline, method_key


LOCAL_ROOT = REPO_ROOT / "opengu_adapted_baselines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenGU-adapted baselines under the local HASI protocol.")
    parser.add_argument("--baseline", required=True, choices=ADAPTED_BASELINES)
    parser.add_argument("--dataset_name", default="pubmed")
    parser.add_argument("--unlearning_type", required=True, choices=["node", "edge", "feature"])
    parser.add_argument("--forget_set_file", default="", help="JSON or text forget-set protocol file.")
    parser.add_argument("--forget_ratio", type=float, default=0.1)
    parser.add_argument("--forget_nodes", default="")
    parser.add_argument("--forget_edges", default="")
    parser.add_argument("--forget_features", default="")
    parser.add_argument("--data_root", default=str(REPO_ROOT / "data" / "raw"))
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--model_type", default="GCN", choices=["GCN", "GAT", "GraphSAGE"])
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--train_epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42, help="Shared-base/model seed.")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", default=None)
    parser.add_argument("--base_artifact_root", default=str(REPO_ROOT / "results" / "shared_base"))
    parser.add_argument("--base_artifact_dir", default="")
    parser.add_argument("--experiment_name", default="default_main")
    parser.add_argument("--artifact_root", default="", help="Artifact root. Defaults to --output_root; legacy opengu_adapted_baselines/artifacts roots are redirected there.")
    parser.add_argument("--output_root", default=str(LOCAL_ROOT / "results"))
    parser.add_argument("--output", default="")
    parser.add_argument("--rebuild_artifact", action="store_true")
    parser.add_argument("--prepare_artifact_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    _set_training_seed(args.seed)

    timer = RuntimeTracker()
    timer.start_total()

    with timer.track("load_dataset"):
        bundle = load_dataset(args.dataset_name, args.data_root, download=args.allow_download)
        data = bundle.data

    with timer.track("prepare_model"):
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

        model = model_factory(data)
        trainer = GNNTrainer(model, TrainingConfig(lr=args.lr, epochs=args.train_epochs, device=args.device))
        baseline = get_adapted_baseline(args.baseline)
        if hasattr(baseline, "seed"):
            baseline.seed = int(args.seed)

    base_artifact_dir = _resolve_base_artifact_dir(args, bundle.name)
    base_artifact_metadata = None
    with timer.track("load_base"):
        if base_artifact_dir is not None:
            base_artifact_metadata, logits_before, embeddings_before = load_base_artifact(
                base_artifact_dir,
                trainer=trainer,
                dataset_name=bundle.name,
                seed=args.seed,
                model_config=model_config,
            )
            base_training = dict(base_artifact_metadata.get("base_training", {}))
        else:
            base_training = trainer.train_full_batch(data, epochs=args.train_epochs).as_dict()
            logits_before, embeddings_before = trainer.predict_with_embeddings(data)

    with timer.track("reconstruct_split"):
        if base_artifact_metadata is not None:
            data = _apply_base_artifact_split(data, base_artifact_metadata, args.seed)

    graph_artifact_dir = None
    if hasattr(baseline, "prepare_artifact"):
        graph_artifact_dir = default_artifact_dir(
            _resolve_artifact_root(args),
            dataset=bundle.name,
            experiment_name=args.experiment_name,
            method=baseline.name,
            unlearning_type=args.unlearning_type,
            seed=args.seed,
        )
        if args.rebuild_artifact or not (graph_artifact_dir / "metadata.json").exists():
            with timer.track("prepare_grapheraser_artifact"):
                metadata = baseline.prepare_artifact(
                    data,
                    model_factory,
                    graph_artifact_dir,
                    dataset_name=bundle.name,
                    seed=args.seed,
                    unlearning_type=args.unlearning_type,
                    model_config=model_config,
                    base_artifact={"path": str(base_artifact_dir), "training": (base_artifact_metadata or {}).get("training")},
                    epochs=args.train_epochs,
                    device=args.device,
                    overwrite=True,
                )
                metadata["offline_preprocessing_seconds"] = timer.times.get("prepare_grapheraser_artifact")
                (graph_artifact_dir / "metadata.json").write_text(json.dumps(json_safe(metadata), indent=2) + "\n", encoding="utf-8")
        if args.prepare_artifact_only:
            timer.end_total()
            print(json.dumps(json_safe({"artifact_dir": str(graph_artifact_dir), "time": timer.to_dict()}), indent=2))
            return

    with timer.track("load_forget_set"):
        forget_targets, forget_set_info = _resolve_forget_targets(args, data, bundle.name)

    previous_unlearn_time = timer.times.get("unlearn_or_retrain", 0.0)
    with timer.track("unlearn_or_retrain"):
        if args.unlearning_type == "node":
            result = baseline.run_node_unlearning(
                data,
                forget_targets,
                trainer=trainer,
                model_factory=model_factory,
                epochs=args.train_epochs,
                artifact_dir=graph_artifact_dir,
                device=args.device,
            )
        elif args.unlearning_type == "edge":
            result = baseline.run_edge_unlearning(
                data,
                forget_targets,
                trainer=trainer,
                model_factory=model_factory,
                epochs=args.train_epochs,
                artifact_dir=graph_artifact_dir,
                device=args.device,
            )
        else:
            result = baseline.run_feature_unlearning(
                data,
                forget_targets,
                trainer=trainer,
                model_factory=model_factory,
                epochs=args.train_epochs,
                artifact_dir=graph_artifact_dir,
                device=args.device,
            )
    unlearn_time = timer.times.get("unlearn_or_retrain", 0.0) - previous_unlearn_time

    after_trainer = result.trainer or trainer
    if result.logits is not None:
        logits_after = result.logits
        embeddings_after = result.embeddings
    else:
        with timer.track("predict_after"):
            logits_after, embeddings_after = after_trainer.predict_with_embeddings(result.data)

    offline_preprocessing_seconds = None
    if isinstance(result.training, dict):
        artifact_info = result.training.get("artifact")
        if isinstance(artifact_info, dict):
            offline_preprocessing_seconds = artifact_info.get("offline_preprocessing_seconds")

    with timer.track("evaluate"):
        metrics = build_experiment_metrics(
            method=result.method,
            dataset=bundle.name,
            unlearning_type=args.unlearning_type,
            data_before=data,
            data_after=result.data,
            logits_before=logits_before,
            logits_after=logits_after,
            embeddings_before=embeddings_before,
            embeddings_after=embeddings_after,
            forget_targets=forget_targets,
            unlearn_time_seconds=unlearn_time,
            retrain_time_seconds=None,
            online_wall_clock_seconds=timer.total,
            time_breakdown=timer.times,
            offline_preprocessing_seconds=offline_preprocessing_seconds,
        )
        metrics.pop("rq_summary", None)

    timer.end_total()
    output = {
        "dataset": bundle.name,
        "baseline": args.baseline,
        "method": result.method,
        "source_family": "OpenGU-adapted",
        "forget_set": forget_set_info,
        "base_training": base_training,
        "base_artifact": _base_artifact_result(base_artifact_dir, base_artifact_metadata),
        "opengu_artifact": _artifact_result(graph_artifact_dir, result.training),
        "result": result.as_dict(),
        "metrics": metrics,
        "runtime": timer.to_dict(),
    }

    output_path = Path(args.output) if args.output else _default_output_path(args, bundle.name, result.method, forget_set_info)
    output["metrics_path"] = str(save_metrics(output, output_path))
    print(json.dumps(json_safe(output), indent=2))


def _validate_args(args: argparse.Namespace) -> None:
    if args.baseline == "gif" and args.unlearning_type != "edge":
        raise SystemExit("OpenGU-adapted GIF currently supports only edge unlearning.")


def _model_config(args: argparse.Namespace, bundle) -> dict[str, Any]:
    return {
        "type": args.model_type,
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "in_channels": bundle.num_features,
        "out_channels": bundle.num_classes,
    }


def _resolve_artifact_root(args: argparse.Namespace) -> Path:
    if not args.artifact_root:
        return Path(args.output_root)
    artifact_root = Path(args.artifact_root)
    legacy_root = LOCAL_ROOT / "artifacts"
    try:
        if artifact_root.resolve().is_relative_to(legacy_root.resolve()):
            return Path(args.output_root)
    except RuntimeError:
        pass
    return artifact_root


def _resolve_base_artifact_dir(args: argparse.Namespace, dataset_name: str) -> Path | None:
    if args.base_artifact_dir:
        return Path(args.base_artifact_dir)
    if args.base_artifact_root:
        return default_base_artifact_dir(args.base_artifact_root, dataset_name, args.seed)
    return None


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


def _resolve_forget_targets(args: argparse.Namespace, data, dataset_name: str):
    if args.forget_set_file:
        spec = load_forget_set(
            args.forget_set_file,
            expected_type=args.unlearning_type,
            expected_dataset=dataset_name,
        )
        info = spec.as_dict()
        info["source"] = "forget_set_file"
        info["path"] = str(spec.path)
        return spec.targets, info

    if args.unlearning_type == "node":
        explicit = parse_forget_targets(args.forget_nodes, "node")
        source = "cli" if explicit else "missing_explicit_forget_set"
    elif args.unlearning_type == "edge":
        explicit = parse_forget_targets(args.forget_edges, "edge")
        source = "cli" if explicit else "missing_explicit_forget_set"
    else:
        explicit = parse_forget_targets(args.forget_features, "feature")
        source = "cli" if explicit else "missing_explicit_forget_set"
    if not explicit:
        raise SystemExit("OpenGU-adapted runs require --forget_set_file or explicit forget targets.")
    return explicit, {
        "source": source,
        "dataset": dataset_name,
        "unlearning_type": args.unlearning_type,
        "ratio": args.forget_ratio,
        "targets": explicit,
    }


def _base_artifact_result(path: Path | None, metadata: dict[str, Any] | None) -> dict[str, Any]:
    if path is None:
        return {"loaded": False, "path": None}
    return {
        "loaded": metadata is not None,
        "path": str(path),
        "base_training": (metadata or {}).get("base_training"),
        "training": (metadata or {}).get("training"),
    }


def _artifact_result(path: Path | None, training: dict[str, Any] | None) -> dict[str, Any]:
    if path is None:
        return {"loaded": False, "path": None}
    artifact = training.get("artifact") if isinstance(training, dict) else None
    return {
        "loaded": bool(artifact) or (path / "metadata.json").exists(),
        "path": str(path),
        "artifact": artifact,
    }


def _default_output_path(args: argparse.Namespace, dataset: str, method: str, forget_set_info: dict[str, Any]) -> Path:
    ratio = forget_set_info.get("ratio", args.forget_ratio)
    ratio_label = str(ratio).replace(".", "p")
    selection = forget_set_info.get("selection") or "manual"
    forget_seed = forget_set_info.get("seed")
    parts = [
        method_key(method),
        dataset,
        args.unlearning_type,
        f"r{ratio_label}",
        str(selection),
        f"base{args.seed}",
    ]
    if forget_seed is not None:
        parts.append(f"fseed{forget_seed}")
    filename = "_".join(parts) + ".json"
    root = Path(args.output_root) / dataset
    if args.experiment_name and args.experiment_name != "__root__":
        root = root / args.experiment_name
    return root / "baselines" / method_key(method) / args.unlearning_type / filename


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
