from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baselines import OfficialBaselineUnavailable, baseline_registry, get_baseline, official_specs_as_dict
from data import (
    apply_stratified_split,
    load_dataset,
    load_forget_set,
    parse_forget_targets,
    select_forget_edges,
    select_forget_features,
    select_forget_nodes,
    with_protocol_semantics,
)
from evaluation import (
    build_experiment_metrics,
    default_metrics_path,
    load_exact_retrain_reference,
    save_exact_retrain_reference,
    save_metrics,
)
from evaluation.metrics import json_safe
from utils import RuntimeTracker
from models import (
    GNNTrainer,
    TrainingConfig,
    build_gnn_model,
    default_base_artifact_dir,
    load_base_artifact,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run graph unlearning baselines.")
    parser.add_argument("--baseline", default="retrain")
    parser.add_argument("--list_baselines", action="store_true", help="List registered baselines and exit.")
    parser.add_argument("--data_root", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--allow_download", action="store_true", help="Allow this command to download missing datasets.")
    parser.add_argument("--dataset_name", default="cora", choices=["cora", "citeseer", "pubmed", "primekg", "primekg-homo", "primekg-full-nosource", "primekg-disease-gene-small", "primekg-disease-gene-small-nosource", "hetionet-small-nosource", "hetionet-full-nosource", "ppi-homo-sl-filtered", "ppi-inductive-sl-filtered", "ppi-inductive-sl-mostfreq-filtered", "ppi-inductive-sl-balanced20-filtered", "ppi-inductive-sl-balanced10-filtered", "reddit"])
    parser.add_argument("--unlearning_type", default="node", choices=["node", "edge", "feature"])
    parser.add_argument("--forget_ratio", type=float, default=0.1)
    parser.add_argument("--forget_nodes", default="")
    parser.add_argument("--forget_edges", default="")
    parser.add_argument("--forget_features", default="")
    parser.add_argument("--forget_set_file", default="", help="JSON or text forget-set protocol file.")
    parser.add_argument("--model_type", default="GCN", choices=["GCN", "GAT", "GraphSAGE"])
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--train_epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42, help="Training seed for model init and dropout randomness.")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", default=None)
    parser.add_argument("--base_artifact_root", default="", help="Root containing shared base artifacts as <root>/<dataset>/seed<seed>.")
    parser.add_argument("--base_artifact_dir", default="", help="Explicit shared base artifact directory for this run.")
    parser.add_argument("--grapheraser_artifact_dir", default="", help="Offline GraphEraser artifact directory for online unlearning.")
    parser.add_argument("--output", default=None, help="Metrics JSON path. Defaults to results/baselines/<baseline>/<baseline>_*.json.")
    parser.add_argument("--save_exact_retrain_reference", default="", help="Save retrain logits/embeddings as a reusable reference artifact.")
    parser.add_argument("--exact_retrain_reference", default="", help="Load a matching exact-retrain reference for Edge diagnostics.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.list_baselines:
        registry = baseline_registry(ROOT)
        print(
            json.dumps(
                {
                    "registered": sorted(registry),
                    "official_sources": official_specs_as_dict(ROOT),
                    "note": "Use *-surrogate only for sanity checks; do not report them as official baselines.",
                },
                indent=2,
            )
        )
        return

    timer = RuntimeTracker()
    timer.start_total()
    with timer.track("prepare"):
        _set_training_seed(args.seed)
        baseline = get_baseline(args.baseline, ROOT)
        if hasattr(baseline, "seed"):
            baseline.seed = int(args.seed)
        trainer_config = TrainingConfig(lr=args.lr, epochs=args.train_epochs, device=args.device)

    with timer.track("load_dataset"):
        bundle = load_dataset(args.dataset_name, args.data_root, download=args.allow_download)
        data = bundle.data

    with timer.track("prepare"):
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
        trainer = GNNTrainer(model, trainer_config)
        model_config = _model_config(args, bundle)
        base_artifact_dir = _resolve_base_artifact_dir(args, bundle.name)

    base_artifact_metadata = None
    if base_artifact_dir is not None:
        with timer.track("load_base"):
            base_artifact_metadata, logits_before, embeddings_before = load_base_artifact(
                base_artifact_dir,
                trainer=trainer,
                dataset_name=bundle.name,
                seed=args.seed,
                model_config=model_config,
            )
            base_training = dict(base_artifact_metadata.get("base_training", {}))
        with timer.track("reconstruct_split"):
            data = _apply_base_artifact_split(data, base_artifact_metadata, args.seed)
    else:
        with timer.track("prepare"):
            base_training = trainer.train_full_batch(data, epochs=args.train_epochs).as_dict()
        with timer.track("predict"):
            logits_before, embeddings_before = trainer.predict_with_embeddings(data)

    with timer.track("load_forget_set"):
        forget_targets, forget_set_info = _resolve_forget_targets(args, data, bundle.name)

    grapheraser_artifact_dir = Path(args.grapheraser_artifact_dir) if args.grapheraser_artifact_dir else None
    previous_unlearn_time = timer.times.get("unlearn_or_retrain", 0.0)
    with timer.track("unlearn_or_retrain"):
        try:
            if args.unlearning_type == "node":
                result = baseline.run_node_unlearning(
                    data,
                    forget_targets,
                    trainer=trainer,
                    model_factory=model_factory,
                    epochs=args.train_epochs,
                    artifact_dir=grapheraser_artifact_dir,
                    device=args.device,
                )
            elif args.unlearning_type == "edge":
                result = baseline.run_edge_unlearning(
                    data,
                    forget_targets,
                    trainer=trainer,
                    model_factory=model_factory,
                    epochs=args.train_epochs,
                    artifact_dir=grapheraser_artifact_dir,
                    device=args.device,
                )
            else:
                result = baseline.run_feature_unlearning(
                    data,
                    forget_targets,
                    trainer=trainer,
                    model_factory=model_factory,
                    epochs=args.train_epochs,
                    artifact_dir=grapheraser_artifact_dir,
                    device=args.device,
                )
        except OfficialBaselineUnavailable as exc:
            raise SystemExit(str(exc)) from exc
    unlearn_time = timer.times.get("unlearn_or_retrain", 0.0) - previous_unlearn_time

    after_trainer = result.trainer or trainer
    if result.logits is not None:
        logits_after = result.logits
        embeddings_after = result.embeddings
    else:
        with timer.track("predict"):
            logits_after, embeddings_after = after_trainer.predict_with_embeddings(result.data)

    exact_reference_payload = None
    if args.save_exact_retrain_reference and args.exact_retrain_reference:
        raise SystemExit("Use only one of --save_exact_retrain_reference and --exact_retrain_reference.")
    if args.save_exact_retrain_reference:
        if args.baseline.lower() != "retrain" or args.unlearning_type != "edge":
            raise SystemExit("--save_exact_retrain_reference requires --baseline retrain --unlearning_type edge.")
        if not forget_set_info.get("path"):
            raise SystemExit("--save_exact_retrain_reference requires --forget_set_file.")
        with timer.track("save_exact_retrain_reference"):
            reference_metadata = save_exact_retrain_reference(
                args.save_exact_retrain_reference,
                logits=logits_after,
                embeddings=embeddings_after,
                dataset=bundle.name,
                unlearning_type=args.unlearning_type,
                forget_set_path=forget_set_info["path"],
                base_artifact_path=base_artifact_dir,
                seed=args.seed,
                model_config=model_config,
                training=result.training,
            )
        exact_reference_payload = {
            "metadata": reference_metadata,
            "logits": logits_after,
            "embeddings": embeddings_after,
            "path": args.save_exact_retrain_reference,
        }
    elif args.exact_retrain_reference:
        if args.unlearning_type != "edge" or not forget_set_info.get("path"):
            raise SystemExit("--exact_retrain_reference requires Edge --forget_set_file input.")
        with timer.track("load_exact_retrain_reference"):
            exact_reference_payload = load_exact_retrain_reference(
                args.exact_retrain_reference,
                dataset=bundle.name,
                unlearning_type=args.unlearning_type,
                forget_set_path=forget_set_info["path"],
                base_artifact_path=base_artifact_dir,
            )

    offline_preprocessing_seconds = None
    if isinstance(result.training, dict):
        artifact_info = result.training.get("artifact")
        if isinstance(artifact_info, dict):
            offline_preprocessing_seconds = artifact_info.get("offline_preprocessing_seconds")

    with timer.track("evaluate"):
        metrics = build_experiment_metrics(
            method=args.baseline,
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
            retrain_time_seconds=unlearn_time if args.baseline.lower() == "retrain" else None,
            online_wall_clock_seconds=timer.total,
            time_breakdown=timer.times,
            offline_preprocessing_seconds=offline_preprocessing_seconds,
            mia_seed=int(forget_set_info.get("seed", args.seed) or 0),
            exact_retrain_logits=(exact_reference_payload or {}).get("logits"),
            exact_retrain_embeddings=(exact_reference_payload or {}).get("embeddings"),
            exact_retrain_reference=(
                {
                    **(exact_reference_payload or {}).get("metadata", {}),
                    "path": (exact_reference_payload or {}).get("path"),
                }
                if exact_reference_payload else None
            ),
        )
        metrics.pop("rq_summary", None)

    output = {
        "dataset": bundle.name,
        "baseline": args.baseline,
        "forget_set": forget_set_info,
        "base_training": base_training,
        "base_artifact": _base_artifact_result(base_artifact_dir, base_artifact_metadata),
        "grapheraser_artifact": _grapheraser_artifact_result(grapheraser_artifact_dir, result.training),
        "exact_retrain_reference": {
            "loaded": exact_reference_payload is not None,
            "path": (exact_reference_payload or {}).get("path"),
            "metadata": (exact_reference_payload or {}).get("metadata"),
        },
        "result": result.as_dict(),
        "metrics": metrics,
    }
    output_path = Path(args.output) if args.output else default_metrics_path(
        ROOT,
        args.baseline,
        bundle.name,
        args.unlearning_type,
        forget_set_info.get("ratio", args.forget_ratio),
        selection=forget_set_info.get("selection"),
        seed=forget_set_info.get("seed", args.seed),
    )
    output["metrics_path"] = str(_save_with_runtime(output, output_path, timer))
    print(json.dumps(json_safe(output), indent=2))


def _save_with_runtime(result: dict, output_path: Path, timer: RuntimeTracker) -> Path:
    result["metrics_path"] = str(output_path)
    start = time.perf_counter()
    save_metrics(result, output_path)
    timer.add("save", time.perf_counter() - start)
    timer.end_total()
    metrics = result.get("metrics")
    if isinstance(metrics, dict):
        efficiency = metrics.setdefault("efficiency", {})
        runtime = timer.to_dict()
        efficiency["online_wall_clock_seconds"] = runtime["online_wall_clock_seconds"]
        efficiency["time_breakdown"] = runtime["time_breakdown"]
    return save_metrics(result, output_path)


def _grapheraser_artifact_result(path: Path | None, training: dict | None) -> dict:
    if path is None:
        return {"loaded": False, "path": None}
    artifact = training.get("artifact") if isinstance(training, dict) else None
    return {
        "loaded": bool(artifact),
        "path": str(path),
        "offline_preprocessing_seconds": artifact.get("offline_preprocessing_seconds") if isinstance(artifact, dict) else None,
        "partition_method": artifact.get("partition_method") if isinstance(artifact, dict) else None,
        "num_shards": artifact.get("num_shards") if isinstance(artifact, dict) else None,
    }


def _model_config(args, bundle) -> dict:
    return {
        "type": args.model_type,
        "hidden_channels": args.hidden_channels,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "in_channels": bundle.num_features,
        "out_channels": bundle.num_classes,
    }


def _resolve_base_artifact_dir(args, dataset_name: str) -> Path | None:
    if args.base_artifact_dir:
        return Path(args.base_artifact_dir)
    if args.base_artifact_root:
        return default_base_artifact_dir(args.base_artifact_root, dataset_name, args.seed)
    return None


def _apply_base_artifact_split(data, metadata: dict | None, seed: int):
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


def _base_artifact_result(path: Path | None, metadata: dict | None) -> dict:
    if path is None:
        return {"loaded": False, "path": None}
    return {
        "loaded": metadata is not None,
        "path": str(path),
        "base_training": (metadata or {}).get("base_training"),
    }


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


def _resolve_forget_targets(args, data, dataset_name: str):
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
        targets = explicit or select_forget_nodes(data, args.forget_ratio)
        source = "cli" if explicit else "generated_default"
    elif args.unlearning_type == "edge":
        explicit = parse_forget_targets(args.forget_edges, "edge")
        targets = explicit or select_forget_edges(data, args.forget_ratio)
        source = "cli" if explicit else "generated_default"
    else:
        explicit = parse_forget_targets(args.forget_features, "feature")
        targets = explicit or select_forget_features(data, args.forget_ratio)
        source = "cli" if explicit else "generated_default"

    return targets, {
        "source": source,
        "dataset": dataset_name,
        "unlearning_type": args.unlearning_type,
        "ratio": args.forget_ratio,
        "targets": targets,
        "protocol": with_protocol_semantics(args.unlearning_type, None),
    }


if __name__ == "__main__":
    main()
