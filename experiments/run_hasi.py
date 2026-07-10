from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import (
    apply_stratified_split,
    load_dataset,
    load_forget_set,
    parse_forget_targets,
    select_forget_edges,
    select_forget_features,
    select_forget_nodes,
)
from evaluation import build_experiment_metrics, default_metrics_path, save_metrics
from evaluation.metrics import json_safe
from hasi import HASIConfig, HASIUnlearner
from hasi.dar import DARConfig
from hasi.hub_identification import HubScoreConfig
from utils import RuntimeTracker
from models import (
    GNNTrainer,
    TrainingConfig,
    build_gnn_model,
    default_base_artifact_dir,
    load_base_artifact,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run HASI preprocessing or unlearning.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "hasi_default.yaml"), help="YAML config path.")
    parser.add_argument("--data_root", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--allow_download", action="store_true", help="Allow this command to download missing datasets.")
    parser.add_argument("--dataset_name", default=None, choices=["cora", "citeseer", "pubmed", "primekg", "primekg-homo", "primekg-disease-gene-small", "primekg-disease-gene-small-nosource", "hetionet-small-nosource", "ppi-homo-sl-filtered", "ppi-inductive-sl-filtered", "ppi-inductive-sl-mostfreq-filtered", "ppi-inductive-sl-balanced20-filtered", "ppi-inductive-sl-balanced10-filtered", "reddit"])
    parser.add_argument("--mode", default="plan", choices=["plan", "unlearn"])
    parser.add_argument("--unlearning_type", default=None, choices=["node", "edge", "feature"])
    parser.add_argument("--forget_ratio", type=float, default=None)
    parser.add_argument("--forget_nodes", default="", help="Comma-separated node ids.")
    parser.add_argument("--forget_edges", default="", help="Comma-separated edges like '1-2,3-4'.")
    parser.add_argument("--forget_features", default="", help="Comma-separated feature ids.")
    parser.add_argument("--forget_set_file", default="", help="JSON or text forget-set protocol file.")
    parser.add_argument(
        "--edge_forget_scope",
        default="all",
        choices=["all", "train_subgraph"],
        help="Scope for generated edge forget targets when --forget_set_file/--forget_edges are not provided.",
    )
    parser.add_argument(
        "--generated_forget_seed",
        type=int,
        default=None,
        help="Random seed for generated forget targets. Defaults to --seed after config resolution.",
    )
    parser.add_argument("--model_type", default=None, choices=["GCN", "GAT", "GraphSAGE"])
    parser.add_argument("--hidden_channels", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--train_epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Training seed for model init, dropout, and inpainting randomness.")
    parser.add_argument("--finetune_epochs", type=int, default=None)
    parser.add_argument("--finetune_lr", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Base model training learning rate.")
    parser.add_argument("--hub_alpha", type=float, default=None)
    parser.add_argument("--hub_beta", type=float, default=None)
    parser.add_argument("--hub_gamma", type=float, default=None)
    parser.add_argument("--hub_filter_ratio", type=float, default=None)
    parser.add_argument("--primary_ratio", type=float, default=None)
    parser.add_argument("--secondary_ratio", type=float, default=None)
    parser.add_argument("--anchor_lambda1", type=float, default=None)
    parser.add_argument("--anchor_lambda2", type=float, default=None)
    parser.add_argument(
        "--anchor_mode",
        default=None,
        choices=["hierarchical", "none"],
        help="Anchor behavior. Defaults to hierarchical, matching previous HASI behavior.",
    )
    parser.add_argument("--erf_alpha", type=float, default=None)
    parser.add_argument("--erf_k_steps", type=int, default=None)
    parser.add_argument("--erf_threshold", type=float, default=None)
    parser.add_argument("--inpainting_mode", default=None, choices=["none", "local_only", "full"])
    parser.add_argument("--inpainting_method", default=None, choices=["mgae"])
    parser.add_argument("--inpainting_cc_drop_threshold", type=float, default=None)
    parser.add_argument("--inpainting_min_damage_ratio", type=float, default=None)
    parser.add_argument("--inpainting_hidden_channels", type=int, default=None)
    parser.add_argument("--inpainting_embedding_channels", type=int, default=None)
    parser.add_argument("--inpainting_train_epochs", type=int, default=None)
    parser.add_argument("--inpainting_lr", type=float, default=None)
    parser.add_argument("--inpainting_mask_ratio", type=float, default=None)
    parser.add_argument("--inpainting_edge_threshold", type=float, default=None)
    parser.add_argument("--inpainting_max_added_edges", type=int, default=None)
    parser.add_argument("--inpainting_repair_ratio", type=float, default=None)
    parser.add_argument("--inpainting_max_candidate_nodes", type=int, default=None)
    parser.add_argument("--inpainting_max_candidate_edges", type=int, default=None)
    parser.add_argument("--dar_enabled", "--dar-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dar_k", type=int, default=None)
    parser.add_argument(
        "--dar_strategy",
        default=None,
        choices=["hubscore", "proximity_weighted", "privacy_constrained", "distributed"],
    )
    parser.add_argument("--dar_min_distance", type=int, default=None)
    parser.add_argument("--dar_small_component_threshold", type=int, default=None)
    parser.add_argument("--dar_gumbel_tau", type=float, default=None)
    parser.add_argument("--dar_alpha_score", type=float, default=None)
    parser.add_argument("--dar_beta_score", type=float, default=None)
    parser.add_argument("--dar_max_search_radius", type=int, default=None)
    parser.add_argument("--dar_lambda2", type=float, default=None, help="Distributed anchor total weight. Defaults to anchor_lambda2.")
    parser.add_argument("--subgraph_finetune", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--subgraph_min_nodes", type=int, default=None)
    parser.add_argument("--feature_drift_threshold", type=float, default=None)
    parser.add_argument("--feature_anchor_to_h_new", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--forget_weight", type=float, default=None)
    parser.add_argument("--edge_forget_loss_mode", default=None, choices=["original_kl", "uniform", "none"])
    parser.add_argument(
        "--gradient_hub_score",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use task-gradient sensitivity in HubScore. Defaults to on for unlearn mode and off for plan mode.",
    )
    parser.add_argument("--gradient_hub_passes", type=int, default=1)
    parser.add_argument("--gradient_hub_dropout", action="store_true", help="Average gradient scores with dropout enabled.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--base_artifact_root", default="", help="Root containing shared base artifacts as <root>/<dataset>/seed<seed>.")
    parser.add_argument("--base_artifact_dir", default="", help="Explicit shared base artifact directory for this run.")
    parser.add_argument("--method_name", default="hasi", help="Method label stored in output metrics.")
    parser.add_argument("--output", default=None, help="Metrics JSON path. Defaults to results/hasi/hasi_*.json.")
    return parser.parse_args()


def main():
    args = parse_args()
    timer = RuntimeTracker()
    timer.start_total()

    with timer.track("load_config"):
        file_config, config_path = _load_yaml_config(args.config)
        hasi_config, resolved_config = _resolve_runtime_config(args, file_config)
        _set_training_seed(args.seed)

    with timer.track("load_dataset"):
        bundle = load_dataset(args.dataset_name, args.data_root, download=args.allow_download)
        data = bundle.data

    with timer.track("prepare"):
        model_config = _model_config(args, bundle)
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
            TrainingConfig(lr=args.lr, epochs=args.train_epochs, device=args.device),
        )
        base_artifact_dir = _resolve_base_artifact_dir(args, bundle.name)

    base_artifact_metadata = None
    logits_before = None
    embeddings_before = None
    if base_artifact_dir is not None:
        with timer.track("load_base"):
            base_artifact_metadata, logits_before, embeddings_before = load_base_artifact(
                base_artifact_dir,
                trainer=trainer,
                dataset_name=bundle.name,
                seed=args.seed,
                model_config=model_config,
            )
        with timer.track("reconstruct_split"):
            data = _apply_base_artifact_split(data, base_artifact_metadata, args.seed)

    with timer.track("prepare"):
        unlearner = HASIUnlearner(
            model=model,
            data=data,
            config=hasi_config,
        )

    with timer.track("load_forget_set"):
        forget_targets, forget_set_info = _resolve_forget_targets(args, data, bundle.name)

    use_gradient_hub_score = args.gradient_hub_score
    if use_gradient_hub_score is None:
        use_gradient_hub_score = args.mode == "unlearn"

    result = {
        "dataset": bundle.name,
        "method": args.method_name,
        "config": {
            "path": str(config_path) if config_path else None,
            "resolved": resolved_config,
        },
        "forget_set": forget_set_info,
        "gradient_hub_scoring": {
            "enabled": bool(use_gradient_hub_score),
            "passes": args.gradient_hub_passes,
            "dropout": args.gradient_hub_dropout,
        },
        "base_artifact": _base_artifact_result(base_artifact_dir, base_artifact_metadata),
    }
    if args.mode == "plan":
        with timer.track("prepare"):
            if use_gradient_hub_score:
                result["base_training"] = _ensure_base_model(
                    trainer,
                    data,
                    args.train_epochs,
                    base_artifact_metadata,
                )
                gradient_scores = trainer.gradient_sensitivity(
                    data,
                    passes=args.gradient_hub_passes,
                    use_dropout=args.gradient_hub_dropout,
                )
                result["gradient_hub_scoring"]["num_scored_nodes"] = len(gradient_scores)
            else:
                gradient_scores = None
            summary = unlearner.preprocess(gradient_scores=gradient_scores)
            result["preprocess"] = summary
            if args.unlearning_type == "node":
                result["plan"] = unlearner.plan_node_unlearning(forget_targets)
            else:
                result["plan"] = {
                    "type": args.unlearning_type,
                    "forget_targets": forget_targets,
                    "status": "planning is implemented for node unlearning",
                }
    else:
        with timer.track("prepare"):
            result["base_training"] = _ensure_base_model(
                trainer,
                data,
                args.train_epochs,
                base_artifact_metadata,
            )
            gradient_scores = None
            if use_gradient_hub_score:
                gradient_scores = trainer.gradient_sensitivity(
                    data,
                    passes=args.gradient_hub_passes,
                    use_dropout=args.gradient_hub_dropout,
                )
                result["gradient_hub_scoring"]["num_scored_nodes"] = len(gradient_scores)
            summary = unlearner.preprocess(gradient_scores=gradient_scores)
            result["preprocess"] = summary
            anchor_nodes_for_metrics = _anchor_nodes_from_unlearner(unlearner)
        if logits_before is None or embeddings_before is None:
            with timer.track("predict"):
                logits_before, embeddings_before = trainer.predict_with_embeddings(data)

        previous_unlearn_time = timer.times.get("unlearn_or_retrain", 0.0)
        with timer.track("unlearn_or_retrain"):
            if args.unlearning_type == "node":
                result["unlearning"] = unlearner.unlearn_nodes(
                    forget_targets,
                    trainer=trainer,
                    finetune_epochs=args.finetune_epochs,
                    finetune_lr=args.finetune_lr,
                    forget_weight=args.forget_weight,
                )
            elif args.unlearning_type == "edge":
                result["unlearning"] = unlearner.unlearn_edges(
                    forget_targets,
                    trainer=trainer,
                    finetune_epochs=args.finetune_epochs,
                    finetune_lr=args.finetune_lr,
                    forget_weight=args.forget_weight,
                )
            else:
                result["unlearning"] = unlearner.unlearn_features(
                    forget_targets,
                    trainer=trainer,
                    finetune_epochs=args.finetune_epochs,
                    finetune_lr=args.finetune_lr,
                    forget_weight=args.forget_weight,
                )
        unlearn_time = timer.times.get("unlearn_or_retrain", 0.0) - previous_unlearn_time

        with timer.track("predict"):
            logits_after, embeddings_after = trainer.predict_with_embeddings(unlearner.data)
        with timer.track("evaluate"):
            result["metrics"] = build_experiment_metrics(
                method=args.method_name,
                dataset=bundle.name,
                unlearning_type=args.unlearning_type,
                data_before=data,
                data_after=unlearner.data,
                logits_before=logits_before,
                logits_after=logits_after,
                embeddings_before=embeddings_before,
                embeddings_after=embeddings_after,
                forget_targets=forget_targets,
                unlearn_time_seconds=unlearn_time,
                online_wall_clock_seconds=timer.total,
                time_breakdown=timer.times,
                primary_anchor_nodes=anchor_nodes_for_metrics["primary"],
                secondary_anchor_nodes=anchor_nodes_for_metrics["secondary"],
            )
            result["metrics"].pop("rq_summary", None)
        output_path = Path(args.output) if args.output else default_metrics_path(
            ROOT,
            "hasi",
            bundle.name,
            args.unlearning_type,
            forget_set_info.get("ratio", args.forget_ratio),
            selection=forget_set_info.get("selection"),
            seed=forget_set_info.get("seed", args.seed),
        )
        result["metrics_path"] = str(_save_with_runtime(result, output_path, timer))

    print(json.dumps(json_safe(result), indent=2))


def _anchor_nodes_from_unlearner(unlearner: HASIUnlearner) -> dict[str, list[int]]:
    """Return reference anchor nodes for evaluation, independent of anchor_mode.

    RQ3 disables the anchor manager for no-anchor runs, but anchor-specific drift
    still needs the same would-be primary/secondary hub nodes for fair comparison.
    """

    hub_scores = getattr(unlearner, "hub_scores", None) or {}
    hub_scorer = getattr(unlearner, "hub_scorer", None)
    if hub_scores and hub_scorer is not None and hasattr(hub_scorer, "classify_anchors"):
        primary, secondary, _ = hub_scorer.classify_anchors(hub_scores)
        return {
            "primary": sorted(int(node) for node in primary),
            "secondary": sorted(int(node) for node in secondary),
        }

    anchor_manager = getattr(unlearner, "anchor_manager", None)
    anchors = getattr(anchor_manager, "anchors", None)
    if anchors is None:
        return {"primary": [], "secondary": []}
    return {
        "primary": sorted(int(node) for node in getattr(anchors, "primary", set())),
        "secondary": sorted(int(node) for node in getattr(anchors, "secondary", set())),
    }


def _model_config(args, bundle) -> dict[str, Any]:
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


def _base_artifact_result(path: Path | None, metadata: dict[str, Any] | None) -> dict[str, Any]:
    if path is None:
        return {"loaded": False, "path": None}
    return {
        "loaded": metadata is not None,
        "path": str(path),
        "base_training": (metadata or {}).get("base_training"),
    }


def _ensure_base_model(
    trainer: GNNTrainer,
    data,
    train_epochs: int,
    base_artifact_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    if base_artifact_metadata is not None:
        return dict(base_artifact_metadata.get("base_training", {}))
    return trainer.train_full_batch(data, epochs=train_epochs).as_dict()


def _save_with_runtime(result: dict[str, Any], output_path: Path, timer: RuntimeTracker) -> Path:
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


def _load_yaml_config(path: str | None) -> tuple[dict[str, Any], Path | None]:
    if not path:
        return {}, None

    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required for --config support. Install pyyaml or pass --config ''.") from exc

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {config_path}")
    return payload, config_path


def _resolve_runtime_config(args, file_config: Mapping[str, Any]) -> tuple[HASIConfig, dict[str, Any]]:
    model_cfg = _section(file_config, "model")
    hub_cfg = _section(file_config, "hub_identification")
    anchor_cfg = _section(file_config, "anchor_stabilization")
    erf_cfg = _section(file_config, "erf_partitioning")
    inpainting_cfg = _section(file_config, "inpainting")
    dar_cfg = _section(file_config, "dar")
    unlearning_cfg = _section(file_config, "unlearning")
    optimization_cfg = _section(file_config, "optimization")
    feature_cfg = _section(file_config, "feature_unlearning")
    training_cfg = _section(file_config, "training")

    args.dataset_name = _arg(args, "dataset_name", "cora")
    args.unlearning_type = _arg(args, "unlearning_type", unlearning_cfg.get("type", "node"))
    args.forget_ratio = _arg(args, "forget_ratio", unlearning_cfg.get("ratio", 0.1))

    args.model_type = _arg(args, "model_type", model_cfg.get("type", "GCN"))
    args.hidden_channels = _arg(args, "hidden_channels", model_cfg.get("hidden_channels", 64))
    args.num_layers = _arg(args, "num_layers", model_cfg.get("num_layers", 2))
    args.dropout = _arg(args, "dropout", model_cfg.get("dropout", 0.5))

    cli_lr = args.lr
    args.train_epochs = _arg(args, "train_epochs", training_cfg.get("epochs", 200))
    args.lr = _arg(args, "lr", training_cfg.get("lr", 0.01))
    args.seed = _arg(args, "seed", training_cfg.get("seed", 42))
    args.finetune_epochs = _arg(args, "finetune_epochs", unlearning_cfg.get("finetune_epochs", 50))
    finetune_lr_default = cli_lr if cli_lr is not None else unlearning_cfg.get("finetune_lr", args.lr)
    args.finetune_lr = _arg(args, "finetune_lr", finetune_lr_default)
    args.forget_weight = _arg(args, "forget_weight", unlearning_cfg.get("forget_weight", 0.0))
    args.edge_forget_loss_mode = _arg(args, "edge_forget_loss_mode", unlearning_cfg.get("edge_forget_loss_mode", "original_kl"))

    hub_config = HubScoreConfig(
        alpha=_arg(args, "hub_alpha", hub_cfg.get("alpha", 0.4)),
        beta=_arg(args, "hub_beta", hub_cfg.get("beta", 0.3)),
        gamma=_arg(args, "hub_gamma", hub_cfg.get("gamma", 0.3)),
        filter_ratio=_arg(args, "hub_filter_ratio", hub_cfg.get("filter_ratio", 0.1)),
        primary_ratio=_arg(args, "primary_ratio", hub_cfg.get("primary_ratio", 0.01)),
        secondary_ratio=_arg(args, "secondary_ratio", hub_cfg.get("secondary_ratio", 0.05)),
    )
    args.anchor_lambda1 = _arg(args, "anchor_lambda1", anchor_cfg.get("lambda1", 2.0))
    args.anchor_lambda2 = _arg(args, "anchor_lambda2", anchor_cfg.get("lambda2", 0.5))
    args.anchor_mode = _arg(args, "anchor_mode", anchor_cfg.get("mode", "hierarchical"))
    args.erf_alpha = _arg(args, "erf_alpha", erf_cfg.get("alpha", 0.15))
    args.erf_k_steps = _arg(args, "erf_k_steps", erf_cfg.get("k_steps", 3))
    args.erf_threshold = _arg(args, "erf_threshold", erf_cfg.get("threshold", 0.01))

    args.inpainting_mode = _arg(args, "inpainting_mode", inpainting_cfg.get("mode", "full"))
    args.inpainting_method = _arg(args, "inpainting_method", inpainting_cfg.get("method", "mgae"))
    args.inpainting_cc_drop_threshold = _arg(args, "inpainting_cc_drop_threshold", inpainting_cfg.get("cc_drop_threshold", 0.30))
    args.inpainting_min_damage_ratio = _arg(args, "inpainting_min_damage_ratio", inpainting_cfg.get("min_damage_ratio", 0.10))
    args.inpainting_hidden_channels = _arg(args, "inpainting_hidden_channels", inpainting_cfg.get("hidden_channels", 64))
    args.inpainting_embedding_channels = _arg(args, "inpainting_embedding_channels", inpainting_cfg.get("embedding_channels", 32))
    args.inpainting_train_epochs = _arg(args, "inpainting_train_epochs", inpainting_cfg.get("train_epochs", 80))
    args.inpainting_lr = _arg(args, "inpainting_lr", inpainting_cfg.get("lr", 0.01))
    args.inpainting_mask_ratio = _arg(args, "inpainting_mask_ratio", inpainting_cfg.get("mask_ratio", 0.15))
    args.inpainting_edge_threshold = _arg(args, "inpainting_edge_threshold", inpainting_cfg.get("edge_threshold", 0.50))
    args.inpainting_max_added_edges = _arg(args, "inpainting_max_added_edges", inpainting_cfg.get("max_added_edges", 256))
    args.inpainting_repair_ratio = _arg(args, "inpainting_repair_ratio", inpainting_cfg.get("repair_ratio", 0.35))
    args.inpainting_max_candidate_nodes = _arg(args, "inpainting_max_candidate_nodes", inpainting_cfg.get("max_candidate_nodes", 512))
    args.inpainting_max_candidate_edges = _arg(args, "inpainting_max_candidate_edges", inpainting_cfg.get("max_candidate_edges", 20000))

    dar_lambda2 = _arg(args, "dar_lambda2", dar_cfg.get("lambda2", args.anchor_lambda2))
    dar_config = DARConfig(
        enabled=_arg(args, "dar_enabled", dar_cfg.get("enabled", True)),
        k=_arg(args, "dar_k", dar_cfg.get("k", 5)),
        strategy=_arg(args, "dar_strategy", dar_cfg.get("strategy", "distributed")),
        min_distance=_arg(args, "dar_min_distance", dar_cfg.get("min_distance", 2)),
        small_component_threshold=_arg(args, "dar_small_component_threshold", dar_cfg.get("small_component_threshold", 10)),
        lambda2=dar_lambda2,
        gumbel_tau=_arg(args, "dar_gumbel_tau", dar_cfg.get("gumbel_tau", 0.1)),
        alpha_score=_arg(args, "dar_alpha_score", dar_cfg.get("alpha_score", 0.6)),
        beta_score=_arg(args, "dar_beta_score", dar_cfg.get("beta_score", 0.4)),
        max_search_radius=_arg(args, "dar_max_search_radius", dar_cfg.get("max_search_radius", None)),
        seed=args.seed,
    )
    args.subgraph_finetune = _arg(args, "subgraph_finetune", optimization_cfg.get("subgraph_finetune", True))
    args.subgraph_min_nodes = _arg(args, "subgraph_min_nodes", optimization_cfg.get("subgraph_min_nodes", 5000))
    args.feature_drift_threshold = _arg(args, "feature_drift_threshold", feature_cfg.get("drift_threshold", 1e-6))
    args.feature_anchor_to_h_new = _arg(args, "feature_anchor_to_h_new", feature_cfg.get("anchor_to_h_new", True))

    hasi_config = HASIConfig(
        hub_identification=hub_config,
        erf_alpha=args.erf_alpha,
        erf_k_steps=args.erf_k_steps,
        erf_threshold=args.erf_threshold,
        inpainting_mode=args.inpainting_mode,
        inpainting_method=args.inpainting_method,
        inpainting_cc_drop_threshold=args.inpainting_cc_drop_threshold,
        inpainting_min_damage_ratio=args.inpainting_min_damage_ratio,
        inpainting_hidden_channels=args.inpainting_hidden_channels,
        inpainting_embedding_channels=args.inpainting_embedding_channels,
        inpainting_train_epochs=args.inpainting_train_epochs,
        inpainting_lr=args.inpainting_lr,
        inpainting_mask_ratio=args.inpainting_mask_ratio,
        inpainting_edge_threshold=args.inpainting_edge_threshold,
        inpainting_max_added_edges=args.inpainting_max_added_edges,
        inpainting_repair_ratio=args.inpainting_repair_ratio,
        inpainting_max_candidate_nodes=args.inpainting_max_candidate_nodes,
        inpainting_max_candidate_edges=args.inpainting_max_candidate_edges,
        dar=dar_config,
        anchor_mode=args.anchor_mode,
        anchor_lambda1=args.anchor_lambda1,
        anchor_lambda2=args.anchor_lambda2,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        forget_weight=args.forget_weight,
        edge_forget_loss_mode=args.edge_forget_loss_mode,
        subgraph_finetune=args.subgraph_finetune,
        subgraph_min_nodes=args.subgraph_min_nodes,
        feature_drift_threshold=args.feature_drift_threshold,
        feature_anchor_to_h_new=args.feature_anchor_to_h_new,
    )
    resolved = {
        "model": {
            "type": args.model_type,
            "hidden_channels": args.hidden_channels,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
        },
        "training": {"epochs": args.train_epochs, "lr": args.lr, "seed": args.seed},
        "hub_identification": hub_config,
        "anchor_stabilization": {
            "mode": args.anchor_mode,
            "lambda1": args.anchor_lambda1,
            "lambda2": args.anchor_lambda2,
        },
        "erf_partitioning": {"alpha": args.erf_alpha, "k_steps": args.erf_k_steps, "threshold": args.erf_threshold},
        "inpainting": {
            "mode": args.inpainting_mode,
            "method": args.inpainting_method,
            "cc_drop_threshold": args.inpainting_cc_drop_threshold,
            "min_damage_ratio": args.inpainting_min_damage_ratio,
            "hidden_channels": args.inpainting_hidden_channels,
            "embedding_channels": args.inpainting_embedding_channels,
            "train_epochs": args.inpainting_train_epochs,
            "lr": args.inpainting_lr,
            "mask_ratio": args.inpainting_mask_ratio,
            "edge_threshold": args.inpainting_edge_threshold,
            "max_added_edges": args.inpainting_max_added_edges,
            "repair_ratio": args.inpainting_repair_ratio,
            "max_candidate_nodes": args.inpainting_max_candidate_nodes,
            "max_candidate_edges": args.inpainting_max_candidate_edges,
        },
        "dar": dar_config,
        "optimization": {
            "subgraph_finetune": args.subgraph_finetune,
            "subgraph_min_nodes": args.subgraph_min_nodes,
        },
        "feature_unlearning": {
            "drift_threshold": args.feature_drift_threshold,
            "anchor_to_h_new": args.feature_anchor_to_h_new,
        },
        "unlearning": {
            "type": args.unlearning_type,
            "ratio": args.forget_ratio,
            "forget_weight": args.forget_weight,
            "edge_forget_loss_mode": args.edge_forget_loss_mode,
            "finetune_epochs": args.finetune_epochs,
            "finetune_lr": args.finetune_lr,
        },
    }
    return hasi_config, resolved


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


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section {name!r} must be a mapping.")
    return value


def _arg(args, name: str, default):
    value = getattr(args, name)
    return default if value is None else value


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

    selection_seed = int(args.generated_forget_seed if args.generated_forget_seed is not None else args.seed)
    source = "generated_default"
    candidate_scope = None
    candidate_count = None

    if args.unlearning_type == "node":
        explicit = parse_forget_targets(args.forget_nodes, "node")
        targets = explicit or select_forget_nodes(data, args.forget_ratio, seed=selection_seed)
        source = "cli" if explicit else "generated_train_mask"
        candidate_scope = "train_mask"
        candidate_count = _mask_count(getattr(data, "train_mask", None)) or int(data.num_nodes)
    elif args.unlearning_type == "edge":
        explicit = parse_forget_targets(args.forget_edges, "edge")
        if explicit:
            targets = explicit
            source = "cli"
        elif args.edge_forget_scope == "train_subgraph":
            targets, candidate_count = _select_train_subgraph_edges(data, args.forget_ratio, selection_seed)
            source = "generated_train_subgraph"
            candidate_scope = "train_subgraph_edges"
        else:
            targets = select_forget_edges(data, args.forget_ratio, seed=selection_seed)
            source = "generated_all_edges"
            candidate_scope = "all_edges"
            candidate_count = int(data.edge_index.shape[1])
    else:
        explicit = parse_forget_targets(args.forget_features, "feature")
        targets = explicit or select_forget_features(data, args.forget_ratio, seed=selection_seed)
        source = "cli" if explicit else "generated_feature_dims"
        candidate_scope = "feature_dimensions"
        candidate_count = int(data.x.shape[1]) if hasattr(data, "x") and data.x is not None else None

    info = {
        "source": source,
        "dataset": dataset_name,
        "unlearning_type": args.unlearning_type,
        "ratio": args.forget_ratio,
        "selection_seed": selection_seed,
        "candidate_scope": candidate_scope,
        "candidate_count": candidate_count,
        "targets": targets,
    }
    if args.unlearning_type == "edge":
        info["edge_forget_scope"] = args.edge_forget_scope
    return targets, info


def _select_train_subgraph_edges(data, ratio: float, seed: int) -> tuple[list[tuple[int, int]], int]:
    train_mask = getattr(data, "train_mask", None)
    if train_mask is None:
        raise ValueError("--edge_forget_scope train_subgraph requires data.train_mask.")
    if not hasattr(data, "edge_index") or data.edge_index is None:
        return [], 0

    import torch

    edge_index = data.edge_index.detach().cpu()
    mask = train_mask.detach().cpu().bool()
    keep = mask[edge_index[0]] & mask[edge_index[1]]
    candidate_indices = torch.nonzero(keep, as_tuple=False).view(-1)
    candidate_count = int(candidate_indices.numel())
    count = _sample_count(candidate_count, ratio)
    if count <= 0:
        return [], candidate_count
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    order = torch.randperm(candidate_count, generator=generator)[:count]
    selected = candidate_indices[order].tolist()
    return [(int(edge_index[0, idx]), int(edge_index[1, idx])) for idx in selected], candidate_count


def _mask_count(mask) -> int:
    if mask is None:
        return 0
    return int(mask.detach().cpu().bool().sum().item())


def _sample_count(total: int, ratio: float) -> int:
    if total <= 0:
        return 0
    if ratio <= 0:
        return 0
    if ratio >= 1 and float(ratio).is_integer():
        return min(total, int(ratio))
    return max(1, min(total, int(round(total * ratio))))


if __name__ == "__main__":
    main()
