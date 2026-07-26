from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = {
    "anchor_lambda1": 2.0,
    "anchor_lambda2": 0.5,
    "edge_forget_loss_mode": "original_kl",
    "node_forget_loss_mode": "uniform",
    "forget_weight": 0.1,
    "finetune_lr": 0.01,
    "finetune_epochs": 50,
    "inpainting_repair_ratio": 0.35,
    "inpainting_edge_threshold": 0.50,
    "inpainting_max_added_edges": 256,
    "inpainting_cc_drop_threshold": 0.30,
    "inpainting_min_damage_ratio": 0.10,
}
CONFIG_KEYS = (
    "anchor_lambda1",
    "anchor_lambda2",
    "edge_forget_loss_mode",
    "node_forget_loss_mode",
    "forget_weight",
    "finetune_lr",
    "finetune_epochs",
    "inpainting_repair_ratio",
    "inpainting_edge_threshold",
    "inpainting_max_added_edges",
    "inpainting_cc_drop_threshold",
    "inpainting_min_damage_ratio",
)
PAIRED_DEFAULT_METRICS = (
    "overall_mia_auc",
    "privacy_gap",
    "privacy_score",
    "val_accuracy_drop",
    "val_f1_macro_drop",
    "test_accuracy_drop",
    "test_f1_macro_drop",
    "degree_kl_abs",
    "clustering_change_abs",
    "component_change_abs",
    "exact_retrain_js_mean",
    "exact_retrain_tv_mean",
    "exact_retrain_disagreement_rate",
    "unlearn_time_seconds",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Paper-grade HASI hyperparameter sweep with validation scoring and multi-seed aggregation.")
    parser.add_argument("--dataset", default="pubmed", choices=["cora", "citeseer", "pubmed", "primekg-full-nosource", "primekg-disease-gene-small", "primekg-disease-gene-small-nosource", "hetionet-small-nosource", "hetionet-full-nosource", "ppi-homo-sl-filtered", "ppi-inductive-sl-filtered", "ppi-inductive-sl-mostfreq-filtered", "ppi-inductive-sl-balanced20-filtered", "ppi-inductive-sl-balanced10-filtered"])
    parser.add_argument("--unlearning_type", default="node", choices=["node", "edge", "feature"])
    parser.add_argument("--ratios", default="0.1", help="Comma-separated forget ratios.")
    parser.add_argument("--seeds", default="42,123,2024", help="Comma-separated seeds used for shared base artifacts and runs.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "hasi_default.yaml"))
    parser.add_argument("--data_root", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--base_artifact_root", default=str(ROOT / "results" / "shared_base"))
    parser.add_argument("--output_root", default=str(ROOT / "results" / "tuning"))
    parser.add_argument("--output_dir", default="", help="Explicit task output directory. Overrides --output_root/<dataset>/<type> when set.")
    parser.add_argument("--conda_env", default="graphunlearning")
    parser.add_argument("--python", default="", help="Python executable. Defaults to conda run -n <conda_env> python.")
    parser.add_argument("--device", default=None, help="Device forwarded to experiments/run_hasi.py, e.g. cuda:0.")
    parser.add_argument("--graph_compute_backend", default=None, choices=["auto", "cpu", "torch"])
    parser.add_argument("--hub_ppr_batch_size", type=int, default=None)
    parser.add_argument("--hub_score_cache_root", default="")
    parser.add_argument("--hub_score_cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--require_hub_score_cache_hit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--exact_retrain_reference_root",
        default="",
        help="Root for Edge exact-retrain references, laid out as <root>/r0p05/base42_fseed42.pt.",
    )
    parser.add_argument("--forget_set_file", default="", help="Optional fixed forget-set file used for every run.")
    parser.add_argument(
        "--forget_set_dir",
        default="",
        help=(
            "Directory of pre-generated forget-set JSON files. "
            "Defaults to experiments/forget_sets/<dataset> when present. "
            "Matching files for --unlearning_type/--ratios/--seeds are used."
        ),
    )
    parser.add_argument(
        "--edge_forget_scope",
        default="all",
        choices=["all", "train_subgraph"],
        help="Scope for generated edge forget targets when no fixed forget-set file is used.",
    )
    parser.add_argument("--grid", default="coarse", choices=["coarse", "full", "privacy_refine", "feature_wide_refine", "feature_utility_privacy_refine", "feature_primekg_utility_guarded_refine", "node_privacy_wide_refine", "node_hetionet_robust_pareto_refine", "node_pubmed_refine", "node_pubmed_privacy_refine", "node_pubmed_privacy_guarded_refine", "node_primekg_privacy_guarded_refine", "edge_refine", "edge_hetionet_privacy_guarded_refine", "edge_hetionet_robust_pareto_refine", "edge_primekg_privacy_guarded_refine", "edge_pubmed_privacy_guarded_refine", "edge_pubmed_privacy_strict_refine", "edge_pubmed_privacy_aggressive_refine", "edge_repair_refine", "edge_comprehensive_refine"])
    parser.add_argument(
        "--candidate_configs_file",
        default="",
        help="Optional JSON list of candidate configs, for validating top configs from an earlier search stage.",
    )
    parser.add_argument(
        "--reference_config_file",
        default="",
        help=(
            "Optional JSON config used as the paired reference instead of the "
            "built-in HASI default. The reference is still identified by "
            "is_default_reference in sweep outputs."
        ),
    )
    parser.add_argument(
        "--score_mode",
        default="structure",
        choices=["quick", "formal", "structure", "privacy", "privacy_guarded", "privacy_improvement_guarded", "privacy_robust_pareto", "edge_privacy_guarded", "edge_privacy_strict", "edge_privacy_aggressive", "utility_privacy", "feature_utility"],
    )
    parser.add_argument("--max_configs", type=int, default=0, help="Limit candidate hyperparameter configs; 0 means all.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top configs to write separately.")
    parser.add_argument(
        "--top_configs_require_hard_constraints",
        action="store_true",
        help="Write top_configs only from candidates that pass hard constraints.",
    )
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--append_existing_configs",
        action="store_true",
        help=(
            "Append new configs in an existing output directory. Existing config JSONs are "
            "matched by hyperparameters and skipped; new configs continue from the next cfg id."
        ),
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--include_default_reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--accuracy_drop_limit", type=float, default=0.05)
    parser.add_argument("--f1_macro_drop_limit", type=float, default=0.05)
    parser.add_argument("--mia_auc_limit", type=float, default=0.60)
    parser.add_argument("--runtime_multiplier", type=float, default=2.0)
    parser.add_argument(
        "--require_privacy_improvement_vs_default",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require candidate privacy_gap_mean to improve over the default reference.",
    )
    parser.add_argument(
        "--privacy_gap_improvement_margin",
        type=float,
        default=0.0,
        help="Minimum privacy_gap_mean decrease required when --require_privacy_improvement_vs_default is set.",
    )
    parser.add_argument(
        "--paired_hard_constraints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply default-relative hard constraints to every matching ratio/seed pair instead of aggregate means.",
    )
    parser.add_argument(
        "--val_accuracy_drop_delta_limit_vs_default",
        type=float,
        default=None,
        help="Maximum allowed candidate validation accuracy-drop increase over the default reference.",
    )
    parser.add_argument(
        "--val_f1_macro_drop_delta_limit_vs_default",
        type=float,
        default=None,
        help="Maximum allowed candidate validation macro-F1-drop increase over the default reference.",
    )
    parser.add_argument(
        "--test_accuracy_drop_delta_limit_vs_default",
        type=float,
        default=None,
        help="Maximum allowed candidate test accuracy-drop increase over the default reference.",
    )
    parser.add_argument(
        "--test_f1_macro_drop_delta_limit_vs_default",
        type=float,
        default=None,
        help="Maximum allowed candidate test macro-F1-drop increase over the default reference.",
    )
    parser.add_argument(
        "--degree_kl_abs_delta_limit_vs_default",
        type=float,
        default=None,
        help="Maximum allowed candidate degree-KL absolute-damage increase over the default reference.",
    )
    parser.add_argument(
        "--clustering_change_abs_delta_limit_vs_default",
        type=float,
        default=None,
        help="Maximum allowed candidate clustering absolute-damage increase over the default reference.",
    )
    parser.add_argument(
        "--component_change_abs_delta_limit_vs_default",
        type=float,
        default=None,
        help="Maximum allowed candidate component-count absolute-damage increase over the default reference.",
    )
    parser.add_argument(
        "--exact_retrain_js_delta_limit_vs_default",
        type=float,
        default=None,
        help="Maximum allowed candidate JS-distance-to-exact-retrain increase over the default reference.",
    )
    parser.add_argument(
        "--exact_retrain_tv_delta_limit_vs_default",
        type=float,
        default=None,
        help="Maximum allowed candidate TV-distance-to-exact-retrain increase over the default reference.",
    )
    parser.add_argument(
        "--exact_retrain_disagreement_delta_limit_vs_default",
        type=float,
        default=None,
        help="Maximum allowed candidate prediction-disagreement-to-exact-retrain increase over the default reference.",
    )
    parser.add_argument(
        "--unlearn_time_delta_limit_vs_default",
        type=float,
        default=None,
        help="Maximum allowed candidate unlearning-time increase over the default reference.",
    )
    parser.add_argument(
        "--runtime_ratio_mean_limit_vs_default",
        type=float,
        default=None,
        help="Maximum allowed mean runtime ratio vs the default reference.",
    )
    parser.add_argument("--stability_weight", type=float, default=0.10, help="Extra penalty for cross-seed/ratio std in config-level score.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grid.startswith("edge_") and args.unlearning_type != "edge":
        raise ValueError(f"--grid {args.grid} is only valid with --unlearning_type edge.")
    ratios = _parse_floats(args.ratios)
    seeds = _parse_ints(args.seeds)
    forget_specs = _resolve_forget_specs(args, ratios, seeds)
    if args.output_dir:
        dataset_dir = Path(args.output_dir)
        dataset_root = dataset_dir.parent
    else:
        dataset_root = Path(args.output_root) / args.dataset
        dataset_dir = dataset_root / args.unlearning_type
    runs_dir = dataset_dir / "runs"
    configs_dir = dataset_dir / "configs"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    configs = _candidate_configs_from_file(args.candidate_configs_file) if args.candidate_configs_file else _grid(args.grid)
    if args.max_configs > 0:
        configs = configs[: args.max_configs]

    existing_signatures, next_config_index = _existing_config_signatures(configs_dir)
    run_specs: list[tuple[str, dict[str, Any], bool]] = []
    if args.include_default_reference:
        default_cfg = (
            _reference_config_from_file(args.reference_config_file)
            if args.reference_config_file
            else dict(DEFAULT_CONFIG)
        )
        if not args.append_existing_configs or _config_signature(default_cfg) not in existing_signatures:
            run_specs.append(("default", default_cfg, True))
    for idx, cfg in enumerate(configs):
        signature = _config_signature(cfg)
        if args.append_existing_configs and signature in existing_signatures:
            continue
        if args.append_existing_configs:
            config_id = f"{next_config_index:04d}"
            next_config_index += 1
        else:
            config_id = f"{idx:04d}"
        existing_signatures.add(signature)
        run_specs.append((config_id, cfg, False))

    rows: list[dict[str, Any]] = _read_csv(dataset_dir / "run_summary.csv") if args.append_existing_configs else []
    total = len(run_specs) * len(forget_specs)
    run_index = 0
    for config_id, cfg, is_reference in run_specs:
        config_path = configs_dir / f"config_{config_id}.json"
        config_path.write_text(json.dumps({"is_default_reference": is_reference, **cfg}, indent=2) + "\n", encoding="utf-8")
        for spec in forget_specs:
            run_index += 1
            output_path = runs_dir / f"hasi_{spec.output_stem}_cfg{config_id}.json"
            row = {
                "config_id": config_id,
                "is_default_reference": is_reference,
                "dataset": args.dataset,
                "unlearning_type": args.unlearning_type,
                "ratio": spec.ratio,
                "seed": spec.seed,
                "forget_set_file": str(spec.path) if spec.path is not None else "",
                **cfg,
                "output": str(output_path),
            }
            if output_path.exists() and args.skip_existing:
                payload = _load_json(output_path)
                row.update(_metrics_from_payload(payload))
                row["status"] = "ok"
                rows.append(row)
                continue

            cmd = _run_command(args, cfg, spec, output_path)
            print(json.dumps({"run": run_index, "total": total, "config_id": config_id, "reference": is_reference, "cmd": cmd}, ensure_ascii=False), flush=True)
            if args.dry_run:
                row["status"] = "dry_run"
                rows.append(row)
                continue

            completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            row["returncode"] = completed.returncode
            if completed.returncode != 0:
                row["status"] = "failed"
                row["stderr_tail"] = completed.stderr[-2000:]
            else:
                row["status"] = "ok"
                payload = _load_json(output_path)
                row.update(_metrics_from_payload(payload))
            rows.append(row)
            _write_outputs(dataset_dir, rows, args)

    output = _write_outputs(dataset_dir, rows, args)
    best = output["best"]
    if best:
        _write_best_config(dataset_dir / "best_config.yaml", best[0], args)
        print(json.dumps({"best_config": best[0], "output_dir": str(dataset_dir), "dataset_root": str(dataset_root)}, indent=2))
    else:
        print(json.dumps({"output_dir": str(dataset_dir), "dataset_root": str(dataset_root), "best_config": None}, indent=2))


def _grid(name: str) -> list[dict[str, Any]]:
    if name == "privacy_refine":
        return _privacy_refine_grid()
    if name == "feature_wide_refine":
        return _feature_wide_refine_grid()
    if name == "feature_utility_privacy_refine":
        return _feature_utility_privacy_refine_grid()
    if name == "feature_primekg_utility_guarded_refine":
        return _feature_primekg_utility_guarded_refine_grid()
    if name == "node_privacy_wide_refine":
        return _node_privacy_wide_refine_grid()
    if name == "node_hetionet_robust_pareto_refine":
        return _node_hetionet_robust_pareto_refine_grid()
    if name == "node_pubmed_refine":
        return _node_pubmed_refine_grid()
    if name == "node_pubmed_privacy_refine":
        return _node_pubmed_privacy_refine_grid()
    if name == "node_pubmed_privacy_guarded_refine":
        return _node_pubmed_privacy_guarded_refine_grid()
    if name == "node_primekg_privacy_guarded_refine":
        return _node_primekg_privacy_guarded_refine_grid()
    if name == "edge_refine":
        return _edge_refine_grid()
    if name == "edge_hetionet_privacy_guarded_refine":
        return _edge_hetionet_privacy_guarded_refine_grid()
    if name == "edge_hetionet_robust_pareto_refine":
        return _edge_hetionet_robust_pareto_refine_grid()
    if name == "edge_primekg_privacy_guarded_refine":
        return _edge_primekg_privacy_guarded_refine_grid()
    if name == "edge_pubmed_privacy_guarded_refine":
        return _edge_pubmed_privacy_guarded_refine_grid()
    if name == "edge_pubmed_privacy_strict_refine":
        return _edge_pubmed_privacy_strict_refine_grid()
    if name == "edge_pubmed_privacy_aggressive_refine":
        return _edge_pubmed_privacy_aggressive_refine_grid()
    if name == "edge_repair_refine":
        return _edge_repair_refine_grid()
    if name == "edge_comprehensive_refine":
        return _edge_comprehensive_refine_grid()
    if name == "full":
        lambda1 = [0.5, 1.0, 2.0, 5.0]
        lambda2 = [0.1, 0.5, 1.0]
        forget_weight = [0.0, 0.01, 0.05, 0.1]
        finetune_lr = [0.003, 0.005, 0.01]
    else:
        lambda1 = [1.0, 2.0, 5.0]
        lambda2 = [0.1, 0.5]
        forget_weight = [0.0, 0.05, 0.1]
        finetune_lr = [0.003, 0.005]
    return [
        _candidate(anchor_lambda1=l1, anchor_lambda2=l2, forget_weight=fw, finetune_lr=lr)
        for l1, l2, fw, lr in itertools.product(lambda1, lambda2, forget_weight, finetune_lr)
    ]


def _candidate_configs_from_file(path: str) -> list[dict[str, Any]]:
    payload = _load_json(Path(path))
    if isinstance(payload, dict):
        payload = payload.get("configs") or payload.get("top_configs") or payload.get("candidates")
    if not isinstance(payload, list):
        raise ValueError(f"--candidate_configs_file must contain a JSON list of configs: {path}")

    configs: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"candidate config #{index} is not an object in {path}")
        missing = [
            key
            for key in CONFIG_KEYS
            if key not in item and key != "node_forget_loss_mode"
        ]
        if missing:
            raise ValueError(f"candidate config #{index} is missing keys {missing} in {path}")
        cfg = {
            key: item.get(key, "uniform")
            if key == "node_forget_loss_mode"
            else item[key]
            for key in CONFIG_KEYS
        }
        signature = _config_signature(cfg)
        if signature in seen:
            continue
        seen.add(signature)
        configs.append(cfg)
    if not configs:
        raise ValueError(f"--candidate_configs_file did not yield any configs: {path}")
    return configs


def _reference_config_from_file(path: str) -> dict[str, Any]:
    payload = _load_json(Path(path))
    if not isinstance(payload, dict):
        raise ValueError(f"--reference_config_file must contain one JSON config object: {path}")
    missing = [
        key
        for key in CONFIG_KEYS
        if key not in payload and key != "node_forget_loss_mode"
    ]
    if missing:
        raise ValueError(f"reference config is missing keys {missing} in {path}")
    return {
        key: payload.get(key, "uniform")
        if key == "node_forget_loss_mode"
        else payload[key]
        for key in CONFIG_KEYS
    }


def _candidate(**overrides: Any) -> dict[str, Any]:
    if "edge_forget_loss_mode" in overrides and "node_forget_loss_mode" not in overrides:
        overrides["node_forget_loss_mode"] = overrides["edge_forget_loss_mode"]
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(overrides)
    return cfg


def _dedupe_configs(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for cfg in configs:
        signature = tuple(cfg.get(key) for key in CONFIG_KEYS)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(cfg)
    return deduped


def _node_pubmed_refine_grid() -> list[dict[str, Any]]:
    """Node round-2 sweep around PubMed node coarse winners: stronger anchoring, modest forget loss."""
    return [
        _candidate(anchor_lambda1=1.0, anchor_lambda2=0.5, forget_weight=0.05, finetune_lr=0.003),
        _candidate(anchor_lambda1=1.0, anchor_lambda2=0.5, forget_weight=0.05, finetune_lr=0.005),
        _candidate(anchor_lambda1=2.0, anchor_lambda2=0.5, forget_weight=0.05, finetune_lr=0.003),
        _candidate(anchor_lambda1=2.0, anchor_lambda2=0.5, forget_weight=0.05, finetune_lr=0.005),
        _candidate(anchor_lambda1=5.0, anchor_lambda2=0.5, forget_weight=0.05, finetune_lr=0.003),
        _candidate(anchor_lambda1=5.0, anchor_lambda2=0.5, forget_weight=0.05, finetune_lr=0.005),
        _candidate(anchor_lambda1=5.0, anchor_lambda2=0.5, forget_weight=0.1, finetune_lr=0.003),
        _candidate(anchor_lambda1=5.0, anchor_lambda2=0.5, forget_weight=0.1, finetune_lr=0.005),
        _candidate(anchor_lambda1=2.0, anchor_lambda2=1.0, forget_weight=0.05, finetune_lr=0.005),
        _candidate(anchor_lambda1=5.0, anchor_lambda2=1.0, forget_weight=0.05, finetune_lr=0.005),
    ]


def _node_pubmed_privacy_refine_grid() -> list[dict[str, Any]]:
    """PubMed node privacy sweep with guarded utility: stronger forgetting, weaker anchoring."""
    configs: list[dict[str, Any]] = []

    for anchor_lambda1, anchor_lambda2, forget_weight, (lr, epochs) in itertools.product(
        [2.0, 5.0],
        [0.1, 0.5],
        [0.1, 0.2, 0.3, 0.5],
        [(0.003, 80), (0.005, 80)],
    ):
        configs.append(
            _candidate(
                anchor_lambda1=anchor_lambda1,
                anchor_lambda2=anchor_lambda2,
                forget_weight=forget_weight,
                edge_forget_loss_mode="original_kl",
                finetune_lr=lr,
                finetune_epochs=epochs,
                inpainting_repair_ratio=0.35,
            )
        )

    for forget_weight, (anchor_lambda1, anchor_lambda2), repair_ratio in itertools.product(
        [0.3, 0.5, 0.8, 1.0],
        [(1.0, 0.1), (1.0, 0.05), (0.5, 0.05), (0.25, 0.02), (0.0, 0.02)],
        [0.05, 0.15],
    ):
        configs.append(
            _candidate(
                anchor_lambda1=anchor_lambda1,
                anchor_lambda2=anchor_lambda2,
                forget_weight=forget_weight,
                edge_forget_loss_mode="uniform",
                finetune_lr=0.003,
                finetune_epochs=120,
                inpainting_repair_ratio=repair_ratio,
            )
        )

    for forget_weight, (anchor_lambda1, anchor_lambda2), (lr, epochs) in itertools.product(
        [0.5, 0.8, 1.0],
        [(0.5, 0.05), (0.25, 0.02), (0.0, 0.0)],
        [(0.003, 150), (0.005, 120)],
    ):
        configs.append(
            _candidate(
                anchor_lambda1=anchor_lambda1,
                anchor_lambda2=anchor_lambda2,
                forget_weight=forget_weight,
                edge_forget_loss_mode="uniform",
                finetune_lr=lr,
                finetune_epochs=epochs,
                inpainting_repair_ratio=0.05,
            )
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for cfg in configs:
        signature = tuple(sorted(cfg.items()))
        if signature not in seen:
            seen.add(signature)
            deduped.append(cfg)
    return deduped


def _node_pubmed_privacy_guarded_refine_grid() -> list[dict[str, Any]]:
    """PubMed node privacy sweep that keeps utility/runtime close to the default reference."""
    configs: list[dict[str, Any]] = []

    for anchor_lambda1, anchor_lambda2, forget_weight, loss_mode, lr in itertools.product(
        [2.0, 5.0],
        [0.1, 0.5],
        [0.1, 0.2, 0.3, 0.5],
        ["original_kl", "uniform"],
        [0.003, 0.005],
    ):
        configs.append(
            _candidate(
                anchor_lambda1=anchor_lambda1,
                anchor_lambda2=anchor_lambda2,
                forget_weight=forget_weight,
                edge_forget_loss_mode=loss_mode,
                finetune_lr=lr,
                finetune_epochs=50,
                inpainting_repair_ratio=0.35,
            )
        )

    for forget_weight, (anchor_lambda1, anchor_lambda2), repair_ratio, (lr, epochs) in itertools.product(
        [0.2, 0.3, 0.5, 0.8],
        [(2.0, 0.1), (1.0, 0.1), (1.0, 0.05), (0.5, 0.05)],
        [0.15, 0.35],
        [(0.003, 50), (0.003, 80)],
    ):
        configs.append(
            _candidate(
                anchor_lambda1=anchor_lambda1,
                anchor_lambda2=anchor_lambda2,
                forget_weight=forget_weight,
                edge_forget_loss_mode="uniform",
                finetune_lr=lr,
                finetune_epochs=epochs,
                inpainting_repair_ratio=repair_ratio,
            )
        )

    for forget_weight, (anchor_lambda1, anchor_lambda2), repair_ratio in itertools.product(
        [0.5, 0.8],
        [(0.5, 0.05), (0.25, 0.02)],
        [0.05, 0.15],
    ):
        configs.append(
            _candidate(
                anchor_lambda1=anchor_lambda1,
                anchor_lambda2=anchor_lambda2,
                forget_weight=forget_weight,
                edge_forget_loss_mode="uniform",
                finetune_lr=0.003,
                finetune_epochs=80,
                inpainting_repair_ratio=repair_ratio,
            )
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for cfg in configs:
        signature = tuple(sorted(cfg.items()))
        if signature not in seen:
            seen.add(signature)
            deduped.append(cfg)
    return deduped


def _node_primekg_privacy_guarded_refine_grid() -> list[dict[str, Any]]:
    """PrimeKG-full node privacy sweep with all HASI components kept active."""
    configs: list[dict[str, Any]] = []

    def add(
        *,
        anchor_lambda1: float,
        anchor_lambda2: float,
        forget_weight: float,
        edge_forget_loss_mode: str,
        finetune_lr: float,
        finetune_epochs: int,
        inpainting_repair_ratio: float,
    ) -> None:
        configs.append(
            _candidate(
                anchor_lambda1=anchor_lambda1,
                anchor_lambda2=anchor_lambda2,
                forget_weight=forget_weight,
                edge_forget_loss_mode=edge_forget_loss_mode,
                finetune_lr=finetune_lr,
                finetune_epochs=finetune_epochs,
                inpainting_repair_ratio=inpainting_repair_ratio,
                inpainting_edge_threshold=0.5,
                inpainting_max_added_edges=256,
                inpainting_cc_drop_threshold=0.3,
                inpainting_min_damage_ratio=0.1,
            )
        )

    # The first candidate mirrors the single-seed direction that improved PrimeKG node MIA,
    # then nearby variants trade a small utility allowance for stronger privacy pressure.
    add(anchor_lambda1=2.0, anchor_lambda2=0.10, forget_weight=0.50, edge_forget_loss_mode="original_kl", finetune_lr=0.010, finetune_epochs=50, inpainting_repair_ratio=0.20)
    add(anchor_lambda1=2.0, anchor_lambda2=0.10, forget_weight=0.30, edge_forget_loss_mode="original_kl", finetune_lr=0.010, finetune_epochs=50, inpainting_repair_ratio=0.20)
    add(anchor_lambda1=2.0, anchor_lambda2=0.10, forget_weight=0.20, edge_forget_loss_mode="original_kl", finetune_lr=0.010, finetune_epochs=50, inpainting_repair_ratio=0.20)
    add(anchor_lambda1=2.0, anchor_lambda2=0.20, forget_weight=0.50, edge_forget_loss_mode="original_kl", finetune_lr=0.005, finetune_epochs=50, inpainting_repair_ratio=0.20)
    add(anchor_lambda1=2.0, anchor_lambda2=0.20, forget_weight=0.30, edge_forget_loss_mode="original_kl", finetune_lr=0.005, finetune_epochs=50, inpainting_repair_ratio=0.25)
    add(anchor_lambda1=2.0, anchor_lambda2=0.30, forget_weight=0.30, edge_forget_loss_mode="original_kl", finetune_lr=0.005, finetune_epochs=50, inpainting_repair_ratio=0.25)
    add(anchor_lambda1=2.0, anchor_lambda2=0.50, forget_weight=0.20, edge_forget_loss_mode="original_kl", finetune_lr=0.005, finetune_epochs=50, inpainting_repair_ratio=0.35)

    add(anchor_lambda1=5.0, anchor_lambda2=0.50, forget_weight=0.20, edge_forget_loss_mode="original_kl", finetune_lr=0.005, finetune_epochs=50, inpainting_repair_ratio=0.35)
    add(anchor_lambda1=5.0, anchor_lambda2=0.50, forget_weight=0.30, edge_forget_loss_mode="original_kl", finetune_lr=0.005, finetune_epochs=50, inpainting_repair_ratio=0.35)
    add(anchor_lambda1=5.0, anchor_lambda2=0.10, forget_weight=0.30, edge_forget_loss_mode="original_kl", finetune_lr=0.005, finetune_epochs=50, inpainting_repair_ratio=0.20)
    add(anchor_lambda1=5.0, anchor_lambda2=0.10, forget_weight=0.50, edge_forget_loss_mode="original_kl", finetune_lr=0.005, finetune_epochs=50, inpainting_repair_ratio=0.20)

    add(anchor_lambda1=1.0, anchor_lambda2=0.10, forget_weight=0.30, edge_forget_loss_mode="uniform", finetune_lr=0.005, finetune_epochs=80, inpainting_repair_ratio=0.20)
    add(anchor_lambda1=1.0, anchor_lambda2=0.10, forget_weight=0.50, edge_forget_loss_mode="uniform", finetune_lr=0.003, finetune_epochs=80, inpainting_repair_ratio=0.15)
    add(anchor_lambda1=2.0, anchor_lambda2=0.10, forget_weight=0.50, edge_forget_loss_mode="uniform", finetune_lr=0.005, finetune_epochs=50, inpainting_repair_ratio=0.15)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for cfg in configs:
        signature = tuple(cfg.get(key) for key in CONFIG_KEYS)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(cfg)
    return deduped


def _node_hetionet_robust_pareto_refine_grid() -> list[dict[str, Any]]:
    """Small all-components-active Hetionet node grid around the full default."""
    configs = [
        # Isolate modest forgetting-strength changes with the default anchors and repair.
        _candidate(forget_weight=0.05),
        _candidate(forget_weight=0.075),
        _candidate(forget_weight=0.125),
        _candidate(forget_weight=0.15),
        # Lower-rate variants test whether gentler optimization improves privacy stability.
        _candidate(forget_weight=0.075, finetune_lr=0.005),
        _candidate(forget_weight=0.10, finetune_lr=0.005),
        _candidate(forget_weight=0.15, finetune_lr=0.005),
        _candidate(forget_weight=0.20, finetune_lr=0.005),
        # Reduce anchoring gradually; both anchor terms remain active.
        _candidate(anchor_lambda1=2.0, anchor_lambda2=0.25, forget_weight=0.10, finetune_lr=0.005),
        _candidate(anchor_lambda1=2.0, anchor_lambda2=0.25, forget_weight=0.20, finetune_lr=0.005),
        _candidate(anchor_lambda1=1.0, anchor_lambda2=0.10, forget_weight=0.10, finetune_lr=0.005),
        _candidate(anchor_lambda1=1.0, anchor_lambda2=0.10, forget_weight=0.20, finetune_lr=0.005),
        # Conservative uniform-loss probes, far below the failed cfg0037 pressure.
        _candidate(anchor_lambda1=0.5, anchor_lambda2=0.05, edge_forget_loss_mode="uniform", forget_weight=0.20, finetune_lr=0.003, finetune_epochs=80, inpainting_repair_ratio=0.20),
        _candidate(anchor_lambda1=0.5, anchor_lambda2=0.05, edge_forget_loss_mode="uniform", forget_weight=0.30, finetune_lr=0.003, finetune_epochs=80, inpainting_repair_ratio=0.20),
    ]
    return _dedupe_configs(configs)


def _privacy_refine_grid() -> list[dict[str, Any]]:
    """Small privacy-oriented sweep for node/feature tasks after default privacy is weak."""
    return [
        _candidate(forget_weight=0.2, anchor_lambda2=0.1, inpainting_repair_ratio=0.20),
        _candidate(forget_weight=0.5, anchor_lambda2=0.1, inpainting_repair_ratio=0.20),
        _candidate(forget_weight=0.2, anchor_lambda1=1.0, anchor_lambda2=0.1, finetune_lr=0.005, finetune_epochs=80),
        _candidate(forget_weight=0.5, anchor_lambda1=1.0, anchor_lambda2=0.05, finetune_lr=0.005, finetune_epochs=80, inpainting_repair_ratio=0.15),
        _candidate(forget_weight=0.3, anchor_lambda1=1.0, anchor_lambda2=0.05, edge_forget_loss_mode="uniform", finetune_lr=0.005, finetune_epochs=80, inpainting_repair_ratio=0.25),
        _candidate(forget_weight=0.5, anchor_lambda1=0.5, anchor_lambda2=0.05, edge_forget_loss_mode="uniform", finetune_lr=0.003, finetune_epochs=100, inpainting_repair_ratio=0.15),
    ]


def _feature_wide_refine_grid() -> list[dict[str, Any]]:
    """Wider feature sweep around hetionet feature winners from the privacy-refine round."""
    anchor_lambda1 = [0.75, 1.0, 1.25]
    anchor_lambda2 = [0.03, 0.05, 0.10]
    forget_weight = [0.2, 0.3, 0.4]
    loss_mode = ["uniform", "original_kl"]
    lr_epochs = [(0.003, 100), (0.005, 80)]
    repair_ratio = [0.15, 0.25]
    return [
        _candidate(
            anchor_lambda1=l1,
            anchor_lambda2=l2,
            forget_weight=fw,
            edge_forget_loss_mode=loss,
            finetune_lr=lr,
            finetune_epochs=epochs,
            inpainting_repair_ratio=repair,
        )
        for l1, l2, fw, loss, (lr, epochs), repair in itertools.product(
            anchor_lambda1,
            anchor_lambda2,
            forget_weight,
            loss_mode,
            lr_epochs,
            repair_ratio,
        )
    ]


def _feature_utility_privacy_refine_grid() -> list[dict[str, Any]]:
    """Feature sweep for utility-preserving privacy refinement on Hetionet."""
    configs: list[dict[str, Any]] = []

    for anchor_lambda1, anchor_lambda2, forget_weight, (lr, epochs) in itertools.product(
        [0.5, 0.75, 1.0],
        [0.02, 0.03, 0.05],
        [0.15, 0.2, 0.25],
        [(0.003, 100)],
    ):
        configs.append(
            _candidate(
                anchor_lambda1=anchor_lambda1,
                anchor_lambda2=anchor_lambda2,
                forget_weight=forget_weight,
                edge_forget_loss_mode="uniform",
                finetune_lr=lr,
                finetune_epochs=epochs,
                inpainting_repair_ratio=0.25,
            )
        )

    for anchor_lambda1, anchor_lambda2, forget_weight, (lr, epochs) in itertools.product(
        [0.75, 1.0],
        [0.03, 0.05],
        [0.3],
        [(0.003, 100), (0.005, 80)],
    ):
        configs.append(
            _candidate(
                anchor_lambda1=anchor_lambda1,
                anchor_lambda2=anchor_lambda2,
                forget_weight=forget_weight,
                edge_forget_loss_mode="uniform",
                finetune_lr=lr,
                finetune_epochs=epochs,
                inpainting_repair_ratio=0.25,
            )
        )

    for anchor_lambda1, anchor_lambda2, forget_weight in itertools.product(
        [0.5, 0.75],
        [0.02],
        [0.3, 0.4],
    ):
        configs.append(
            _candidate(
                anchor_lambda1=anchor_lambda1,
                anchor_lambda2=anchor_lambda2,
                forget_weight=forget_weight,
                edge_forget_loss_mode="uniform",
                finetune_lr=0.003,
                finetune_epochs=120,
                inpainting_repair_ratio=0.25,
            )
        )

    for anchor_lambda2, forget_weight in itertools.product([0.03, 0.05], [0.2, 0.25]):
        configs.append(
            _candidate(
                anchor_lambda1=0.75,
                anchor_lambda2=anchor_lambda2,
                forget_weight=forget_weight,
                edge_forget_loss_mode="original_kl",
                finetune_lr=0.003,
                finetune_epochs=100,
                inpainting_repair_ratio=0.25,
            )
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    for cfg in configs:
        signature = tuple(sorted(cfg.items()))
        if signature not in seen:
            seen.add(signature)
            deduped.append(cfg)
    return deduped


def _feature_primekg_utility_guarded_refine_grid() -> list[dict[str, Any]]:
    """PrimeKG-full feature sweep centered on the full-dataset default behavior."""
    configs = [
        # Conservative forgetting-strength probes around the full-dataset default.
        _candidate(forget_weight=0.05),
        _candidate(forget_weight=0.075),
        _candidate(forget_weight=0.125),
        _candidate(forget_weight=0.15),
        # Reduce anchoring gradually without jumping to the Hetionet-scale regime.
        _candidate(anchor_lambda2=0.25),
        _candidate(anchor_lambda2=0.10),
        _candidate(anchor_lambda1=1.5, anchor_lambda2=0.25),
        _candidate(anchor_lambda1=1.0, anchor_lambda2=0.10),
        # Lower learning-rate controls that retain the default loss and repair policy.
        _candidate(finetune_lr=0.005),
        _candidate(finetune_lr=0.005, finetune_epochs=80),
        _candidate(finetune_epochs=80),
        # Moderately smaller repair regions for utility preservation.
        _candidate(inpainting_repair_ratio=0.25, inpainting_max_added_edges=128),
        _candidate(anchor_lambda1=1.5, anchor_lambda2=0.10, finetune_lr=0.005, finetune_epochs=80, inpainting_repair_ratio=0.25),
        # Uniform-loss probes remain close to default and keep every component active.
        _candidate(anchor_lambda1=1.0, anchor_lambda2=0.10, edge_forget_loss_mode="uniform", forget_weight=0.10, finetune_lr=0.005, finetune_epochs=80, inpainting_repair_ratio=0.25),
        _candidate(anchor_lambda1=1.0, anchor_lambda2=0.05, edge_forget_loss_mode="uniform", forget_weight=0.15, finetune_lr=0.005, finetune_epochs=80, inpainting_repair_ratio=0.25),
    ]
    return _dedupe_configs(configs)


def _node_privacy_wide_refine_grid() -> list[dict[str, Any]]:
    """Second-round node sweep for stronger privacy around Hetionet node cfg0005."""
    configs: list[dict[str, Any]] = [
        _candidate(
            anchor_lambda1=0.5,
            anchor_lambda2=0.05,
            forget_weight=0.5,
            edge_forget_loss_mode="uniform",
            finetune_lr=0.003,
            finetune_epochs=100,
            inpainting_repair_ratio=0.15,
        )
    ]

    for forget_weight in [0.8, 1.0, 1.5]:
        for anchor_lambda1, anchor_lambda2 in [(0.5, 0.05), (0.25, 0.05), (0.25, 0.02), (0.0, 0.02)]:
            for repair_ratio in [0.05, 0.10]:
                configs.append(
                    _candidate(
                        anchor_lambda1=anchor_lambda1,
                        anchor_lambda2=anchor_lambda2,
                        forget_weight=forget_weight,
                        edge_forget_loss_mode="uniform",
                        finetune_lr=0.003,
                        finetune_epochs=120,
                        inpainting_repair_ratio=repair_ratio,
                    )
                )

    for forget_weight in [0.8, 1.0, 1.5]:
        for anchor_lambda1, anchor_lambda2 in [(0.25, 0.02), (0.0, 0.02), (0.0, 0.0)]:
            for finetune_lr, finetune_epochs in [(0.003, 150), (0.005, 120)]:
                configs.append(
                    _candidate(
                        anchor_lambda1=anchor_lambda1,
                        anchor_lambda2=anchor_lambda2,
                        forget_weight=forget_weight,
                        edge_forget_loss_mode="uniform",
                        finetune_lr=finetune_lr,
                        finetune_epochs=finetune_epochs,
                        inpainting_repair_ratio=0.05,
                    )
                )

    return configs

def _edge_refine_grid() -> list[dict[str, Any]]:
    """First-pass edge sweep: focus on privacy loss and repair strength."""
    edge_forget_loss_mode = ["original_kl", "uniform", "none"]
    forget_weight = [0.05, 0.1, 0.2, 0.5]
    finetune_lr = [0.003, 0.005, 0.01]
    repair_ratio = [0.2, 0.35, 0.5]
    edge_threshold = [0.4, 0.5, 0.6]
    return [
        _candidate(
            edge_forget_loss_mode=loss_mode,
            forget_weight=fw,
            finetune_lr=lr,
            inpainting_repair_ratio=repair,
            inpainting_edge_threshold=threshold,
        )
        for loss_mode, fw, lr, repair, threshold in itertools.product(
            edge_forget_loss_mode,
            forget_weight,
            finetune_lr,
            repair_ratio,
            edge_threshold,
        )
    ]


def _edge_hetionet_privacy_guarded_refine_grid() -> list[dict[str, Any]]:
    """Hetionet full edge sweep: improve privacy while guarding utility, structure, and retrain alignment."""
    configs: list[dict[str, Any]] = []

    def add(**overrides: Any) -> None:
        base = {
            "anchor_lambda1": 2.0,
            "anchor_lambda2": 0.5,
            "edge_forget_loss_mode": "original_kl",
            "forget_weight": 0.05,
            "finetune_lr": 0.005,
            "finetune_epochs": 50,
            "inpainting_repair_ratio": 0.35,
            "inpainting_edge_threshold": 0.5,
            "inpainting_max_added_edges": 128,
            "inpainting_cc_drop_threshold": 0.4,
            "inpainting_min_damage_ratio": 0.1,
        }
        base.update(overrides)
        configs.append(_candidate(**base))

    # Low privacy-gap pocket found in the previous Hetionet edge search.
    for forget_weight in [0.03, 0.05, 0.075]:
        for max_edges in [96, 128, 160]:
            add(
                forget_weight=forget_weight,
                inpainting_max_added_edges=max_edges,
                inpainting_cc_drop_threshold=0.4,
                inpainting_min_damage_ratio=0.1,
            )

    # Tighten/loosen repair slightly around the same pocket, while keeping graph damage guarded.
    for repair_ratio, min_damage in [(0.25, 0.05), (0.30, 0.05), (0.30, 0.1), (0.35, 0.05)]:
        add(
            forget_weight=0.05,
            finetune_lr=0.005,
            finetune_epochs=50,
            inpainting_repair_ratio=repair_ratio,
            inpainting_max_added_edges=128,
            inpainting_cc_drop_threshold=0.4,
            inpainting_min_damage_ratio=min_damage,
        )

    # Check whether lower learning rate or more fine-tuning preserves utility/alignment without losing privacy.
    for forget_weight in [0.05, 0.075]:
        for finetune_lr, finetune_epochs in [(0.003, 60), (0.003, 80), (0.005, 60), (0.007, 50)]:
            add(
                forget_weight=forget_weight,
                finetune_lr=finetune_lr,
                finetune_epochs=finetune_epochs,
                inpainting_max_added_edges=128,
            )

    # Include the prior full-dataset edge utility/alignment winner and nearby variants as controls.
    for forget_weight, finetune_lr, finetune_epochs in [(0.1, 0.003, 60), (0.1, 0.005, 50), (0.125, 0.003, 60)]:
        add(
            forget_weight=forget_weight,
            finetune_lr=finetune_lr,
            finetune_epochs=finetune_epochs,
            inpainting_repair_ratio=0.35,
            inpainting_edge_threshold=0.5,
            inpainting_max_added_edges=128,
            inpainting_cc_drop_threshold=0.4,
            inpainting_min_damage_ratio=0.1,
        )

    # A few uniform-loss probes from PubMed privacy tuning, kept conservative for Hetionet utility.
    for forget_weight, anchor_lambda1, anchor_lambda2 in [(0.1, 1.0, 0.1), (0.15, 1.0, 0.1), (0.2, 0.5, 0.05)]:
        add(
            anchor_lambda1=anchor_lambda1,
            anchor_lambda2=anchor_lambda2,
            edge_forget_loss_mode="uniform",
            forget_weight=forget_weight,
            finetune_lr=0.003,
            finetune_epochs=80,
            inpainting_repair_ratio=0.25,
            inpainting_edge_threshold=0.6,
            inpainting_max_added_edges=96,
            inpainting_cc_drop_threshold=0.35,
            inpainting_min_damage_ratio=0.05,
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for cfg in configs:
        signature = tuple(cfg.get(key) for key in CONFIG_KEYS)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(cfg)
    return deduped


def _edge_hetionet_robust_pareto_refine_grid() -> list[dict[str, Any]]:
    """Default-near Hetionet edge grid for paired multi-seed Pareto screening."""
    configs = [
        # Keep the default graph-repair policy and isolate forgetting pressure.
        _candidate(forget_weight=0.01),
        _candidate(forget_weight=0.02),
        _candidate(forget_weight=0.03),
        _candidate(forget_weight=0.05),
        _candidate(forget_weight=0.075),
        # Lower-rate variants around the seed-42 privacy pocket.
        _candidate(forget_weight=0.01, finetune_lr=0.005),
        _candidate(forget_weight=0.02, finetune_lr=0.005),
        _candidate(forget_weight=0.03, finetune_lr=0.005),
        _candidate(forget_weight=0.05, finetune_lr=0.005),
        _candidate(forget_weight=0.075, finetune_lr=0.005),
        # Repair-cap controls from the previous search, with every component active.
        _candidate(forget_weight=0.02, finetune_lr=0.005, inpainting_max_added_edges=96, inpainting_cc_drop_threshold=0.4),
        _candidate(forget_weight=0.03, finetune_lr=0.005, inpainting_max_added_edges=96, inpainting_cc_drop_threshold=0.4),
        _candidate(forget_weight=0.02, finetune_lr=0.005, inpainting_max_added_edges=128, inpainting_cc_drop_threshold=0.4),
        _candidate(forget_weight=0.03, finetune_lr=0.005, inpainting_max_added_edges=128, inpainting_cc_drop_threshold=0.4),
    ]
    return _dedupe_configs(configs)


def _edge_pubmed_privacy_guarded_refine_grid() -> list[dict[str, Any]]:
    """PubMed edge privacy sweep that uses multi-seed search and guards utility/runtime/alignment."""
    configs: list[dict[str, Any]] = []

    for forget_weight, finetune_lr, repair_ratio, threshold in itertools.product(
        [0.05, 0.075, 0.1, 0.125, 0.15],
        [0.005, 0.007, 0.01],
        [0.2, 0.35, 0.5],
        [0.4, 0.5, 0.6],
    ):
        configs.append(
            _candidate(
                edge_forget_loss_mode="original_kl",
                forget_weight=forget_weight,
                finetune_lr=finetune_lr,
                finetune_epochs=50,
                inpainting_repair_ratio=repair_ratio,
                inpainting_edge_threshold=threshold,
            )
        )

    for loss_mode, forget_weight, finetune_lr, repair_ratio, threshold in itertools.product(
        ["uniform", "none"],
        [0.05, 0.1, 0.2],
        [0.005, 0.01],
        [0.2, 0.35],
        [0.4, 0.5],
    ):
        configs.append(
            _candidate(
                edge_forget_loss_mode=loss_mode,
                forget_weight=forget_weight,
                finetune_lr=finetune_lr,
                finetune_epochs=50,
                inpainting_repair_ratio=repair_ratio,
                inpainting_edge_threshold=threshold,
            )
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for cfg in configs:
        signature = tuple(cfg.get(key) for key in CONFIG_KEYS)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(cfg)
    return deduped


def _edge_pubmed_privacy_strict_refine_grid() -> list[dict[str, Any]]:
    """PubMed edge privacy sweep with stronger privacy pressure and utility/runtime guards."""
    configs: list[dict[str, Any]] = []

    def add(**overrides: Any) -> None:
        base = {
            "anchor_lambda1": 2.0,
            "anchor_lambda2": 0.5,
            "edge_forget_loss_mode": "original_kl",
            "forget_weight": 0.125,
            "finetune_lr": 0.007,
            "finetune_epochs": 50,
            "inpainting_repair_ratio": 0.20,
            "inpainting_edge_threshold": 0.60,
            "inpainting_max_added_edges": 256,
            "inpainting_cc_drop_threshold": 0.30,
            "inpainting_min_damage_ratio": 0.10,
        }
        base.update(overrides)
        configs.append(_candidate(**base))

    add()
    add(
        anchor_lambda1=1.0,
        anchor_lambda2=0.1,
        edge_forget_loss_mode="uniform",
        forget_weight=0.5,
        finetune_lr=0.003,
        finetune_epochs=50,
        inpainting_repair_ratio=0.35,
        inpainting_edge_threshold=0.50,
    )

    for anchor_lambda1, anchor_lambda2 in [(1.0, 0.1), (0.5, 0.05), (0.25, 0.02), (0.0, 0.02)]:
        for forget_weight in [0.2, 0.3, 0.5, 0.8]:
            for repair_ratio in [0.05, 0.10, 0.20]:
                add(
                    anchor_lambda1=anchor_lambda1,
                    anchor_lambda2=anchor_lambda2,
                    edge_forget_loss_mode="uniform",
                    forget_weight=forget_weight,
                    finetune_lr=0.003,
                    finetune_epochs=80,
                    inpainting_repair_ratio=repair_ratio,
                    inpainting_edge_threshold=0.60,
                    inpainting_max_added_edges=128,
                    inpainting_cc_drop_threshold=0.30,
                    inpainting_min_damage_ratio=0.10,
                )

    for anchor_lambda1, anchor_lambda2 in [(1.0, 0.1), (0.5, 0.05), (0.25, 0.02)]:
        for forget_weight in [0.2, 0.3, 0.5]:
            for threshold, max_edges in [(0.60, 64), (0.70, 128), (0.70, 256)]:
                add(
                    anchor_lambda1=anchor_lambda1,
                    anchor_lambda2=anchor_lambda2,
                    edge_forget_loss_mode="original_kl",
                    forget_weight=forget_weight,
                    finetune_lr=0.005,
                    finetune_epochs=60,
                    inpainting_repair_ratio=0.10,
                    inpainting_edge_threshold=threshold,
                    inpainting_max_added_edges=max_edges,
                    inpainting_cc_drop_threshold=0.25,
                    inpainting_min_damage_ratio=0.05,
                )

    for loss_mode, forget_weight, repair_ratio in itertools.product(
        ["uniform", "none"],
        [0.5, 0.8],
        [0.05, 0.10],
    ):
        add(
            anchor_lambda1=0.0,
            anchor_lambda2=0.0,
            edge_forget_loss_mode=loss_mode,
            forget_weight=forget_weight,
            finetune_lr=0.003,
            finetune_epochs=100,
            inpainting_repair_ratio=repair_ratio,
            inpainting_edge_threshold=0.70,
            inpainting_max_added_edges=64,
            inpainting_cc_drop_threshold=0.25,
            inpainting_min_damage_ratio=0.05,
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for cfg in configs:
        signature = tuple(cfg.get(key) for key in CONFIG_KEYS)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(cfg)
    return deduped


def _edge_pubmed_privacy_aggressive_refine_grid() -> list[dict[str, Any]]:
    """PubMed edge sweep that prioritizes privacy while bounding utility/runtime damage."""
    configs: list[dict[str, Any]] = []

    def add(**overrides: Any) -> None:
        base = {
            "anchor_lambda1": 0.5,
            "anchor_lambda2": 0.05,
            "edge_forget_loss_mode": "uniform",
            "forget_weight": 0.8,
            "finetune_lr": 0.003,
            "finetune_epochs": 80,
            "inpainting_repair_ratio": 0.05,
            "inpainting_edge_threshold": 0.60,
            "inpainting_max_added_edges": 128,
            "inpainting_cc_drop_threshold": 0.30,
            "inpainting_min_damage_ratio": 0.10,
        }
        base.update(overrides)
        configs.append(_candidate(**base))

    # Reference-near configs from the last strict run and the current formal edge tuned config.
    add()
    add(
        anchor_lambda1=2.0,
        anchor_lambda2=0.5,
        edge_forget_loss_mode="uniform",
        forget_weight=0.5,
        finetune_lr=0.003,
        finetune_epochs=50,
        inpainting_repair_ratio=0.50,
        inpainting_edge_threshold=0.50,
        inpainting_max_added_edges=256,
        inpainting_cc_drop_threshold=0.30,
        inpainting_min_damage_ratio=0.10,
    )

    for anchor_lambda1, anchor_lambda2 in [(1.0, 0.10), (0.5, 0.05), (0.25, 0.02), (0.1, 0.01)]:
        for forget_weight in [0.8, 1.0, 1.5, 2.0]:
            for repair_ratio in [0.03, 0.05, 0.10, 0.15]:
                add(
                    anchor_lambda1=anchor_lambda1,
                    anchor_lambda2=anchor_lambda2,
                    edge_forget_loss_mode="uniform",
                    forget_weight=forget_weight,
                    finetune_lr=0.003,
                    finetune_epochs=120,
                    inpainting_repair_ratio=repair_ratio,
                    inpainting_edge_threshold=0.70,
                    inpainting_max_added_edges=64,
                    inpainting_cc_drop_threshold=0.25,
                    inpainting_min_damage_ratio=0.05,
                )

    for anchor_lambda1, anchor_lambda2 in [(0.5, 0.05), (0.25, 0.02), (0.1, 0.01)]:
        for forget_weight in [0.8, 1.0, 1.5]:
            for finetune_lr, finetune_epochs in [(0.003, 150), (0.005, 120), (0.007, 80)]:
                add(
                    anchor_lambda1=anchor_lambda1,
                    anchor_lambda2=anchor_lambda2,
                    edge_forget_loss_mode="uniform",
                    forget_weight=forget_weight,
                    finetune_lr=finetune_lr,
                    finetune_epochs=finetune_epochs,
                    inpainting_repair_ratio=0.03,
                    inpainting_edge_threshold=0.75,
                    inpainting_max_added_edges=32,
                    inpainting_cc_drop_threshold=0.20,
                    inpainting_min_damage_ratio=0.05,
                )

    for loss_mode, forget_weight in itertools.product(["original_kl", "uniform"], [0.8, 1.0, 1.5]):
        for anchor_lambda1, anchor_lambda2 in [(0.5, 0.05), (0.25, 0.02), (0.1, 0.01)]:
            add(
                anchor_lambda1=anchor_lambda1,
                anchor_lambda2=anchor_lambda2,
                edge_forget_loss_mode=loss_mode,
                forget_weight=forget_weight,
                finetune_lr=0.005,
                finetune_epochs=120,
                inpainting_repair_ratio=0.03,
                inpainting_edge_threshold=0.75,
                inpainting_max_added_edges=32,
                inpainting_cc_drop_threshold=0.20,
                inpainting_min_damage_ratio=0.05,
            )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for cfg in configs:
        signature = tuple(cfg.get(key) for key in CONFIG_KEYS)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(cfg)
    return deduped


def _edge_repair_refine_grid() -> list[dict[str, Any]]:
    """Second-pass edge sweep: tune graph-repair limits around promising configs."""
    edge_forget_loss_mode = ["original_kl", "uniform", "none"]
    forget_weight = [0.1, 0.2, 0.5]
    finetune_lr = [0.005]
    repair_ratio = [0.35]
    edge_threshold = [0.5]
    max_added_edges = [128, 256, 512]
    cc_drop_threshold = [0.2, 0.3, 0.4]
    min_damage_ratio = [0.05, 0.1, 0.2]
    return [
        _candidate(
            edge_forget_loss_mode=loss_mode,
            forget_weight=fw,
            finetune_lr=lr,
            inpainting_repair_ratio=repair,
            inpainting_edge_threshold=threshold,
            inpainting_max_added_edges=max_edges,
            inpainting_cc_drop_threshold=cc_threshold,
            inpainting_min_damage_ratio=min_damage,
        )
        for loss_mode, fw, lr, repair, threshold, max_edges, cc_threshold, min_damage in itertools.product(
            edge_forget_loss_mode,
            forget_weight,
            finetune_lr,
            repair_ratio,
            edge_threshold,
            max_added_edges,
            cc_drop_threshold,
            min_damage_ratio,
        )
    ]


def _edge_primekg_privacy_guarded_refine_grid() -> list[dict[str, Any]]:
    """PrimeKG-full edge sweep that pressures privacy without disabling HASI components."""
    configs = [
        # Stronger forgetting around the utility/alignment-friendly PrimeKG control.
        _candidate(forget_weight=0.20, finetune_lr=0.005, inpainting_max_added_edges=128),
        _candidate(forget_weight=0.30, finetune_lr=0.005, inpainting_max_added_edges=128),
        _candidate(anchor_lambda2=0.25, forget_weight=0.20, finetune_lr=0.005, inpainting_max_added_edges=128),
        _candidate(anchor_lambda2=0.10, forget_weight=0.30, finetune_lr=0.005, inpainting_max_added_edges=128),
        _candidate(anchor_lambda1=1.0, anchor_lambda2=0.10, forget_weight=0.30, finetune_lr=0.005, inpainting_repair_ratio=0.25, inpainting_max_added_edges=128),
        _candidate(anchor_lambda1=0.5, anchor_lambda2=0.05, forget_weight=0.50, finetune_lr=0.003, finetune_epochs=80, inpainting_repair_ratio=0.20, inpainting_max_added_edges=96),
        # Uniform-loss probes cover the stronger privacy-pressure regime.
        _candidate(anchor_lambda1=1.0, anchor_lambda2=0.10, edge_forget_loss_mode="uniform", forget_weight=0.20, finetune_lr=0.005, finetune_epochs=80, inpainting_repair_ratio=0.25, inpainting_max_added_edges=96),
        _candidate(anchor_lambda1=1.0, anchor_lambda2=0.10, edge_forget_loss_mode="uniform", forget_weight=0.30, finetune_lr=0.003, finetune_epochs=80, inpainting_repair_ratio=0.20, inpainting_max_added_edges=96),
        _candidate(anchor_lambda1=0.5, anchor_lambda2=0.05, edge_forget_loss_mode="uniform", forget_weight=0.30, finetune_lr=0.003, finetune_epochs=100, inpainting_repair_ratio=0.20, inpainting_max_added_edges=96),
        _candidate(anchor_lambda1=0.5, anchor_lambda2=0.05, edge_forget_loss_mode="uniform", forget_weight=0.50, finetune_lr=0.003, finetune_epochs=100, inpainting_repair_ratio=0.15, inpainting_max_added_edges=64),
        # Lower-rate controls preserve utility and exact-retrain alignment.
        _candidate(finetune_lr=0.003, finetune_epochs=60, inpainting_max_added_edges=128),
        _candidate(anchor_lambda2=0.25, forget_weight=0.15, finetune_lr=0.003, finetune_epochs=60, inpainting_repair_ratio=0.25, inpainting_max_added_edges=128),
        _candidate(anchor_lambda1=1.0, anchor_lambda2=0.10, forget_weight=0.15, finetune_lr=0.003, finetune_epochs=80, inpainting_repair_ratio=0.25, inpainting_max_added_edges=96),
        _candidate(anchor_lambda1=1.0, anchor_lambda2=0.05, edge_forget_loss_mode="uniform", forget_weight=0.15, finetune_lr=0.003, finetune_epochs=80, inpainting_repair_ratio=0.20, inpainting_max_added_edges=96),
    ]
    return _dedupe_configs(configs)


def _edge_comprehensive_refine_grid() -> list[dict[str, Any]]:
    """Second-round Hetionet edge sweep around cfg0007/cfg0008 for balanced score."""
    configs: list[dict[str, Any]] = []

    def add(**overrides: Any) -> None:
        base = {
            "anchor_lambda1": 2.0,
            "anchor_lambda2": 0.5,
            "edge_forget_loss_mode": "original_kl",
            "forget_weight": 0.1,
            "finetune_lr": 0.005,
            "finetune_epochs": 50,
            "inpainting_repair_ratio": 0.35,
            "inpainting_edge_threshold": 0.5,
            "inpainting_max_added_edges": 128,
            "inpainting_cc_drop_threshold": 0.4,
            "inpainting_min_damage_ratio": 0.1,
        }
        base.update(overrides)
        configs.append(_candidate(**base))

    for min_damage in [0.05, 0.1, 0.15, 0.2]:
        add(inpainting_min_damage_ratio=min_damage)

    for cc_threshold in [0.3, 0.35, 0.4, 0.45]:
        add(inpainting_cc_drop_threshold=cc_threshold)

    for max_edges in [96, 128, 160, 192, 256]:
        add(inpainting_max_added_edges=max_edges)

    for forget_weight in [0.05, 0.075, 0.1, 0.125, 0.15]:
        for max_edges in [96, 128, 160]:
            add(forget_weight=forget_weight, inpainting_max_added_edges=max_edges)

    for repair_ratio in [0.25, 0.30, 0.35, 0.40]:
        for min_damage in [0.1, 0.2]:
            add(inpainting_repair_ratio=repair_ratio, inpainting_min_damage_ratio=min_damage)

    for finetune_lr, finetune_epochs in [(0.003, 60), (0.005, 40), (0.005, 60), (0.007, 50)]:
        add(finetune_lr=finetune_lr, finetune_epochs=finetune_epochs)

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for cfg in configs:
        signature = tuple(cfg.get(key) for key in CONFIG_KEYS)
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(cfg)
    return deduped


class ForgetSpec:
    def __init__(
        self,
        *,
        ratio: float,
        seed: int,
        output_stem: str,
        path: Path | None = None,
        base_seed: int | None = None,
        forget_seed: int | None = None,
    ) -> None:
        self.ratio = ratio
        self.seed = seed
        self.base_seed = int(base_seed if base_seed is not None else seed)
        self.forget_seed = int(forget_seed if forget_seed is not None else seed)
        self.output_stem = output_stem
        self.path = path


def _resolve_forget_specs(args, ratios: list[float], seeds: list[int]) -> list[ForgetSpec]:
    if args.forget_set_file:
        path = Path(args.forget_set_file)
        payload = _load_json(path)
        return [
            ForgetSpec(
                ratio=float(payload.get("ratio", ratios[0] if ratios else 0.0)),
                seed=int(payload.get("seed", seeds[0] if seeds else 0)),
                output_stem=path.stem,
                path=path,
                base_seed=_base_seed_from_payload(payload),
                forget_seed=_forget_seed_from_payload(payload),
            )
        ]

    forget_set_dir = Path(args.forget_set_dir) if args.forget_set_dir else ROOT / "experiments" / "forget_sets" / args.dataset
    if forget_set_dir.exists():
        specs = _forget_specs_from_dir(forget_set_dir, args, ratios, seeds)
        if specs:
            return specs

    return [
        ForgetSpec(
            ratio=ratio,
            seed=seed,
            output_stem=f"{args.dataset}_{args.unlearning_type}_r{_ratio_label(ratio)}_seed{seed}",
            base_seed=seed,
            forget_seed=seed,
        )
        for ratio, seed in itertools.product(ratios, seeds)
    ]


def _forget_specs_from_dir(forget_set_dir: Path, args, ratios: list[float], seeds: list[int]) -> list[ForgetSpec]:
    ratio_set = {_ratio_label(ratio) for ratio in ratios}
    seed_set = set(seeds)
    specs: list[ForgetSpec] = []
    for path in sorted(forget_set_dir.glob(f"{args.dataset}_{args.unlearning_type}_*.json")):
        payload = _load_json(path)
        if payload.get("dataset") != args.dataset or payload.get("unlearning_type") != args.unlearning_type:
            continue
        ratio = _number(payload.get("ratio"))
        seed = payload.get("seed")
        if ratio is None or seed is None:
            continue
        seed = int(seed)
        if _ratio_label(ratio) not in ratio_set or seed not in seed_set:
            continue
        specs.append(
            ForgetSpec(
                ratio=ratio,
                seed=seed,
                output_stem=path.stem,
                path=path,
                base_seed=_base_seed_from_payload(payload),
                forget_seed=_forget_seed_from_payload(payload),
            )
        )
    return specs


def _run_command(args, cfg: dict[str, Any], spec: ForgetSpec, output_path: Path) -> list[str]:
    if args.python:
        cmd = [args.python]
    else:
        cmd = ["conda", "run", "-n", args.conda_env, "python"]
    cmd += [
        "experiments/run_hasi.py",
        "--config", args.config,
        "--dataset_name", args.dataset,
        "--mode", "unlearn",
        "--unlearning_type", args.unlearning_type,
        "--forget_ratio", str(spec.ratio),
        "--seed", str(spec.base_seed),
        "--base_artifact_root", args.base_artifact_root,
        "--data_root", args.data_root,
        "--method_name", "hasi_tuning",
        "--output", str(output_path),
        "--anchor_lambda1", str(cfg["anchor_lambda1"]),
        "--anchor_lambda2", str(cfg["anchor_lambda2"]),
        "--edge_forget_loss_mode", str(cfg["edge_forget_loss_mode"]),
        "--node_forget_loss_mode", str(cfg["node_forget_loss_mode"]),
        "--forget_weight", str(cfg["forget_weight"]),
        "--finetune_lr", str(cfg["finetune_lr"]),
        "--finetune_epochs", str(cfg["finetune_epochs"]),
        "--inpainting_repair_ratio", str(cfg["inpainting_repair_ratio"]),
        "--inpainting_edge_threshold", str(cfg["inpainting_edge_threshold"]),
        "--inpainting_max_added_edges", str(cfg["inpainting_max_added_edges"]),
        "--inpainting_cc_drop_threshold", str(cfg["inpainting_cc_drop_threshold"]),
        "--inpainting_min_damage_ratio", str(cfg["inpainting_min_damage_ratio"]),
        "--edge_forget_scope", args.edge_forget_scope,
    ]
    for flag, value in (
        ("--device", args.device),
        ("--graph_compute_backend", args.graph_compute_backend),
        ("--hub_ppr_batch_size", args.hub_ppr_batch_size),
        ("--hub_score_cache_root", args.hub_score_cache_root),
    ):
        if value is not None and value != "":
            cmd += [flag, str(value)]
    for flag, value in (
        ("--hub_score_cache", args.hub_score_cache),
        ("--require_hub_score_cache_hit", args.require_hub_score_cache_hit),
    ):
        if value is True:
            cmd.append(flag)
        elif value is False:
            cmd.append(f"--no-{flag.removeprefix('--')}")
    if spec.path is not None:
        cmd += ["--forget_set_file", str(spec.path)]
    if args.unlearning_type == "edge" and args.exact_retrain_reference_root:
        cmd += ["--exact_retrain_reference", str(_exact_retrain_reference_path(args, spec))]
    return cmd


def _base_seed_from_payload(payload: dict[str, Any]) -> int | None:
    protocol = payload.get("protocol") or {}
    for key in ("shared_base_seed", "split_seed"):
        if protocol.get(key) is not None:
            return int(protocol[key])
    return None


def _forget_seed_from_payload(payload: dict[str, Any]) -> int | None:
    protocol = payload.get("protocol") or {}
    if protocol.get("forget_seed") is not None:
        return int(protocol["forget_seed"])
    if payload.get("seed") is not None:
        return int(payload["seed"])
    return None


def _exact_retrain_reference_path(args, spec: ForgetSpec) -> Path:
    ratio_dir = f"r{_ratio_label(spec.ratio)}"
    return Path(args.exact_retrain_reference_root) / ratio_dir / f"base{spec.base_seed}_fseed{spec.forget_seed}.pt"


def _metrics_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    protocol = metrics.get("evaluation_protocol", {})
    test_utility = metrics.get("utility", {})
    utility = metrics.get("validation_utility", test_utility)
    privacy = metrics.get("privacy", {})
    efficiency = metrics.get("efficiency", {})
    structure = metrics.get("structure", {})
    exact = metrics.get("exact_retrain_alignment", {})
    mia_auc = _number(privacy.get("overall_mia_auc"))
    degree_kl = _number(structure.get("degree_kl_divergence"))
    cc_change = _number(structure.get("clustering_coefficient_change"))
    component_change = _number(structure.get("component_count_change"))
    return {
        "val_accuracy_after": _number(utility.get("accuracy_after")),
        "val_accuracy_drop": _number(utility.get("accuracy_drop")),
        "val_f1_macro_after": _number(utility.get("f1_macro_after")),
        "val_f1_macro_drop": _number(utility.get("f1_macro_drop")),
        "test_accuracy_after": _number(test_utility.get("accuracy_after")),
        "test_accuracy_drop": _number(test_utility.get("accuracy_drop")),
        "test_f1_macro_after": _number(test_utility.get("f1_macro_after")),
        "test_f1_macro_drop": _number(test_utility.get("f1_macro_drop")),
        "accuracy_after": _number(utility.get("accuracy_after")),
        "accuracy_drop": _number(utility.get("accuracy_drop")),
        "f1_macro_after": _number(utility.get("f1_macro_after")),
        "f1_macro_drop": _number(utility.get("f1_macro_drop")),
        "evaluation_protocol_version": protocol.get("version"),
        "privacy_status": privacy.get("status"),
        "privacy_applicable": privacy.get("applicable"),
        "overall_mia_auc": mia_auc,
        "privacy_gap": abs(mia_auc - 0.5) if mia_auc is not None else None,
        "privacy_score": _number(privacy.get("privacy_score")),
        "strong_auc_null_mean": _number(privacy.get("strong_auc_null_mean")),
        "strong_auc_null_std": _number(privacy.get("strong_auc_null_std")),
        "strong_auc_pvalue": _number(privacy.get("strong_auc_pvalue")),
        "degree_kl_divergence": degree_kl,
        "clustering_coefficient_change": cc_change,
        "component_count_change": component_change,
        "degree_kl_abs": abs(degree_kl) if degree_kl is not None else None,
        "clustering_change_abs": abs(cc_change) if cc_change is not None else None,
        "component_change_abs": abs(component_change) if component_change is not None else None,
        "exact_retrain_status": exact.get("status"),
        "exact_retrain_js_mean": _number(exact.get("unlearned_to_retrain_js_mean")),
        "exact_retrain_tv_mean": _number(exact.get("unlearned_to_retrain_tv_mean")),
        "exact_retrain_disagreement_rate": _number(exact.get("prediction_disagreement_rate")),
        "unlearn_time_seconds": _number(efficiency.get("unlearn_time_seconds")),
    }


def _write_outputs(dataset_dir: Path, rows: list[dict[str, Any]], args) -> dict[str, Any]:
    run_rows = _score_run_rows(rows, args)
    config_rows = _score_config_rows(_aggregate_configs(run_rows), args)
    candidates = [row for row in config_rows if not row.get("is_default_reference")]
    passed = [row for row in candidates if row.get("passes_hard_constraints")]
    best = passed[: args.top_k] if (passed or args.top_configs_require_hard_constraints) else candidates[: args.top_k]

    _write_json(dataset_dir / "run_summary.json", run_rows)
    _write_json(dataset_dir / "config_summary.json", config_rows)
    _write_json(dataset_dir / "sweep_summary.json", config_rows)
    _write_csv(dataset_dir / "run_summary.csv", run_rows)
    _write_csv(dataset_dir / "config_summary.csv", config_rows)
    _write_csv(dataset_dir / "sweep_summary.csv", config_rows)
    _write_json(dataset_dir / "top_configs.json", best)
    _write_csv(dataset_dir / "top_configs.csv", best)
    return {"runs": run_rows, "configs": config_rows, "best": best}


def _score_run_rows(rows: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    scored = [dict(row) for row in rows]
    runtime_refs = _runtime_reference_by_run(scored)
    default_refs = {
        (row.get("ratio"), row.get("seed")): row
        for row in scored
        if row.get("is_default_reference")
    }
    for row in scored:
        run_key = (row.get("ratio"), row.get("seed"))
        ref = runtime_refs.get(run_key)
        runtime = _number(row.get("unlearn_time_seconds"))
        if ref is not None and runtime is not None:
            row["default_runtime_seconds"] = ref
            row["runtime_ratio_vs_default"] = runtime / ref if ref > 0 else None
            row["passes_runtime_reference"] = runtime <= args.runtime_multiplier * ref
        else:
            row["default_runtime_seconds"] = ref
            row["runtime_ratio_vs_default"] = None
            row["passes_runtime_reference"] = None
        default_row = default_refs.get(run_key)
        for metric in PAIRED_DEFAULT_METRICS:
            delta_key = f"{metric}_paired_delta_vs_default"
            value = _number(row.get(metric))
            reference = _number(default_row.get(metric)) if default_row is not None else None
            row[delta_key] = value - reference if value is not None and reference is not None else None
    return scored


def _aggregate_configs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("config_id")), []).append(row)

    metric_names = [
        "val_accuracy_drop",
        "val_f1_macro_drop",
        "test_accuracy_drop",
        "test_f1_macro_drop",
        "overall_mia_auc",
        "privacy_gap",
        "privacy_score",
        "strong_auc_null_mean",
        "strong_auc_null_std",
        "strong_auc_pvalue",
        "degree_kl_abs",
        "clustering_change_abs",
        "component_change_abs",
        "exact_retrain_js_mean",
        "exact_retrain_tv_mean",
        "exact_retrain_disagreement_rate",
        "unlearn_time_seconds",
        "runtime_ratio_vs_default",
        *(f"{metric}_paired_delta_vs_default" for metric in PAIRED_DEFAULT_METRICS),
    ]
    summaries = []
    for config_id, group in groups.items():
        first = group[0]
        summary: dict[str, Any] = {
            "config_id": config_id,
            "is_default_reference": bool(first.get("is_default_reference")),
            "dataset": first.get("dataset"),
            "unlearning_type": first.get("unlearning_type"),
            "num_runs": len(group),
            "num_ok_runs": sum(1 for row in group if row.get("status") == "ok"),
            "ratios": ",".join(str(item) for item in sorted({row.get("ratio") for row in group})),
            "seeds": ",".join(str(item) for item in sorted({row.get("seed") for row in group})),
            "evaluation_protocol_versions": ",".join(
                sorted({str(row.get("evaluation_protocol_version")) for row in group if row.get("evaluation_protocol_version")})
            ),
            "privacy_statuses": ",".join(
                sorted({str(row.get("privacy_status")) for row in group if row.get("privacy_status")})
            ),
            "privacy_applicable_values": ",".join(
                sorted({str(row.get("privacy_applicable")) for row in group if row.get("privacy_applicable") is not None})
            ),
            "exact_retrain_statuses": ",".join(
                sorted({str(row.get("exact_retrain_status")) for row in group if row.get("exact_retrain_status")})
            ),
        }
        for key in CONFIG_KEYS:
            summary[key] = first.get(key)
        for metric in metric_names:
            values = [_number(row.get(metric)) for row in group]
            clean = [value for value in values if value is not None]
            summary[f"{metric}_mean"] = _mean(clean)
            summary[f"{metric}_std"] = _std(clean)
            summary[f"{metric}_max"] = max(clean) if clean else None
        runtime_flags = [row.get("passes_runtime_reference") for row in group if row.get("passes_runtime_reference") is not None]
        summary["passes_runtime_reference"] = all(runtime_flags) if runtime_flags else None
        summaries.append(summary)
    return summaries


def _score_config_rows(rows: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    scored = [dict(row) for row in rows]
    _add_default_relative_metrics(scored)
    candidate_rows = [row for row in scored if not row.get("is_default_reference")]
    metric_bounds = {
        "val_accuracy_drop_mean": _minmax([_number(row.get("val_accuracy_drop_mean")) for row in candidate_rows]),
        "val_f1_macro_drop_mean": _minmax([_number(row.get("val_f1_macro_drop_mean")) for row in candidate_rows]),
        "privacy_gap_mean": _minmax([_number(row.get("privacy_gap_mean")) for row in candidate_rows]),
        "privacy_gap_paired_delta_vs_default_max": _minmax(
            [_number(row.get("privacy_gap_paired_delta_vs_default_max")) for row in candidate_rows]
        ),
        "unlearn_time_seconds_mean": _minmax([_number(row.get("unlearn_time_seconds_mean")) for row in candidate_rows]),
        "degree_kl_abs_mean": _minmax([_number(row.get("degree_kl_abs_mean")) for row in candidate_rows]),
        "clustering_change_abs_mean": _minmax([_number(row.get("clustering_change_abs_mean")) for row in candidate_rows]),
        "component_change_abs_mean": _minmax([_number(row.get("component_change_abs_mean")) for row in candidate_rows]),
        "exact_retrain_js_mean_mean": _minmax([_number(row.get("exact_retrain_js_mean_mean")) for row in candidate_rows]),
        "exact_retrain_tv_mean_mean": _minmax([_number(row.get("exact_retrain_tv_mean_mean")) for row in candidate_rows]),
        "exact_retrain_disagreement_rate_mean": _minmax(
            [_number(row.get("exact_retrain_disagreement_rate_mean")) for row in candidate_rows]
        ),
        "val_accuracy_drop_std": _minmax([_number(row.get("val_accuracy_drop_std")) for row in candidate_rows]),
        "val_f1_macro_drop_std": _minmax([_number(row.get("val_f1_macro_drop_std")) for row in candidate_rows]),
        "privacy_gap_std": _minmax([_number(row.get("privacy_gap_std")) for row in candidate_rows]),
    }
    for row in scored:
        if row.get("is_default_reference"):
            row["passes_hard_constraints"] = None
            row["score_mode"] = args.score_mode
            row["score"] = None
            continue
        structure_damage = _avg(
            [
                _norm(row.get("degree_kl_abs_mean"), metric_bounds["degree_kl_abs_mean"]),
                _norm(row.get("clustering_change_abs_mean"), metric_bounds["clustering_change_abs_mean"]),
                _norm(row.get("component_change_abs_mean"), metric_bounds["component_change_abs_mean"]),
            ]
        )
        exact_alignment_damage = _avg(
            [
                _norm(row.get("exact_retrain_js_mean_mean"), metric_bounds["exact_retrain_js_mean_mean"]),
                _norm(row.get("exact_retrain_tv_mean_mean"), metric_bounds["exact_retrain_tv_mean_mean"]),
                _norm(
                    row.get("exact_retrain_disagreement_rate_mean"),
                    metric_bounds["exact_retrain_disagreement_rate_mean"],
                ),
            ]
        )
        stability_penalty = _avg(
            [
                _norm(row.get("val_accuracy_drop_std"), metric_bounds["val_accuracy_drop_std"]),
                _norm(row.get("val_f1_macro_drop_std"), metric_bounds["val_f1_macro_drop_std"]),
                _norm(row.get("privacy_gap_std"), metric_bounds["privacy_gap_std"]),
            ]
        )
        quick_score = (
            0.50 * _norm(row.get("val_accuracy_drop_mean"), metric_bounds["val_accuracy_drop_mean"])
            + 0.50 * _norm(row.get("val_f1_macro_drop_mean"), metric_bounds["val_f1_macro_drop_mean"])
        )
        formal_score = (
            0.40 * _norm(row.get("val_accuracy_drop_mean"), metric_bounds["val_accuracy_drop_mean"])
            + 0.25 * _norm(row.get("val_f1_macro_drop_mean"), metric_bounds["val_f1_macro_drop_mean"])
            + 0.25 * _norm(row.get("privacy_gap_mean"), metric_bounds["privacy_gap_mean"])
            + 0.10 * _norm(row.get("unlearn_time_seconds_mean"), metric_bounds["unlearn_time_seconds_mean"])
        )
        structure_score = (
            0.35 * _norm(row.get("val_accuracy_drop_mean"), metric_bounds["val_accuracy_drop_mean"])
            + 0.20 * _norm(row.get("val_f1_macro_drop_mean"), metric_bounds["val_f1_macro_drop_mean"])
            + 0.25 * _norm(row.get("privacy_gap_mean"), metric_bounds["privacy_gap_mean"])
            + 0.10 * structure_damage
            + 0.10 * _norm(row.get("unlearn_time_seconds_mean"), metric_bounds["unlearn_time_seconds_mean"])
        )
        privacy_score = (
            0.50 * _norm(row.get("privacy_gap_mean"), metric_bounds["privacy_gap_mean"])
            + 0.20 * _norm(row.get("val_f1_macro_drop_mean"), metric_bounds["val_f1_macro_drop_mean"])
            + 0.15 * _norm(row.get("val_accuracy_drop_mean"), metric_bounds["val_accuracy_drop_mean"])
            + 0.10 * structure_damage
            + 0.05 * _norm(row.get("unlearn_time_seconds_mean"), metric_bounds["unlearn_time_seconds_mean"])
        )
        privacy_guarded_score = (
            0.40 * _norm(row.get("privacy_gap_mean"), metric_bounds["privacy_gap_mean"])
            + 0.20 * _norm(row.get("val_accuracy_drop_mean"), metric_bounds["val_accuracy_drop_mean"])
            + 0.20 * _norm(row.get("val_f1_macro_drop_mean"), metric_bounds["val_f1_macro_drop_mean"])
            + 0.15 * _norm(row.get("unlearn_time_seconds_mean"), metric_bounds["unlearn_time_seconds_mean"])
            + 0.05 * structure_damage
        )
        edge_privacy_guarded_score = (
            0.30 * _norm(row.get("privacy_gap_mean"), metric_bounds["privacy_gap_mean"])
            + 0.20 * _norm(row.get("val_accuracy_drop_mean"), metric_bounds["val_accuracy_drop_mean"])
            + 0.15 * _norm(row.get("val_f1_macro_drop_mean"), metric_bounds["val_f1_macro_drop_mean"])
            + 0.15 * exact_alignment_damage
            + 0.10 * structure_damage
            + 0.10 * _norm(row.get("unlearn_time_seconds_mean"), metric_bounds["unlearn_time_seconds_mean"])
        )
        edge_privacy_strict_score = (
            0.50 * _norm(row.get("privacy_gap_mean"), metric_bounds["privacy_gap_mean"])
            + 0.15 * _norm(row.get("val_accuracy_drop_mean"), metric_bounds["val_accuracy_drop_mean"])
            + 0.12 * _norm(row.get("val_f1_macro_drop_mean"), metric_bounds["val_f1_macro_drop_mean"])
            + 0.10 * exact_alignment_damage
            + 0.08 * structure_damage
            + 0.05 * _norm(row.get("unlearn_time_seconds_mean"), metric_bounds["unlearn_time_seconds_mean"])
        )
        edge_privacy_aggressive_score = (
            0.70 * _norm(row.get("privacy_gap_mean"), metric_bounds["privacy_gap_mean"])
            + 0.07 * _norm(row.get("val_accuracy_drop_mean"), metric_bounds["val_accuracy_drop_mean"])
            + 0.07 * _norm(row.get("val_f1_macro_drop_mean"), metric_bounds["val_f1_macro_drop_mean"])
            + 0.06 * exact_alignment_damage
            + 0.05 * structure_damage
            + 0.05 * _norm(row.get("unlearn_time_seconds_mean"), metric_bounds["unlearn_time_seconds_mean"])
        )
        privacy_improvement_guarded_score = (
            0.70 * _norm(row.get("privacy_gap_mean"), metric_bounds["privacy_gap_mean"])
            + 0.10 * _norm(row.get("val_accuracy_drop_mean"), metric_bounds["val_accuracy_drop_mean"])
            + 0.10 * _norm(row.get("val_f1_macro_drop_mean"), metric_bounds["val_f1_macro_drop_mean"])
            + 0.05 * structure_damage
            + 0.05 * _norm(row.get("unlearn_time_seconds_mean"), metric_bounds["unlearn_time_seconds_mean"])
        )
        privacy_robust_pareto_score = (
            0.75
            * _norm(
                row.get("privacy_gap_paired_delta_vs_default_max"),
                metric_bounds["privacy_gap_paired_delta_vs_default_max"],
            )
            + 0.10 * _norm(row.get("val_accuracy_drop_mean"), metric_bounds["val_accuracy_drop_mean"])
            + 0.10 * _norm(row.get("val_f1_macro_drop_mean"), metric_bounds["val_f1_macro_drop_mean"])
            + 0.05 * _norm(row.get("unlearn_time_seconds_mean"), metric_bounds["unlearn_time_seconds_mean"])
        )
        utility_privacy_score = (
            0.35 * _norm(row.get("privacy_gap_mean"), metric_bounds["privacy_gap_mean"])
            + 0.25 * _norm(row.get("val_accuracy_drop_mean"), metric_bounds["val_accuracy_drop_mean"])
            + 0.25 * _norm(row.get("val_f1_macro_drop_mean"), metric_bounds["val_f1_macro_drop_mean"])
            + 0.10 * structure_damage
            + 0.05 * _norm(row.get("unlearn_time_seconds_mean"), metric_bounds["unlearn_time_seconds_mean"])
        )
        feature_utility_score = (
            0.45 * _norm(row.get("val_accuracy_drop_mean"), metric_bounds["val_accuracy_drop_mean"])
            + 0.35 * _norm(row.get("val_f1_macro_drop_mean"), metric_bounds["val_f1_macro_drop_mean"])
            + 0.10 * structure_damage
            + 0.10 * _norm(row.get("unlearn_time_seconds_mean"), metric_bounds["unlearn_time_seconds_mean"])
        )
        base_score = {
            "quick": quick_score,
            "formal": formal_score,
            "structure": structure_score,
            "privacy": privacy_score,
            "privacy_guarded": privacy_guarded_score,
            "privacy_improvement_guarded": privacy_improvement_guarded_score,
            "privacy_robust_pareto": privacy_robust_pareto_score,
            "edge_privacy_guarded": edge_privacy_guarded_score,
            "edge_privacy_strict": edge_privacy_strict_score,
            "edge_privacy_aggressive": edge_privacy_aggressive_score,
            "utility_privacy": utility_privacy_score,
            "feature_utility": feature_utility_score,
        }[args.score_mode]
        row["score_mode"] = args.score_mode
        row["structure_damage_norm"] = structure_damage
        row["exact_alignment_damage_norm"] = exact_alignment_damage
        row["stability_penalty_norm"] = stability_penalty
        row["score_without_stability"] = base_score
        row["score"] = base_score + float(args.stability_weight) * stability_penalty
        row["passes_hard_constraints"] = _passes_config_constraints(row, args)
    return sorted(scored, key=lambda row: (bool(row.get("is_default_reference")), not bool(row.get("passes_hard_constraints")), _sort_number(row.get("score"))))


def _add_default_relative_metrics(rows: list[dict[str, Any]]) -> None:
    default = next((row for row in rows if row.get("is_default_reference")), None)
    metric_pairs = {
        "overall_mia_auc_mean": "overall_mia_auc_delta_vs_default",
        "privacy_gap_mean": "privacy_gap_delta_vs_default",
        "privacy_score_mean": "privacy_score_delta_vs_default",
        "val_accuracy_drop_mean": "val_accuracy_drop_delta_vs_default",
        "val_f1_macro_drop_mean": "val_f1_macro_drop_delta_vs_default",
        "test_accuracy_drop_mean": "test_accuracy_drop_delta_vs_default",
        "test_f1_macro_drop_mean": "test_f1_macro_drop_delta_vs_default",
        "unlearn_time_seconds_mean": "unlearn_time_seconds_delta_vs_default",
        "degree_kl_abs_mean": "degree_kl_abs_delta_vs_default",
        "clustering_change_abs_mean": "clustering_change_abs_delta_vs_default",
        "component_change_abs_mean": "component_change_abs_delta_vs_default",
        "exact_retrain_js_mean_mean": "exact_retrain_js_delta_vs_default",
        "exact_retrain_tv_mean_mean": "exact_retrain_tv_delta_vs_default",
        "exact_retrain_disagreement_rate_mean": "exact_retrain_disagreement_delta_vs_default",
    }
    for row in rows:
        for metric, delta_key in metric_pairs.items():
            if default is None:
                row[delta_key] = None
                continue
            value = _number(row.get(metric))
            reference = _number(default.get(metric))
            row[delta_key] = value - reference if value is not None and reference is not None else None


def _constraint_delta(row: dict[str, Any], args, aggregate_key: str, run_metric: str) -> Any:
    if args.paired_hard_constraints:
        return row.get(f"{run_metric}_paired_delta_vs_default_max")
    return row.get(aggregate_key)


def _passes_config_constraints(row: dict[str, Any], args) -> bool:
    if _number(row.get("num_ok_runs")) != _number(row.get("num_runs")):
        return False
    accuracy_drop_key = "val_accuracy_drop_max" if args.paired_hard_constraints else "val_accuracy_drop_mean"
    f1_drop_key = "val_f1_macro_drop_max" if args.paired_hard_constraints else "val_f1_macro_drop_mean"
    mia_auc_key = "overall_mia_auc_max" if args.paired_hard_constraints else "overall_mia_auc_mean"
    if _gt(row.get(accuracy_drop_key), args.accuracy_drop_limit):
        return False
    if _gt(row.get(f1_drop_key), args.f1_macro_drop_limit):
        return False
    privacy_applicable_values = str(row.get("privacy_applicable_values") or "")
    privacy_is_applicable = row.get("unlearning_type") != "feature" and privacy_applicable_values != "False"
    if privacy_is_applicable and _gt(row.get(mia_auc_key), args.mia_auc_limit):
        return False
    if privacy_is_applicable and args.require_privacy_improvement_vs_default:
        privacy_delta = _number(
            _constraint_delta(row, args, "privacy_gap_delta_vs_default", "privacy_gap")
        )
        required_delta = -float(args.privacy_gap_improvement_margin)
        if privacy_delta is None or privacy_delta >= required_delta:
            return False
    if args.val_accuracy_drop_delta_limit_vs_default is not None:
        if _gt(_constraint_delta(row, args, "val_accuracy_drop_delta_vs_default", "val_accuracy_drop"), args.val_accuracy_drop_delta_limit_vs_default):
            return False
    if args.val_f1_macro_drop_delta_limit_vs_default is not None:
        if _gt(_constraint_delta(row, args, "val_f1_macro_drop_delta_vs_default", "val_f1_macro_drop"), args.val_f1_macro_drop_delta_limit_vs_default):
            return False
    if args.test_accuracy_drop_delta_limit_vs_default is not None:
        if _gt(_constraint_delta(row, args, "test_accuracy_drop_delta_vs_default", "test_accuracy_drop"), args.test_accuracy_drop_delta_limit_vs_default):
            return False
    if args.test_f1_macro_drop_delta_limit_vs_default is not None:
        if _gt(_constraint_delta(row, args, "test_f1_macro_drop_delta_vs_default", "test_f1_macro_drop"), args.test_f1_macro_drop_delta_limit_vs_default):
            return False
    if args.degree_kl_abs_delta_limit_vs_default is not None:
        if _gt(_constraint_delta(row, args, "degree_kl_abs_delta_vs_default", "degree_kl_abs"), args.degree_kl_abs_delta_limit_vs_default):
            return False
    if args.clustering_change_abs_delta_limit_vs_default is not None:
        if _gt(_constraint_delta(row, args, "clustering_change_abs_delta_vs_default", "clustering_change_abs"), args.clustering_change_abs_delta_limit_vs_default):
            return False
    if args.component_change_abs_delta_limit_vs_default is not None:
        if _gt(_constraint_delta(row, args, "component_change_abs_delta_vs_default", "component_change_abs"), args.component_change_abs_delta_limit_vs_default):
            return False
    if args.exact_retrain_js_delta_limit_vs_default is not None:
        if _gt(_constraint_delta(row, args, "exact_retrain_js_delta_vs_default", "exact_retrain_js_mean"), args.exact_retrain_js_delta_limit_vs_default):
            return False
    if args.exact_retrain_tv_delta_limit_vs_default is not None:
        if _gt(_constraint_delta(row, args, "exact_retrain_tv_delta_vs_default", "exact_retrain_tv_mean"), args.exact_retrain_tv_delta_limit_vs_default):
            return False
    if args.exact_retrain_disagreement_delta_limit_vs_default is not None:
        if _gt(_constraint_delta(row, args, "exact_retrain_disagreement_delta_vs_default", "exact_retrain_disagreement_rate"), args.exact_retrain_disagreement_delta_limit_vs_default):
            return False
    if args.unlearn_time_delta_limit_vs_default is not None:
        if _gt(_constraint_delta(row, args, "unlearn_time_seconds_delta_vs_default", "unlearn_time_seconds"), args.unlearn_time_delta_limit_vs_default):
            return False
    if args.runtime_ratio_mean_limit_vs_default is not None:
        runtime_ratio_key = "runtime_ratio_vs_default_max" if args.paired_hard_constraints else "runtime_ratio_vs_default_mean"
        if _gt(row.get(runtime_ratio_key), args.runtime_ratio_mean_limit_vs_default):
            return False
    runtime_flag = row.get("passes_runtime_reference")
    if runtime_flag is False:
        return False
    return True


def _runtime_reference_by_run(rows: list[dict[str, Any]]) -> dict[tuple[Any, Any], float]:
    refs = {}
    for row in rows:
        if not row.get("is_default_reference"):
            continue
        runtime = _number(row.get("unlearn_time_seconds"))
        if runtime is not None:
            refs[(row.get("ratio"), row.get("seed"))] = runtime
    return refs


def _write_best_config(path: Path, row: dict[str, Any], args) -> None:
    content = f"""anchor_stabilization:
  lambda1: {row['anchor_lambda1']}
  lambda2: {row['anchor_lambda2']}

inpainting:
  cc_drop_threshold: {row['inpainting_cc_drop_threshold']}
  min_damage_ratio: {row['inpainting_min_damage_ratio']}
  edge_threshold: {row['inpainting_edge_threshold']}
  max_added_edges: {row['inpainting_max_added_edges']}
  repair_ratio: {row['inpainting_repair_ratio']}

unlearning:
  edge_forget_loss_mode: {row['edge_forget_loss_mode']}
  node_forget_loss_mode: {row['node_forget_loss_mode']}
  forget_weight: {row['forget_weight']}
  finetune_epochs: {row['finetune_epochs']}
  finetune_lr: {row['finetune_lr']}

# Selected by experiments/sweep_hasi_params.py
# grid: {args.grid}
# score_mode: {args.score_mode}
# score: {row.get('score')}
# score_without_stability: {row.get('score_without_stability')}
# stability_penalty_norm: {row.get('stability_penalty_norm')}
# passes_hard_constraints: {row.get('passes_hard_constraints')}
# val_accuracy_drop_mean: {row.get('val_accuracy_drop_mean')}
# val_accuracy_drop_std: {row.get('val_accuracy_drop_std')}
# val_f1_macro_drop_mean: {row.get('val_f1_macro_drop_mean')}
# val_f1_macro_drop_std: {row.get('val_f1_macro_drop_std')}
# overall_mia_auc_mean: {row.get('overall_mia_auc_mean')}
# privacy_gap_mean: {row.get('privacy_gap_mean')}
# privacy_gap_delta_vs_default: {row.get('privacy_gap_delta_vs_default')}
# val_accuracy_drop_delta_vs_default: {row.get('val_accuracy_drop_delta_vs_default')}
# val_f1_macro_drop_delta_vs_default: {row.get('val_f1_macro_drop_delta_vs_default')}
# structure_damage_norm: {row.get('structure_damage_norm')}
# exact_alignment_damage_norm: {row.get('exact_alignment_damage_norm')}
# unlearn_time_seconds_mean: {row.get('unlearn_time_seconds_mean')}
# unlearn_time_seconds_delta_vs_default: {row.get('unlearn_time_seconds_delta_vs_default')}
# runtime_ratio_vs_default_mean: {row.get('runtime_ratio_vs_default_mean')}
"""
    path.write_text(content, encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [_coerce_csv_row(row) for row in csv.DictReader(handle)]


def _coerce_csv_row(row: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(row)
    for key in ("is_default_reference", "passes_runtime_reference"):
        if coerced.get(key) == "True":
            coerced[key] = True
        elif coerced.get(key) == "False":
            coerced[key] = False
        elif coerced.get(key) == "":
            coerced[key] = None
    return coerced


def _config_signature(cfg: dict[str, Any]) -> str:
    normalized = {
        key: (
            cfg.get(key, "uniform")
            if key == "node_forget_loss_mode"
            else cfg.get(key)
        )
        for key in CONFIG_KEYS
    }
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _existing_config_signatures(configs_dir: Path) -> tuple[set[str], int]:
    signatures: set[str] = set()
    max_index = -1
    for path in configs_dir.glob("config_*.json"):
        payload = _load_json(path)
        signatures.add(_config_signature(payload))
        stem = path.stem.removeprefix("config_")
        if stem.isdigit():
            max_index = max(max_index, int(stem))
    return signatures, max_index + 1


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _ratio_label(value: float) -> str:
    return str(value).replace(".", "p")


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _gt(value: Any, threshold: float) -> bool:
    number = _number(value)
    return number is not None and number > threshold


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _std(values: list[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    mean = _mean(values)
    if mean is None:
        return None
    return math.sqrt(sum((value - mean) ** 2 for value in values) / float(len(values) - 1))


def _avg(values: list[float | None]) -> float:
    clean = [value for value in values if value is not None]
    if not clean:
        return 1.0
    return sum(clean) / float(len(clean))


def _minmax(values: list[float | None]) -> tuple[float, float] | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return min(clean), max(clean)


def _norm(value: Any, bounds: tuple[float, float] | None) -> float:
    number = _number(value)
    if number is None or bounds is None:
        return 1.0
    low, high = bounds
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (number - low) / (high - low)))


def _sort_number(value: Any) -> float:
    number = _number(value)
    return number if number is not None else float("inf")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
    return value


if __name__ == "__main__":
    main()
