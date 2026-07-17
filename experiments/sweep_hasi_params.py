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
    "forget_weight",
    "finetune_lr",
    "finetune_epochs",
    "inpainting_repair_ratio",
    "inpainting_edge_threshold",
    "inpainting_max_added_edges",
    "inpainting_cc_drop_threshold",
    "inpainting_min_damage_ratio",
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
    parser.add_argument("--grid", default="coarse", choices=["coarse", "full", "privacy_refine", "feature_wide_refine", "feature_utility_privacy_refine", "node_privacy_wide_refine", "node_pubmed_refine", "edge_refine", "edge_repair_refine", "edge_comprehensive_refine"])
    parser.add_argument("--score_mode", default="structure", choices=["quick", "formal", "structure", "privacy", "utility_privacy"])
    parser.add_argument("--max_configs", type=int, default=0, help="Limit candidate hyperparameter configs; 0 means all.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top configs to write separately.")
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

    configs = _grid(args.grid)
    if args.max_configs > 0:
        configs = configs[: args.max_configs]

    existing_signatures, next_config_index = _existing_config_signatures(configs_dir)
    run_specs: list[tuple[str, dict[str, Any], bool]] = []
    if args.include_default_reference:
        default_cfg = dict(DEFAULT_CONFIG)
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
    if name == "node_privacy_wide_refine":
        return _node_privacy_wide_refine_grid()
    if name == "node_pubmed_refine":
        return _node_pubmed_refine_grid()
    if name == "edge_refine":
        return _edge_refine_grid()
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


def _candidate(**overrides: Any) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(overrides)
    return cfg


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
    def __init__(self, *, ratio: float, seed: int, output_stem: str, path: Path | None = None) -> None:
        self.ratio = ratio
        self.seed = seed
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
        specs.append(ForgetSpec(ratio=ratio, seed=seed, output_stem=path.stem, path=path))
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
        "--seed", str(spec.seed),
        "--base_artifact_root", args.base_artifact_root,
        "--data_root", args.data_root,
        "--method_name", "hasi_tuning",
        "--output", str(output_path),
        "--anchor_lambda1", str(cfg["anchor_lambda1"]),
        "--anchor_lambda2", str(cfg["anchor_lambda2"]),
        "--edge_forget_loss_mode", str(cfg["edge_forget_loss_mode"]),
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
    if spec.path is not None:
        cmd += ["--forget_set_file", str(spec.path)]
    return cmd


def _metrics_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    test_utility = metrics.get("utility", {})
    utility = metrics.get("validation_utility", test_utility)
    privacy = metrics.get("privacy", {})
    efficiency = metrics.get("efficiency", {})
    structure = metrics.get("structure", {})
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
        "overall_mia_auc": mia_auc,
        "privacy_gap": abs(mia_auc - 0.5) if mia_auc is not None else None,
        "privacy_score": _number(privacy.get("privacy_score")),
        "degree_kl_divergence": degree_kl,
        "clustering_coefficient_change": cc_change,
        "component_count_change": component_change,
        "degree_kl_abs": abs(degree_kl) if degree_kl is not None else None,
        "clustering_change_abs": abs(cc_change) if cc_change is not None else None,
        "component_change_abs": abs(component_change) if component_change is not None else None,
        "unlearn_time_seconds": _number(efficiency.get("unlearn_time_seconds")),
    }


def _write_outputs(dataset_dir: Path, rows: list[dict[str, Any]], args) -> dict[str, Any]:
    run_rows = _score_run_rows(rows, args)
    config_rows = _score_config_rows(_aggregate_configs(run_rows), args)
    candidates = [row for row in config_rows if not row.get("is_default_reference")]
    passed = [row for row in candidates if row.get("passes_hard_constraints")]
    best = passed[: args.top_k] if passed else candidates[: args.top_k]

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
    for row in scored:
        ref = runtime_refs.get((row.get("ratio"), row.get("seed")))
        runtime = _number(row.get("unlearn_time_seconds"))
        if ref is not None and runtime is not None:
            row["default_runtime_seconds"] = ref
            row["runtime_ratio_vs_default"] = runtime / ref if ref > 0 else None
            row["passes_runtime_reference"] = runtime <= args.runtime_multiplier * ref
        else:
            row["default_runtime_seconds"] = ref
            row["runtime_ratio_vs_default"] = None
            row["passes_runtime_reference"] = None
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
        "degree_kl_abs",
        "clustering_change_abs",
        "component_change_abs",
        "unlearn_time_seconds",
        "runtime_ratio_vs_default",
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
    candidate_rows = [row for row in scored if not row.get("is_default_reference")]
    metric_bounds = {
        "val_accuracy_drop_mean": _minmax([_number(row.get("val_accuracy_drop_mean")) for row in candidate_rows]),
        "val_f1_macro_drop_mean": _minmax([_number(row.get("val_f1_macro_drop_mean")) for row in candidate_rows]),
        "privacy_gap_mean": _minmax([_number(row.get("privacy_gap_mean")) for row in candidate_rows]),
        "unlearn_time_seconds_mean": _minmax([_number(row.get("unlearn_time_seconds_mean")) for row in candidate_rows]),
        "degree_kl_abs_mean": _minmax([_number(row.get("degree_kl_abs_mean")) for row in candidate_rows]),
        "clustering_change_abs_mean": _minmax([_number(row.get("clustering_change_abs_mean")) for row in candidate_rows]),
        "component_change_abs_mean": _minmax([_number(row.get("component_change_abs_mean")) for row in candidate_rows]),
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
        utility_privacy_score = (
            0.35 * _norm(row.get("privacy_gap_mean"), metric_bounds["privacy_gap_mean"])
            + 0.25 * _norm(row.get("val_accuracy_drop_mean"), metric_bounds["val_accuracy_drop_mean"])
            + 0.25 * _norm(row.get("val_f1_macro_drop_mean"), metric_bounds["val_f1_macro_drop_mean"])
            + 0.10 * structure_damage
            + 0.05 * _norm(row.get("unlearn_time_seconds_mean"), metric_bounds["unlearn_time_seconds_mean"])
        )
        base_score = {"quick": quick_score, "formal": formal_score, "structure": structure_score, "privacy": privacy_score, "utility_privacy": utility_privacy_score}[args.score_mode]
        row["score_mode"] = args.score_mode
        row["structure_damage_norm"] = structure_damage
        row["stability_penalty_norm"] = stability_penalty
        row["score_without_stability"] = base_score
        row["score"] = base_score + float(args.stability_weight) * stability_penalty
        row["passes_hard_constraints"] = _passes_config_constraints(row, args)
    return sorted(scored, key=lambda row: (bool(row.get("is_default_reference")), not bool(row.get("passes_hard_constraints")), _sort_number(row.get("score"))))


def _passes_config_constraints(row: dict[str, Any], args) -> bool:
    if _number(row.get("num_ok_runs")) != _number(row.get("num_runs")):
        return False
    if _gt(row.get("val_accuracy_drop_mean"), args.accuracy_drop_limit):
        return False
    if _gt(row.get("val_f1_macro_drop_mean"), args.f1_macro_drop_limit):
        return False
    if _gt(row.get("overall_mia_auc_mean"), args.mia_auc_limit):
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
# structure_damage_norm: {row.get('structure_damage_norm')}
# unlearn_time_seconds_mean: {row.get('unlearn_time_seconds_mean')}
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
    return json.dumps({key: cfg.get(key) for key in CONFIG_KEYS}, sort_keys=True, separators=(",", ":"))


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
