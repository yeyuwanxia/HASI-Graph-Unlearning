from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

RATIO_LABELS = {0.05: "0p05", 0.1: "0p1"}
HASI_METHODS = {"hasi_tuned", "hasi_best", "hasi_default", "hasi_prev_tuned"}
BASELINE_METHODS = {"retrain", "grapheraser_bekm", "grapheraser_blpa"}
BASELINE_KEYS = {
    "retrain": "retrain",
    "grapheraser_bekm": "grapheraser-bekm",
    "grapheraser_blpa": "grapheraser-blpa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PubMed node comparison with fixed shared_base forget sets.")
    parser.add_argument("--output_root", default="results/pubmed")
    parser.add_argument("--ratios", default="0.05,0.1")
    parser.add_argument("--seeds", default="42,123,2024")
    parser.add_argument(
        "--methods",
        default="hasi_tuned,hasi_default,retrain,grapheraser_bekm,grapheraser_blpa",
    )
    parser.add_argument("--base_artifact_root", default="results/shared_base")
    parser.add_argument("--forget_sets_root", default="experiments/forget_sets")
    parser.add_argument("--data_root", default="data/raw")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conda_env", default="graphunlearning")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    methods = _parse_strings(args.methods)
    ratios = _parse_floats(args.ratios)
    seeds = _parse_ints(args.seeds)
    commands = []
    for ratio in ratios:
        ratio_label = _ratio_label(ratio)
        for seed in seeds:
            forget_set = Path(args.forget_sets_root) / "pubmed" / f"pubmed_node_r{ratio_label}_random_train_seed{seed}.json"
            if not forget_set.exists():
                raise FileNotFoundError(f"Missing forget set: {forget_set}")
            for method in methods:
                commands.append(_build_command(args, output_root, method, ratio, ratio_label, seed, forget_set))

    if args.dry_run:
        for cmd, output, log in commands:
            print(_shell_join(cmd))
        return

    records = []
    for idx, (cmd, output, log) in enumerate(commands, 1):
        output.parent.mkdir(parents=True, exist_ok=True)
        log.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and args.skip_existing:
            print(json.dumps({"event": "skip_existing", "run": idx, "total": len(commands), "output": str(output)}), flush=True)
            records.append(_record_from_output(output, skipped=True))
            continue
        print(json.dumps({"event": "start", "run": idx, "total": len(commands), "output": str(output), "log": str(log)}), flush=True)
        start = time.perf_counter()
        with log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True)
        elapsed = time.perf_counter() - start
        if proc.returncode != 0:
            tail = ""
            if log.exists():
                tail = "\n".join(log.read_text(errors="replace").splitlines()[-80:])
            print(json.dumps({"event": "failed", "output": str(output), "log": str(log), "returncode": proc.returncode, "elapsed_seconds": elapsed}), flush=True)
            print(tail, flush=True)
            raise SystemExit(proc.returncode)
        rec = _record_from_output(output, skipped=False)
        rec["wall_time_seconds"] = elapsed
        records.append(rec)
        print(json.dumps({"event": "done", **_brief(rec)}), flush=True)

    summary_json = output_root / "node_comparison_summary.json"
    summary_csv = output_root / "node_comparison_summary.csv"
    summary_json.write_text(json.dumps({"count": len(records), "records": records}, indent=2) + "\n", encoding="utf-8")
    _write_csv(summary_csv, records)
    print(json.dumps({"event": "all_done", "summary_json": str(summary_json), "summary_csv": str(summary_csv), "count": len(records)}, indent=2), flush=True)


def _build_command(args, output_root: Path, method: str, ratio: float, ratio_label: str, seed: int, forget_set: Path):
    prefix = ["conda", "run", "-n", args.conda_env, "python"]
    common = [
        "--dataset_name", "pubmed",
        "--unlearning_type", "node",
        "--forget_ratio", str(ratio),
        "--forget_set_file", str(forget_set),
        "--seed", str(seed),
        "--base_artifact_root", args.base_artifact_root,
        "--data_root", args.data_root,
        "--device", args.device,
    ]
    if method == "hasi_tuned":
        output = output_root / "default_main" / "hasi" / "node" / f"hasi_tuned_pubmed_node_r{ratio_label}_seed{seed}.json"
        cmd = [*prefix, "experiments/run_hasi.py", "--mode", "unlearn", "--config", "configs/tuned/by_dataset/pubmed/node.yaml", "--method_name", method, *common]
    elif method == "hasi_best":
        output = output_root / "hasi_best" / "node" / f"hasi_best_pubmed_node_r{ratio_label}_seed{seed}.json"
        cmd = [*prefix, "experiments/run_hasi.py", "--mode", "unlearn", "--config", "configs/hasi_default.yaml", "--method_name", method, *common, *best_overrides()]
    elif method == "hasi_default":
        output = output_root / "default_main" / "hasi" / "node" / f"hasi_default_pubmed_node_r{ratio_label}_seed{seed}.json"
        cmd = [*prefix, "experiments/run_hasi.py", "--mode", "unlearn", "--config", "configs/hasi_default.yaml", "--method_name", method, *common]
    elif method == "hasi_prev_tuned":
        output = output_root / "hasi_prev_tuned" / "node" / f"hasi_prev_tuned_pubmed_node_r{ratio_label}_seed{seed}.json"
        cmd = [*prefix, "experiments/run_hasi.py", "--mode", "unlearn", "--config", "configs/tuned/by_dataset/pubmed/node.yaml", "--method_name", method, *common]
    elif method in BASELINE_METHODS:
        output = output_root / "default_main" / "baselines" / method / "node" / f"{method}_pubmed_node_r{ratio_label}_seed{seed}.json"
        cmd = [*prefix, "experiments/run_baselines.py", "--baseline", BASELINE_KEYS[method], *common]
        if method.startswith("grapheraser_"):
            artifact_dir = output_root / "default_main" / "baselines" / method / "node" / "artifacts" / f"seed{seed}"
            cmd.extend(["--grapheraser_artifact_dir", str(artifact_dir)])
    else:
        raise ValueError(f"Unsupported method: {method}")
    cmd.extend(["--output", str(output)])
    log = output_root / "logs" / "node_comparison" / f"{method}_pubmed_node_r{ratio_label}_seed{seed}.log"
    return cmd, output, log


def best_overrides() -> list[str]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required to read results/tuning/pubmed/node/best_config.yaml") from exc
    path = Path("results/tuning/pubmed/node/best_config.yaml")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    anchor = cfg.get("anchor_stabilization", {})
    unlearning = cfg.get("unlearning", {})
    return [
        "--anchor_lambda1", str(anchor.get("lambda1", 2.0)),
        "--anchor_lambda2", str(anchor.get("lambda2", 0.5)),
        "--forget_weight", str(unlearning.get("forget_weight", 0.1)),
        "--finetune_lr", str(unlearning.get("finetune_lr", 0.005)),
        "--finetune_epochs", str(unlearning.get("finetune_epochs", 50)),
    ]


def _record_from_output(path: Path, skipped: bool) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    method = payload.get("method") or payload.get("baseline") or metrics.get("method")
    forget = payload.get("forget_set", {})
    utility = metrics.get("utility", {})
    val = metrics.get("validation_utility", {})
    privacy = metrics.get("privacy", {})
    eff = metrics.get("efficiency", {})
    return {
        "method": method,
        "dataset": payload.get("dataset", metrics.get("dataset")),
        "unlearning_type": metrics.get("unlearning_type"),
        "ratio": forget.get("ratio"),
        "seed": forget.get("seed"),
        "selection": forget.get("selection"),
        "forget_count": metrics.get("forget_count"),
        "val_accuracy_drop": val.get("accuracy_drop"),
        "val_f1_macro_drop": val.get("f1_macro_drop"),
        "test_accuracy_drop": utility.get("accuracy_drop"),
        "test_f1_macro_drop": utility.get("f1_macro_drop"),
        "overall_mia_auc": privacy.get("overall_mia_auc"),
        "privacy_score": privacy.get("privacy_score"),
        "unlearn_time_seconds": eff.get("unlearn_time_seconds"),
        "online_wall_clock_seconds": eff.get("online_wall_clock_seconds"),
        "offline_preprocessing_seconds": eff.get("offline_preprocessing_seconds"),
        "output": str(path),
        "skipped": skipped,
    }


def _brief(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        k: rec.get(k)
        for k in (
            "method",
            "ratio",
            "seed",
            "val_accuracy_drop",
            "test_accuracy_drop",
            "overall_mia_auc",
            "unlearn_time_seconds",
            "online_wall_clock_seconds",
            "output",
        )
    }


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    fields = []
    for rec in records:
        for key in rec.keys():
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _parse_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_floats(value: str) -> list[float]:
    return [float(item) for item in _parse_strings(value)]


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in _parse_strings(value)]


def _ratio_label(ratio: float) -> str:
    for key, label in RATIO_LABELS.items():
        if abs(float(ratio) - key) < 1e-9:
            return label
    return str(ratio).replace(".", "p")


def _shell_join(parts: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(str(part)) for part in parts)


if __name__ == "__main__":
    main()
