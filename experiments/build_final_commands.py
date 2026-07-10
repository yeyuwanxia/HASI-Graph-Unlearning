from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FinalRun:
    method: str
    kind: str
    dataset: str
    unlearning_type: str
    ratio_label: str
    ratio: float
    selection: str
    seed: int
    forget_set_file: Path
    output: Path
    config: Path | None
    baseline: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and expand the final experiment matrix.")
    parser.add_argument("--matrix", default="configs/final/main_matrix.yaml", help="Matrix YAML path.")
    parser.add_argument("--dataset", action="append", help="Limit to one or more datasets.")
    parser.add_argument("--unlearning_type", action="append", choices=["node", "edge", "feature"], help="Limit types.")
    parser.add_argument("--method", action="append", help="Limit to one or more matrix method keys.")
    parser.add_argument("--seed", action="append", type=int, help="Limit to one or more seeds.")
    parser.add_argument("--ratio", action="append", help="Limit to one or more ratio labels, e.g. 0p05.")
    parser.add_argument("--device", default=None, help="Optional device value to append as --device.")
    parser.add_argument("--print_commands", action="store_true", help="Print runnable commands.")
    parser.add_argument("--output_script", default=None, help="Write commands to a shell script.")
    parser.add_argument("--include_aggregate", action="store_true", help="Append aggregate_results.py command.")
    parser.add_argument("--json_summary", action="store_true", help="Print a JSON validation summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_path = _resolve(args.matrix)
    matrix = _load_yaml(matrix_path)
    runs = list(_expand_runs(matrix, args))
    missing = _missing_inputs(runs)

    if missing:
        print(json.dumps({"matrix": str(matrix_path), "missing_inputs": missing}, indent=2), file=sys.stderr)
        raise SystemExit(1)

    commands = [_command_for_run(matrix, run, args.device) for run in runs]
    if args.include_aggregate:
        commands.append(_aggregate_command(matrix))

    summary = _summary(matrix_path, matrix, runs)
    if args.json_summary or not args.print_commands:
        print(json.dumps(summary, indent=2))

    if args.print_commands:
        print("\n".join(commands))

    if args.output_script:
        script_path = _resolve(args.output_script)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script = ["#!/usr/bin/env bash", "set -euo pipefail", "", *commands, ""]
        script_path.write_text("\n".join(script), encoding="utf-8")
        script_path.chmod(0o755)
        print(json.dumps({"output_script": str(script_path), "num_commands": len(commands)}, indent=2))


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("PyYAML is required. Run this inside the graphunlearning environment.") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Matrix must be a mapping: {path}")
    return payload


def _expand_runs(matrix: dict[str, Any], args: argparse.Namespace) -> Iterable[FinalRun]:
    output_root = matrix["outputs"]["root"]
    result_template = matrix["outputs"]["result_template"]
    forget_template = matrix["forget_sets"]["path_template"]

    for dataset in _filter(matrix["datasets"], args.dataset):
        for unlearning_type, type_cfg in _filter_items(matrix["unlearning_types"], args.unlearning_type):
            selection = type_cfg["selection"]
            for ratio_cfg in _filter_ratios(matrix["ratios"], args.ratio):
                ratio_label = ratio_cfg["label"]
                ratio_value = float(ratio_cfg["value"])
                for seed in _filter(matrix["seeds"], args.seed):
                    for method_cfg in _filter_methods(matrix["methods"], args.method):
                        method = method_cfg["key"]
                        values = {
                            "root": output_root,
                            "dataset": dataset,
                            "method": method,
                            "unlearning_type": unlearning_type,
                            "ratio_label": ratio_label,
                            "selection": selection,
                            "seed": int(seed),
                        }
                        config = _method_config(method_cfg, dataset, unlearning_type)
                        yield FinalRun(
                            method=method,
                            kind=method_cfg["kind"],
                            dataset=dataset,
                            unlearning_type=unlearning_type,
                            ratio_label=ratio_label,
                            ratio=ratio_value,
                            selection=selection,
                            seed=int(seed),
                            forget_set_file=_resolve(forget_template.format(**values)),
                            output=_resolve(result_template.format(**values)),
                            config=_resolve(config) if config else None,
                            baseline=method_cfg.get("baseline"),
                        )


def _method_config(method_cfg: dict[str, Any], dataset: str, unlearning_type: str) -> str | None:
    if "config_template" in method_cfg:
        return method_cfg["config_template"].format(dataset=dataset, unlearning_type=unlearning_type)
    return method_cfg.get("config")


def _command_for_run(matrix: dict[str, Any], run: FinalRun, device: str | None) -> str:
    command_prefix = list(matrix.get("command", {}).get("python", [sys.executable]))
    if run.kind == "hasi":
        if run.config is None:
            raise ValueError(f"HASI run {run.method} needs a config path.")
        parts = [
            *command_prefix,
            "experiments/run_hasi.py",
            "--mode",
            "unlearn",
            "--config",
            _rel(run.config),
            "--method_name",
            run.method,
            "--dataset_name",
            run.dataset,
            "--unlearning_type",
            run.unlearning_type,
            "--forget_set_file",
            _rel(run.forget_set_file),
            "--seed",
            str(run.seed),
        ]
    elif run.kind == "baseline":
        if not run.baseline:
            raise ValueError(f"Baseline run {run.method} needs a baseline key.")
        parts = [
            *command_prefix,
            "experiments/run_baselines.py",
            "--baseline",
            run.baseline,
            "--dataset_name",
            run.dataset,
            "--unlearning_type",
            run.unlearning_type,
            "--forget_set_file",
            _rel(run.forget_set_file),
            "--seed",
            str(run.seed),
        ]
    else:
        raise ValueError(f"Unsupported method kind: {run.kind}")

    if device:
        parts.extend(["--device", device])
    parts.extend(["--output", _rel(run.output)])
    return _shell_join(parts)


def _aggregate_command(matrix: dict[str, Any]) -> str:
    output_root = matrix["outputs"]["root"]
    parts = [
        *matrix.get("command", {}).get("python", [sys.executable]),
        "experiments/aggregate_results.py",
        "--input_dir",
        output_root,
        "--output_json",
        matrix["outputs"]["aggregate_json"].format(root=output_root),
        "--output_csv",
        matrix["outputs"]["aggregate_csv"].format(root=output_root),
    ]
    return _shell_join(parts)


def _missing_inputs(runs: Iterable[FinalRun]) -> list[dict[str, str]]:
    missing = []
    seen: set[Path] = set()
    for run in runs:
        inputs = [run.forget_set_file]
        if run.config is not None:
            inputs.append(run.config)
        for path in inputs:
            if path in seen:
                continue
            seen.add(path)
            if not path.exists():
                missing.append({"path": str(path), "kind": "input"})
    return missing


def _summary(matrix_path: Path, matrix: dict[str, Any], runs: list[FinalRun]) -> dict[str, Any]:
    by_method: dict[str, int] = {}
    by_dataset: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for run in runs:
        by_method[run.method] = by_method.get(run.method, 0) + 1
        by_dataset[run.dataset] = by_dataset.get(run.dataset, 0) + 1
        by_type[run.unlearning_type] = by_type.get(run.unlearning_type, 0) + 1
    return {
        "matrix": str(matrix_path),
        "name": matrix.get("name"),
        "num_runs": len(runs),
        "methods": by_method,
        "datasets": by_dataset,
        "unlearning_types": by_type,
        "forget_seeds": by_forget_seed,
        "outputs_root": matrix["outputs"]["root"],
    }



def _method_applies(method_cfg: dict[str, Any], dataset: str, unlearning_type: str) -> bool:
    datasets = method_cfg.get("datasets")
    if datasets is not None and str(dataset) not in {str(item) for item in datasets}:
        return False
    types = method_cfg.get("unlearning_types")
    if types is not None and unlearning_type not in set(types):
        return False
    return True


def _base_artifact_root(matrix: dict[str, Any]) -> str | None:
    shared_base = matrix.get("shared_base") or {}
    return shared_base.get("root") or matrix.get("base_artifact_root")

def _filter(values: list[Any], selected: list[Any] | None) -> list[Any]:
    if not selected:
        return values
    selected_set = {str(item) for item in selected}
    return [value for value in values if str(value) in selected_set]


def _filter_items(values: dict[str, Any], selected: list[str] | None) -> list[tuple[str, Any]]:
    if not selected:
        return list(values.items())
    selected_set = set(selected)
    return [(key, value) for key, value in values.items() if key in selected_set]


def _filter_ratios(ratios: list[dict[str, Any]], selected: list[str] | None) -> list[dict[str, Any]]:
    if not selected:
        return ratios
    selected_set = set(selected)
    return [ratio for ratio in ratios if ratio["label"] in selected_set]


def _filter_methods(methods: list[dict[str, Any]], selected: list[str] | None) -> list[dict[str, Any]]:
    if not selected:
        return methods
    selected_set = set(selected)
    return [method for method in methods if method["key"] in selected_set]


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _shell_join(parts: Iterable[str]) -> str:
    quoted = []
    for part in parts:
        if part.startswith("$") or part.startswith("${"):
            quoted.append(part)
        else:
            quoted.append(shlex.quote(part))
    return " ".join(quoted)


if __name__ == "__main__":
    main()
