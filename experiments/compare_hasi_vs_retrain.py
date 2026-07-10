from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

METRICS = [
    ("accuracy_after", "utility__accuracy_after_mean", "higher"),
    ("accuracy_drop", "utility__accuracy_drop_mean", "lower"),
    ("f1_macro_after", "utility__f1_macro_after_mean", "higher"),
    ("mia_auc", "privacy__overall_mia_auc_mean", "closer_0p5"),
    ("privacy_score", "privacy__privacy_score_mean", "higher"),
    ("degree_kl", "structure__degree_kl_divergence_mean", "lower"),
    ("cc_change", "structure__clustering_coefficient_change_mean", "closer_0"),
    ("component_change", "structure__component_count_change_mean", "closer_0"),
    ("time_seconds", "efficiency__unlearn_time_seconds_mean", "lower"),
]
KEY_FIELDS = ["dataset", "unlearning_type", "ratio", "selection", "seed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a wide HASI-vs-retrain comparison table from aggregate_summary.json.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "results" / "aggregate_summary.json"),
        help="Path to aggregate_summary.json produced by experiments/aggregate_results.py.",
    )
    parser.add_argument(
        "--output_csv",
        default=str(ROOT / "results" / "comparison_hasi_vs_retrain.csv"),
        help="CSV path for the wide comparison table.",
    )
    parser.add_argument(
        "--output_md",
        default=str(ROOT / "results" / "comparison_hasi_vs_retrain.md"),
        help="Markdown summary path.",
    )
    parser.add_argument("--method", default="hasi", help="Primary method name in aggregate_summary.json.")
    parser.add_argument("--baseline", default="retrain", help="Baseline method name in aggregate_summary.json.")
    return parser.parse_args()


def numeric(row: dict[str, Any], column: str) -> float | None:
    value = row.get(column)
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value) >= 100:
        return f"{value:.3f}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def fmt_seeds(value: Any) -> str:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return "" if value is None else str(value)


def winner(rule: str, method_value: float | None, baseline_value: float | None, method: str, baseline: str) -> str:
    if method_value is None or baseline_value is None:
        return ""
    eps = 1e-12
    if rule == "higher":
        method_score, baseline_score = method_value, baseline_value
    elif rule == "lower":
        method_score, baseline_score = -method_value, -baseline_value
    elif rule == "closer_0p5":
        method_score, baseline_score = -abs(method_value - 0.5), -abs(baseline_value - 0.5)
    elif rule == "closer_0":
        method_score, baseline_score = -abs(method_value), -abs(baseline_value)
    else:
        return ""
    if method_score > baseline_score + eps:
        return method
    if baseline_score > method_score + eps:
        return baseline
    return "tie"


def build_comparison(summary: list[dict[str, Any]], method: str, baseline: str) -> list[dict[str, str]]:
    by_key: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in summary:
        row_method = row.get("method")
        if row_method not in {method, baseline}:
            continue
        key = tuple(row.get(field) for field in KEY_FIELDS)
        by_key.setdefault(key, {})[row_method] = row

    comparison: list[dict[str, str]] = []
    for key, methods in sorted(by_key.items(), key=lambda item: tuple(str(part) for part in item[0])):
        if method not in methods or baseline not in methods:
            continue
        method_row = methods[method]
        baseline_row = methods[baseline]
        output = {field: str(value) for field, value in zip(KEY_FIELDS, key)}
        output["seed"] = str(method_row.get("seed") if method_row.get("seed") is not None else baseline_row.get("seed"))

        method_time = numeric(method_row, "efficiency__unlearn_time_seconds_mean")
        baseline_time = numeric(baseline_row, "efficiency__unlearn_time_seconds_mean")
        output[f"computed_speedup_{baseline}_over_{method}"] = fmt(
            baseline_time / method_time if method_time not in (None, 0) and baseline_time is not None else None
        )

        for name, column, rule in METRICS:
            method_value = numeric(method_row, column)
            baseline_value = numeric(baseline_row, column)
            output[f"{method}_{name}"] = fmt(method_value)
            output[f"{baseline}_{name}"] = fmt(baseline_value)
            output[f"delta_{name}_{method}_minus_{baseline}"] = fmt(
                None if method_value is None or baseline_value is None else method_value - baseline_value
            )
            output[f"better_{name}"] = winner(rule, method_value, baseline_value, method, baseline)
        comparison.append(output)
    return comparison


def fieldnames(method: str, baseline: str) -> list[str]:
    names = [*KEY_FIELDS, f"computed_speedup_{baseline}_over_{method}"]
    for name, _, _ in METRICS:
        names.extend([
            f"{method}_{name}",
            f"{baseline}_{name}",
            f"delta_{name}_{method}_minus_{baseline}",
            f"better_{name}",
        ])
    return names


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]], method: str, baseline: str) -> None:
    speed_key = f"computed_speedup_{baseline}_over_{method}"
    lines = [
        f"# {method.upper()} vs {baseline.capitalize()} Comparison",
        "",
        "Matched by dataset, unlearning_type, ratio, selection, and seed. Delta columns are primary method minus baseline.",
        "",
        "| dataset | type | ratio | selection | seed | acc H/R/delta | f1 H/R/delta | MIA AUC H/R/delta | privacy H/R/delta | time H/R/speedup |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {unlearning_type} | {ratio} | {selection} | {seeds} | "
            "{ma}/{ba}/{da} | {mf}/{bf}/{df} | {mm}/{bm}/{dm} | {mp}/{bp}/{dp} | {mt}/{bt}/{sp}x |".format(
                dataset=row["dataset"],
                unlearning_type=row["unlearning_type"],
                ratio=row["ratio"],
                selection=row["selection"],
                seeds=row["seed"],
                ma=row[f"{method}_accuracy_after"],
                ba=row[f"{baseline}_accuracy_after"],
                da=row[f"delta_accuracy_after_{method}_minus_{baseline}"],
                mf=row[f"{method}_f1_macro_after"],
                bf=row[f"{baseline}_f1_macro_after"],
                df=row[f"delta_f1_macro_after_{method}_minus_{baseline}"],
                mm=row[f"{method}_mia_auc"],
                bm=row[f"{baseline}_mia_auc"],
                dm=row[f"delta_mia_auc_{method}_minus_{baseline}"],
                mp=row[f"{method}_privacy_score"],
                bp=row[f"{baseline}_privacy_score"],
                dp=row[f"delta_privacy_score_{method}_minus_{baseline}"],
                mt=row[f"{method}_time_seconds"],
                bt=row[f"{baseline}_time_seconds"],
                sp=row[speed_key],
            )
        )

    if rows:
        lines.extend(["", "## Quick Read", ""])
        for row in rows:
            lines.append(
                f"- {row['dataset']} {row['unlearning_type']} r={row['ratio']}: "
                f"accuracy delta {row[f'delta_accuracy_after_{method}_minus_{baseline}']}, "
                f"privacy_score delta {row[f'delta_privacy_score_{method}_minus_{baseline}']}, "
                f"computed speedup {row[speed_key]}x."
            )
    else:
        lines.extend(["", "No matched method/baseline pairs found."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    aggregate = json.loads(input_path.read_text(encoding="utf-8"))
    summary = aggregate.get("summary", [])
    rows = build_comparison(summary, args.method, args.baseline)
    fields = fieldnames(args.method, args.baseline)
    output_csv = Path(args.output_csv)
    output_md = Path(args.output_md)
    write_csv(output_csv, rows, fields)
    write_markdown(output_md, rows, args.method, args.baseline)
    print(json.dumps({"output_csv": str(output_csv.resolve()), "output_md": str(output_md.resolve()), "matched_pairs": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
