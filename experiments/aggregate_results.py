from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_GROUP_BY = (
    "method",
    "dataset",
    "unlearning_type",
    "ratio",
    "selection",
    "seed",
)
DEFAULT_METRICS = (
    "metrics.utility.accuracy_after",
    "metrics.utility.accuracy_drop",
    "metrics.utility.f1_macro_after",
    "metrics.structure.degree_kl_divergence",
    "metrics.structure.clustering_coefficient_change",
    "metrics.structure.component_count_change",
    "metrics.privacy.weak_auc",
    "metrics.privacy.medium_auc",
    "metrics.privacy.strong_auc",
    "metrics.privacy.overall_mia_auc",
    "metrics.privacy.privacy_score",
    "metrics.efficiency.unlearn_time_seconds",
    "metrics.efficiency.online_wall_clock_seconds",
    "metrics.efficiency.offline_preprocessing_seconds",
    "metrics.efficiency.speedup_vs_retrain",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate experiment JSON files into summaries.")
    parser.add_argument("--input_dir", default=str(ROOT / "results"))
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--output_json", default=str(ROOT / "results" / "aggregate_summary.json"))
    parser.add_argument("--output_csv", default=str(ROOT / "results" / "aggregate_summary.csv"))
    parser.add_argument("--group_by", default=",".join(DEFAULT_GROUP_BY), help="Comma-separated record fields.")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help="Comma-separated JSON paths to aggregate.")
    parser.add_argument("--include_records", action="store_true", help="Include per-file records in JSON output.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    group_by = _parse_list(args.group_by)
    metric_paths = _parse_list(args.metrics)

    records = []
    skipped = []
    output_json_path = Path(args.output_json).resolve()
    for path in _iter_input_files(input_dir, args.pattern):
        if _should_skip_input(path, output_json_path):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload.get("metrics"), dict):
                raise ValueError("missing metrics object")
            records.append(_record_from_payload(payload, path, metric_paths))
        except Exception as exc:  # noqa: BLE001 - aggregation should report bad files and continue.
            skipped.append({"path": str(path), "error": str(exc)})

    summary = _aggregate(records, group_by, metric_paths)
    output = {
        "input_dir": str(input_dir),
        "pattern": args.pattern,
        "recursive": True,
        "num_files": len(records),
        "num_groups": len(summary),
        "group_by": group_by,
        "metrics": metric_paths,
        "summary": summary,
        "skipped": skipped,
    }
    if args.include_records:
        output["records"] = records

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_json_safe(output), indent=2) + "\n", encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output_csv, summary, group_by, metric_paths)

    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_csv": str(output_csv),
                "num_files": len(records),
                "num_groups": len(summary),
                "num_skipped": len(skipped),
            },
            indent=2,
        )
    )


def _record_from_payload(payload: dict[str, Any], path: Path, metric_paths: Iterable[str]) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    forget_set = payload.get("forget_set", {})
    config = payload.get("config", {}).get("resolved", {})
    unlearning_cfg = config.get("unlearning", {})

    record = {
        "path": str(path),
        "method": metrics.get("method") or payload.get("method") or payload.get("baseline"),
        "dataset": metrics.get("dataset") or payload.get("dataset"),
        "unlearning_type": metrics.get("unlearning_type") or forget_set.get("unlearning_type") or unlearning_cfg.get("type"),
        "ratio": forget_set.get("ratio") if forget_set.get("ratio") is not None else unlearning_cfg.get("ratio"),
        "seed": forget_set.get("seed") if forget_set.get("seed") is not None else config.get("training", {}).get("seed"),
        "selection": forget_set.get("selection") or "unspecified",
        "forget_count": metrics.get("forget_count") or len(forget_set.get("targets", [])),
    }
    for metric_path in metric_paths:
        record[_metric_key(metric_path)] = _as_float(_get_path(payload, metric_path))
    return record


def _aggregate(records: list[dict[str, Any]], group_by: list[str], metric_paths: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        key = tuple(record.get(field) for field in group_by)
        groups.setdefault(key, []).append(record)

    summary = []
    for key, items in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        row = {field: value for field, value in zip(group_by, key)}
        row["n"] = len(items)
        row["seeds"] = sorted({item.get("seed") for item in items if item.get("seed") is not None})
        row["paths"] = [item["path"] for item in items]
        for metric_path in metric_paths:
            metric_key = _metric_key(metric_path)
            values = [item[metric_key] for item in items if item.get(metric_key) is not None]
            row[f"{metric_key}_n"] = len(values)
            row[f"{metric_key}_mean"] = statistics.fmean(values) if values else None
            row[f"{metric_key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0 if values else None
            row[f"{metric_key}_min"] = min(values) if values else None
            row[f"{metric_key}_max"] = max(values) if values else None
        summary.append(row)
    return summary


def _iter_input_files(input_dir: Path, pattern: str) -> list[Path]:
    if "**" in pattern:
        return sorted(input_dir.glob(pattern))
    return sorted(input_dir.rglob(pattern))


def _should_skip_input(path: Path, output_json_path: Path) -> bool:
    if path.resolve() == output_json_path:
        return True
    return path.name.startswith("aggregate_summary")


def _write_csv(path: Path, summary: list[dict[str, Any]], group_by: list[str], metric_paths: list[str]) -> None:
    metric_columns = []
    for metric_path in metric_paths:
        metric_key = _metric_key(metric_path)
        metric_columns.extend(
            [
                f"{metric_key}_n",
                f"{metric_key}_mean",
                f"{metric_key}_std",
                f"{metric_key}_min",
                f"{metric_key}_max",
            ]
        )
    seed_columns = [] if "seed" in group_by else ["seeds"]
    fieldnames = [*group_by, "n", *seed_columns, *metric_columns]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in summary:
            csv_row = dict(row)
            csv_row["seeds"] = ",".join(str(seed) for seed in row.get("seeds", []))
            writer.writerow(csv_row)


def _get_path(payload: dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric_key(path: str) -> str:
    prefix = "metrics."
    if path.startswith(prefix):
        path = path[len(prefix) :]
    return path.replace(".", "__")


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


if __name__ == "__main__":
    main()
