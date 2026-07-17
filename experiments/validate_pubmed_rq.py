from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SEEDS = {42, 123, 2024}
EXPECTED = {
    "rq1": {"files": 18, "groups": 6},
    "rq3": {"files": 9, "groups": 3},
    "rq4": {"files": 6, "groups": 2},
    "rq5": {"files": 6, "groups": 2},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate formal PubMed RQ result semantics.")
    parser.add_argument("--rq", required=True, choices=sorted(EXPECTED))
    parser.add_argument("--input_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir)
    files = sorted(path for path in root.rglob("*.json") if not path.name.startswith("aggregate_summary"))
    errors: list[str] = []
    records: list[dict[str, Any]] = []

    expected_count = EXPECTED[args.rq]["files"]
    if len(files) != expected_count:
        errors.append(f"expected {expected_count} result JSONs, found {len(files)}")

    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - validator must report every bad file.
            errors.append(f"{path}: cannot parse JSON: {exc}")
            continue
        _validate_common(path, payload, errors)
        records.append(payload)

    if args.rq == "rq1":
        _validate_rq1(records, errors)
    elif args.rq == "rq3":
        _validate_rq3(records, errors)
    elif args.rq == "rq4":
        _validate_rq4(records, errors)
    else:
        _validate_rq5(records, errors)
    _validate_aggregate(root, args.rq, errors)

    report = {
        "rq": args.rq,
        "input_dir": str(root),
        "num_result_files": len(files),
        "status": "ok" if not errors else "failed",
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


def _validate_common(path: Path, payload: dict[str, Any], errors: list[str]) -> None:
    prefix = str(path)
    metrics = payload.get("metrics", {})
    forget = payload.get("forget_set", {})
    unlearning = payload.get("unlearning", {})
    resolved = payload.get("config", {}).get("resolved", {})
    privacy = metrics.get("privacy", {})
    diagnostics = unlearning.get("affected_region_diagnostics", {})
    protocol = metrics.get("evaluation_protocol", {})
    base = payload.get("base_artifact", {})
    forget_protocol = forget.get("protocol", {})
    optimization = resolved.get("optimization", {})

    _require(payload.get("dataset") == "pubmed", prefix, "dataset must be pubmed", errors)
    _require(protocol.get("version") == "paper_eval_20260715_v1", prefix, "wrong evaluation protocol", errors)
    _require(metrics.get("unlearning_type") == "node", prefix, "unlearning_type must be node", errors)
    _require(_as_int(forget.get("seed")) in SEEDS, prefix, "unexpected forget/base seed", errors)
    seed = _as_int(forget.get("seed"))
    expected_base = f"results/shared_base/pubmed/seed{seed}"
    _require(base.get("loaded") is True, prefix, "shared base was not loaded", errors)
    _require(base.get("path") == expected_base, prefix, "wrong shared-base path", errors)
    _require(str(forget.get("path", "")).startswith("experiments/rq_forget_sets/pubmed/"), prefix, "wrong forget-set path", errors)
    _require(forget_protocol.get("split_source") == "shared_base", prefix, "forget set is not shared-base bound", errors)
    _require(forget_protocol.get("base_artifact_dir") == expected_base, prefix, "forget-set base path mismatch", errors)
    _require(forget_protocol.get("base_training_graph") == "train_subgraph", prefix, "wrong base training graph", errors)
    _require(forget_protocol.get("selection_scope") == "train_mask_nodes", prefix, "RQ forget set must target training nodes", errors)
    _require(_as_float(resolved.get("unlearning", {}).get("ratio")) == _as_float(forget.get("ratio")), prefix, "resolved/forget ratio mismatch", errors)
    _require(privacy.get("status") == "ok", prefix, "node MIA status is not ok", errors)
    _require(privacy.get("medium_evaluation") == "held_out_target_split", prefix, "Medium MIA is not held-out", errors)
    _require((privacy.get("medium_train_size") or 0) > 0, prefix, "Medium MIA train split is empty", errors)
    _require((privacy.get("medium_eval_size") or 0) > 0, prefix, "Medium MIA eval split is empty", errors)
    _require(privacy.get("medium_auc") is not None, prefix, "missing Medium MIA AUC", errors)
    for field in ("strong_auc", "strong_auc_null_mean", "strong_auc_null_std", "strong_auc_pvalue"):
        _require(privacy.get(field) is not None, prefix, f"missing privacy field {field}", errors)
    _require(diagnostics.get("returned_region_size", 0) > 0, prefix, "affected region is empty", errors)
    _require(diagnostics.get("selection_policy") == "absolute_threshold_with_ranked_minimum", prefix, "wrong affected-region policy", errors)
    _require(metrics.get("structure", {}).get("evaluation_scope") == "retained_nodes", prefix, "structural metrics are not retained-node scoped", errors)
    backend = diagnostics.get("compute_backend", {})
    _require(backend.get("used_backend") == "torch", prefix, "missing PPR backend diagnostics", errors)
    cache = payload.get("hub_score_cache", {})
    _require(cache.get("enabled") is True and bool(cache.get("key")), prefix, "HubScore cache metadata missing", errors)
    _require(cache.get("hit") is True, prefix, "formal run must use a warm HubScore artifact", errors)
    _require(cache.get("artifact_metadata", {}).get("implementation_version") == "hub_score_cache_schema_v2_torch_ppr_v1", prefix, "wrong HubScore cache implementation", errors)
    _require(
        cache.get("offline_preprocessing_seconds") is not None,
        prefix,
        "HubScore offline preprocessing time missing",
        errors,
    )
    _require(
        "/hasi/artifacts/hub_scores/" in str(cache.get("path", "")),
        prefix,
        "HubScore artifact is outside the formal result tree",
        errors,
    )
    _require(
        metrics.get("efficiency", {}).get("offline_preprocessing_seconds")
        == cache.get("offline_preprocessing_seconds"),
        prefix,
        "efficiency/offline HubScore time mismatch",
        errors,
    )
    _require(optimization.get("graph_compute_backend") == "torch", prefix, "graph backend must be torch", errors)
    _require(optimization.get("hub_ppr_batch_size") == 64, prefix, "HubScore PPR batch size must be 64", errors)


def _validate_rq1(records: list[dict[str, Any]], errors: list[str]) -> None:
    expected = {
        (seed, ratio, selection)
        for seed in SEEDS
        for ratio in (0.05, 0.1)
        for selection in ("random_train", "hub_train", "low_degree_train")
    }
    observed = set()
    for payload in records:
        method = payload.get("method")
        _require(method == "hasi_default_rq1_mia_v2", "RQ1", f"unexpected method {method}", errors)
        forget = payload.get("forget_set", {})
        observed.add((_as_int(forget.get("seed")), _as_float(forget.get("ratio")), forget.get("selection")))
    _require(observed == expected, "RQ1", "seed/ratio/selection matrix is incomplete or duplicated", errors)


def _validate_rq3(records: list[dict[str, Any]], errors: list[str]) -> None:
    variants = {
        "hasi_no_anchor_rq3_mia_v2_daroff": ("none", 0.0, 0.0),
        "hasi_hier_anchor_rq3_mia_v2_daroff": ("hierarchical", 2.0, 0.5),
        "hasi_strong_anchor_rq3_mia_v2_daroff": ("hierarchical", 5.0, 1.0),
    }
    observed = set()
    for payload in records:
        method = payload.get("method")
        expected = variants.get(method)
        _require(expected is not None, "RQ3", f"unexpected method {method}", errors)
        if expected is None:
            continue
        resolved = payload.get("config", {}).get("resolved", {})
        anchor = resolved.get("anchor_stabilization", {})
        dar = resolved.get("dar", {})
        _require(
            (anchor.get("mode"), _as_float(anchor.get("lambda1")), _as_float(anchor.get("lambda2"))) == expected,
            method,
            "anchor configuration mismatch",
            errors,
        )
        _require(dar.get("enabled") is False, method, "DAR must be disabled for every RQ3 variant", errors)
        observed.add((method, _as_int(payload.get("forget_set", {}).get("seed"))))
    _require(observed == {(method, seed) for method in variants for seed in SEEDS}, "RQ3", "variant/seed matrix incomplete", errors)


def _validate_rq4(records: list[dict[str, Any]], errors: list[str]) -> None:
    variants = {
        "hasi_no_inpaint_rq4_mia_v2": "none",
        "hasi_full_inpaint_rq4_mia_v2": "full",
    }
    observed = set()
    for payload in records:
        method = payload.get("method")
        mode = variants.get(method)
        _require(mode is not None, "RQ4", f"unexpected method {method}", errors)
        if mode is None:
            continue
        resolved_mode = payload.get("config", {}).get("resolved", {}).get("inpainting", {}).get("mode")
        _require(resolved_mode == mode, method, "inpainting mode mismatch", errors)
        inpainting = payload.get("unlearning", {}).get("inpainting", {})
        if mode == "full":
            _require(inpainting.get("triggered") is True, method, "full inpainting did not trigger", errors)
            _require(inpainting.get("stats", {}).get("edges_added", 0) > 0, method, "full inpainting added no edges", errors)
        else:
            _require(inpainting.get("triggered") is False, method, "no-inpainting variant unexpectedly triggered", errors)
        observed.add((method, _as_int(payload.get("forget_set", {}).get("seed"))))
    _require(observed == {(method, seed) for method in variants for seed in SEEDS}, "RQ4", "variant/seed matrix incomplete", errors)


def _validate_rq5(records: list[dict[str, Any]], errors: list[str]) -> None:
    variants = {
        "hasi_dar_off_rq5_mia_v2": False,
        "hasi_dar_on_rq5_mia_v2": True,
    }
    observed = set()
    for payload in records:
        method = payload.get("method")
        enabled = variants.get(method)
        _require(enabled is not None, "RQ5", f"unexpected method {method}", errors)
        if enabled is None:
            continue
        resolved = payload.get("config", {}).get("resolved", {}).get("dar", {}).get("enabled")
        _require(resolved is enabled, method, "DAR enabled flag mismatch", errors)
        contexts = payload.get("unlearning", {}).get("dar_contexts", [])
        anchors = payload.get("unlearning", {}).get("dar_anchors", [])
        if enabled:
            _require(len(contexts) > 0, method, "DAR-on produced no deletion contexts", errors)
            _require(len(anchors) > 0, method, "DAR-on produced no replacement anchors", errors)
        else:
            _require(len(contexts) == 0 and len(anchors) == 0, method, "DAR-off produced DAR outputs", errors)
        observed.add((method, _as_int(payload.get("forget_set", {}).get("seed"))))
    _require(observed == {(method, seed) for method in variants for seed in SEEDS}, "RQ5", "variant/seed matrix incomplete", errors)


def _validate_aggregate(root: Path, rq: str, errors: list[str]) -> None:
    path = root / "aggregate_summary.json"
    if not path.is_file():
        errors.append(f"{path}: aggregate summary missing")
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("num_files") == EXPECTED[rq]["files"], str(path), "aggregate file count mismatch", errors)
    _require(payload.get("num_groups") == EXPECTED[rq]["groups"], str(path), "aggregate group count mismatch", errors)


def _require(condition: bool, prefix: str, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(f"{prefix}: {message}")


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
