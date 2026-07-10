from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import dataset_cache_exists, dataset_cache_paths, load_dataset
from evaluation.metrics import json_safe


DEFAULT_DATASETS = ("cora", "citeseer", "pubmed")


def parse_args():
    parser = argparse.ArgumentParser(description="Download and prepare graph datasets before experiments.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="Comma-separated dataset names.")
    parser.add_argument("--data_root", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--manifest", default=str(ROOT / "data" / "dataset_manifest.json"))
    parser.add_argument("--force_load", action="store_true", help="Load datasets even if local cache appears present.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    records = []

    for dataset_name in _parse_list(args.datasets):
        was_cached = dataset_cache_exists(dataset_name, data_root)
        if was_cached and not args.force_load:
            records.append(
                {
                    "dataset": dataset_name,
                    "status": "exists",
                    "cache_paths": [str(path) for path in dataset_cache_paths(dataset_name, data_root)],
                }
            )
            continue

        bundle = load_dataset(dataset_name, data_root, download=True)
        data = bundle.data
        records.append(
            {
                "dataset": bundle.name,
                "status": "downloaded" if not was_cached else "loaded",
                "num_nodes": int(getattr(data, "num_nodes", 0)),
                "num_edges": int(data.edge_index.shape[1]) if hasattr(data, "edge_index") else None,
                "num_features": bundle.num_features,
                "num_classes": bundle.num_classes,
                "root": str(bundle.root),
                "cache_paths": [str(path) for path in dataset_cache_paths(bundle.name, data_root)],
            }
        )

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"data_root": str(data_root), "records": records}
    manifest_path.write_text(json.dumps(json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json_safe({"manifest": str(manifest_path), "records": records}), indent=2))


def _parse_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    main()
