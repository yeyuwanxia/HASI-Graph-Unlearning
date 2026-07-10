from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import load_dataset, save_forget_set
from evaluation.metrics import json_safe
from experiments.generate_forget_sets import _default_output_path, _prepare_protocol_data, _select_targets


DEFAULT_DATASETS = ("cora", "citeseer", "pubmed", "primekg")
DEFAULT_TYPES = ("node", "edge", "feature")
DEFAULT_RATIOS = (0.05, 0.1)
DEFAULT_SEEDS = (42, 123, 2024)
DEFAULT_NODE_SELECTIONS = ("random_train",)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a matrix of fixed forget-set protocol files.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS), help="Comma-separated dataset names.")
    parser.add_argument("--unlearning_types", default=",".join(DEFAULT_TYPES), help="Comma-separated: node,edge,feature.")
    parser.add_argument("--ratios", default=",".join(str(item) for item in DEFAULT_RATIOS), help="Comma-separated ratios.")
    parser.add_argument("--seeds", default=",".join(str(item) for item in DEFAULT_SEEDS), help="Comma-separated seeds.")
    parser.add_argument(
        "--shared_base_seeds",
        default=None,
        help="Optional comma-separated shared-base seeds. When set with --forget_seeds, generate every pair.",
    )
    parser.add_argument(
        "--forget_seeds",
        default=None,
        help="Optional comma-separated forget-set sampling seeds. When omitted, uses --seeds.",
    )
    parser.add_argument(
        "--node_selections",
        default=",".join(DEFAULT_NODE_SELECTIONS),
        help="Comma-separated node selection policies: random_train,random_all,hub,low_degree.",
    )
    parser.add_argument(
        "--edge_feature_selection",
        default="random_all",
        choices=["random_train", "random_all"],
        help="Selection label for edge/feature protocols. Edge scope is controlled separately.",
    )
    parser.add_argument(
        "--split_source",
        default="shared_base",
        choices=["shared_base", "dataset"],
        help="Use shared_base metadata to reconstruct each seed split, or keep dataset-provided masks.",
    )
    parser.add_argument("--base_artifact_root", default=str(ROOT / "results" / "shared_base"))
    parser.add_argument(
        "--edge_scope",
        default="train_subgraph",
        choices=["train_subgraph", "full"],
        help="For edge unlearning, sample from train-train edges or the full edge_index.",
    )
    parser.add_argument("--output_dir", default=str(ROOT / "experiments" / "forget_sets"))
    parser.add_argument(
        "--layout",
        default="flat",
        choices=["flat", "shared_base_forget_seed"],
        help="Use flat legacy filenames, or nested shared_base_seed*/forget_seed*/ directories.",
    )
    parser.add_argument("--manifest", default=None, help="Defaults to <output_dir>/manifest.json.")
    parser.add_argument("--data_root", default=str(ROOT / "data" / "raw"))
    parser.add_argument("--allow_download", action="store_true", help="Allow this command to download missing datasets.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate files that already exist.")
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = _parse_strings(args.datasets)
    unlearning_types = _parse_strings(args.unlearning_types)
    ratios = _parse_floats(args.ratios)
    seeds = _parse_ints(args.seeds)
    if args.shared_base_seeds or args.forget_seeds:
        shared_base_seeds = _parse_ints(args.shared_base_seeds) if args.shared_base_seeds else seeds
        forget_seeds = _parse_ints(args.forget_seeds) if args.forget_seeds else seeds
        seed_pairs = [(shared_base_seed, forget_seed) for shared_base_seed in shared_base_seeds for forget_seed in forget_seeds]
    else:
        shared_base_seeds = seeds
        forget_seeds = seeds
        seed_pairs = [(seed, seed) for seed in seeds]
    node_selections = _parse_strings(args.node_selections)
    output_dir = Path(args.output_dir)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "manifest.json"
    data_root = Path(args.data_root)

    records = []
    split_cache = {}
    for dataset_name in datasets:
        bundle = load_dataset(dataset_name, data_root, download=args.allow_download)
        for unlearning_type in unlearning_types:
            selections = node_selections if unlearning_type == "node" else (args.edge_feature_selection,)
            for selection in selections:
                for ratio in ratios:
                    for shared_base_seed, forget_seed in seed_pairs:
                        split_args = SimpleNamespace(
                            dataset_name=bundle.name,
                            unlearning_type=unlearning_type,
                            forget_ratio=ratio,
                            seed=shared_base_seed,
                            selection=selection,
                            split_source=args.split_source,
                            base_artifact_root=args.base_artifact_root,
                            base_artifact_dir="",
                            edge_scope=args.edge_scope,
                        )
                        target_args = SimpleNamespace(**vars(split_args))
                        target_args.seed = forget_seed
                        output_path = _output_path(
                            output_dir=output_dir,
                            layout=args.layout,
                            dataset=bundle.name,
                            unlearning_type=unlearning_type,
                            ratio=ratio,
                            shared_base_seed=shared_base_seed,
                            forget_seed=forget_seed,
                            selection=selection,
                        )
                        if output_path.exists() and not args.overwrite:
                            payload = json.loads(output_path.read_text(encoding="utf-8"))
                            status = "exists"
                            count = len(payload.get("targets", []))
                            protocol = payload.get("protocol", {})
                        else:
                            cache_key = (bundle.name, shared_base_seed, args.split_source)
                            if cache_key not in split_cache:
                                split_cache[cache_key] = _prepare_protocol_data(bundle.data, split_args, bundle.name)
                            data, split_protocol = split_cache[cache_key]
                            targets, target_protocol = _select_targets(data, target_args)
                            protocol = dict(split_protocol)
                            protocol.update(target_protocol)
                            protocol["shared_base_seed"] = int(shared_base_seed)
                            protocol["forget_seed"] = int(forget_seed)
                            save_forget_set(
                                output_path,
                                dataset=bundle.name,
                                unlearning_type=unlearning_type,
                                ratio=ratio,
                                seed=forget_seed,
                                selection=selection,
                                targets=targets,
                                protocol_metadata=protocol,
                            )
                            status = "written"
                            count = len(targets)
                        records.append(
                            {
                                "path": str(output_path),
                                "dataset": bundle.name,
                                "unlearning_type": unlearning_type,
                                "ratio": ratio,
                                "shared_base_seed": shared_base_seed,
                                "forget_seed": forget_seed,
                                "seed": forget_seed,
                                "selection": selection,
                                "forget_count": count,
                                "selection_scope": protocol.get("selection_scope"),
                                "candidate_count": protocol.get("candidate_count"),
                                "split_source": protocol.get("split_source"),
                                "base_artifact_dir": protocol.get("base_artifact_dir"),
                                "base_training_graph": protocol.get("base_training_graph"),
                                "status": status,
                            }
                        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "count": len(records),
        "datasets": datasets,
        "unlearning_types": unlearning_types,
        "ratios": ratios,
        "seeds": seeds,
        "shared_base_seeds": shared_base_seeds,
        "forget_seeds": forget_seeds,
        "split_source": args.split_source,
        "edge_scope": args.edge_scope,
        "layout": args.layout,
        "records": records,
    }
    manifest_path.write_text(json.dumps(json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    print(json.dumps(json_safe({"manifest": str(manifest_path), "count": len(records), "records": records}), indent=2))


def _parse_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in _parse_strings(value))


def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _parse_strings(value))


def _output_path(
    *,
    output_dir: Path,
    layout: str,
    dataset: str,
    unlearning_type: str,
    ratio: float,
    shared_base_seed: int,
    forget_seed: int,
    selection: str,
) -> Path:
    if layout == "shared_base_forget_seed":
        ratio_label = str(ratio).replace(".", "p")
        filename = f"{dataset}_{unlearning_type}_r{ratio_label}_{selection}.json"
        return (
            Path(output_dir)
            / dataset
            / f"shared_base_seed{int(shared_base_seed)}"
            / f"forget_seed{int(forget_seed)}"
            / filename
        )
    return _default_output_path(
        ROOT,
        output_dir,
        dataset=dataset,
        unlearning_type=unlearning_type,
        ratio=ratio,
        seed=forget_seed,
        selection=selection,
    )


if __name__ == "__main__":
    main()
