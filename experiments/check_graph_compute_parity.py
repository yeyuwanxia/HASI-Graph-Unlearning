from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import networkx as nx
import torch

from hasi.dar import DARConfig, DARPipeline
from hasi.erf_partitioning import PPRComputer
from hasi.graph_compute import ShortestPathComputer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check CPU/torch graph-compute parity.")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA is unavailable for requested device {device}")

    graph = nx.barabasi_albert_graph(300, 3, seed=11)
    graph.add_node(400)
    seeds = [0, 17, 31]

    cpu_ppr = PPRComputer(backend="cpu")
    torch_ppr = PPRComputer(backend="torch", device=str(device))
    expected = cpu_ppr.compute_seed_ppr(graph, seeds)
    actual = torch_ppr.compute_seed_ppr(graph, seeds)
    max_ppr_error = max(abs(expected[node] - actual[node]) for node in expected)
    region_equal = cpu_ppr.affected_region(graph, seeds) == torch_ppr.affected_region(graph, seeds)

    cpu_bfs = ShortestPathComputer("cpu").single_source(graph, 0)
    torch_bfs = ShortestPathComputer("torch", str(device)).single_source(graph, 0)
    bfs_equal = cpu_bfs == torch_bfs

    scores = {node: float(graph.degree(node)) for node in graph}
    dar_common = dict(k=3, min_distance=2, small_component_threshold=1, gumbel_tau=0.0, seed=7)
    cpu_dar = DARPipeline(DARConfig(**dar_common, compute_backend="cpu"))
    torch_dar = DARPipeline(DARConfig(**dar_common, compute_backend="torch", device=str(device)))
    cpu_context = cpu_dar.run_phase1(graph, 0, scores)
    torch_context = torch_dar.run_phase1(graph, 0, scores)
    dar_context_equal = (
        cpu_context.candidate_distances == torch_context.candidate_distances
        and cpu_context.preselected_candidates == torch_context.preselected_candidates
    )

    report = {
        "status": "ok",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "ppr_max_abs_error": max_ppr_error,
        "ppr_affected_region_equal": region_equal,
        "bfs_distances_equal": bfs_equal,
        "dar_phase1_equal": dar_context_equal,
    }
    if max_ppr_error >= 1e-10 or not region_equal or not bfs_equal or not dar_context_equal:
        report["status"] = "failed"
        print(json.dumps(report, indent=2))
        raise SystemExit(1)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
