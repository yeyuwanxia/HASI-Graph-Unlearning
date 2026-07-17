from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import networkx as nx
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluation.metrics import structural_metrics
from hasi.erf_partitioning import PPRComputer
from hasi.unlearner import HASIUnlearner


def test_multi_seed_ppr_falls_back_to_ranked_minimum():
    graph = nx.path_graph(12)
    seeds = [0, 2, 4, 6]
    ppr = PPRComputer(alpha=0.15, k_steps=3, threshold=2.0)

    region = ppr.affected_region(graph, seeds, excluded_nodes=seeds)

    assert len(region) == len(seeds)
    assert region.isdisjoint(seeds)
    assert ppr.last_diagnostics["threshold_selected_count"] == 0
    assert ppr.last_diagnostics["fallback_used"] is True
    assert ppr.last_diagnostics["returned_region_size"] == len(seeds)


def test_multi_seed_ppr_deduplicates_and_ignores_invalid_seeds():
    graph = nx.path_graph(6)
    ppr = PPRComputer(alpha=0.15, k_steps=2, threshold=2.0)

    region = ppr.affected_region(graph, [0, 0, 2, 99])

    assert len(region) == 2
    assert ppr.last_diagnostics["valid_seed_count"] == 2
    assert ppr.last_diagnostics["minimum_region_size"] == 2


def test_structural_metrics_exclude_forgotten_nodes_from_both_graphs():
    graph_before = nx.Graph([(0, 1), (1, 2)])
    graph_after = nx.Graph()
    graph_after.add_nodes_from([0, 1, 2])
    graph_after.add_edge(1, 2)

    raw = structural_metrics(graph_before, graph_after)
    retained = structural_metrics(graph_before, graph_after, excluded_nodes=[0])

    assert raw["component_count_change"] == 1
    assert retained["evaluation_scope"] == "retained_nodes"
    assert retained["excluded_node_count"] == 1
    assert retained["component_count_change"] == 0
    assert retained["num_nodes_before"] == 2
    assert retained["num_nodes_after"] == 2


def test_adaptive_mask_never_falls_back_to_validation_or_test_nodes():
    unlearner = SimpleNamespace(
        config=SimpleNamespace(subgraph_finetune=True, subgraph_min_nodes=2),
    )
    data = SimpleNamespace(
        num_nodes=4,
        train_mask=torch.tensor([True, True, False, False]),
    )

    mask = HASIUnlearner._adaptive_train_mask(unlearner, data, [2, 3])

    assert mask is None


if __name__ == "__main__":
    test_multi_seed_ppr_falls_back_to_ranked_minimum()
    test_multi_seed_ppr_deduplicates_and_ignores_invalid_seeds()
    test_structural_metrics_exclude_forgotten_nodes_from_both_graphs()
    test_adaptive_mask_never_falls_back_to_validation_or_test_nodes()
    print("all PPR and retained-structure tests passed")
