from __future__ import annotations

import pathlib
import sys

import networkx as nx
import pytest
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hasi.dar import DARConfig, DARPipeline
from hasi.erf_partitioning import PPRComputer
from hasi.graph_compute import ShortestPathComputer, TorchSparseGraph
from hasi.hub_identification import HubScoreConfig, HubScorer


def _graphs():
    path = nx.path_graph(9)
    path.add_node(20)
    branched = nx.Graph([(0, 1), (1, 2), (1, 3), (3, 4), (3, 5), (5, 6)])
    branched.add_edge(6, 6)
    directed = nx.DiGraph([(0, 1), (1, 2), (2, 0), (2, 3)])
    return [path, branched, directed]


@pytest.mark.parametrize("graph", _graphs())
def test_torch_ppr_matches_python_cpu(graph):
    seeds = [node for node in [0, 3, 20] if node in graph]
    cpu = PPRComputer(alpha=0.15, k_steps=3, threshold=0.01, backend="cpu")
    torch_cpu = PPRComputer(alpha=0.15, k_steps=3, threshold=0.01, backend="torch", device="cpu")

    expected = cpu.compute_seed_ppr(graph, seeds)
    actual = torch_cpu.compute_seed_ppr(graph, seeds)

    assert expected.keys() == actual.keys()
    assert max(abs(expected[node] - actual[node]) for node in expected) < 1e-12
    assert cpu.affected_region(graph, seeds) == torch_cpu.affected_region(graph, seeds)


@pytest.mark.parametrize("graph", _graphs())
def test_torch_bfs_matches_networkx_cpu(graph):
    source = 0
    cpu = ShortestPathComputer("cpu")
    torch_cpu = ShortestPathComputer("torch", "cpu")

    assert cpu.single_source(graph, source) == torch_cpu.single_source(graph, source)
    assert cpu.single_source(graph, source, cutoff=2) == torch_cpu.single_source(graph, source, cutoff=2)


@pytest.mark.parametrize("graph", _graphs())
def test_torch_hub_pagerank_and_eigenvector_match_networkx(graph):
    sparse = TorchSparseGraph(graph, "cpu")

    expected_pagerank = nx.pagerank(graph, alpha=0.85, max_iter=100, tol=1e-6)
    actual_pagerank = sparse.pagerank(alpha=0.85, max_iter=100, tol=1e-6)
    for node, value in zip(sparse.nodes, actual_pagerank.tolist()):
        assert value == pytest.approx(expected_pagerank[node], abs=1e-12)

    expected_eigen = nx.eigenvector_centrality(graph, max_iter=200, tol=1e-6)
    actual_eigen = sparse.eigenvector_centrality(max_iter=200, tol=1e-6)
    for node, value in zip(sparse.nodes, actual_eigen.tolist()):
        assert value == pytest.approx(expected_eigen[node], abs=1e-12)


@pytest.mark.parametrize("graph", _graphs())
def test_batched_hub_personalized_pagerank_matches_networkx(graph):
    seeds = list(graph.nodes)[: min(4, graph.number_of_nodes())]
    sparse = TorchSparseGraph(graph, "cpu")
    actual = sparse.personalized_pagerank_off_diagonal_mass(
        seeds, alpha=0.15, max_iter=50, tol=1e-6, batch_size=2
    )

    for seed in seeds:
        expected_rank = nx.pagerank(
            graph,
            alpha=0.15,
            personalization={node: 1.0 if node == seed else 0.0 for node in graph.nodes},
            max_iter=50,
            tol=1e-6,
        )
        expected = sum(expected_rank.values()) - expected_rank[seed]
        assert actual[seed] == pytest.approx(expected, abs=1e-12)


def test_hub_scorer_torch_backend_matches_cpu_scores():
    graph = nx.barabasi_albert_graph(40, 2, seed=17)
    common = dict(filter_ratio=0.25, ppr_max_iter=50, ppr_tol=1e-6)
    cpu = HubScorer(HubScoreConfig(**common, compute_backend="cpu"))
    torch_cpu = HubScorer(
        HubScoreConfig(
            **common,
            compute_backend="torch",
            compute_device="cpu",
            ppr_batch_size=3,
        )
    )
    expected = cpu.compute_hub_scores(graph)
    actual = torch_cpu.compute_hub_scores(graph)

    assert expected.keys() == actual.keys()
    assert max(abs(expected[node] - actual[node]) for node in expected) < 1e-9
    assert torch_cpu.last_diagnostics["centrality_backend"] == "torch_sparse"
    assert torch_cpu.last_diagnostics["erf_backend"] == "torch_sparse_batched"


def test_dar_selection_is_backend_invariant():
    graph = nx.cycle_graph(12)
    graph.add_edges_from([(0, 6), (0, 7), (3, 9)])
    scores = {node: float(node) / 11.0 for node in graph}
    common = dict(k=3, min_distance=2, small_component_threshold=1, gumbel_tau=0.0, seed=7)
    cpu = DARPipeline(DARConfig(**common, compute_backend="cpu"))
    torch_cpu = DARPipeline(DARConfig(**common, compute_backend="torch", device="cpu"))

    cpu_context = cpu.run_phase1(graph, 0, scores)
    torch_context = torch_cpu.run_phase1(graph, 0, scores)
    assert cpu_context.candidate_distances == torch_context.candidate_distances
    assert cpu_context.preselected_candidates == torch_context.preselected_candidates

    repaired = graph.copy()
    repaired.remove_node(0)
    cpu_anchors = cpu.run_phase2(cpu_context, repaired)
    torch_anchors = torch_cpu.run_phase2(torch_context, repaired)
    assert [(item.node, item.cached_distance) for item in cpu_anchors] == [
        (item.node, item.cached_distance) for item in torch_anchors
    ]


def test_undirected_dar_reuses_bfs_from_selected_anchors():
    graph = nx.barabasi_albert_graph(120, 3, seed=5)
    scores = {node: float(graph.degree(node)) for node in graph}
    pipeline = DARPipeline(
        DARConfig(k=4, min_distance=2, small_component_threshold=1, gumbel_tau=0.0)
    )

    pipeline.run_phase1(graph, 0, scores)

    # One BFS is rooted at the deleted node. Diversity checks then need at
    # most one exact BFS for each selected candidate, not one per candidate.
    assert len(pipeline.distance_computer._distance_cache) <= 1 + 2 * pipeline.config.k


def test_repeated_distance_queries_reuse_graph_key_and_distance_map():
    class CountingGraph(nx.Graph):
        edge_count_calls = 0

        def number_of_edges(self, u=None, v=None):
            self.edge_count_calls += 1
            return super().number_of_edges(u, v)

    graph = CountingGraph(nx.path_graph(20))
    computer = ShortestPathComputer("cpu")

    assert computer.distance(graph, 0, 19) == 19
    for target in range(19):
        assert computer.distance(graph, 0, target) == target

    assert graph.edge_count_calls == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_ppr_and_bfs_match_cpu():
    graph = nx.barabasi_albert_graph(200, 3, seed=11)
    seeds = [0, 17, 31]
    cpu_ppr = PPRComputer(backend="cpu")
    gpu_ppr = PPRComputer(backend="torch", device="cuda:0")
    expected = cpu_ppr.compute_seed_ppr(graph, seeds)
    actual = gpu_ppr.compute_seed_ppr(graph, seeds)
    assert max(abs(expected[node] - actual[node]) for node in expected) < 1e-10
    assert cpu_ppr.affected_region(graph, seeds) == gpu_ppr.affected_region(graph, seeds)

    common = dict(filter_ratio=0.1, ppr_max_iter=50, ppr_tol=1e-6)
    cpu_hub = HubScorer(HubScoreConfig(**common, compute_backend="cpu"))
    gpu_hub = HubScorer(
        HubScoreConfig(
            **common,
            compute_backend="torch",
            compute_device="cuda:0",
            ppr_batch_size=8,
        )
    )
    expected_hub = cpu_hub.compute_hub_scores(graph)
    actual_hub = gpu_hub.compute_hub_scores(graph)
    assert max(abs(expected_hub[node] - actual_hub[node]) for node in expected_hub) < 1e-9

    assert ShortestPathComputer("cpu").single_source(graph, 0) == ShortestPathComputer(
        "torch", "cuda:0"
    ).single_source(graph, 0)
