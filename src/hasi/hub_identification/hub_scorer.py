from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set

import networkx as nx


ScoreMap = Dict[int, float]


@dataclass(frozen=True)
class HubScoreConfig:
    """Weights and ratios for HASI hub identification."""

    alpha: float = 0.4
    beta: float = 0.3
    gamma: float = 0.3
    filter_ratio: float = 0.1
    primary_ratio: float = 0.01
    secondary_ratio: float = 0.05
    ppr_alpha: float = 0.15
    ppr_max_iter: int = 50
    ppr_tol: float = 1e-6


class HubScorer:
    """Compute HASI HubScore for graph nodes.

    The implementation is intentionally dependency-light: topology scores use
    NetworkX, and model-aware gradient scores can be supplied by the caller.
    If no gradient scores are supplied, the gradient term is treated as zero
    rather than silently inventing task information.
    """

    def __init__(self, config: Optional[HubScoreConfig] = None):
        self.config = config or HubScoreConfig()

    def compute_hub_scores(
        self,
        graph: nx.Graph,
        gradient_scores: Optional[Mapping[int, float]] = None,
        candidate_nodes: Optional[Iterable[int]] = None,
    ) -> ScoreMap:
        """Return weighted HubScore for every node in the graph.

        Filter-and-refine is applied by first ranking nodes with a centrality
        score, then computing ERF/PPR influence only for the retained set.
        Non-candidate nodes remain in the returned map with score 0.0 so
        downstream anchor classification can stay total over V.
        """

        nodes = list(candidate_nodes) if candidate_nodes is not None else list(graph.nodes)
        if not nodes:
            return {}

        centrality = self.compute_centrality_scores(graph, nodes)
        retained = self._retain_candidates(centrality)
        gradient = self._normalize_dict(dict(gradient_scores or {}), retained)
        erf = self.compute_erf_influence(graph, retained)

        scores = {int(node): 0.0 for node in graph.nodes}
        for node in retained:
            scores[int(node)] = (
                self.config.alpha * gradient.get(node, 0.0)
                + self.config.beta * centrality.get(node, 0.0)
                + self.config.gamma * erf.get(node, 0.0)
            )
        return scores

    def classify_anchors(self, scores: Mapping[int, float]) -> tuple[Set[int], Set[int], Set[int]]:
        """Split nodes into Primary, Secondary, and Regular anchors."""

        ranked = sorted(scores, key=lambda node: scores[node], reverse=True)
        n_nodes = len(ranked)
        if n_nodes == 0:
            return set(), set(), set()

        primary_count = max(1, int(round(n_nodes * self.config.primary_ratio)))
        secondary_count = max(primary_count, int(round(n_nodes * self.config.secondary_ratio)))

        primary = set(ranked[:primary_count])
        secondary = set(ranked[primary_count:secondary_count])
        regular = set(ranked[secondary_count:])
        return primary, secondary, regular

    def compute_centrality_scores(self, graph: nx.Graph, nodes: Optional[Sequence[int]] = None) -> ScoreMap:
        """Compute normalized topology centrality score.

        The default follows the HASI document: PageRank, degree, and
        eigenvector-style centrality. Eigenvector centrality can fail to
        converge on some graphs, so PageRank is used as the stable fallback.
        """

        nodes = list(nodes) if nodes is not None else list(graph.nodes)
        if not nodes:
            return {}

        pagerank = nx.pagerank(graph, alpha=0.85) if graph.number_of_edges() else {node: 1.0 for node in graph.nodes}
        degree = dict(graph.degree())
        try:
            eigen = nx.eigenvector_centrality(graph, max_iter=200)
        except (nx.NetworkXException, ZeroDivisionError):
            eigen = pagerank

        pr_norm = self._normalize_dict(pagerank, nodes)
        degree_norm = self._normalize_dict(degree, nodes)
        eigen_norm = self._normalize_dict(eigen, nodes)

        return {
            int(node): 0.6 * pr_norm.get(node, 0.0)
            + 0.3 * degree_norm.get(node, 0.0)
            + 0.1 * eigen_norm.get(node, 0.0)
            for node in nodes
        }

    def compute_erf_influence(self, graph: nx.Graph, nodes: Iterable[int]) -> ScoreMap:
        """Approximate ERF influence with personalized PageRank mass."""

        raw_scores: ScoreMap = {}
        for node in nodes:
            if node not in graph:
                continue
            ppr = nx.pagerank(
                graph,
                alpha=self.config.ppr_alpha,
                personalization={n: 1.0 if n == node else 0.0 for n in graph.nodes},
                max_iter=self.config.ppr_max_iter,
                tol=self.config.ppr_tol,
            )
            raw_scores[int(node)] = sum(ppr.values()) - ppr.get(node, 0.0)
        return self._normalize_dict(raw_scores, raw_scores.keys())

    def _retain_candidates(self, centrality: Mapping[int, float]) -> Set[int]:
        if self.config.filter_ratio >= 1.0:
            return set(centrality.keys())
        count = max(1, int(round(len(centrality) * self.config.filter_ratio)))
        ranked = sorted(centrality, key=lambda node: centrality[node], reverse=True)
        return set(ranked[:count])

    @staticmethod
    def _normalize_dict(values: Mapping[int, float], nodes: Iterable[int]) -> ScoreMap:
        nodes = list(nodes)
        if not nodes:
            return {}
        selected = {int(node): float(values.get(node, 0.0)) for node in nodes}
        min_value = min(selected.values())
        max_value = max(selected.values())
        if max_value == min_value:
            return {node: 0.0 for node in selected}
        return {node: (value - min_value) / (max_value - min_value) for node, value in selected.items()}
