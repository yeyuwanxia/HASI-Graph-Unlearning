from __future__ import annotations

from typing import Dict, Iterable, Set

import networkx as nx


class PPRComputer:
    """Personalized PageRank based ERF approximation."""

    def __init__(self, alpha: float = 0.15, k_steps: int = 3, threshold: float = 0.01):
        self.alpha = alpha
        self.k_steps = k_steps
        self.threshold = threshold

    def compute_seed_ppr(self, graph: nx.Graph, seeds: Iterable[int]) -> Dict[int, float]:
        seeds = [seed for seed in seeds if seed in graph]
        if not seeds:
            return {}

        rank = {node: 0.0 for node in graph.nodes}
        seed_mass = 1.0 / len(seeds)
        for seed in seeds:
            rank[seed] = seed_mass

        for _ in range(self.k_steps):
            next_rank = {node: 0.0 for node in graph.nodes}
            for node, value in rank.items():
                neighbors = list(graph.neighbors(node))
                if not neighbors:
                    next_rank[node] += (1.0 - self.alpha) * value
                    continue
                share = (1.0 - self.alpha) * value / len(neighbors)
                for neighbor in neighbors:
                    next_rank[neighbor] += share
            for seed in seeds:
                next_rank[seed] += self.alpha * seed_mass
            rank = next_rank

        return rank

    def affected_region(self, graph: nx.Graph, seeds: Iterable[int]) -> Set[int]:
        ppr = self.compute_seed_ppr(graph, seeds)
        return {node for node, value in ppr.items() if value > self.threshold}
