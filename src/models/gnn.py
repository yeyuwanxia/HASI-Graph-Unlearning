from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


ModelType = Literal["GCN", "GAT", "GraphSAGE", "SAGE"]


class UnlearnableGNN(nn.Module):
    """Small full-batch GNN used by HASI experiments.

    The module exposes the penultimate node embeddings because HASI anchor
    losses and MIA diagnostics both need a stable representation hook.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        model_type: ModelType = "GCN",
        num_layers: int = 2,
        dropout: float = 0.5,
        heads: int = 4,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        self.model_type = _normalize_model_type(model_type)
        self.dropout = float(dropout)
        self.num_layers = int(num_layers)
        self.hidden_channels = int(hidden_channels)
        self.out_channels = int(out_channels)

        self.convs = nn.ModuleList()
        if self.model_type == "GCN":
            conv_cls = _pyg_conv("GCNConv")
            dims = [in_channels] + [hidden_channels] * max(0, num_layers - 1) + [out_channels]
            for idx in range(num_layers):
                self.convs.append(conv_cls(dims[idx], dims[idx + 1]))
        elif self.model_type == "SAGE":
            conv_cls = _pyg_conv("SAGEConv")
            dims = [in_channels] + [hidden_channels] * max(0, num_layers - 1) + [out_channels]
            for idx in range(num_layers):
                self.convs.append(conv_cls(dims[idx], dims[idx + 1]))
        elif self.model_type == "GAT":
            conv_cls = _pyg_conv("GATConv")
            if num_layers == 1:
                self.convs.append(conv_cls(in_channels, out_channels, heads=1, concat=False, dropout=dropout))
            else:
                self.convs.append(conv_cls(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout))
                hidden_in = hidden_channels * heads
                for _ in range(num_layers - 2):
                    self.convs.append(conv_cls(hidden_in, hidden_channels, heads=heads, concat=True, dropout=dropout))
                    hidden_in = hidden_channels * heads
                self.convs.append(conv_cls(hidden_in, out_channels, heads=1, concat=False, dropout=dropout))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, return_embeddings: bool = False):
        embedding = x
        for idx, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            is_last = idx == len(self.convs) - 1
            if not is_last:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                embedding = x
            elif len(self.convs) == 1:
                embedding = x
        if return_embeddings:
            return x, embedding
        return x


class GCN(UnlearnableGNN):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__(in_channels, hidden_channels, out_channels, "GCN", num_layers, dropout)


class GAT(UnlearnableGNN):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        heads: int = 4,
    ):
        super().__init__(in_channels, hidden_channels, out_channels, "GAT", num_layers, dropout, heads=heads)


class GraphSAGE(UnlearnableGNN):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__(in_channels, hidden_channels, out_channels, "SAGE", num_layers, dropout)


def build_gnn_model(
    model_type: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    num_layers: int = 2,
    dropout: float = 0.5,
    heads: int = 4,
) -> UnlearnableGNN:
    model_key = _normalize_model_type(model_type)
    if model_key == "GCN":
        return GCN(in_channels, hidden_channels, out_channels, num_layers, dropout)
    if model_key == "GAT":
        return GAT(in_channels, hidden_channels, out_channels, num_layers, dropout, heads=heads)
    if model_key == "SAGE":
        return GraphSAGE(in_channels, hidden_channels, out_channels, num_layers, dropout)
    raise ValueError(f"Unsupported model_type {model_type!r}")


def _normalize_model_type(model_type: str) -> str:
    value = model_type.upper()
    if value in {"GRAPHSAGE", "SAGE"}:
        return "SAGE"
    if value in {"GCN", "GAT"}:
        return value
    raise ValueError("model_type must be one of GCN, GAT, GraphSAGE")


def _pyg_conv(name: str):
    try:
        import torch_geometric.nn as pyg_nn
    except ImportError as exc:
        if name == "GCNConv":
            return SimpleGCNConv
        if name == "SAGEConv":
            return SimpleSAGEConv
        raise SystemExit(
            "torch-geometric is required for GAT models. "
            "Install torch-geometric or choose GCN/GraphSAGE for the built-in fallback."
        ) from exc
    return getattr(pyg_nn, name)


class SimpleGCNConv(nn.Module):
    """Dependency-light GCN layer used when torch-geometric is unavailable."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        source, target = _with_self_loops(edge_index, num_nodes, x.device)
        support = self.lin(x)
        degree = torch.bincount(target, minlength=num_nodes).to(x.dtype).clamp(min=1)
        norm = degree[source].pow(-0.5) * degree[target].pow(-0.5)
        out = support.new_zeros((num_nodes, support.shape[1]))
        out.index_add_(0, target, support[source] * norm.unsqueeze(-1))
        return out + self.bias


class SimpleSAGEConv(nn.Module):
    """Mean GraphSAGE layer used when torch-geometric is unavailable."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.root = nn.Linear(in_channels, out_channels)
        self.neighbor = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        num_nodes = x.shape[0]
        source, target = edge_index
        agg = x.new_zeros(x.shape)
        degree = x.new_zeros(num_nodes)
        agg.index_add_(0, target, x[source])
        degree.index_add_(0, target, torch.ones_like(target, dtype=x.dtype))
        agg = agg / degree.clamp(min=1).unsqueeze(-1)
        return self.root(x) + self.neighbor(agg)


def _with_self_loops(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    loops = torch.arange(num_nodes, device=device)
    source = torch.cat([edge_index[0], loops])
    target = torch.cat([edge_index[1], loops])
    return source, target
