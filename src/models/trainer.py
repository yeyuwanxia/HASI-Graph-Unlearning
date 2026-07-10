from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F


ExtraLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class TrainingConfig:
    lr: float = 0.01
    weight_decay: float = 5e-4
    epochs: int = 200
    patience: int = 50
    device: Optional[str] = None


@dataclass(frozen=True)
class TrainingResult:
    epochs_ran: int
    train_loss: float
    train_accuracy: float
    val_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "epochs_ran": self.epochs_ran,
            "train_loss": self.train_loss,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "test_accuracy": self.test_accuracy,
        }


class GNNTrainer:
    def __init__(self, model: torch.nn.Module, config: Optional[TrainingConfig] = None, **overrides):
        if config is None:
            config = TrainingConfig(**overrides)
        elif overrides:
            values = config.__dict__ | overrides
            config = TrainingConfig(**values)
        self.model = model
        self.config = config
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

    def train_full_batch(
        self,
        data,
        train_mask=None,
        val_mask=None,
        test_mask=None,
        epochs: Optional[int] = None,
        extra_loss_fn: Optional[ExtraLossFn] = None,
    ) -> TrainingResult:
        return self._fit(data, train_mask, val_mask, test_mask, epochs or self.config.epochs, extra_loss_fn)

    def fine_tune(
        self,
        data,
        train_mask=None,
        val_mask=None,
        test_mask=None,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        extra_loss_fn: Optional[ExtraLossFn] = None,
    ) -> TrainingResult:
        original_lr = self.config.lr
        if lr is not None:
            self.config = TrainingConfig(
                lr=lr,
                weight_decay=self.config.weight_decay,
                epochs=self.config.epochs,
                patience=self.config.patience,
                device=str(self.device),
            )
        try:
            return self._fit(data, train_mask, val_mask, test_mask, epochs or self.config.epochs, extra_loss_fn)
        finally:
            if lr is not None:
                self.config = TrainingConfig(
                    lr=original_lr,
                    weight_decay=self.config.weight_decay,
                    epochs=self.config.epochs,
                    patience=self.config.patience,
                    device=str(self.device),
                )

    @torch.no_grad()
    def predict_logits(self, data) -> torch.Tensor:
        self.model.eval()
        data = self._to_device(data)
        return self.model(data.x, data.edge_index).detach().cpu()

    @torch.no_grad()
    def predict_with_embeddings(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        data = self._to_device(data)
        logits, embeddings = self.model(data.x, data.edge_index, return_embeddings=True)
        return logits.detach().cpu(), embeddings.detach().cpu()

    @torch.no_grad()
    def evaluate_accuracy(self, data, mask=None) -> float:
        self.model.eval()
        data = self._to_device(data)
        mask = self._resolve_mask(data, mask, "test_mask")
        if mask.sum().item() == 0:
            return 0.0
        logits = self.model(data.x, data.edge_index)
        pred = logits.argmax(dim=-1)
        y = _labels(data)
        return float((pred[mask] == y[mask]).float().mean().item())

    def gradient_sensitivity(
        self,
        data,
        mask=None,
        passes: int = 1,
        use_dropout: bool = False,
    ) -> Dict[int, float]:
        """Return ||dL/dh_v||_2 for every node embedding h_v.

        This is the task-aware HubScore term used by HASI. The loss is computed
        on the training mask by default, and gradients are read from the
        penultimate node embeddings exposed by `return_embeddings=True`.
        """

        data = self._to_device(data)
        train_mask = self._resolve_mask(data, mask, "train_mask")
        y = _labels(data)
        num_nodes = int(data.num_nodes)
        scores = torch.zeros(num_nodes, dtype=torch.float32, device=self.device)
        passes = max(1, int(passes))
        was_training = self.model.training

        for _ in range(passes):
            self.model.zero_grad(set_to_none=True)
            if use_dropout:
                self.model.train()
            else:
                self.model.eval()
            logits, embeddings = self.model(data.x, data.edge_index, return_embeddings=True)
            embeddings.retain_grad()
            loss = F.cross_entropy(logits[train_mask], y[train_mask])
            loss.backward()
            if embeddings.grad is not None:
                scores = scores + embeddings.grad.detach().norm(p=2, dim=1)

        self.model.zero_grad(set_to_none=True)
        self.model.train(was_training)
        scores = scores / float(passes)
        return {int(node): float(value) for node, value in enumerate(scores.detach().cpu().tolist())}

    def _fit(
        self,
        data,
        train_mask=None,
        val_mask=None,
        test_mask=None,
        epochs: int = 200,
        extra_loss_fn: Optional[ExtraLossFn] = None,
    ) -> TrainingResult:
        data = self._to_device(data)
        train_mask = self._resolve_mask(data, train_mask, "train_mask")
        val_mask = self._resolve_optional_mask(data, val_mask, "val_mask")
        test_mask = self._resolve_optional_mask(data, test_mask, "test_mask")
        y = _labels(data)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        best_state = copy.deepcopy(self.model.state_dict())
        best_score = float("-inf")
        stale_epochs = 0
        final_loss = 0.0
        epochs_ran = 0

        for epoch in range(int(epochs)):
            self.model.train()
            optimizer.zero_grad()
            logits, embeddings = self.model(data.x, data.edge_index, return_embeddings=True)
            loss = F.cross_entropy(logits[train_mask], y[train_mask])
            if extra_loss_fn is not None:
                loss = loss + extra_loss_fn(logits, embeddings)
            loss.backward()
            optimizer.step()

            final_loss = float(loss.detach().cpu().item())
            epochs_ran = epoch + 1
            current_score = (
                self._accuracy_from_logits(logits.detach(), y, val_mask)
                if val_mask is not None
                else -final_loss
            )
            if current_score > best_score:
                best_score = current_score
                best_state = copy.deepcopy(self.model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if val_mask is not None and stale_epochs >= self.config.patience:
                break

        self.model.load_state_dict(best_state)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(data.x, data.edge_index)
            train_acc = self._accuracy_from_logits(logits, y, train_mask)
            val_acc = self._accuracy_from_logits(logits, y, val_mask) if val_mask is not None else None
            test_acc = self._accuracy_from_logits(logits, y, test_mask) if test_mask is not None else None

        return TrainingResult(
            epochs_ran=epochs_ran,
            train_loss=final_loss,
            train_accuracy=train_acc,
            val_accuracy=val_acc,
            test_accuracy=test_acc,
        )

    def _to_device(self, data):
        if hasattr(data, "to"):
            return data.to(self.device)
        return data

    @staticmethod
    def _resolve_mask(data, mask, attr_name: str) -> torch.Tensor:
        resolved = GNNTrainer._resolve_optional_mask(data, mask, attr_name)
        if resolved is not None and resolved.sum().item() > 0:
            return resolved
        return torch.ones(data.num_nodes, dtype=torch.bool, device=data.x.device)

    @staticmethod
    def _resolve_optional_mask(data, mask, attr_name: str) -> Optional[torch.Tensor]:
        if mask is None and hasattr(data, attr_name):
            mask = getattr(data, attr_name)
        if mask is None:
            return None
        if mask.dim() > 1:
            mask = mask[:, 0]
        return mask.to(device=data.x.device, dtype=torch.bool)

    @staticmethod
    def _accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor]) -> float:
        if mask is None or mask.sum().item() == 0:
            return 0.0
        pred = logits.argmax(dim=-1)
        return float((pred[mask] == y[mask]).float().mean().item())


def _labels(data) -> torch.Tensor:
    y = data.y
    if y.dim() > 1:
        y = y.squeeze(-1)
    return y.to(device=data.x.device, dtype=torch.long)
