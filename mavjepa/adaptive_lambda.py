"""Adaptive per-edge lambda controller for MAV-JEPA."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class AdaptiveLambda:
    lambda_base: float = 1.0
    lambda_min: float = 0.05
    lambda_max: float = 4.0
    beta: float = 0.95
    warmup_steps: int = 50
    eps: float = 1e-12
    steps: int = 0
    ema: dict[str, float] = field(default_factory=dict)

    def update(self, edge_name: str, loss_value: float) -> None:
        if not edge_name:
            return
        value = float(loss_value)
        if not math.isfinite(value):
            return
        value = max(value, 0.0)
        current = self.ema.get(edge_name, value)
        self.ema[edge_name] = self.beta * current + (1.0 - self.beta) * value

    def update_many(self, losses: dict[str, float]) -> None:
        self.steps += 1
        for edge_name, loss_value in losses.items():
            self.update(edge_name, loss_value)

    def lambda_for(self, edge_name: str) -> float:
        if self.steps < self.warmup_steps:
            return float(self.lambda_base)
        if edge_name not in self.ema:
            return float(self.lambda_base)
        mean_ema = self.mean_ema()
        if mean_ema <= self.eps:
            return float(self.lambda_base)
        value = self.lambda_base * self.ema[edge_name] / mean_ema
        return float(min(self.lambda_max, max(self.lambda_min, value)))

    def lambda_for_inverse_loss(self, edge_name: str) -> float:
        if self.steps < self.warmup_steps:
            return float(self.lambda_base)
        mean_ema = self.mean_ema()
        if mean_ema <= self.eps:
            return float(self.lambda_base)
        edge_ema = self.ema.get(edge_name, mean_ema)
        value = self.lambda_base * mean_ema / max(edge_ema, self.eps)
        value = min(float(self.lambda_base), value)
        return float(min(self.lambda_max, max(self.lambda_min, value)))

    def lambdas(self, edge_names: list[str] | None = None) -> dict[str, float]:
        names = edge_names if edge_names is not None else sorted(self.ema)
        return {name: self.lambda_for(name) for name in names}

    def mean_ema(self) -> float:
        values = [value for value in self.ema.values() if math.isfinite(value)]
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def state_dict(self) -> dict[str, object]:
        return {
            "lambda_base": self.lambda_base,
            "lambda_min": self.lambda_min,
            "lambda_max": self.lambda_max,
            "beta": self.beta,
            "warmup_steps": self.warmup_steps,
            "steps": self.steps,
            "ema": dict(self.ema),
            "lambda_by_edge": self.lambdas(),
        }
