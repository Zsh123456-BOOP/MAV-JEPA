"""Edge sampling for MAV-JEPA."""

from __future__ import annotations

import math
import random
from typing import Any


class EdgeSampler:
    def __init__(
        self,
        mode: str = "none",
        edge_budget: int = 1,
        p_min: float = 0.05,
        seed: int = 0,
        eps: float = 1e-8,
        nan_blacklist_threshold: int = 3,
        blacklist_steps: int = 50,
    ):
        if mode not in {"none", "random", "adaptive"}:
            raise ValueError(f"Unknown edge dropout mode: {mode}")
        self.mode = mode
        self.edge_budget = max(0, int(edge_budget))
        self.p_min = max(0.0, float(p_min))
        self.eps = float(eps)
        self.rng = random.Random(seed)
        self.ema_loss: dict[str, float] = {}
        self.nan_counts: dict[str, int] = {}
        self.blacklist_until: dict[str, int] = {}
        self.nan_blacklist_threshold = max(1, int(nan_blacklist_threshold))
        self.blacklist_steps = max(1, int(blacklist_steps))
        self.updates = 0

    def probabilities(self, edges: list[dict[str, Any]]) -> dict[str, float]:
        edges = self.active_edges(edges)
        if not edges:
            return {}
        if self.mode != "adaptive":
            p = 1.0 / len(edges)
            return {edge["name"]: p for edge in edges}
        scores = []
        for edge in edges:
            loss = self.ema_loss.get(edge["name"], 1.0)
            quality = safe_quality(edge.get("quality", 1.0))
            score = math.sqrt(max(loss, 0.0) + self.eps) * quality
            if not math.isfinite(score):
                score = 0.0
            scores.append(score)
        total = sum(scores)
        if total <= 0:
            p = 1.0 / len(edges)
            return {edge["name"]: p for edge in edges}
        probs = [max(score / total, self.p_min) for score in scores]
        norm = sum(probs)
        return {edge["name"]: prob / norm for edge, prob in zip(edges, probs)}

    def sample(self, edges: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
        edges = self.active_edges(edges)
        if not edges or self.edge_budget == 0:
            return [], {}
        if self.mode == "none" or self.edge_budget >= len(edges):
            probs = self.probabilities(edges)
            return list(edges), probs
        probs = self.probabilities(edges)
        if self.mode == "random":
            return self.rng.sample(edges, k=min(self.edge_budget, len(edges))), probs
        names = [edge["name"] for edge in edges]
        weights = [probs[name] for name in names]
        selected: list[dict[str, Any]] = []
        pool = list(zip(edges, weights))
        for _ in range(min(self.edge_budget, len(edges))):
            total = sum(weight for _, weight in pool)
            pick = self.rng.random() * total
            acc = 0.0
            for idx, (edge, weight) in enumerate(pool):
                acc += weight
                if acc >= pick:
                    selected.append(edge)
                    pool.pop(idx)
                    break
        return selected, probs

    def update_loss(self, edge_name: str, loss_value: float, beta: float = 0.95) -> None:
        self.updates += 1
        value = float(loss_value)
        if not math.isfinite(value):
            self.nan_counts[edge_name] = self.nan_counts.get(edge_name, 0) + 1
            if self.nan_counts[edge_name] >= self.nan_blacklist_threshold:
                self.blacklist_until[edge_name] = self.updates + self.blacklist_steps
            return
        self.nan_counts[edge_name] = 0
        current = self.ema_loss.get(edge_name, value)
        self.ema_loss[edge_name] = beta * current + (1.0 - beta) * value

    def active_edges(self, edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        active = []
        for edge in edges:
            name = edge.get("name")
            if not name:
                continue
            if self.blacklist_until.get(name, -1) > self.updates:
                continue
            active.append(edge)
        return active


def safe_quality(value: Any) -> float:
    try:
        quality = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(quality):
        return 1.0
    return max(0.0, quality)
