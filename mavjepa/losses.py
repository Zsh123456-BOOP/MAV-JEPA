"""Loss utilities for multi-view JEPA."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def last_token_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_idx, lengths]


def jepa_loss(
    source_hidden: torch.Tensor,
    target_hidden: torch.Tensor,
    loss_type: str = "cosine",
    detach_target: bool = False,
) -> torch.Tensor:
    if detach_target:
        target_hidden = target_hidden.detach()
    if loss_type == "cosine":
        return 1.0 - F.cosine_similarity(source_hidden, target_hidden, dim=-1).mean()
    if loss_type == "mse":
        return F.mse_loss(source_hidden, target_hidden)
    if loss_type == "l2":
        return torch.linalg.norm(source_hidden - target_hidden, ord=2, dim=-1).mean()
    raise ValueError(f"Unknown MV-JEPA loss type: {loss_type}")
