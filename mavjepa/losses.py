"""Loss utilities for multi-view JEPA."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def last_token_hidden(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.long().sum(dim=1).clamp(min=1) - 1
    batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    return hidden_states[batch_idx, lengths]


def pooled_hidden(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str = "last",
    last_k: int = 64,
) -> torch.Tensor:
    if mode == "last":
        return last_token_hidden(hidden_states, attention_mask)
    if mode == "mean":
        mask = attention_mask.bool()
        masked = hidden_states * mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        return masked.sum(dim=1) / denom
    if mode == "mean_last_k":
        vectors = []
        for idx in range(hidden_states.shape[0]):
            length = int(attention_mask[idx].sum().item())
            if length <= 0:
                vectors.append(hidden_states[idx, 0])
                continue
            start = max(0, length - max(1, int(last_k)))
            vectors.append(hidden_states[idx, start:length].mean(dim=0))
        return torch.stack(vectors)
    raise ValueError(f"Unknown pooling mode: {mode}")


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
    if loss_type == "normalized_mse":
        source_hidden = F.normalize(source_hidden, dim=-1)
        target_hidden = F.normalize(target_hidden, dim=-1)
        return F.mse_loss(source_hidden, target_hidden)
    if loss_type == "l2":
        return torch.linalg.norm(source_hidden - target_hidden, ord=2, dim=-1).mean()
    raise ValueError(f"Unknown MV-JEPA loss type: {loss_type}")
