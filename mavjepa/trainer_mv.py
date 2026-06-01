"""Minimal multi-view JEPA trainer used by finetune_mv.py."""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .adaptive_lambda import AdaptiveLambda
from .edge_sampler import EdgeSampler
from .logging_utils import append_jsonl
from .losses import jepa_loss, last_token_hidden


class MVJEPADataset(Dataset):
    def __init__(self, jsonl_path: str | Path, tokenizer: Any, max_length: int, view_max_length: int, model_name: str):
        self.records = []
        with Path(jsonl_path).open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    self.records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise json.JSONDecodeError(
                        f"{exc.msg} in {jsonl_path} at physical line {line_no}",
                        exc.doc,
                        exc.pos,
                    ) from exc
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.view_max_length = view_max_length
        self.model_name = model_name

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        input_ids, attention_mask, labels = tokenize_messages(
            self.tokenizer, record["messages"], self.max_length
        )
        return {
            "id": record["id"],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "views": record.get("views", {}),
            "edges": record.get("edges", []),
        }


def tokenize_messages(tokenizer: Any, messages: list[dict[str, str]], max_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if getattr(tokenizer, "chat_template", None):
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
    else:
        full_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        prompt_text = "\n".join(f"{m['role']}: {m['content']}" for m in messages[:2]) + "\nassistant:"
    full = tokenizer(full_text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    prompt = tokenizer(prompt_text, truncation=True, max_length=max_length, padding=False, return_tensors="pt")
    input_ids = full["input_ids"].squeeze(0)
    attention_mask = full["attention_mask"].squeeze(0)
    labels = input_ids.clone()
    prompt_len = min(prompt["input_ids"].shape[1], labels.shape[0])
    labels[:prompt_len] = -100
    labels[attention_mask == 0] = -100
    if torch.all(labels == -100):
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
    return input_ids, attention_mask, labels


def collate_mv(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ids": [item["id"] for item in batch],
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "views": [item["views"] for item in batch],
        "edges": [item["edges"] for item in batch],
    }


class MultiViewTrainer:
    def __init__(self, model: Any, tokenizer: Any, args: Any, logger: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampler = EdgeSampler(args.edge_dropout, args.edge_budget, args.edge_p_min, seed=args.finetune_seed)
        self.lambda_controller = AdaptiveLambda(
            lambda_base=args.lambda_base,
            lambda_min=args.lambda_min,
            lambda_max=args.lambda_max,
            beta=args.lambda_ema_beta,
            warmup_steps=args.lambda_warmup_steps,
        )
        self.edge_counts: Counter[str] = Counter()
        self.view_config = load_view_config(args.view_config)
        self.config_edges = {edge["name"]: edge for edge in self.view_config.get("edges", [])}
        self.view_max_lengths = {
            name: int(cfg.get("max_length", args.view_max_length))
            for name, cfg in self.view_config.get("views", {}).items()
            if isinstance(cfg, dict)
        }
        self.total_params = sum(param.numel() for param in self.model.parameters())
        self.trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)

    def train(self, train_file: str, eval_file: str | None = None) -> dict[str, Any]:
        dataset = MVJEPADataset(
            train_file,
            tokenizer=self.tokenizer,
            max_length=self.args.max_length,
            view_max_length=self.args.view_max_length,
            model_name=self.args.model_name,
        )
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=collate_mv)
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.args.learning_rate)
        self.model.train()
        metrics_path = Path(self.args.output_dir) / "metrics.jsonl"
        start = time.time()
        total_loss_sum = 0.0
        ce_loss_sum = 0.0
        jepa_loss_sum = 0.0
        steps = 0
        tokens = 0
        view_tokens = 0
        estimated_total_flops = 0
        selected_edges_total = 0
        used_edges_total = 0
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.args.num_epochs):
            for batch_idx, batch in enumerate(loader):
                step_start = time.time()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
                ce_loss = outputs.loss
                mv_loss, edge_log, selected_count, used_count, batch_view_tokens = self.compute_mv_loss(batch)
                if not torch.isfinite(mv_loss):
                    self.logger.warning("Non-finite MV loss at step %s; using CE only", steps)
                    mv_loss = ce_loss.new_tensor(0.0)
                total_loss = self.args.gamma * ce_loss + mv_loss
                (total_loss / self.args.grad_accum).backward()
                if (batch_idx + 1) % self.args.grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                steps += 1
                batch_tokens = int(attention_mask.sum().item())
                tokens += batch_tokens
                view_tokens += batch_view_tokens
                selected_edges_total += selected_count
                used_edges_total += used_count
                step_flops = estimate_step_flops(self.total_params, batch_tokens + batch_view_tokens, self.args.track_flop)
                if step_flops is not None:
                    estimated_total_flops += step_flops
                total_loss_sum += float(total_loss.detach().cpu())
                ce_loss_sum += float(ce_loss.detach().cpu())
                jepa_loss_sum += float(mv_loss.detach().cpu())
                elapsed_step = time.time() - step_start
                record = {
                    "step": steps,
                    "epoch": epoch + (batch_idx + 1) / max(1, len(loader)),
                    "ce_loss": float(ce_loss.detach().cpu()),
                    "jepa_loss": float(mv_loss.detach().cpu()),
                    "total_loss": float(total_loss.detach().cpu()),
                    "learning_rate": self.args.learning_rate,
                    "active_edges": [item["name"] for item in edge_log],
                    "loss_by_edge": {item["name"]: item["raw_loss"] for item in edge_log},
                    "lambda_by_edge": {item["name"]: item["lambda"] for item in edge_log},
                    "weighted_loss_by_edge": {item["name"]: item["weighted_loss"] for item in edge_log},
                    "edge_sampling_prob": {item["name"]: item.get("prob") for item in edge_log},
                    "step_time_sec": elapsed_step,
                    "gpu_memory_gb": current_gpu_memory_gb(),
                    "mv_edges_selected": selected_count,
                    "mv_edges_used": used_count,
                    "tokens": batch_tokens,
                    "view_tokens": batch_view_tokens,
                    "estimated_step_flops": step_flops,
                }
                append_jsonl(metrics_path, record)
                if steps % max(1, self.args.log_steps) == 0:
                    self.logger.info(
                        "step=%s ce_loss=%.4f jepa_loss=%.4f total_loss=%.4f edges=%s",
                        steps,
                        record["ce_loss"],
                        record["jepa_loss"],
                        record["total_loss"],
                        record["active_edges"],
                    )

        if steps % self.args.grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        wall = time.time() - start
        save_dir = Path(self.args.output_dir) / "checkpoint-final"
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        return {
            "status": "success",
            "train_steps": steps,
            "train_loss": total_loss_sum / max(1, steps),
            "ce_loss": ce_loss_sum / max(1, steps),
            "jepa_loss": jepa_loss_sum / max(1, steps),
            "wall_clock_sec": wall,
            "gpu_hours": wall * max(1, torch.cuda.device_count()) / 3600,
            "peak_vram_gb": peak_gpu_memory_gb(),
            "avg_steps_per_sec": steps / max(wall, 1e-6),
            "avg_tokens_per_sec": (tokens + view_tokens) / max(wall, 1e-6),
            "train_tokens": tokens,
            "view_tokens": view_tokens,
            "estimated_total_flops": estimated_total_flops if self.args.track_flop else None,
            "jepa_edges_per_step": selected_edges_total / max(1, steps),
            "jepa_edges_used_per_step": used_edges_total / max(1, steps),
            "edge_sampling_frequency": dict(self.edge_counts),
            "lambda_history": self.lambda_controller.state_dict() if self.args.adaptive_lambda else None,
            "same_flop_accuracy": None,
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "eval_file": eval_file,
        }

    def compute_mv_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, list[dict[str, Any]], int, int, int]:
        if not self.args.mv_jepa:
            return next(self.model.parameters()).new_tensor(0.0), [], 0, 0, 0
        losses = []
        edge_log = []
        selected_count = 0
        view_tokens = 0
        raw_losses_by_edge: dict[str, list[float]] = defaultdict(list)
        for views, edges in zip(batch["views"], batch["edges"]):
            candidate_edges = self.candidate_edges(views, edges)
            selected, probs = self.sampler.sample(candidate_edges)
            selected_count += len(selected)
            for edge in selected:
                src_text = views.get(edge["src"])
                tgt_text = views.get(edge["tgt"])
                if not src_text or not tgt_text:
                    continue
                src_hidden, src_tokens = self.encode_view(src_text, view_name=edge["src"], predictors=self.args.predictors)
                tgt_hidden, tgt_tokens = self.encode_view(tgt_text, view_name=edge["tgt"], predictors=0)
                view_tokens += src_tokens + tgt_tokens
                raw_loss = jepa_loss(src_hidden, tgt_hidden, self.args.mv_loss_type, self.args.detach_target)
                edge_name = edge["name"]
                lam = (
                    self.lambda_controller.lambda_for(edge_name)
                    if self.args.adaptive_lambda
                    else float(self.args.lambda_base)
                )
                weighted = raw_loss * lam
                losses.append(weighted)
                self.edge_counts[edge_name] += 1
                self.sampler.update_loss(edge_name, float(raw_loss.detach().cpu()))
                raw_losses_by_edge[edge_name].append(float(raw_loss.detach().cpu()))
                edge_log.append(
                    {
                        "name": edge_name,
                        "raw_loss": float(raw_loss.detach().cpu()),
                        "lambda": lam,
                        "weighted_loss": float(weighted.detach().cpu()),
                        "prob": probs.get(edge_name),
                    }
                )
        if raw_losses_by_edge and self.args.adaptive_lambda:
            self.lambda_controller.update_many(
                {name: sum(values) / len(values) for name, values in raw_losses_by_edge.items()}
            )
        if not losses:
            return next(self.model.parameters()).new_tensor(0.0), edge_log, selected_count, 0, view_tokens
        return torch.stack(losses).sum(), edge_log, selected_count, len(losses), view_tokens

    def candidate_edges(self, views: dict[str, str], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.config_edges:
            return edges
        sample_by_name = {edge.get("name"): edge for edge in edges if edge.get("name")}
        candidates = []
        for name, config_edge in self.config_edges.items():
            if sample_by_name and name not in sample_by_name:
                continue
            edge = {**config_edge, **sample_by_name.get(name, {})}
            if edge.get("src") in views and edge.get("tgt") in views:
                candidates.append(edge)
        return candidates

    def encode_view(self, text: str, view_name: str, predictors: int = 0) -> tuple[torch.Tensor, int]:
        if predictors > 0:
            text = text + "".join(f"<|predictor_{idx}|>" for idx in range(1, predictors + 1))
        max_length = min(self.args.view_max_length, self.view_max_lengths.get(view_name, self.args.view_max_length))
        tokenized = tokenize_view_left_truncate(self.tokenizer, text, max_length)
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
        return last_token_hidden(outputs.hidden_states[-1], attention_mask), int(attention_mask.sum().item())


def tokenize_view_left_truncate(tokenizer: Any, text: str, max_length: int) -> dict[str, torch.Tensor]:
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    input_ids = list(encoded["input_ids"])
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
    attention = [1] * len(input_ids)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if len(input_ids) < max_length:
        pad_len = max_length - len(input_ids)
        input_ids.extend([pad_id] * pad_len)
        attention.extend([0] * pad_len)
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention], dtype=torch.long),
    }


def load_view_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        if path.endswith(".json"):
            return json.load(handle)
        import yaml

        return yaml.safe_load(handle) or {}


def estimate_step_flops(total_params: int, token_count: int, enabled: bool) -> int | None:
    if not enabled:
        return None
    return int(6 * total_params * token_count)


def current_gpu_memory_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.memory_allocated() / (1024**3), 4)


def peak_gpu_memory_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / (1024**3), 4)
