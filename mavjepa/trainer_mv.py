"""Minimal multi-view JEPA trainer used by finetune_mv.py."""

from __future__ import annotations

import json
import random
import re
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .adaptive_lambda import AdaptiveLambda
from .edge_sampler import EdgeSampler
from .logging_utils import append_jsonl
from .losses import jepa_loss, pooled_hidden
from .view_builders import strip_final_answer


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
            "meta": record.get("meta", {}),
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
        "meta": [item["meta"] for item in batch],
    }


class MultiViewTrainer:
    def __init__(self, model: Any, tokenizer: Any, args: Any, logger: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampler = EdgeSampler(args.edge_dropout, args.edge_budget, args.edge_p_min, seed=args.finetune_seed)
        self.rng = random.Random(args.finetune_seed)
        self.lambda_controller = AdaptiveLambda(
            lambda_base=args.lambda_base,
            lambda_min=args.lambda_min,
            lambda_max=args.lambda_max,
            beta=args.lambda_ema_beta,
            warmup_steps=args.lambda_warmup_steps,
        )
        self.edge_counts: Counter[str] = Counter()
        self.filtered_edge_counts: Counter[str] = Counter()
        self.sampled_edge_counts: Counter[str] = Counter()
        self.view_config = load_view_config(args.view_config)
        self.config_edges = {edge["name"]: edge for edge in self.view_config.get("edges", [])}
        self.view_max_lengths = {
            name: int(cfg.get("max_length", args.view_max_length))
            for name, cfg in self.view_config.get("views", {}).items()
            if isinstance(cfg, dict)
        }
        self.view_truncation_sides = {
            name: str(cfg.get("truncation_side", "left"))
            for name, cfg in self.view_config.get("views", {}).items()
            if isinstance(cfg, dict)
        }
        self.allowed_edges = parse_allowed_edges(getattr(args, "allowed_edges", None))
        self.lambda_mode = getattr(args, "lambda_mode", "fixed")
        self.last_mv_loss_info: dict[str, Any] = {}
        self.view_stats_accumulator: dict[str, list[float]] = defaultdict(list)
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
        filtered_edges_total = 0
        candidates_before_total = 0
        candidates_after_total = 0
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.args.num_epochs):
            for batch_idx, batch in enumerate(loader):
                step_start = time.time()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
                ce_loss = outputs.loss
                mv_loss, edge_log, selected_count, used_count, batch_view_tokens = self.compute_mv_loss(
                    batch=batch,
                    ce_loss=ce_loss,
                    step=steps,
                )
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
                filtered_edges_total += int(self.last_mv_loss_info.get("filtered_edges_step", 0) or 0)
                candidates_before_total += int(self.last_mv_loss_info.get("candidate_edges_before_filter", 0) or 0)
                candidates_after_total += int(self.last_mv_loss_info.get("candidate_edges_after_filter", 0) or 0)
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
                    "jepa_loss_raw": self.last_mv_loss_info.get("raw_loss"),
                    "jepa_loss_after_warmup": self.last_mv_loss_info.get("loss_after_warmup"),
                    "jepa_warmup": self.last_mv_loss_info.get("warmup"),
                    "jepa_step_prob_effective": self.last_mv_loss_info.get("step_prob_effective"),
                    "jepa_skipped": self.last_mv_loss_info.get("skipped"),
                    "jepa_skip_reason": self.last_mv_loss_info.get("skip_reason"),
                    "jepa_ce_ratio_cap_value": self.last_mv_loss_info.get("ce_ratio_cap_value"),
                    "jepa_reduce": getattr(self.args, "jepa_reduce", "sum"),
                    "total_loss": float(total_loss.detach().cpu()),
                    "learning_rate": self.args.learning_rate,
                    "active_edges": [item["name"] for item in edge_log],
                    "selected_main_edges": self.last_mv_loss_info.get("selected_main_edges", []),
                    "selected_weak_edges": self.last_mv_loss_info.get("selected_weak_edges", []),
                    "candidate_edges_before_filter": self.last_mv_loss_info.get("candidate_edges_before_filter"),
                    "candidate_edges_after_filter": self.last_mv_loss_info.get("candidate_edges_after_filter"),
                    "filtered_edge_counts": dict(self.filtered_edge_counts),
                    "filtered_edges_step": self.last_mv_loss_info.get("filtered_edges_step"),
                    "loss_by_edge": {item["name"]: item["raw_loss"] for item in edge_log},
                    "lambda_by_edge": {item["name"]: item["lambda"] for item in edge_log},
                    "weighted_loss_by_edge": {item["name"]: item["weighted_loss"] for item in edge_log},
                    "edge_sampling_prob": {item["name"]: item.get("prob") for item in edge_log},
                    "step_time_sec": elapsed_step,
                    "gpu_memory_gb": current_gpu_memory_gb(),
                    "mv_edges_selected": selected_count,
                    "mv_edges_used": used_count,
                    "main_edges_used": self.last_mv_loss_info.get("main_edges_used", 0),
                    "weak_edges_used": self.last_mv_loss_info.get("weak_edges_used", 0),
                    "mv_edges_filtered": self.last_mv_loss_info.get("filtered_edges_step"),
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
            "jepa_edges_sampled_per_step": selected_edges_total / max(1, steps),
            "jepa_edges_used_per_step": used_edges_total / max(1, steps),
            "jepa_edges_filtered_per_step": filtered_edges_total / max(1, steps),
            "jepa_candidate_edges_before_filter_per_step": candidates_before_total / max(1, steps),
            "jepa_candidate_edges_after_filter_per_step": candidates_after_total / max(1, steps),
            "edge_sampling_frequency": dict(self.edge_counts),
            "edge_sampled_frequency": dict(self.sampled_edge_counts),
            "filtered_edge_counts": dict(self.filtered_edge_counts),
            "lambda_history": self.lambda_controller.state_dict()
            if self.lambda_mode in {"current_adaptive", "inverse_loss"}
            else None,
            "same_flop_accuracy": None,
            "trainable_params": self.trainable_params,
            "total_params": self.total_params,
            "view_stats": summarize_numeric_lists(self.view_stats_accumulator),
            "eval_file": eval_file,
        }

    def compute_mv_loss(
        self,
        batch: dict[str, Any],
        ce_loss: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, list[dict[str, Any]], int, int, int]:
        if not self.args.mv_jepa:
            return self.zero_mv_result(skip_reason="disabled", ce_loss=ce_loss)
        if step < getattr(self.args, "jepa_start_step", 0):
            return self.zero_mv_result(skip_reason="before_start_step", ce_loss=ce_loss, step=step)

        warmup = self.jepa_warmup(step)
        step_prob = self.current_jepa_step_prob(warmup)
        if self.rng.random() > step_prob:
            return self.zero_mv_result(
                skip_reason="step_dropout",
                ce_loss=ce_loss,
                step=step,
                warmup=warmup,
                step_prob=step_prob,
            )

        losses = []
        edge_log = []
        selected_count = 0
        candidate_before_count = 0
        candidate_after_count = 0
        filtered_count_start = sum(self.filtered_edge_counts.values())
        view_tokens = 0
        raw_losses_by_edge: dict[str, list[float]] = defaultdict(list)
        selected_main_names: list[str] = []
        selected_weak_names: list[str] = []
        main_edges_used = 0
        weak_edges_used = 0
        for views, edges, meta in zip(batch["views"], batch["edges"], batch.get("meta", [{}] * len(batch["views"]))):
            self.collect_view_stats(meta)
            candidate_edges = self.candidate_edges(views, edges, step=step)
            candidate_before_count += self.last_mv_loss_info.get("candidate_edges_before_filter", 0) or 0
            candidate_after_count += len(candidate_edges)
            main_candidates, weak_candidates = split_main_weak_edges(candidate_edges)
            selected, probs = self.sampler.sample(main_candidates)
            weak_selected, weak_probs = self.sample_weak_edges(weak_candidates, step=step)
            selected_main_names.extend(edge["name"] for edge in selected)
            selected_weak_names.extend(edge["name"] for edge in weak_selected)
            selected = selected + weak_selected
            probs = {**probs, **weak_probs}
            selected_count += len(selected)
            for edge in selected:
                self.sampled_edge_counts[edge["name"]] += 1
                src_text = self.prepare_view_text(edge["src"], views.get(edge["src"]))
                tgt_text = self.prepare_view_text(edge["tgt"], views.get(edge["tgt"]))
                if not src_text or not tgt_text:
                    continue
                src_hidden, src_tokens = self.encode_view(
                    src_text,
                    view_name=edge["src"],
                    predictors=self.args.predictors,
                    grad=True,
                    pooling=edge.get("source_pooling", edge.get("pooling", getattr(self.args, "pooling", "last"))),
                )
                tgt_hidden, tgt_tokens = self.encode_view(
                    tgt_text,
                    view_name=edge["tgt"],
                    predictors=0,
                    grad=not getattr(self.args, "target_no_grad", False),
                    pooling=edge.get("target_pooling", getattr(self.args, "target_pooling", "last")),
                )
                view_tokens += src_tokens + tgt_tokens
                raw_loss = jepa_loss(src_hidden, tgt_hidden, self.args.mv_loss_type, self.args.detach_target)
                edge_name = edge["name"]
                lam = self.lambda_for_edge(edge)
                weighted = raw_loss * lam
                ratio_cap = edge_ratio_cap(edge, getattr(self.args, "jepa_ce_ratio_cap", 0.0))
                if ratio_cap > 0.0:
                    weighted = torch.minimum(weighted, ratio_cap * ce_loss.detach())
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
                        "ce_ratio_cap": ratio_cap,
                        "prob": probs.get(edge_name),
                        "edge_pool": "weak" if edge.get("weak_only") else "main",
                    }
                )
                if edge.get("weak_only"):
                    weak_edges_used += 1
                else:
                    main_edges_used += 1
        filtered_edges_step = sum(self.filtered_edge_counts.values()) - filtered_count_start
        if raw_losses_by_edge and self.lambda_mode in {"current_adaptive", "inverse_loss"}:
            self.lambda_controller.update_many(
                {name: sum(values) / len(values) for name, values in raw_losses_by_edge.items()}
            )
        if not losses:
            zero, _, _, _, _ = self.zero_mv_result(
                skip_reason="no_usable_edges",
                ce_loss=ce_loss,
                step=step,
                warmup=warmup,
                step_prob=step_prob,
            )
            self.last_mv_loss_info.update(
                {
                    "candidate_edges_before_filter": candidate_before_count,
                    "candidate_edges_after_filter": candidate_after_count,
                    "filtered_edges_step": filtered_edges_step,
                    "selected_main_edges": selected_main_names,
                    "selected_weak_edges": selected_weak_names,
                    "main_edges_used": main_edges_used,
                    "weak_edges_used": weak_edges_used,
                }
            )
            return zero, edge_log, selected_count, 0, view_tokens

        loss_stack = torch.stack(losses)
        raw_mv_loss = loss_stack.mean() if getattr(self.args, "jepa_reduce", "sum") == "mean" else loss_stack.sum()
        loss_after_warmup = raw_mv_loss * warmup
        mv_loss = loss_after_warmup
        cap_value = None
        ratio_cap = float(getattr(self.args, "jepa_ce_ratio_cap", 0.0) or 0.0)
        if ratio_cap > 0.0:
            cap = ratio_cap * ce_loss.detach()
            cap_value = float(cap.detach().cpu())
            mv_loss = torch.minimum(mv_loss, cap)
        self.last_mv_loss_info = {
            "raw_loss": float(raw_mv_loss.detach().cpu()),
            "loss_after_warmup": float(loss_after_warmup.detach().cpu()),
            "warmup": warmup,
            "step_prob_effective": step_prob,
            "skipped": False,
            "skip_reason": None,
            "ce_ratio_cap_value": cap_value,
            "candidate_edges_before_filter": candidate_before_count,
            "candidate_edges_after_filter": candidate_after_count,
            "filtered_edges_step": filtered_edges_step,
            "selected_main_edges": selected_main_names,
            "selected_weak_edges": selected_weak_names,
            "main_edges_used": main_edges_used,
            "weak_edges_used": weak_edges_used,
        }
        return mv_loss, edge_log, selected_count, len(losses), view_tokens

    def sample_weak_edges(self, weak_candidates: list[dict[str, Any]], step: int) -> tuple[list[dict[str, Any]], dict[str, float]]:
        if not weak_candidates:
            return [], {}
        weak_start = int(getattr(self.args, "weak_edge_start_step", 0) or 0)
        if step < weak_start:
            return [], {}
        weak_prob = min(1.0, max(0.0, float(getattr(self.args, "weak_edge_step_prob", 0.0) or 0.0)))
        if weak_prob <= 0.0 or self.rng.random() > weak_prob:
            return [], {}
        weights = [max(0.0, float(edge.get("prior", edge.get("quality", 1.0)) or 0.0)) for edge in weak_candidates]
        if not any(weights):
            weights = [1.0 for _ in weak_candidates]
        selected = self.rng.choices(weak_candidates, weights=weights, k=1)
        total = sum(weights)
        probs = {edge["name"]: weight / total for edge, weight in zip(weak_candidates, weights)}
        return selected, probs

    def collect_view_stats(self, meta: dict[str, Any]) -> None:
        stats = meta.get("view_stats") if isinstance(meta, dict) else None
        if not isinstance(stats, dict):
            return
        for key, value in stats.items():
            try:
                self.view_stats_accumulator[key].append(float(value))
            except (TypeError, ValueError):
                continue

    def candidate_edges(self, views: dict[str, str], edges: list[dict[str, Any]], step: int | None = None) -> list[dict[str, Any]]:
        if not self.config_edges:
            candidates = list(edges)
        else:
            sample_by_name = {edge.get("name"): edge for edge in edges if edge.get("name")}
            candidates = []
            for name, config_edge in self.config_edges.items():
                if sample_by_name and name not in sample_by_name:
                    continue
                edge = {**config_edge, **sample_by_name.get(name, {})}
                if edge.get("src") in views and edge.get("tgt") in views:
                    candidates.append(edge)
        if self.allowed_edges is not None:
            candidates = [edge for edge in candidates if edge.get("name") in self.allowed_edges]
        before_filter = len(candidates)
        filtered = self.apply_edge_filters_before_sampling(candidates, views, step=step)
        self.last_mv_loss_info = {
            **self.last_mv_loss_info,
            "candidate_edges_before_filter": before_filter,
            "candidate_edges_after_filter": len(filtered),
        }
        return filtered

    def apply_edge_filters_before_sampling(
        self,
        candidates: list[dict[str, Any]],
        views: dict[str, str],
        step: int | None = None,
    ) -> list[dict[str, Any]]:
        filtered = []
        for edge in candidates:
            name = edge.get("name", "unknown")
            src_name = edge.get("src")
            tgt_name = edge.get("tgt")
            if getattr(self.args, "disable_answer_target_edges", False) and tgt_name in {"A", "A_STMT"}:
                self.filtered_edge_counts[f"{name}:answer_target_disabled"] += 1
                continue
            edge_start_step = int(edge.get("start_step", getattr(self.args, "jepa_start_step", 0)) or 0)
            if step is not None and step < edge_start_step:
                self.filtered_edge_counts[f"{name}:before_edge_start_step"] += 1
                continue
            src_text = self.prepare_view_text(src_name, views.get(src_name))
            tgt_text = self.prepare_view_text(tgt_name, views.get(tgt_name))
            if not src_text:
                self.filtered_edge_counts[f"{name}:missing_source"] += 1
                continue
            if not tgt_text:
                self.filtered_edge_counts[f"{name}:missing_target"] += 1
                continue
            min_target_tokens = int(edge.get("min_target_tokens", getattr(self.args, "min_target_tokens", 0)) or 0)
            if min_target_tokens > 0:
                estimated_target_tokens = self.count_view_tokens(tgt_text, view_name=tgt_name)
                if estimated_target_tokens < min_target_tokens:
                    self.filtered_edge_counts[f"{name}:short_target"] += 1
                    continue
            filtered.append(edge)
        return filtered

    def lambda_for_edge(self, edge: dict[str, Any]) -> float:
        edge_name = edge["name"]
        if "lambda" in edge:
            return float(edge["lambda"])
        if self.lambda_mode == "current_adaptive":
            return self.lambda_controller.lambda_for(edge_name)
        if self.lambda_mode == "inverse_loss":
            return self.lambda_controller.lambda_for_inverse_loss(edge_name)
        return float(self.args.lambda_base)

    def zero_mv_result(
        self,
        skip_reason: str,
        ce_loss: torch.Tensor,
        step: int | None = None,
        warmup: float = 0.0,
        step_prob: float = 0.0,
    ) -> tuple[torch.Tensor, list[dict[str, Any]], int, int, int]:
        ratio_cap = float(getattr(self.args, "jepa_ce_ratio_cap", 0.0) or 0.0)
        cap_value = float((ratio_cap * ce_loss.detach()).cpu()) if ratio_cap > 0.0 else None
        self.last_mv_loss_info = {
            "raw_loss": 0.0,
            "loss_after_warmup": 0.0,
            "warmup": warmup,
            "step_prob_effective": step_prob,
            "skipped": True,
            "skip_reason": skip_reason,
            "ce_ratio_cap_value": cap_value,
            "step": step,
            "selected_main_edges": [],
            "selected_weak_edges": [],
            "main_edges_used": 0,
            "weak_edges_used": 0,
        }
        return next(self.model.parameters()).new_tensor(0.0), [], 0, 0, 0

    def jepa_warmup(self, step: int) -> float:
        start_step = int(getattr(self.args, "jepa_start_step", 0) or 0)
        configured_warmup = int(getattr(self.args, "jepa_warmup_steps", 1) or 0)
        if configured_warmup <= 0:
            return 1.0
        warmup_steps = max(1, configured_warmup)
        return min(1.0, max(0.0, (step - start_step) / warmup_steps))

    def current_jepa_step_prob(self, warmup: float) -> float:
        base_prob = min(1.0, max(0.0, float(getattr(self.args, "jepa_step_prob", 1.0) or 0.0)))
        if getattr(self.args, "jepa_step_schedule", "constant") == "linear_warmup":
            return base_prob * warmup
        return base_prob

    def prepare_view_text(self, view_name: str, text: str | None) -> str | None:
        if text is None:
            return None
        if getattr(self.args, "strip_answer_from_reasoning", False) and view_name == "R":
            return strip_final_answer(text)
        return text

    def count_view_tokens(self, text: str, view_name: str) -> int:
        max_length = min(self.args.view_max_length, self.view_max_lengths.get(view_name, self.args.view_max_length))
        encoded = self.tokenizer(text, add_special_tokens=True, truncation=False)
        return min(len(list(encoded["input_ids"])), max_length)

    def encode_view(
        self,
        text: str,
        view_name: str,
        predictors: int = 0,
        grad: bool = True,
        pooling: str = "last",
    ) -> tuple[torch.Tensor, int]:
        if predictors > 0:
            text = text + "".join(f"<|predictor_{idx}|>" for idx in range(1, predictors + 1))
        max_length = min(self.args.view_max_length, self.view_max_lengths.get(view_name, self.args.view_max_length))
        tokenized = tokenize_view(
            self.tokenizer,
            text,
            max_length,
            truncation_side=self.view_truncation_sides.get(view_name, "left"),
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        context = nullcontext() if grad else torch.no_grad()
        with context:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
            hidden = pooled_hidden(
                outputs.hidden_states[-1],
                attention_mask,
                mode=pooling,
                last_k=getattr(self.args, "pool_last_k", 64),
            )
        return hidden, int(attention_mask.sum().item())


def parse_allowed_edges(value: str | None) -> set[str] | None:
    if value is None:
        return None
    names = {part.strip() for part in value.split(",") if part.strip()}
    return names or None


def edge_ratio_cap(edge: dict[str, Any], default: float) -> float:
    value = edge.get("ce_ratio_cap", default)
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return max(0.0, float(default or 0.0))


def split_main_weak_edges(candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    main_edges = [edge for edge in candidates if not edge.get("weak_only")]
    weak_edges = [edge for edge in candidates if edge.get("weak_only")]
    return main_edges, weak_edges


def tokenize_view(
    tokenizer: Any,
    text: str,
    max_length: int,
    truncation_side: str = "left",
) -> dict[str, torch.Tensor]:
    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    input_ids = list(encoded["input_ids"])
    if len(input_ids) > max_length:
        if truncation_side == "right":
            input_ids = input_ids[:max_length]
        elif truncation_side == "middle":
            left = max_length // 2
            right = max_length - left
            input_ids = input_ids[:left] + input_ids[-right:]
        else:
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


def tokenize_view_left_truncate(tokenizer: Any, text: str, max_length: int) -> dict[str, torch.Tensor]:
    return tokenize_view(tokenizer, text, max_length, truncation_side="left")


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


def summarize_numeric_lists(values: dict[str, list[float]]) -> dict[str, float]:
    summary = {}
    for key, nums in values.items():
        if nums:
            summary[f"{key}_mean"] = sum(nums) / len(nums)
    return summary
