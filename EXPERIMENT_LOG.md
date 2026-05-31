# MAV-JEPA Experiment Log

## Task 00 bootstrap notes

- Official source commit: see `artifacts_source_commit.txt`.
- The upstream `spider_data.zip` Git LFS object could not be downloaded during clone because the repository LFS budget was exceeded. Spider JSONL files are present; SQLite DBs will be restored from `spider_data.zip` only if available, otherwise Spider execution metrics will use the fallback rules in the task card.
- Server execution target: `10.154.22.11`, single idle RTX 4090 first, with `CUDA_VISIBLE_DEVICES=0` unless a later GPU check selects another idle card.
- Server Python environment: isolated conda env `mav-jepa` under `/home/zsh/anaconda3/envs/mav-jepa`; run commands through `/home/zsh/anaconda3/bin/conda run -n mav-jepa ...`.
- GitHub repository: `https://github.com/Zsh123456-BOOP/MAV-JEPA`, branch `master`. The official upstream remote and temporary server bare repo were removed; the GitHub history is a clean MAV-JEPA root commit without the unusable upstream LFS pointer.
- Model source policy: use ModelScope for server-side model downloads first. If ModelScope cannot provide the requested model snapshot, download locally and transfer the resolved local model directory to the server. Record the final `model_name_or_path`, `model_source`, and any fallback in each `run_config.json`.
- Task 00 server acceptance passed on `10.154.22.11` with `CUDA_VISIBLE_DEVICES=0`: Python 3.11.15, PyTorch 2.7.1+cu128, Transformers 4.55.2, Datasets 4.8.5, Accelerate 1.13.0, PEFT 0.17.0, ModelScope 1.37.1, one visible RTX 4090, BF16 supported.

## Required efficiency accounting

Every training or evaluation run must record the following efficiency fields in `run_config.json`, `metrics.jsonl`, `results.json`, or derived aggregate CSVs:

- `wall_clock_sec`
- `gpu_hours`
- `peak_vram_gb`
- `avg_steps_per_sec`
- `avg_tokens_per_sec`
- `estimated_total_flops`
- `jepa_edges_per_step`
- `lambda_history`
- `edge_sampling_frequency`
- `same_flop_accuracy` or a null value when not applicable

Per-step trace records should include `step_time_sec`, `gpu_memory_gb`, CE loss, JEPA loss, total loss, learning rate, active edges, lambda by edge, and edge sampling probabilities.

## Task 01 smoke-test design

- `scripts/20_smoke_test.sh` resolves model IDs through ModelScope first and passes the resulting local snapshot path to `finetune.py`.
- The default model is `Qwen/Qwen2.5-1.5B-Instruct`; gated Llama/Gemma requests fall back to Qwen and write the fallback metadata into each run's `run_config.json`.
- Original-code smoke runs use `CUDA_VISIBLE_DEVICES=0`, LoRA rank 16, 64 GSM8K train/eval records, one epoch, and per-run efficiency artifacts under `outputs/smoke/<run_name>/`.

## Task 01 acceptance

Server: `10.154.22.11`, commit `b0b2a6e`, conda env `mav-jepa`, `CUDA_VISIBLE_DEVICES=0`.

- `outputs/help/finetune_help.txt` created.
- `data/debug/gsm8k_64_train.jsonl` and `data/debug/gsm8k_64_test.jsonl` created with 64 records each.
- `outputs/smoke/original_sft`: success, finite loss, 1 checkpoint, `wall_clock_sec=33`, `gpu_hours=0.009167`, `peak_vram_gb=6.137`.
- `outputs/smoke/original_jepa`: success, finite loss, 1 checkpoint, `train.log` contains `llm_loss` and `jepa_loss`, `wall_clock_sec=305`, `gpu_hours=0.084722`, `peak_vram_gb=8.355`.
