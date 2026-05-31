# MAV-JEPA

This branch implements MAV-JEPA on top of the official LLM-JEPA repository.

Execution order follows `MAV_JEPA_Codex_Task_Card.md`: finish each task, run its acceptance check or smoke test, commit locally, sync the committed branch to `10.154.22.11`, and run GPU checks on the server with one idle RTX 4090 before scaling up.

## Server Workflow

Work locally first, commit each completed task, push `master` to the project GitHub repository, then pull and run acceptance on the server:

```bash
git push origin master
ssh zsh@10.154.22.11 'cd /home/zsh/projects/MAV-JEPA/llm-jepa && git pull --ff-only origin master'
ssh zsh@10.154.22.11 'cd /home/zsh/projects/MAV-JEPA/llm-jepa && CUDA_VISIBLE_DEVICES=0 /home/zsh/anaconda3/bin/conda run -n mav-jepa bash scripts/00_env_check.sh'
```

Do not run experiments from uncommitted local changes. Keep server runs tied to a Git commit.

## Conda Environment

Create or refresh the isolated server environment with:

```bash
bash scripts/00_create_conda_env.sh
```

On `10.154.22.11`, conda is available at `/home/zsh/anaconda3/bin/conda`; use `conda run -n mav-jepa ...` for all smoke tests and experiments.

## Model Source Policy

Server runs should use ModelScope first because Hugging Face downloads are unavailable in the target environment. The default MVP model remains the Qwen 2.5 1.5B instruct model, resolved through ModelScope when possible and then passed to Transformers as a local snapshot path.

If ModelScope cannot resolve a requested Llama, Gemma, or Qwen model, download the model on the local machine, transfer the local model directory to the server, and record the fallback in `outputs/<run_name>/run_config.json`.

## Efficiency Metrics

MAV-JEPA must report more than accuracy. Runs must record wall-clock time, GPU-hours, peak VRAM, throughput, estimated FLOPs, selected JEPA edges per step, adaptive lambda history, edge sampling frequency, and same-FLOP accuracy where applicable.

The expected run artifacts are:

```text
outputs/<run_name>/run_config.json
outputs/<run_name>/metrics.jsonl
outputs/<run_name>/train.log
outputs/<run_name>/results.json
```

## Task 00

Run:

```bash
bash scripts/00_env_check.sh
```

The script writes `outputs/env/env_report.json` and fails early if no CUDA GPU is visible.

Server acceptance command:

```bash
CUDA_VISIBLE_DEVICES=0 /home/zsh/anaconda3/bin/conda run -n mav-jepa bash scripts/00_env_check.sh
```

## Task 01

Run the original-code smoke test on one server GPU:

```bash
CUDA_VISIBLE_DEVICES=0 /home/zsh/anaconda3/bin/conda run -n mav-jepa bash scripts/20_smoke_test.sh
```

The script resolves Qwen through ModelScope, writes `outputs/help/finetune_help.txt`, creates `data/debug/gsm8k_64_train.jsonl` and `data/debug/gsm8k_64_test.jsonl`, then runs original SFT and original LLM-JEPA smoke tests with LoRA.
