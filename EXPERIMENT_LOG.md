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

## Task 02 acceptance

Server: `10.154.22.11`, commit `b0a09a0`, conda env `mav-jepa`.

- `pytest tests/test_view_builders.py -q`: 5 passed.
- `data/mv/gsm8k/train.jsonl`: 7473 kept, 0 skipped.
- `data/mv/gsm8k/eval.jsonl`: 1319 kept, 0 skipped.
- `data/mv/spider/train.jsonl`: 6587 kept, 0 skipped, 6587 missing DB.
- `data/mv/spider/eval.jsonl`: 1447 kept, 0 skipped, 1447 missing DB.
- Spider SQLite execution views were skipped because upstream `spider_data.zip` is an unavailable LFS pointer; `QS -> SQL`, `Q -> SQL`, and `Q -> S` views/edges were generated.

## Task 03 acceptance

Server: `10.154.22.11`, commit `6fff1c4`, conda env `mav-jepa`, `CUDA_VISIBLE_DEVICES=0`.

- `python finetune_mv.py --help`: passed; required MAV-JEPA CLI flags are present.
- `pytest tests/test_loss_shapes.py -q`: 3 passed.
- `torchrun --nproc_per_node=1 finetune_mv.py ... --track_flop` on `data/mv/gsm8k/train_64.jsonl`: success.
- Output artifacts under `outputs/smoke/mv_gsm8k`: `run_config.json`, `metrics.jsonl`, `train.log`, `results.json`, `checkpoint-final/`.
- Smoke metrics: `train_steps=64`, `train_loss=1.1943`, `ce_loss=0.4977`, `jepa_loss=0.6967`, `wall_clock_sec=21.9263`, `gpu_hours=0.006091`, `peak_vram_gb=8.3506`, `avg_steps_per_sec=2.9189`, `avg_tokens_per_sec=997.7974`, `estimated_total_flops=205012304117760`, `jepa_edges_per_step=1.0`.
- Edge sampling frequency: `Q_to_R=25`, `Q_to_A=22`, `R_to_A=17`.
- Model source: ModelScope snapshot `/home/zsh/.cache/modelscope/hub/models/Qwen/Qwen2___5-1___5B-Instruct`; no fallback was needed for this run.

## Task 04 acceptance

Server: `10.154.22.11`, commit `31dfed4`, conda env `mav-jepa`.

- `pytest tests/test_adaptive_lambda.py -q`: 4 passed.
- `pytest tests/test_loss_shapes.py tests/test_adaptive_lambda.py -q`: 7 passed.
- `python -m py_compile mavjepa/trainer_mv.py finetune_mv.py`: passed.
- Adaptive lambda is wired to `--adaptive_lambda`; default fixed `lambda_base` behavior remains unchanged when the flag is absent.

## Task 05 acceptance

Server: `10.154.22.11`, commit `d3a7e62`, conda env `mav-jepa`.

- `pytest tests/test_edge_sampler.py -q`: 7 passed.
- `pytest tests/test_edge_sampler.py tests/test_adaptive_lambda.py tests/test_loss_shapes.py -q`: 14 passed.
- Edge sampler supports `none`, `random`, and `adaptive`; adaptive probabilities use EMA loss and edge quality, respect `edge_budget`, keep nonzero probability for low-score edges, and skip temporarily blacklisted repeatedly non-finite edges.

## Task 06 acceptance

Server: `10.154.22.11`, commit `501058c`, conda env `mav-jepa`, `CUDA_VISIBLE_DEVICES=2`.

- Ran the one-GPU smoke matrix on the least-occupied visible card at launch, GPU 2.
- Tasks: `gsm8k` and `spider`; model: `Qwen/Qwen2.5-1.5B-Instruct` resolved from ModelScope.
- Methods completed for both tasks: `sft_lora`, `original_llm_jepa_lora`, `original_llm_jepa_random_dropout`, `mv_jepa_fixed_lambda`, `mav_jepa_full`.
- All 10 Task 06 smoke runs wrote `run_status.json` with `status=success`.
- `outputs/aggregate/results.csv`: 13 result rows plus header. The required columns are present: `run_name, task, model, method, seed, lora_rank, lr, accuracy, exact_match, exec_acc, train_loss, jepa_loss, flops, wall_clock_sec, trainable_params`.
- Representative wall-clock seconds: GSM8K SFT 34.32, GSM8K original JEPA 43.40, GSM8K MAV-JEPA full 24.32, Spider SFT 32.30, Spider original JEPA 47.91, Spider MAV-JEPA full 19.96.
- This is the Task 06 smoke matrix with `seed=0`, `lr=2e-5`, `lora_rank=16`, and 64 train/eval samples per task. Larger multi-seed/rank sweeps are intentionally not launched yet while validating the one-card workflow.

## Task 07 acceptance

Server: `10.154.22.11`, commit `51443a6`, conda env `mav-jepa`.

- `pytest tests/test_metrics.py -q`: 4 passed.
- `python scripts/60_evaluate_all.py --outputs_dir outputs --tasks gsm8k spider`: evaluated 13 runs and wrote `outputs/aggregate/evaluation_report.json`.
- `python scripts/70_aggregate_results.py --outputs_dir outputs --output_csv outputs/aggregate/results.csv`: wrote 13 result rows plus header.
- Evaluation handles missing prediction files by writing explicit `null` metrics instead of failing; GSM8K numeric normalization and Spider SQL normalization/code-fence stripping are implemented.

## Task 08 acceptance

Server: `10.154.22.11`, current code commit `2f5f26d`, conda env `mav-jepa`, `CUDA_VISIBLE_DEVICES=2`.

- Reused Task 06 smoke runs for A0, A1, A2, and A5.
- Ran missing A3/A4 smoke ablations for GSM8K and Spider on GPU 2: `mv_jepa_adaptive_lambda` and `mv_jepa_adaptive_edge_dropout`; all 4 wrote `run_status.json` with `status=success`.
- `outputs/aggregate/ablation.csv`: 12 ablation rows plus header, covering A0-A5 for both GSM8K and Spider.
- `outputs/aggregate/ablation.md`: generated and answers the required questions using available smoke-run loss/compute proxies; quality metrics remain `null` until prediction files are generated.
- Optional A6/A7/A8 rank, edge-removal, and same-FLOP sweeps were not launched, per priority constraints.

## Task 09 acceptance

Server: `10.154.22.11`, commit `0bcbe02`, conda env `mav-jepa`.

- `python scripts/70_aggregate_results.py --outputs_dir outputs --make_plots`: passed and regenerated `outputs/aggregate/results.csv`.
- Generated figures: `loss_curves_gsm8k.png`, `loss_curves_spider.png`, `edge_sampling_gsm8k.png`, `edge_sampling_spider.png`, `lambda_dynamics_gsm8k.png`, `lambda_dynamics_spider.png`, `compute_vs_score.png`.
- Generated analysis artifacts: `outputs/analysis/view_edge_table.md` and `outputs/analysis/error_cases.jsonl`.
- Current artifacts are smoke-run diagnostics; qualitative error cases remain empty until prediction/generation files are produced.

## R2 GSM8K rationale-span diagnostic

Server: `10.154.22.11`, commit `056d587`, output dir `outputs/r2_gsm8k_20260602T0856Z_056d587_stage1`.

- `mav_rspan_qrpre_rsuf_p125_l003 seed0` completed full GSM8K evaluation: accuracy `0.479909`, exact match `0.476118`, 1319 eval examples, train wall `1713.563s`, generation wall `4017.326s`.
- Comparable SFT baselines from `outputs/revised_gsm8k_20260602T0208Z_safe_jepa`: seed0 accuracy `0.486732`, exact `0.482942`; seed1 accuracy `0.473844`, exact `0.470811`.
- Continuation gate failed: rationale-span R2 does not beat SFT seed0 and does not justify automatic seed expansion.
- 256-example R2 probes: `mav_qr_p125_l003_cap003 seed0` accuracy `0.484375`, exact `0.476563`; `mav_qr_p125_l003_cap003_nostrip seed0` accuracy `0.500000`, exact `0.500000`; `mav_qr_rspan_prior_p125_l003 seed0` accuracy `0.472656`, exact `0.457031`.
- Decision: do not launch new R2 expansion jobs from this state. The no-strip result suggests `strip_answer_from_reasoning` may damage the R view, but this is diagnostic rather than a confirmed full-eval positive result.

See `docs/experiments/r2_gsm8k_20260602_rspan/README.md`.
