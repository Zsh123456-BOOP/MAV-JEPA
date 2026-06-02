# R2 GSM8K Rationale-Span Diagnostics

Server: `mav-jepa-4090`
Workdir: `/home/zsh/projects/MAV-JEPA/llm-jepa`
Code commit: `056d587`
Output dir: `outputs/r2_gsm8k_20260602T0856Z_056d587_stage1`

## Decision

`mav_rspan_qrpre_rsuf_p125_l003 seed0` completed full GSM8K evaluation on 1319 examples and did not meet the continuation gate.

The result is below SFT seed0 and does not provide a defensible positive signal for expanding this branch to more seeds or a multi-edge validation stage.

## Full-Eval Result

| Method | Seed | Eval examples | Accuracy | Exact match | Train wall sec | Generation wall sec | Train GPU hours | Generation GPU hours |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mav_rspan_qrpre_rsuf_p125_l003` | 0 | 1319 | 0.479909 | 0.476118 | 1713.563 | 4017.326 | 0.475990 | 1.115924 |
| `mv_sft_lora` baseline | 0 | 1319 | 0.486732 | 0.482942 | 1554.120 | 3946.204 | 0.431700 | 1.096168 |
| `mv_sft_lora` baseline | 1 | 1319 | 0.473844 | 0.470811 | 1585.345 | 4190.239 | 0.440374 | 1.163955 |

## 256-Example R2 Probes

| Method | Seed | Eval examples | Accuracy | Exact match | Train wall sec | Notes |
|---|---:|---:|---:|---:|---:|---|
| `mav_rspan_qrpre_rsuf_p125_l003` | 0 | 256 | 0.484375 | 0.480469 | 1713.563 | Full eval later dropped to 0.479909 accuracy. |
| `mav_qr_p125_l003_cap003` | 0 | 256 | 0.484375 | 0.476563 | 1761.029 | Conservative `Q_to_R` control. |
| `mav_qr_rspan_prior_p125_l003` | 0 | 256 | 0.472656 | 0.457031 | 2148.331 | Dual-edge prior was worse and slower. |
| `mav_qr_p125_l003_cap003_nostrip` | 0 | 256 | 0.500000 | 0.500000 | 1750.731 | Diagnostic only; stronger than the stripped Q->R control on the same 256-example slice. |

## Interpretation

- The rationale-span target `QR_PRE -> R_SUF` is safer than direct answer-target edges, but it is not a positive result under the current gate.
- Adding `Q_to_R` plus rationale-span under a prior sampler made the 256-example result worse and increased training time, so this is not a viable next automatic branch.
- The no-strip `Q_to_R` diagnostic improved from `0.484375/0.476563` to `0.500000/0.500000` on the 256-example slice, which suggests `strip_answer_from_reasoning` may be damaging the R view. This is a diagnostic signal, not enough by itself to launch a full expansion after the rationale-span full-eval failure.
- The current evidence points to a method-design issue, not a simple need for more seeds. The answer-related edges and the way multi-edge training interacts with CE need another design pass before spending more GPU time.

## Stop/Continue Gate

Do not launch new R2 expansion jobs from this state.

The completed no-strip probe should be handed to GPT Pro as evidence that view construction may be part of the failure mode. It should not be treated as a confirmed positive result until it is tested on full GSM8K and compared against the same SFT slice/full baseline under a redesigned method gate.
