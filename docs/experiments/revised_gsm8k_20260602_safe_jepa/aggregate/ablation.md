# MAV-JEPA Ablation Smoke Report

This report is generated from available GSM8K and Spider runs. Quality metrics remain `null` until prediction files are generated; current comparisons use smoke-run training loss and compute fields only.

## Runs

| Task | Ablation | Method | Train loss | JEPA loss | FLOPs | Wall clock sec | Exact match | Exec acc |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gsm8k | A10 | mav_qr_stopgrad_p125_l005 | 0.3241018610952227 | 0.0009031353975615914 | 31657027697418240 | 2049.8197391033173 | 0.4806671721000758 | null |
| gsm8k | A11 | mav_qra_safe_all_p25_l005 | 0.32382852811464524 | 0.0006494478290832916 | 30094537450905600 | 1903.696759223938 | 0.46171341925701287 | null |
| gsm8k | A12 | mav_qa_only_p25_l005 | 0.3271772353683715 | 0.003667650913201406 | 30710286536908800 | 2363.2099113464355 | 0.47384382107657314 | null |
| gsm8k | A6 | mv_sft_lora | 0.32297337705275186 | 0.0 | 28630786332672000 | 1570.981427192688 | 0.4829416224412434 | null |
| gsm8k | A7 | mav_qr_stopgrad_p25_l005 | 0.32430609004395317 | 0.0012586331932610939 | 34524988468469760 | 2324.1401817798615 | 0.48597422289613346 | null |
| gsm8k | A8 | mav_qr_stopgrad_p50_l005 | 0.3247850349280875 | 0.0017314181166948274 | 40502205688012800 | 2576.780040502548 | 0.4791508718726308 | null |
| gsm8k | A9 | mav_qr_normmse_p25_l005 | 0.32293963240514145 | 7.106982511512473e-06 | 34511251013591040 | 2400.6368980407715 | 0.46019711902956784 | null |

## Required Questions

1. Does multi-view help over original two-view JEPA? gsm8k: missing A1 or A2
1. Does adaptive lambda reduce tuning sensitivity? gsm8k: missing A2 or A3
1. Does adaptive edge dropout beat random dropout at similar compute? gsm8k: missing A4 or random-dropout baseline
1. Which view edges are useful or harmful? No multi-view ablation runs are available yet.
