# Revised GSM8K Safe-JEPA Experiment Archive

本目录归档 2026-06-02 在 4090x4 服务器上完成的一批 MAV-JEPA / LLM-JEPA 诊断实验。目标不是证明最终方法有效，而是定位当前 MAV-JEPA 实现为什么没有稳定超过同训练器 SFT。

## 结论

当前实现没有稳定正向收益：

| Method | Seeds | Accuracy mean | Notes |
|---|---:|---:|---|
| `mv_sft_lora` | 0,1 | 0.4803 | 同训练器 CE-only 基线 |
| `mav_qr_stopgrad_p25_l005` | 0,1 | 0.4735 | seed0 好，seed1 明显下降 |
| `mav_qr_stopgrad_p125_l005` | 0 | 0.4829 | 接近 SFT seed0，未形成明确优势 |
| `mav_qr_stopgrad_p50_l005` | 0 | 0.4829 | 接近 SFT，但训练更贵 |
| `mav_qra_safe_all_p25_l005` | 0 | 0.4647 | 多边安全门控明显偏低 |
| `mav_qa_only_p25_l005` | 0 | 0.4769 | Q->A 单边无明确收益 |
| `mav_qr_normmse_p25_l005` | 0 | 0.4602 | normalized MSE 版本基本淘汰 |

当前不建议继续直接补同一配置多 seed。更合理的下一步是改方法：保留 Q->R 作为相对安全边，重新设计 answer 相关边，避免直接把答案当 JEPA target。

## 文件结构

- `aggregate/results.csv`: 所有 9 个 run 的聚合结果。
- `aggregate/ablation.csv`: ablation 标号 A6-A13 的结果摘要。
- `aggregate/ablation.md`: 聚合脚本生成的 ablation 报告。
- `logs/*.txt`: 每个后台调度任务的外层日志，包含训练和生成命令输出。
- `runs/*/run_config.json`: 单个 run 的完整配置。
- `runs/*/results.json`: 单个 run 的训练、生成、评估和资源统计。
- `runs/*/train.log.txt`: 单个 run 的训练日志。
- `runs/*/metrics_sample.jsonl`: 从原始 `metrics.jsonl` 下采样得到的关键训练动态。

## 未归档内容

以下内容保留在服务器原始输出目录，不进入公开 Git 仓库：

- `checkpoint-final/adapter_model.safetensors`: 每个 run 约 960MB。
- `predictions.jsonl`: 完整预测输出。
- 原始全量 `metrics.jsonl`: 每个 run 约 10MB，本仓库只保留下采样版本。

服务器原始路径：

```text
/home/zsh/projects/MAV-JEPA/llm-jepa/outputs/revised_gsm8k_20260602T0208Z_safe_jepa
```

## 重点排查方向

建议优先检查：

- `mavjepa/trainer_mv.py`: `compute_mv_loss`、edge filtering、answer target gating、warmup、CE cap、pooling、lambda 计算。
- `finetune_mv.py`: 新增 JEPA 参数默认值和 run config 记录。
- `scripts/run_task06_matrix.py`: A6-A13 方法矩阵是否真实表达了想测的实验假设。
- `configs/views/gsm8k_qra.yaml`: Q/R/A 视图定义是否让 answer 边天然形成 shortcut 或稀疏 target。
- 本目录 `aggregate/results.csv` 和 `runs/*/metrics_sample.jsonl`: 对比 seed 方差、edge sampling frequency、JEPA loss 与 CE loss 关系。
