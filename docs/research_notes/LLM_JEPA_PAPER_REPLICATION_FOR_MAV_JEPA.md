# 面向 MAV-JEPA 的 LLM-JEPA 论文复刻方案

日期：2026-06-02

## 资料来源

- 论文页面：https://openreview.net/forum?id=GbXKPo9QfH
- 官方代码：https://github.com/galilai-group/llm-jepa
- 本地 PDF：`docs/papers/llm-jepa-iclr2026.pdf`
- 本地抽取文本：`docs/papers/llm-jepa-iclr2026.extracted.txt`

论文题目是 `LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures`，ICLR 2026 会议论文，作者为 Hai Huang、Yann LeCun、Randall Balestriero。

## 一句话理解

原论文的核心观点是：LLM 在保持正常生成式 fine-tuning 能力的同时，可以额外加入一个位于 embedding space 的 JEPA 表征预测目标，让两个语义视图之间的表示对齐，从而在多个数据集、模型族和计算预算下提升表现。

对 MAV-JEPA 来说，最合适的复刻切入点是：

> LLM-JEPA 使用一个人工指定的单向视图对；MAV-JEPA 将它推广为多视角边图。在推理和结构化生成任务中，一个样本往往包含多个语义视图和多条候选边。朴素全边训练可能有害，但通过 CE-safe 的边选择、边门控和 loss dropout，可以识别有用推理边并提升或稳定 fine-tuning。

## 关于 baseline 的重要修正

原论文不是只和未训练模型比较。它的主实验比较对象是：

- 标准 LLM fine-tuning，即 next-token / CE loss，论文中写作 `L_LLM`。
- LLM-JEPA fine-tuning，即 CE loss 加 JEPA loss，论文中写作 `L_LLM-JEPA`。

base / no-fine-tune 模型主要用于表征分析、预训练讨论和下界参考。我们的论文可以包含 base/no-fine-tune，但不能只用它作为 baseline。

MAV-JEPA 最低限度需要这些 baseline：

1. base / no fine-tune。
2. 同一 trainer 路径下的 CE-only SFT，即 `mv_sft_lora`。
3. 原始单边 LLM-JEPA。
4. 朴素多边 MAV-JEPA。
5. CE-safe / gated MAV-JEPA。

## 原论文结构与我们的复刻方式

### 摘要

原论文写法：

- 从 LLM 的 input-space reconstruction 目标和视觉 JEPA 的 embedding-space 目标之间的差异切入。
- 提出一个适用于 LLM 预训练和微调的 JEPA 目标。
- 声称在多个数据集、模型族和模型规模上验证。
- 强调不牺牲生成能力。

MAV-JEPA 改写方式：

- 从 LLM-JEPA 只使用一个人工指定的单向视图对切入。
- 提出面向推理和结构化生成任务的多视角边图 formulation。
- 强调不是所有边都有用；贡献在于 CE-safe 边门控、边 dropout 和边级诊断。
- 在多 seed、多模型完成之前，不写“大范围优于 LLM-JEPA”的强结论。

当前 seed0 结果下可用的摘要表述：

> 我们发现，朴素多边 JEPA 会和 CE fine-tuning 发生冲突；而使用 stop-gradient target、step-level loss dropout 和 CE-ratio cap 的 Q-to-reasoning 边，可以在 GSM8K 上相对同一 trainer 的 CE-only baseline 带来小幅正向提升。

### 引言

原论文写法：

- 视觉 JEPA 说明 embedding-space prediction 有价值。
- LLM 训练主要依赖 token-space reconstruction。
- 一些 NLP 数据天然包含同一对象的两个视图，例如自然语言和代码。
- 贡献包括目标函数、实证验证和表征分析。

MAV-JEPA 改写方式：

- LLM-JEPA 证明了二视图 LLM JEPA 可以工作。
- 但推理任务通常不止两个语义视图，例如 question、reasoning、answer、schema、SQL、execution result、retrieved evidence 等。
- 这天然形成一个 view graph，而不是单个 view pair。
- 核心难点是：部分边有用，部分边有噪声，部分边会破坏生成式 CE 学习。

我们的贡献列表建议写成：

1. 提出面向 LLM fine-tuning 的多视角边图 JEPA formulation。
2. 提出 CE-safe MAV-JEPA：stop-gradient target、delayed JEPA warmup、CE-ratio cap、step-level JEPA dropout、answer-target gating。
3. 给出 edge-level diagnostics，说明哪些视图关系有效、哪些有害。
4. 先在 GSM8K 验证，再扩展到 Spider、NQ、HellaSwag 和更多模型。

### 背景

原论文定义：

- 标准 LLM loss `L_LLM`，即 cross-entropy / next-token objective。
- embedding space 中的 JEPA auxiliary objective。
- 用 encoder hidden state 作为 representation。

MAV-JEPA 需要额外定义：

- GSM8K 的 view set：`V = {Q, R, A}`。
- GSM8K 的 edge set：`E = {Q->R, Q->A, R->A}`。
- Spider 可扩展为：`Q`、`Schema`、`Q+Schema`、`SQL`、`ExecutionResult`。
- 边采样/门控函数 `s_t(e)` 和 CE-safe 控制项。

### 方法

原论文目标：

```text
L = L_LLM + lambda * d(Pred(Enc(Text)), Enc(Code))
```

原论文实现选择：

- `Text` 和 `Code` 是两个视图。
- 用最后一层最后一个 token 的 hidden state 作为 embedding。
- metric 首选 cosine similarity。
- 用 `[PRED]` token 借助 LLM 自身权重实现 predictor。
- 自定义 additive attention mask 将两个额外 view forward 降到一个额外 forward。
- random JEPA-loss dropout 用于降低训练计算量。

MAV-JEPA 目标：

```text
L = L_CE
  + alpha_t * m_t * min(
      mean_{e in S_t} lambda_e * d(Pred(h_src(e)), stopgrad(h_tgt(e))),
      rho * stopgrad(L_CE)
    )
```

其中：

- `S_t` 是当前 step 被采样到的 active edge 集合。
- `m_t` 是 step-level JEPA dropout mask。
- `alpha_t` 是 delayed warmup 系数。
- `rho` 是 CE dominance cap。
- `lambda_e` 可以是 fixed，也可以是 conservative inverse-loss weighting。

当前最好的候选配置：

```text
allowed_edges = Q_to_R
lambda_base = 0.05
jepa_step_prob = 0.25
jepa_start_step = 500
jepa_warmup_steps = 1000
jepa_ce_ratio_cap = 0.05
detach_target = true
target_no_grad = true
jepa_reduce = mean
strip_answer_from_reasoning = true
```

关键写法：

- 不要因为当前只有 `Q->R` 明显有效就放弃多边方向。
- 论文应强调：多视角 JEPA 提供候选边集合，使我们能够诊断、筛选和门控边。
- 一个有价值的 MAV-JEPA 结果可以是：在 GSM8K 上，`Q->A` 和 `R->A` 这类 answer-target edge 有害，而 `Q->R` 更有效。

## 原论文实验设计与我们的复刻路线

### 主 fine-tuning 表

原论文 protocol：

- 固定五个 seed：`{82, 23, 37, 84, 4}`。
- 报告 mean 和 standard deviation。
- 使用 paired one-tailed t-test。
- 先为 CE baseline 搜 learning rate。
- 固定 baseline learning rate 后，再搜索 JEPA 专属超参。
- 主实验一般按 4 epochs 评估。

原论文数据集：

- NL-RX-SYNTH。
- NL-RX-TURK。
- GSM8K。
- Spider。
- RottenTomatoes / Yelp，用于 pretraining transfer。
- 后续还包括 NQ-Open 和 HellaSwag。

原论文模型族：

- Llama-3.2-1B-Instruct。
- Gemma-2-2B-it。
- OpenELM-1.1B-Instruct。
- OLMo-2-1B / 7B。
- Llama-3.2-3B-Instruct。
- Llama-3.1-8B-Instruct。
- Qwen3-1.7B。
- DeepSeek-R1-Distill-Qwen-1.5B。

MAV-JEPA 后续复刻矩阵：

| 阶段 | 任务 | 模型 | Seeds | 目的 |
|---|---|---|---:|---|
| S0 | GSM8K | Qwen2.5-1.5B-Instruct | 0/1/2 | 当前 pilot 和方法调试 |
| S1 | GSM8K | Qwen3-1.7B, DeepSeek-R1-Distill-Qwen-1.5B | 5 seeds | 对齐原论文 reasoning-model 表 |
| S2 | Spider | Qwen2.5-1.5B, Llama-3.2-1B | 5 seeds | 多视角结构化生成 |
| S3 | NQ-Open, HellaSwag | Llama-3.2-1B | 5 seeds | 复刻原论文 beyond-code 泛化实验 |
| S4 | NL-RX-SYNTH/TURK | Llama/OpenELM/Gemma/OLMo | 5 seeds | 直接复刻原论文最强二视图设置 |
| S5 | 模型规模扩展 | 1B/3B/7B/8B | 5 seeds | 对齐原论文 model-size appendix |

### 设计 ablation

原论文 ablation 包括：

- Cosine vs L2 vs MSE。
- predictor token prepend vs append。
- 正向 Text->Code vs 反向 Code->Text。
- InfoNCE。
- 平均 hidden state vs last token。
- distinct predictor token vs identical predictor token。
- linear predictor vs `[PRED]` token。
- gamma/lambda ratio。

MAV-JEPA 对应 ablation：

| Ablation | 目的 |
|---|---|
| CE-only same trainer | 区分 trainer/data 路径影响和 MAV-JEPA 影响 |
| `Q->R` only | 测试有效推理边 |
| `Q->A` only | 测试 shortcut / numeric answer target 边 |
| `R->A` only | 测试 reasoning 到 answer 的压缩边 |
| naive all-edge | 说明朴素多边为什么有害 |
| all candidate edges with gating | 说明 MAV-JEPA 是 edge graph + safe selection |
| step prob `0.125/0.25/0.5` | 对齐 JEPA-loss dropout 计算效率叙事 |
| fixed lambda vs inverse-loss lambda | 测试保守 adaptive edge weighting |
| cosine vs normalized MSE | 对齐 metric ablation |
| last-token vs mean-last-k target pooling | 测试 representation extraction |

### 计算量 / FLOPs 部分

原论文的 compute 叙事：

- LLM-JEPA 增加训练计算量。
- 自定义 additive mask 降低 view encoding overhead。
- random JEPA-loss dropout 降低 compute。
- same-PFLOP 对比显示 dropout 可保持或提高性能。

MAV-JEPA 的 compute 叙事：

- 如果朴素实现，多边 JEPA 会比单边 JEPA 更贵。
- edge sampling 和 step-level dropout 不只是效率技巧，也是稳定训练所必需。
- 必须报告：
  - wall-clock，
  - GPU-hours，
  - peak VRAM，
  - estimated FLOPs，
  - JEPA edges used per step，
  - generation wall-clock。

表格模板：

| Method | Acc | Train sec | Gen sec | GPU-hours | FLOPs | JEPA edges/step | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base/no FT | | | | | | 0 | |
| CE-only | | | | | | 0 | |
| LLM-JEPA single-edge | | | | | | 1 or p | |
| naive MAV all-edge | | | | | | 3 | |
| CE-safe MAV gated | | | | | | p | |

### 表征分析

原论文的表征分析包括：

- Text/Code embedding 的 t-SNE。
- `Enc(Text) - Enc(Code)` 的 singular values。
- 从 `Enc(Text)` 到 `Enc(Code)` 的 linear regression error。
- extrapolation toy example。

MAV-JEPA 应该镜像为 edge-aware diagnostics：

1. CE-only、naive all-edge、CE-safe MAV 下 `Q`、`R`、`A` embedding 的 t-SNE。
2. 每条边的 SVD：
   - `Enc(Q) - Enc(R)`
   - `Enc(Q) - Enc(A)`
   - `Enc(R) - Enc(A)`
3. 每条边的 linear regression error。
4. 边有效性表：
   - accuracy impact，
   - loss scale，
   - sampling frequency，
   - gradient/CE conflict。

这是我们和原论文拉开差异的地方：原论文是 pairwise representation analysis，我们是 edge-graph / edge-diagnostic analysis。

## 当前 MAV-JEPA pilot 解读

当前已完成的 seed0 结果：

| Method | Accuracy | Train sec | JEPA edges/step | 解读 |
|---|---:|---:|---:|---|
| `mv_sft_lora` | 0.4867 | 1571 | 0.000 | 同 trainer 的 CE-only sanity |
| `mav_qr_stopgrad_p25_l005` | 0.4898 | 2324 | 0.241 | 当前 seed0 最好，小幅正向 |
| `mav_qr_stopgrad_p50_l005` | 0.4829 | 2577 | 0.481 | JEPA 过频繁会伤效果 |

当前证据：

- 还不能写强 superiority claim。
- 但方向可以作为 multi-view edge-gating study 继续推进。
- `p25` 相对 same-trainer CE-only 有正向信号；原论文在一些 harder reasoning/generalization 设置中也有小幅但显著提升，例如 Qwen3 GSM8K 和 NQ-Open，所以小幅提升并非不能写。

当前等待或正在跑：

- `mav_qr_stopgrad_p25_l005 seed1`
- `mv_sft_lora seed1`
- `mav_qr_normmse_p25_l005 seed0`

如果 `p25 seed1` 也优于 `mv_sft seed1`，立即跑：

```text
mav_qr_stopgrad_p25_l005 seed2
mv_sft_lora seed2
```

## 现在能写和不能写的结论

可以安全写：

- MAV-JEPA 将 LLM-JEPA 从单向 view pair 扩展为 multi-view edge graph。
- 朴素多边 JEPA 会和生成式 CE training 冲突。
- CE-safe controls 可以让 multi-view JEPA 可训练。
- 当前 GSM8K 结果表明 `Q->R` 比 answer-target edges 更有希望。

暂时不能写：

- MAV-JEPA 普遍优于 LLM-JEPA。
- all-edge 多边训练一定更好。
- MAV-JEPA 更省计算量，除非 same-FLOP 或 dropout 对比完成。
- 在少于 3 seeds 或 5 seeds 前写统计显著。

## MAV-JEPA 论文大纲建议

### 标题候选

1. MAV-JEPA: Multi-View Joint Embedding Predictive Fine-Tuning for Large Language Models
2. CE-Safe Multi-View JEPA for Reasoning-Oriented Language Model Fine-Tuning
3. From Pairwise LLM-JEPA to Multi-View Edge-Gated JEPA for Reasoning Tasks

### 摘要骨架

1. 指出 LLM-JEPA 的限制：只使用一个人工选择的 view pair。
2. 提出我们的问题：推理和结构化生成任务天然包含多个 view 和多条候选 edge。
3. 提出方法：multi-view edge graph + CE-safe JEPA gating。
4. 陈述发现：朴素 multi-edge 不稳定或有害；gated Q-to-reasoning 可以改善或稳定 GSM8K；edge diagnostics 解释 answer-target edge 的问题。
5. 陈述后续 scaling：对齐 LLM-JEPA 的模型族和数据集。

### 章节

1. Introduction。
2. Related Work：JEPA、LLM-JEPA、multi-view representation learning、reasoning fine-tuning。
3. Method：
   - multi-view records，
   - edge graph，
   - CE-safe objective，
   - edge sampling / gating，
   - compute accounting。
4. Experiments：
   - GSM8K 主实验，
   - Spider 结构化生成，
   - NQ/HellaSwag 泛化，
   - 多模型 scaling。
5. Ablations：
   - edge type，
   - dropout probability，
   - metric，
   - pooling，
   - lambda control。
6. Representation Analysis：
   - t-SNE，
   - SVD，
   - linear edge maps。
7. Compute Analysis：
   - wall-clock / GPU-hours / FLOPs，
   - same-FLOP results。
8. Conclusion and Limitations。

## 最小下一步实验

Priority 1：

```text
mv_sft_lora seeds 0/1/2
mav_qr_stopgrad_p25_l005 seeds 0/1/2
```

判断：

- 如果 MAV mean 高 `0.3-0.5` 个点以上，继续扩大。
- 如果 MAV mean 持平但 compute 更高，就把它定位为 diagnostic method，而不是 efficiency claim。

Priority 2：

```text
mav_qr_normmse_p25_l005 seed0
```

只有它超过 cosine p25 seed0，才继续多 seed。

Priority 3：

```text
Q_to_A only seed0
R_to_A only seed0
naive all-edge seed0
safe all-candidate with answer gating seed0
```

这是支撑 “multi-edge” claim 必须补的实验，因为它证明 edge graph 本身有研究价值。

Priority 4：

```text
Original LLM-JEPA single-edge same model / comparable setting
Base/no fine-tune eval
```

这两个用于论文式 baseline 表。

Priority 5：

复刻原论文 model-scale 表：

```text
Qwen3-1.7B
DeepSeek-R1-Distill-Qwen-1.5B
Llama-3.2-1B-Instruct
Llama-3.2-3B-Instruct
Llama-3.1-8B-Instruct
```

## 我们需要多接近原论文

前期不需要数值完全接近。需要接近的是实验逻辑：

- 同类 baseline：CE-only fine-tuning。
- 同类报告方式：mean/std over seeds。
- 同类模型和数据集扩展。
- 同类 compute accounting。
- 同类 representation analysis。

当前最大差距不是模型规模，而是实验完整性：

- 原论文用 5 seeds。
- 原论文先为 baseline 调 learning rate。
- 原论文固定 baseline learning rate 后再调 JEPA 专属超参。
- 原论文报告 p-value。
- 原论文有 ablation 和 representation diagnostics。

只要 seed-level 证据稳定，后续扩到他们的模型规模主要是工程和 GPU 调度问题。

## 当前写作策略

不要写：

> MAV-JEPA outperforms LLM-JEPA.

建议写：

> LLM-JEPA demonstrates that pairwise embedding prediction can improve LLM fine-tuning. We ask whether this idea extends to tasks with more than two semantic views. Our results show that multi-view JEPA is not simply an all-edge objective: edge selection and CE dominance are necessary. This turns MAV-JEPA into an edge-gated representation regularizer rather than a uniformly applied multi-edge loss.

中文解释：

> LLM-JEPA 证明了 pairwise embedding prediction 可以提升 LLM fine-tuning。我们进一步研究这个思想能否扩展到包含多个语义视图的任务。结果表明，multi-view JEPA 不能简单地把所有边都打开；边选择和 CE 主导约束是必要的。因此，MAV-JEPA 更适合被表述为一种 edge-gated representation regularizer，而不是均匀施加的全边 loss。

这种表述能保留“多边/多视角”的课题核心，即使最终 accuracy 提升幅度不大，也能形成清晰贡献。
