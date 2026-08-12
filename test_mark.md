# WSSR / SPRING 测试总账（以 C rank-1600 超过 SPRING 为唯一主目标）

更新时间：2026-08-07（America/Vancouver）

## 0. 不可偏离的项目目标

本项目的主目标不是“WSSR 比 KFAC 好”、不是“WSSR 比 MinSR 好”、不是“rank-400 更省”，也不是只解释 WSSR 为什么落后。唯一主目标是：

> 在完全 matched 的 C 原子协议下，让 **WSSR 在精度—计算成本的 Pareto 意义上超过 SPRING**；不能仅靠使用更贵的优化器换取更低 frozen variance。

这里“超过”至少要求：

1. 使用同一个 C KFAC-pre1000 checkpoint；
2. 1000 个训练 walkers、100000 个主训练 step、每步 10 个 MCMC move、`reburn=False`；
3. frozen evaluation 使用 2000 walkers、burn-in 5000、20000 次测量、每次间隔 10 个 MCMC step；
4. 主要终点是 frozen raw local-energy variance，能量不得明显劣化；
5. 相对于两个正式 SPRING replicate，目标应是 variance 小于约 `0.00444 Ha^2`，并最好由至少两个 seed 支持。
6. 必须同时报告实际训练 wall time / seconds per step；同训练时间下 WSSR 的 frozen variance 也必须优于 SPRING。

特别注意：1000 walkers 时 centered current-batch score rank 至多为 999，因而 `rank=1600, eta=0` 不构成当前 batch 的低秩压缩。这样的 arm 即使数值上胜出，也不能支撑“更便宜的 WSSR”结论。

5000-step light frozen evaluation 只作为已经验证过的廉价筛选终点。其 SPRING 参照是 `0.03784061 Ha^2`，不能替代最终 100000-step matched 结论。

后续任何实验在提交前都必须回答一句：**它怎样增加 rank-1600 WSSR 超过 SPRING 的概率？** 如果没有直接答案，就不应占用主线预算。

---

## 1. 当前结论（先说结果）

截至本文件更新时间：

- **目标尚未完成。** 没有一个完成正式 100000-step matched frozen evaluation 的 rank-1600 WSSR 超过 SPRING。
- 最好的传统 rank-1600 WSSR 正式结果是 `eta=0.3, lr=0.04`，variance `0.01746088`，仍是 SPRING replicate2 的 `3.93x`。
- 官方初始 rank-1600 WSSR 配置 `eta=0.8, lr=0.02` 的 variance 是 `0.02166297`，是 SPRING 的 `4.78–4.88x`。
- rank-1600 WSSR 已经稳定超过 MinSR（variance `0.03127745`），但这只是中间里程碑，**不是项目成功**。
- 唯一强阳性结果是 rank-1600 的 **solution recurrence / current-batch residual correction**：E5000 两个 seed 都比 WSSR control 大幅改善，seed1 variance `0.03776839`，略优于同期 SPRING `0.03784061`；seed0 为 `0.06128061`，仍落后 SPRING。两个 seed 的几何均值约 `0.04810894`，仍是同期 SPRING 的 `1.27x`。它尚未做 100000-step 确认。
- 该阳性候选在 `eta=0`、1000 walkers、rank=1600 下，当前 centered batch 的自然秩至多约 999；因此当前 batch 解本身并未被 rank1600 截断。它证明的是 **SPRING 式残差纠错结构有效**，还不能证明“压缩后的 WSSR 历史算子”已经解决问题。
- 真正压缩的 residual-recurrence 已测试：rank800 E5000 两 seed frozen variance 为 `0.04922199/0.05219800`，几何均值 `0.05068816`；等训练时间的 SPRING epoch8000 为 `0.02624066`，rank800 仍落后 `1.93x`。rank400 seed0 为 `0.10475589`，落后 `3.99x`，因此提前淘汰第二 seed。
- rank800 相比未压缩 rank1600 recurrence 的两 seed 几何均值只劣化约 `5.4%`，且每步由约 `0.36–0.38 s` 降至 `0.130 s`。这说明低秩压缩本身可以保留 recurrence，但整个候选仍没有形成相对原生 SPRING 的 accuracy-time Pareto 优势。
- 其他已完成路线——补空间、固定 λ、fp64、exact-first、更多 SSI、普通 proximal/方向平滑、temporal gate、梯度输运、拆分/自适应 `eta_S/eta_g`——均未产生可靠、持续、matched 的 rank-1600 胜出结果。
- 大部分旧机制审计锚定 epoch100000 附近，而 gap curve 后来证明最终差距在 epoch5000 已基本形成。旧审计对晚期状态仍有效，但不能解释早期败局。

最重要的科学认识是：

> WSSR 每个 batch 的独立/加权低秩求解并不差；真正被训练结果正面支持的差别，是 SPRING 用“当前 batch residual”纠正一个完整参数空间的历史方向。普通方向平均、S/RHS 长记忆和梯度 EMA 都没有复现这个作用。

---

## 2. 权威 matched 基线

共同起点：

`/scratch/dexuan1/runs/pilot/C/C_kfac_pre1000_1/checkpoints/1000.npz`

共同训练协议：C `(4,2)`、1000 walkers、100000 updates、10 MCMC/update、clipping 5、inverse-time decay `1e-4`、新 optimizer state、`reburn=False`。

共同正式 frozen 协议：2000 walkers、burn-in 5000、20000 evaluation epochs、10 MCMC/measurement，共 4000 万 local-energy samples。

| 方法 | 关键参数 | frozen energy (Ha) | SEM | frozen variance (Ha²) | 相对 SPRING-1 |
|---|---|---:|---:|---:|---:|
| SPRING official seed0 | lr=.02, mu=.99, λ=.001, C=.001 | -37.8449131762 | 1.213e-5 | **0.0045313892** | 1.00x |
| SPRING replicate2 | 同上 | 见 stats | 见 stats | **0.0044415644** | 0.98x |
| WSSR rank1600 official | eta=.8, lr=.02, damping=.0003, SSI40/2 | -37.8442093734 | 2.680e-5 | 0.0216629652 | 4.78x |
| WSSR rank1600 tuned | eta=.3, lr=.04 | 见 stats | 见 stats | **0.0174608827** | **3.85x** |
| MinSR paper-matched | lr=.1（C 论文配置） | -37.8441982850 | 3.278e-5 | 0.0312774496 | 6.90x |
| KFAC matched | lr=.02 | -37.8440539322 | 4.584e-5 | 0.0652560374 | 14.40x |

权威来源：

- `experiments/C_spring_official_reproduction/COMPARISON_BASELINE.md`
- `/scratch/dexuan1/runs/C_spring_official_kfac1000_E100000/eval/statistics.json`
- `/scratch/dexuan1/runs/C_spring_official_kfac1000_E100000_replicate2/eval/statistics.json`
- `/scratch/dexuan1/runs/C_wssr_official_matched_E100000/E_rank1600_hard_ssi40/eval/statistics.json`
- `/scratch/dexuan1/runs/C_wssr_rank1600_eta03_lr004_E100000_resume1/eval/statistics.json`
- `/scratch/dexuan1/runs/C_minsr_paper_kfac1000_E100000/eval/statistics.json`

注意：项目中还有 KFAC-pre5000、4096 walkers、30k/50k 等旧轨迹。它们可用于机制或历史背景，但不能与上述权威 matched 表直接合并。

---

## 3. 正式 rank / optimizer 结果：已经做到哪里

### 3.1 WSSR rank 扩展

同一 matched 框架下的正式 frozen variance：

| WSSR 版本 | variance (Ha²) | 结论 |
|---|---:|---|
| rank400 hard | 0.34678575 | 很差 |
| rank400 adaptive complement β=.2 | 0.29250414 | 比 hard 略好，仍很差 |
| rank800 hard | 0.11322431 | rank 增大明显有用 |
| rank1600 hard, eta=.8/lr=.02 | 0.02166297 | 大幅提升并超过 MinSR，但远未追平 SPRING |
| rank1600 eta=.3/lr=.04 | 0.01746088 | 当前传统 WSSR 最好正式结果，仍约 3.93x SPRING replicate2 |

rank800 的进一步正式调参：

| eta | lr | frozen variance |
|---:|---:|---:|
| .3 | .03 | 0.04175943 |
| .3 | .04 | 0.03717322 |
| .5 | .02 | 0.05572972 |
| .5 | .04 | 0.04121632 |

这说明 rank 与参数调优都重要，但 rank800 不是主目标，且没有接近 SPRING。

### 3.2 rank1600 lr/eta 粗调

10k training-only screen 中，`eta=.3` 优于 `.5/.8`，`lr=.02/.04` 的末 1000 步 variance 都约 `.0267`；最终选出了 `eta=.3, lr=.04` 的 100k 轨迹。这里的 training variance 不是 formal frozen variance，只能用于筛选。

这个调参结果已经说明固定的大 eta 未必好，但不能由此推出“所有历史都不好”；后面的 solution-recurrence 实验恰好证明历史存放方式才是关键。

---

## 4. 差距在哪里形成：最重要的定位结果

light frozen gap curve：

| epoch | SPRING variance | WSSR rank1600 eta=.3/lr=.04 | WSSR / SPRING |
|---:|---:|---:|---:|
| 5k | 0.03784061 | 0.15170397 | 4.009x |
| 20k | 0.00864293 | 0.06374336 | 7.375x |
| 50k | 0.00553082 | 0.02715343 | 4.910x |
| 100k | 0.00375108 | 0.02123336 | 5.661x |

相对于正式 100k gap，epoch5000 已形成约 101% 的 log-gap；使用同一 light endpoint 算则形成最终 light log-gap 的 80.1%。因此：

- 胜负主要发生在 KFAC-pre1000 后的前 4000 个 optimizer step；
- 20k 后 WSSR 还在相对追赶；
- 在 epoch100000 才做局部修补，不能修复早期进入的较差轨迹；
- E5000 frozen variance 是下一轮候选的首要筛选终点，成本约为旧式 100k 搜索的 1/20。

来源：`/scratch/dexuan1/runs/C_optimizer_gap_curve_light_20260802/report.md`

### 4.1 欧氏 norm constraint / 实际函数空间步长审计（2026-08-09）

对 SPRING replicate2 与 tuned WSSR `rank1600, eta=.3, lr=.04` 的
5k/10k/20k/50k/100k checkpoint 各取 5 个 batch，只读恢复上一轮实际原始方向
`p`，计算

`gain = ||O_bar p_hat||`, `rho = 1/gain`,

以及经过 learning-rate schedule 与 Euclidean norm constraint 后的实际函数空间步
`||Delta theta_actual|| * gain`。代码约定中 score 为 `N_param x N_sample`，所以
`O_bar p` 对应 `score.T @ p`。WSSR 的保存状态允许由
`F=U Sigma`、`ek=V^T epsilon` 精确恢复前一步方向：

`p = F * ek / (Sigma^2 + lambda)`。

| epoch | WSSR/SPRING gain | WSSR/SPRING actual function step | cap active SPRING/WSSR |
|---:|---:|---:|---:|
| 5k | 1.3135 | 1.3135 | 100% / 100% |
| 10k | 1.2523 | 1.9419 | 0% / 100% |
| 20k | 1.0086 | 3.9224 | 0% / 100% |
| 50k | 0.8464 | 7.4562 | 0% / 0% |
| 100k | 0.3915 | 3.8130 | 0% / 0% |

全程结论：

- 单位欧氏方向的函数增益比几何均值为 `.8692`，batch bootstrap 95% CI
  `.7296–1.0214`；它随 epoch 反向，**不存在一个全程恒定的 Fisher-metric
  mismatch 倍率**，不能单独解释 frozen variance gap；
- 实际函数空间步的 WSSR/SPRING 几何均值为 **`3.0329`**（95% CI
  `2.3950–3.8174`），实际欧氏参数步几何均值比为 `3.4895`；
- 100k 单点实际函数步比 `3.8130`，数值上接近正式 frozen variance ratio
  `3.93x`，是强烈的 noise-floor/步长 confound 证据，但尚非因果分解；
- WSSR 在 5k/10k/20k 都被 cap 固定到 `sqrt(.001)=.0316228`，SPRING 只有
  5k 被 cap；
- WSSR 要在这些 checkpoint 退出 cap，base learning rate 分别需低于约
  `.00691/.00986/.02053`。因此此前 rank1600 10k screen 的 `lr=.02/.04`
  在最关键的 5k–10k 区间都处于相同 fixed-radius regime，不能算一次真正覆盖
  早期步长的 lr sweep；这解释了二者 training variance 几乎相同；
- 论文时期原始 commit `18b9b03` 的 SPRING 本身明确使用 Euclidean norm
  constraint，并非 fork 把 Fisher constraint 改错；KFAC 才使用近似 Fisher 范数。

该审计证明现有 WSSR/SPRING 精度对比混入了显著的实际步长差异，但不能从静态
比值声称 3.93x 中的某个确定比例是由它造成的。严格因果验证需要短程 paired
E5000：要么给两方法共同的 function-space trust radius，要么至少把 WSSR 的
早期 lr 扫描下移到约 `.003/.006/.01` 并逐步记录 cap scale。

来源：

- `/scratch/dexuan1/runs/C_euclidean_constraint_confound_audit_20260809/REPORT.md`
- `/scratch/dexuan1/runs/C_euclidean_constraint_confound_audit_20260809/rho_trajectory.csv`

---

## 5. 唯一强阳性：solution recurrence / residual correction

### 5.1 修改内容

两个候选都保留完整参数维度的历史方向 `phi_{t-1}`，`mu=.99`：

- naive：`phi_t = mu phi_{t-1} + solve(current RHS)`；
- residual correction：先算当前 batch 看见的历史作用，再解 residual：

  `r_t = epsilon_t - O_t (mu phi_{t-1})`

  `phi_t = mu phi_{t-1} + O_t^T (O_t O_t^T + lambda I)^(-1) r_t`

第二个公式不是普通 momentum。它只修正当前 batch 能观测到、并与当前方程冲突的历史分量；当前 batch 零空间中的历史信息得以保留。这就是 SPRING 的关键投影/残差纠错结构。

### 5.2 E5000 结果

共同配置：KFAC-pre1000、1000 walkers、rank1600、SSI40/2、`eta=0`、fixed λ=.001、lr=.04、norm=.001、fp64 rank-coordinate arithmetic。

| arm | seed | frozen energy | frozen variance | 相对 eta=.3 control |
|---|---:|---:|---:|---:|
| eta=.3 WSSR control | 0 | -37.84226602 | 0.15199841 | 1.000x |
| residual mu=.99 | 0 | -37.84371689 | 0.06128061 | 0.403x |
| eta=.3 WSSR control | 1 | -37.84236677 | 0.13932486 | 1.000x |
| residual mu=.99 | 1 | -37.84436788 | **0.03776839** | 0.271x |
| naive mu=.99 | 0 | -37.81576350 | 1.10816237 | 7.291x |

关键解释：

- 两个 seed 都证实 residual correction 对传统 WSSR 是巨大改善；几何平均 variance 降到 control 的 `0.3306x`。
- seed1 略优于 SPRING E5k；seed0 仍是 SPRING 的 `1.62x`；两 seed 几何均值仍为 SPRING 的 `1.27x`。
- naive recurrence 灾难性失败，直接证明“加历史/动量”不等于 SPRING；当前 batch residual correction 是必要成分。
- 这是当前唯一值得继续围绕主目标研究的阳性结构。

### 5.3 必须诚实写出的限制

1000 walkers 的 centered `O_t` 秩至多约 999。`eta=0` 时 rank1600 不截断当前 batch 方程，所以这个 arm 虽走 WSSR rank1600 接口，当前 solve 并没有真正被 rank1600 压缩。它更接近“在 WSSR 代码中重现 SPRING recurrence”，而不是已经证明历史算子压缩与 recurrence 完美兼容。

因此它目前回答的是：**为什么 SPRING 好，以及哪种结构能救 WSSR**；它还没有完成“一个有实质 WSSR sketch/averaged-S 成分的 rank1600 方法在 100k 超过 SPRING”。

来源：

- `experiments/C_rank1600_solution_recurrence_E5000_20260802/README.md`
- `/scratch/dexuan1/runs/C_rank1600_solution_recurrence_E5000_20260802/summary.json`

---

## 6. S/history averaging 路线：有局部信号，但训练没有兑现

### 6.1 早期多 batch 离线审计

在 KFAC-pre1000、fixed λ=.001 下：

| depth | held-out residual median improvement | 结果 |
|---:|---:|---|
| K=2 | 6.07% | 6/6 wins，但低于原 10% gate |
| K=3 | 8.22% | 6/6 wins，但低于原 10% gate |

同一审计发现 norm constraint 对所有候选都激活，raw step 被缩到约 9%，最终步长都接近 `sqrt(.001)=.0316228`。因此许多 raw residual/amplitude 优势不会原样进入训练，方向才是主要被消费的信息。

### 6.2 长记忆离线 gate 与训练相反

早期 fixed-theta memory scaling 中，`eta=.8/.95/.99` 的某些方向/压缩指标通过，尤其 bias-corrected `.99` 在 K=16 的 force cosine 增益最大。但随后 matched E5000 直接否定了它：

| arm | seed0 variance | seed1 variance |
|---|---:|---:|
| eta=.3 control | 0.15199841 | 0.13932486 |
| eta=.99 bias-corrected | 0.89954180 | 1.01899506 |

几何平均候选/control = `6.579x`，是明确反效果。

这组结果说明：

- fixed-theta 的 held-out residual / force cosine 不能单独作为训练准入仪器；
- 大 eta 的历史因子在静态方程上可能好，但在早期快速漂移的参数轨迹上会严重 lag；
- “SPRING mu=.99 有效，所以 WSSR eta=.99 也应好”是错误类比。SPRING 保存的是解空间历史并逐批纠错，WSSR eta 保存的是被反复压缩的算子/RHS 历史。

来源：

- `/scratch/dexuan1/runs/C_wssr_rank1600_early_multibatch_audit_pre1000_20260802/FINAL_DECISION.md`
- `/scratch/dexuan1/runs/C_rank1600_early_memory_scaling_fixed_x64_retry1_20260802/report.md`
- `/scratch/dexuan1/runs/C_rank1600_eta_memory_E5000_20260802/report.md`

### 6.3 历史 Jacobian 是否陈旧

history-refresh audit 的同坐标 force-action cosine 为 `0.999919–0.999994`；刷新旧 Jacobian 仅改善 held-out residual `0.034–1.607%`，没有通过 staleness gate。相反，同一参数处合两个 batch 在三个 checkpoint 均改善 `5.43–20.79%`。

所以“一步参数变化使旧 Jacobian 失效”不是晚期主因。旧因子短期仍准确；问题更多在累积/求解结构与非平稳训练，而不是必须昂贵刷新每个历史 Jacobian。

### 6.4 递归压缩是否破坏历史

K=3 时 rank1600 recursive factor 的 force-relevant Gram action relative error 约 `3.7e-4–4.8e-4`，force cosine 大于 `.99994`；production state 更接近。说明算子 action 的短程压缩保真很好。

但是 inverse direction 对尾谱/λ 极敏感，且 raw exact multibatch 在某些 K=3 replay 甚至不优于 current-only。因此不能把“operator action 准”直接等同于“训练方向好”。

---

## 7. λ、SSI、精度与谱：真实病理，但不是最终解法

### 7.1 动态 λ 病理

旧 `damping=.0003` 的相对谱规则对应 effective λ 约 `4.7–6.8e-6`，比 SPRING/fixed λ `.001` 小约 150–210 倍。same-checkpoint crossover 保持 decomposition、rank、key、history factor、RHS 全部相同，只换 inverse coefficient，fixed λ 使 held-out residual 改善 `67–90%`，因果证明确实存在 inverse-tail 放大。

### 7.2 训练只短暂改善

正式 frozen continuation：

| epoch | legacy variance | fixed λ=.001 variance | fixed reduction |
|---:|---:|---:|---:|
| 100500 | 0.01732218 | 0.01310554 | 24.34% |
| 101000 | 0.01790258 | 0.01316556 | 26.46% |
| 101500 | 0.01581649 | 0.01328235 | 16.02% |
| 102000 | 0.01417697 | 0.01421095 | -0.24% |

固定 λ 的收益到 102000 完全消失，所以该路线已经做过且答案是否定的。不得再次以“修 λ 后再训练”为新建议。

### 7.3 SSI / exact SVD

- 同 batch 换三个 SSI key，方向 cosine 约 `1.0`；随机 key 不是 batch-to-batch jitter 来源。
- warm iterations 2→4 仅给极小 coherence 改善，约 45% solver overhead，没有过 gate。
- exact rank1600 相邻方向 cosine 比 native 高约 `.069`，但仍远低于 SPRING，且 held-out residual 反而可变差。
- x64 审计表明 fixed λ 下 rank1600 exact 解相对 full fixed 解 cosine 约 `.9999`，说明合适正则下 rank1600 子空间本身可很好覆盖当前 fixed solve；旧动态 λ 才放大尾部差异。

### 7.4 fp32/fp64

同坐标统一 fp64 只使 frozen local-energy variance 改善 `2.42–3.60%`，不足以解释 4x 左右的 SPRING gap。fp64 对历史抵消与稳定性是合理数值防护，但不是主算法修复。

### 7.5 重尾

固定 λ 在 raw variance 最终无收益时，trim 0.01%–1% 后仍保留约 21–26% bulk variance 改善，说明极端尾样本掩盖了 bulk 改善。但 tail-direction audit 发现 tail leakage 很小（约 `.0064`），clipping 是主要且两个优化器共享的机制；加入单个 tail/complement 方向没有恢复 SPRING gap。

来源：

- `/scratch/dexuan1/runs/C_wssr_rank1600_lambda_crossover_audit_20260802/FINAL_CONCLUSION.md`
- `/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage4_x64_counterfactual/report.md`
- `/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage4_tail_audit/report.md`
- `/scratch/dexuan1/runs/C_wssr_rank1600_uniform_precision_audit_N100k_retry1/report.md`

---

## 8. 补空间 / multilevel complement 路线：已系统性否定

这些实验大多从已有 WSSR epoch100000 继续 5000 步，属于晚期修补，不能逆转早期 gap。主要结果：

- β=.05：训练 tail variance 明显恶化；
- β=.005：严重发散，variance 约 4.7；
- β=.001：比 hard control 更差，且随着继续训练 frozen variance 从 `.01539` 增到 `.01898`；
- complement state decay `rho=.9/.99`：早期偶有改善，但 101k–105k 没有持续优势；
- 绝对 cap `||delta_c|| <= .01 ||delta_r||` 也没有解决长期漂移；
- low-frequency/multilevel period20、period100：个别 checkpoint 下降，整体波动，末期没有稳定胜出；
- “先开 1000/2000 步然后关闭”也没有形成已验证的持续 frozen 改善；
- native-factor proximal E500：variance `0.01355075` vs hard `0.01350200`，ratio `1.00361`，完全无益；
- 六种补空间/谱候选（smooth、cluster、near-tail、force-aware、adaptive complement、projected iterative complement）在短测试中有些改善数值 fidelity，但没有得到能接近 SPRING 的 frozen 结果；最佳 speed–variance tradeoff 仍是 baseline。

hard / complement EMA / multilevel 101k–105k frozen variance 概况：

| arm | 101k | 102k | 103k | 104k | 105k |
|---|---:|---:|---:|---:|---:|
| hard | .01790 | .01418 | .01601 | .01604 | .01554 |
| β=.001, rho=.9 | .01401 | .01620 | .01669 | .01697 | .01793 |
| β=.001, rho=.99 | .01485 | .01489 | .01650 | .01719 | .01570 |
| period20 | .01549 | .01606 | .01425 | .01571 | .01723 |
| period100 | .01485 | .01567 | .02081 | .01662 | .01632 |

结论：问题不在“少了一点补空间”，而在主更新的时间结构。继续微调 β、rho、period 不是高概率主线。

---

## 9. 时间一致性、proximal 与 gate：能平滑，但不能改善目标

### 9.1 晚期 WSSR vs SPRING 方向审计

在各自晚期 checkpoint 的 30 batch fixed-parameter replay：

| 指标 | WSSR | SPRING |
|---|---:|---:|
| consecutive direction cosine | .2717 | .8838 |
| unit resultant length | .4819 | .9356 |
| raw direction SNR | .5482 | 2.6474 |
| mean pairwise cosine | .2057 | .8710 |
| current-batch residual ratio | .5664 | .8824 |

WSSR 对当前 batch 的 SR 方程拟合更好，SPRING 的方向序列却稳定得多。SPRING 方向和 `mu` 历史 anchor 的 median cosine 为 `.9162`。

这支持“SPRING 的历史残差纠错提高时间一致性”，但晚期 replay 不能单独证明它造成了完整 100k gap；gap curve 后来确认该审计做在了太晚的 regime。

### 9.2 普通平滑为何失败

- native proximal 即使机械提高 coherence，E500 frozen 无改善；
- unconditional blend 可把 consecutive cosine 提高约 `.22`，但 held-out 指标基本不变；
- temporal-consensus cosine 与下一 batch force/residual 相关性接近 0，冻结阈值还存在非平稳性；
- three-batch probe selector 的 nonzero 选择率合格，但 coherence 增益不足，probe 与真正下一 batch 的相关仅 `.09–.27`；
- reliability indicators 对未来 raw tail/variance 的最强绝对 Spearman 也只有约 `.142`，没有可靠预警能力。

因此“让最终方向更平滑”不是充分条件。必须像 SPRING 一样，用当前 batch 方程对历史方向进行投影/残差纠错。

---

## 10. 子空间因果审计：不是简单的 rank 边界旋转

rank1600 exact/native 30-batch 审计：

- native consecutive cosine `.2717`；
- exact rank1600 `.3491`；
- exact 只改善约 `.0686`，不足以接近 SPRING `.8838`；
- 相邻更新变化被归类为 **within-subspace coefficient variation**，而不是简单的 rank1600 边界 subspace rotation；
- 选中模式的谱 gap 极小：top32 中约 `98.85%` gap 小于 `1e-2`，约 `27%` 小于 `1e-3`；
- near-degeneracy 使 batch 改变时，稳定大子空间内的系数组合显著变化。

这解释了：同 batch 换 SSI key 完全稳定，但换 walker batch 方向 cosine 只有约 `.27`。它也解释为什么提高 SSI 精度或 exact SVD 只能小幅改善，而不是解决训练差距。

来源：`/scratch/dexuan1/runs/C_wssr_subspace_causal_audit_20260802_retry2/summary.md`

---

## 11. 梯度 EMA、gradient transport、拆分 eta 与 adaptive eta_S

### 11.1 transported-gradient 原型

实现过：

`h_t = eta_g [h_{t-1} - S_ema,prev delta_theta_prev] + (1-eta_g) g_t`

其中 `delta_theta_prev` 是 lr、clipping、norm constraint 后真实应用的参数变化，并增加了 `S_ema`/`S_current` transport norm 日志和开关。

### 11.2 eta_g=.3 / .95 stress

- eta_g=.3 时 `||S delta_theta|| / ||g||` 仅约 `0.5–1.2%`，transport 对训练几乎没有可辨识改善；
- eta_g=.95 时 EMA ratio 约 `.4–.5%`，current-S ratio 约 `1.0–1.2%`；`S_current` 约比 `S_ema` 大 2.4–2.8x，但修正仍很小；
- vanilla/transported 的 training loss 差异在噪声范围内，没有恢复性能。

### 11.3 拆分 eta_S / eta_g 的 ABCD

共同 rank1600、lr=.04、fixed λ=.001、E5000，仅看 training tail（未做 matched frozen）：

| arm | eta_S | eta_g | last100 training variance | 解释 |
|---|---:|---:|---:|---|
| A | .95 | 0 | ~.177 | 大 S lag 有害 |
| B | 0 | .95 | ~11.42 | gradient EMA 单独严重失稳 |
| C | .95 | .95 | ~.191 | 未恢复 |
| D | 0 | 0 | ~.069 | 该网格最好 |

这证明在该早期 fixed-λ/lr=.04 regime 中，大 gradient averaging 很危险；但它不能回溯性解释所有旧 eta=.8/lr=.02 结果，因为协议与 RHS 实现阶段不同。

### 11.4 adaptive eta_S

保持 `eta_g=0`：

- constant eta_S=.95：last100 variance 约 `.1773`；
- linear warmup T=1000：约 `.1447`；
- exponential tau=1000：约 `.1129`；
- eta_S=0：约 `.0692`，仍最好。

adaptive schedule 缓解 lag，但没有超过不用 S averaging 的 early control，更没有接近 SPRING。

### 11.5 exponential eta_g=.2 + transport

正式 light frozen、两个 seed：

| arm | seed0 variance | seed1 variance |
|---|---:|---:|
| expG=.2 vanilla | .44961653 | .31772596 |
| expG=.2 transported | .89192075 | .53050325 |

transported/vanilla 几何比 `1.81995`，明确更差。额外 audit 表明现有 S-action transport 的幅度相对理想变化太小，简单一阶 correction 没有救回 gradient EMA。

结论：默认应保持 `eta_g=0`。gradient EMA/transport 不是当前追平 SPRING 的主线。

---

## 12. 晚期从 SPRING checkpoint 切到 WSSR：一个重要但有限的阳性结果

从同一个 SPRING epoch50k checkpoint 分叉，再跑 5k：

| arm | frozen variance | / SPRING55k |
|---|---:|---:|
| SPRING 续到55k | .00538494 | 1.000x |
| WSSR eta=.3 | **.00379218** | .704x |
| WSSR eta=.8 | .01514191 | 2.812x |
| WSSR eta=.95 | .00424419 | .788x |

这说明：

- WSSR 并非在好 basin 中没有优化能力；eta=.3/.95 可以在 SPRING 已经训练好的晚期状态上继续精修，甚至优于同长度 SPRING continuation；
- 主缺陷高度集中在 KFAC-pre1000 后的早期阶段；
- 这不是 standalone WSSR 超过 SPRING，因为起点已经由 SPRING 训练 50k。

来源：`/scratch/dexuan1/runs/C_late_spring50k_eta_drift_20260803/report.md`

---

## 13. rank400、N2 与基础设施：记录保留，但不应再抢主线

### 13.1 rank400

最近完成的 rank400 `eta_S` exponential→.95、`eta_g=0`、fixed λ=.001、lr=.04、exact-first=False、20k：

- frozen energy `-37.83287886`；
- raw variance `1.20936736`；
- 0.1% winsorized variance `.56638`。

远差于 rank1600 和 SPRING。

截至本文件写入时，另一个 rank400 `exact-first=True + 旧正则逻辑 + lr=.02` 的 20k job `52739626` 仍在运行（约 15.6k/20k），eval job `52739627` pending。它没有最终科学结果，而且不回答 rank1600 超过 SPRING 的主问题。

### 13.2 N2

N2 工作主要用于论文复现与规模验证，不直接决定 C rank1600 主目标。

R=2.016 Bohr、KFAC-pre5000 matched formal frozen：

| 方法 | energy | variance |
|---|---:|---:|
| SPRING mu=.95, lr=.002 | -109.52480719 | .34756254 |
| KFAC lr=.05 | -109.52429111 | .44348302 |
| MinSR lr=.02 | -109.52037575 | .68339028 |

WSSR N2 的 rank400/800 早期/50k 结果 variance 约 `1.48–2.49`，在相应实验中未追上 SPRING/KFAC。R=2.068 是我们早期使用的几何，但论文 equilibrium 是 R=2.016；二者不可混写。

### 13.3 非科学失败

下列只属于环境/API/脚本失败，不应解释成优化器发散：

- paper-era `jaxlib 0.4.14+cuda12.cudnn89` 在当前集群首次 GPU 调用报 `DNN library initialization failed`；
- paper-era VMCNet 对当前 `kfac_jax` 缺少 `pmap_axis_name` 参数；
- diagnostics 曾因 `exact_first_force` 接口不一致报 unexpected keyword；
- 多个 audit pilot 曾 OOM、preempt、collector 路径断言失败；修复/重试后才有科学结果；
- Slurm 维护、坏节点或 dependency pending 不构成算法证据。

---

## 14. 对话与研究过程中的主要偏航/错误

这部分不是科学结果，但必须记录，防止再次重复。

1. **把“rank1600 超过 MinSR”误当成接近成功。** 用户目标一直是超过 SPRING。以后汇报首先给 WSSR/SPRING ratio，而不是只强调超过 MinSR。
2. **过度聚焦晚期 checkpoint。** fixed λ、补空间、tail、proximal、history replay 大量锚定 100k；后来 gap curve 才证明败局在前 5k 已形成。晚期阴性不能排除早期机制。
3. **过度信任离线 held-out residual。** fixed λ 改善 67–90%、long-memory direction gate 通过，训练却不改善或恶化。离线指标不能单独否决/晋级；E5000 frozen 才是验证过的筛选终点。
4. **重复提出已完成的 fixed-λ 训练。** 该实验已经做过，收益到 epoch102000 消失。不得再包装成新建议。
5. **把普通 momentum/EMA 与 SPRING recurrence 混为一谈。** naive recurrence 和 gradient EMA 的失败证明，关键不是平均方向，而是用当前 batch residual 投影纠错历史方向。
6. **把 eta=.99 当成 SPRING mu=.99 的直接对应。** 二者存储的信息位置和压缩方式不同；事实结果一个有效，一个早期恶化 6.58x。
7. **过早说“所有原因都排除了”。** 实际只排除了若干特定实现：补空间不足、单步 Jacobian staleness、SSI key、纯 fp32、普通 proximal、简单 transport 等。solution recurrence 的阳性结果反而说明结构性原因尚未完全解决。
8. **使用“sketched-SPRING”名称时不检查有效秩。** 在 1000 walkers、eta=0、rank1600 下，当前 batch 没有被 rank 截断；不能宣称 2.5x/低秩压缩贡献。
9. **让 rank400/N2/集群故障占据主线。** 这些可以并行保留，但不能改变 C rank1600 > SPRING 的判定标准。
10. **不断产生新 gate，却不先验证 gate 与 frozen variance 的关系。** 多个 temporal/reliability/probe gate 已显示预测力很弱。后续优先做直接 E5000 frozen 因果测试。

---

## 15. 已检验假设总表

| 假设 / 修改 | 直接证据 | 判定 | 对主目标的状态 |
|---|---|---|---|
| 增大 rank | 400→800→1600 variance 大降 | 有效 | 必要但不充分 |
| rank1600 eta/lr 调参 | eta=.3/lr=.04 优于 official | 有效 | 仍落后 3.93x |
| 补空间缺失是主因 | β、EMA、periodic、short-on、tail basis | 否定 | 关闭 |
| 动态 λ 太小 | crossover 改善 residual 67–90% | 真实局部病理 | 训练收益不持续，关闭为主修复 |
| fp32 是主因 | uniform fp64 改善 2–4% | 否定为主因 | 仅数值防护 |
| SSI 随机性/迭代不足 | same-batch cos≈1，warm4 很小 | 否定 | 关闭 |
| rank1600 hard truncation 是主因 | fixed-λ x64 rank1600/full cos≈.9999 | 否定为主要 late fixed-solve 原因 | 不是主线 |
| 一步旧 Jacobian 陈旧 | action cos .9999 | 否定 | 关闭 |
| 多 batch 同参数有益 | 5–21% residual gain | 局部支持 | 未转化为训练胜利 |
| 大 eta 长记忆能直接救 WSSR | eta=.99 E5k 恶化 6.58x | 否定 | 关闭简单 EMA |
| 普通 proximal/平滑方向 | E500 无改善，gate 弱 | 否定 | 关闭 |
| temporal indicator/gate | 相关性弱、非平稳 | 否定所测指标 | 关闭继续造 gate |
| gradient staleness transport | correction 小或使 frozen 更差 | 否定当前实现 | 关闭 |
| early gap | 5k 已形成大部分最终 gap | 强支持 | 所有新测试必须前移 |
| SPRING 式 residual correction | 两 seed E5k 大幅改善，seed1 略胜 | **强阳性** | 当前唯一主线候选 |
| residual correction 在真实 rank 压缩下仍有效 | rank800 geomean 仅比 rank1600 差 5.4%，速度约快 2.8x | 支持压缩兼容性 | 仍比等时间 SPRING 差 1.93x，不构成最终方法 |
| WSSR 在好 basin 可工作 | SPRING50k→WSSR5k 可优于 continuation | 支持 | 缺陷集中早期 |

---

## 16. 真正压缩的 residual-recurrence Pareto 实验

实验目录：`experiments/C_compressed_residual_recurrence_E5000_20260803/`；结果目录：`/scratch/dexuan1/runs/C_compressed_residual_recurrence_E5000_20260803/`。

共同配置：KFAC-pre1000、1000 walkers、E5000、`reburn=False`、residual recurrence `mu=.99`、`eta_S=eta_g=0`、fixed λ=`.001`、lr=`.04`、SSI40/2。rank400/800 都小于 centered current-batch 最大秩 999，因此是真正压缩。

| 方法 | seed | frozen variance | 稳态训练时间 | 相对等时间 SPRING-8k |
|---|---:|---:|---:|---:|
| rank400 recurrence | 0 | 0.10475589 | 602.0 s | 3.992x |
| rank800 recurrence | 0 | 0.04922199 | 654.3 s | 1.876x |
| rank800 recurrence | 1 | 0.05219800 | 648.4 s | 1.989x |
| SPRING replicate2 epoch8000 | - | 0.02624066 | 约 640 s（checkpoint 1k→8k） | 1.000x |

rank800 两 seed几何均值为 `0.05068816`，是等时间 SPRING 的 `1.9317x`。因此：

- 低秩压缩没有摧毁 residual recurrence：rank800 仅比 rank1600 recurrence 的两 seed几何均值差 `5.36%`，而每步快约 `2.8x`；
- 但 rank800 仍比原生 SPRING 更慢（约 `.130 s/step` 对早期约 `.091 s/step`），同时 frozen variance 更差；
- 所以当前 compressed recurrence 不形成 accuracy-time Pareto 优势，不准入 100k；
- “前期复刻 SPRING、后期再上更贵 WSSR”的 hybrid 也不再作为主线，因为它即使降 variance，也不满足成本目标。

## 17. 下一步应怎样围绕唯一目标收敛

在 1000-walker C 协议下，当前证据已经排除了“直接把 residual recurrence 压到 rank400/800 就能超过 SPRING”。若继续，必须同时满足两个量化目标：

1. rank800 E5000 frozen variance 至少从 `0.05069` 降到低于等时间 SPRING 的 `0.02624`，即还需约 `48%` 改善；
2. 每步成本至少从约 `.130 s` 降到不高于 SPRING 早期约 `.091 s`，即还需约 `30%` 加速。

因此不应直接提交新的 100k。只有两类工作仍有科学意义：

- 优化 rank800 求解实现并同时找到能显著改善早期方向质量的机制；单独加速或单独降低 variance 都不够；
- 改到 `N_walkers > rank` 且 SPRING Gram solve 真正成为瓶颈的规模（例如更多 walkers/更大体系），再做等 GPU-hours 对比。此时 rank1600 才可能具有真实低秩计算意义。

任何后续候选仍先走 E5000 frozen + 等训练时间门槛；未同时优于 `0.02624` 和 SPRING wall time，不准入 100k。

---

## 18. 关键证据索引

### 主基线与差距

- `experiments/C_spring_official_reproduction/COMPARISON_BASELINE.md`
- `/scratch/dexuan1/runs/C_optimizer_gap_curve_light_20260802/report.md`
- `/scratch/dexuan1/runs/C_wssr_official_matched_E100000/`
- `/scratch/dexuan1/runs/C_wssr_rank1600_eta03_lr004_E100000_resume1/`
- `/scratch/dexuan1/runs/C_minsr_paper_kfac1000_E100000/`

### 强阳性主线

- `experiments/C_rank1600_solution_recurrence_E5000_20260802/README.md`
- `/scratch/dexuan1/runs/C_rank1600_solution_recurrence_E5000_20260802/summary.json`
- `/scratch/dexuan1/runs/C_compressed_residual_recurrence_E5000_20260803/report.md`
- `/scratch/dexuan1/runs/C_compressed_residual_recurrence_E5000_20260803/summary.json`

### 早期历史与 gap 定位

- `/scratch/dexuan1/runs/C_wssr_rank1600_early_multibatch_audit_pre1000_20260802/FINAL_DECISION.md`
- `/scratch/dexuan1/runs/C_rank1600_early_memory_scaling_fixed_x64_retry1_20260802/report.md`
- `/scratch/dexuan1/runs/C_rank1600_eta_memory_E5000_20260802/report.md`
- `/scratch/dexuan1/runs/C_late_spring50k_eta_drift_20260803/report.md`

### λ / precision / tail

- `/scratch/dexuan1/runs/C_wssr_rank1600_lambda_crossover_audit_20260802/FINAL_CONCLUSION.md`
- `/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage4_x64_counterfactual/report.md`
- `/scratch/dexuan1/runs/C_wssr_rank1600_fixed_lambda_stage4_tail_audit/report.md`
- `/scratch/dexuan1/runs/C_wssr_rank1600_uniform_precision_audit_N100k_retry1/report.md`
- `/scratch/dexuan1/runs/C_wssr_rank1600_tail_direction_audit_20b/report.md`

### 时间一致性与子空间

- `/scratch/dexuan1/runs/C_wssr_spring_mechanism_conclusion_20260802/final_report.md`
- `/scratch/dexuan1/runs/C_wssr_subspace_causal_audit_20260802_retry2/summary.md`
- `/scratch/dexuan1/runs/C_wssr_temporal_consistency_replay_20260802/summary.md`
- `/scratch/dexuan1/runs/C_wssr_probe_selector_replay_20260802/summary.md`
- `/scratch/dexuan1/runs/C_wssr_rank1600_native_proximal_gamma3e4_matched_E500_20260802_results/report.md`

### complement / multilevel

- `/scratch/dexuan1/runs/C_wssr_rank1600_hard_matched_control_frozen/`
- `/scratch/dexuan1/runs/C_wssr_rank1600_beta0001_frozen_audit/`
- `/scratch/dexuan1/runs/C_wssr_rank1600_beta0001_compema_frozen/`
- `/scratch/dexuan1/runs/C_wssr_rank1600_multilevel_comp_frozen/`
- `experiments/C_wssr_six_improvements/results/final_report.md`

### gradient split / transport / adaptive S

- `/scratch/dexuan1/runs/C_gradient_transport_E5000_20260803/comparison/summary.json`
- `/scratch/dexuan1/runs/C_gradient_transport_eta095_E5000_20260803/comparison/summary.json`
- `/scratch/dexuan1/runs/C_wssr_gradient_transport_etaG02_20260803/summary.json`
- `experiments/C_wssr_split_eta_E5000_20260803/README.md`
- `experiments/C_wssr_adaptive_etaS_E5000_20260803/README.md`

---

## 19. 一句话交接

如果之后换人继续这个项目，第一句话必须是：

> 传统 rank1600 WSSR 已超过 MinSR、但仍落后 SPRING；完整历史方向加 current-batch residual correction 是唯一强阳性，但真正压缩到 rank800 后两 seed E5k frozen 仍比等时间 SPRING 差 1.93x。后续目标必须是 accuracy-time Pareto 优势，不能再用更贵的 hybrid 只刷更低 variance。

---

## 20. Adaptive S-mixing indicator 的最终阴性结论（2026-08-07）

为回答“能否用一个在线指示子决定何时混入历史 S”，新增了只读、同 checkpoint 的方向审计。比较 current-only 方向 `d0` 与历史-S 方向 `dh`，两者都经过真实 learning-rate / norm-constraint 缩放；主指标 `G` 是它们在四个后续 batch 上的配对 predicted-descent 相对增益，辅指标 `R` 是更新相关 probe 空间内的 operator drift / sampling noise 比。

结果否定了这两个指标的决策能力：

- 静态固定参数正对照的 `G` 四次全部为负，中位数 `-0.1712`；
- 已知 direct E5000 frozen 表现有害的 early `eta_S=.95` 轨迹，在 epoch1000/3000/5000 的 `G` 中位数反而为 `+0.2624/+0.1161/+0.8485`；
- `R` 中位数为 `.920/1.049/.649`，只会拒绝三个坏点中的一个；
- baseline future dot 全部为正，故不是分母符号或 non-finite 伪影。

因此按预注册停止规则，不实现 adaptive gate，不启动新的 E5000。结论不是“S 永远不能平均”，而是：**当前已经测试的局部指标（held-out residual、transport ratio、candidate cosine、未来 batch direction gain、noise/drift ratio）都不能可靠预测前 5000 步反复混合 S 对 frozen variance 的累积影响。** 若以后再研究 S mixing，判决必须直接交给 paired E5000 frozen variance，不能继续用这些离线量单独放行。

完整报告：`experiments/C_adaptive_S_indicator_20260807/conclusion.md`。

---

## 21. 8 月 7 日新增结果及对旧结论的修正

### 21.1 C stage-wise switch 与 fresh-SPRING control

从同一条 SPRING 轨迹分别在 epoch 5000、20000、50000 切换到 WSSR，继续 5000 步的 light frozen 结果为：

| switch point | WSSR variance | SPRING continuation variance | WSSR/SPRING |
|---:|---:|---:|---:|
| 5000 | .02339315 | .01500384 | 1.5591 |
| 20000 | .00749927 | .00703069 | 1.0666 |
| 50000 | .00379218 | .00538494 | .7042 |

这条曲线仍支持“WSSR 的主要缺陷发生在早期”；但是 epoch50000 的优势原来只有一个 frozen seed，不能单独证明 WSSR 后期优于 SPRING。为检验 optimizer-state reset，新增了 fresh SPRING：从完全相同的 SPRING epoch50000 参数、walkers 和 key 出发，使用与 native continuation 相同的全局连续晚期学习率，但把 SPRING 历史状态清零，再跑 5000 步。

| eval seed | native SPRING variance | fresh SPRING variance | WSSR eta=.3 variance |
|---:|---:|---:|---:|
| 0 | .00538494 | .00418972 | .00379218 |
| 1 | .00398996 | .00428637 | .00488977 |
| arithmetic mean | .00468745 | **.00423805** | .00434098 |

结果的排序跨 seed 翻转。fresh SPRING 的两-seed 平均 variance 比 native continuation 低 9.6%，并比 WSSR 低 2.4%；但 2.4% 不足以建立稳健排序。正确修正是：

- 旧的“seed0 下 WSSR 后期明显胜出”不能提升为方法结论；
- 重置 SPRING 历史状态本身足以解释平均 variance 改善，且不是单纯学习率效应，因为 native/fresh 使用相同晚期学习率；
- stage-wise hybrid 不能作为 WSSR 超过 SPRING 的证据，也不满足原定 accuracy-time Pareto 主目标；
- 如果需要关闭这个次级争议，应对三个现成 checkpoint 做正式 2000 walkers / 20000 measurements frozen evaluation，而不是重新训练。

来源：

- `experiments/C_stagewise_switch_scan_20260807/REPORT.md`
- `/scratch/dexuan1/runs/C_fresh_spring50k_control_20260807/REPORT.md`

### 21.2 full-current Galerkin residual P0

rank800 full-current residual 相对旧 rank-coordinate residual 在两个 seed 上分别改善 6.80% 和 13.65%，几何平均改善 10.29%。E5000 几何平均 variance 从 `.05068816` 降至 `.04547146`，但仍比 SPRING E5000 `.03784061` 高 20.17%，且单步 `.175 s`，约比 SPRING `.091 s` 慢 92%。因此按预注册 15% 门槛停止，P1/P2/P3 未提交。

遥测显示 correctable ratio 从前 500 步约 1.00 降至末 100 步约 `.56-.58`。这说明 rank800 的带宽瓶颈主要在 E5000 后段出现，error feedback 具有明确的机制靶点；但它目前只是未验证候选，不能覆盖 P0 已经失败的 Pareto 判定。

来源：

- `experiments/C_galerkin_residual_recurrence_20260803/P0_REPORT.md`
- `experiments/C_zero_cost_triage_20260804/task1_telemetry/REPORT.md`

### 21.3 delayed-subspace 与大 N 成本试点

delayed SSI refresh period 2/5 没有加速，反而相对每步 refresh 分别降速到 `.730x/.653x`；复用子空间的分支本身更慢，epoch200 online variance 也从 `.1163` 恶化到 `.3450/.9183`。该路线同时因速度和稳定性被拒绝。

pre-W-cache 大 N H100 计时中：

- N=4096：SPRING `.2122 s/step`，rank800 recurrence `.3588 s/step`，后者慢 `1.691x`；rank1600 OOM；
- N=8192：SPRING 能运行到 80 步，约 `.85 s/step` 的非正式观测；rank800/1600 recurrence 均 OOM；
- 没有在已测范围内观察到 WSSR/recurrence 的成本交叉点。

显式 `W=OU` 乘法在 N=4096/rank800 profiler 中约 23 ms，而总步时约 359 ms；即使完全消除该乘法，理论上也不足以扭转 1.69x 的总体成本差。W-cache 等价性测试已经存在，但还没有形成一个能改变 Pareto 结论的 post-cache timing 结果。

来源：

- `experiments/C_delayed_subspace_timing_20260807/REPORT.md`
- `experiments/C_zero_cost_triage_20260804/task2_largeN_timing/REPORT.md`

### 21.4 N2 新结果

正式 N2 R=2.016 rank1400 WSSR 100k frozen：energy `-109.51225494`，variance `.97715459`，明显差于 SPRING `.34756254`、KFAC `.44348302` 和 MinSR `.68339028`。

N2 E5000 full-residual screen 中：

- rank800, eta_S=0：variance `.96195026`；
- rank800, eta_S=.2：variance `1.21904603`，历史 S 使 variance 恶化 26.7%；
- rank1000, eta_S=0：variance `.94808032`，但 1000 walkers 中它是 SPRING-equivalent regression anchor，不是 WSSR 压缩结果；
- rank1000, eta_S=.2：variance `1.12949205`，energy/variance 都恶化。

因此在 N2 早期，固定 S-history mixing 同样没有修复截断损失。

来源：`experiments/N2_wssr_memory_precision_gate_20260807/E5000_RESULTS.md`。

### 21.5 H2O 当前状态

同一 KFAC-pre5000 起点的 H2O 比较截至本次更新：

- SPRING mu=.95/lr=.002：100k + frozen 完成，energy `-76.43252873`，variance `.15757659`；
- MinSR seed0：epoch28082 NaN；
- MinSR seed1 独立 PRNG 重试：epoch28650 NaN；两条独立 sample path 都在约 28k 失败，说明该配置存在可复现的稳定性问题；
- KFAC lr=.05：仍在运行，已产生 epoch45000 checkpoint，尚无 frozen 结果。

来源：`experiments/H2O_N2matched_KFAC5000_SPRING_MinSR_E100000_20260807/REPORT.md`。

---

## 22. 历史 S 的一阶 curvature-transport 审计（2026-08-08）

在 C early fixed-lambda WSSR epoch1000 checkpoint 上做了 8 个只读 one-step
replicate。使用真实 post-lr/post-norm-constraint `delta_theta`，并用中点有限差分

`S_transport v = 2 S(theta + delta/2) v - S(theta) v`

作为对一阶输运非常有利的 oracle 近似。结果：

- 未输运同坐标 action error 中位 `.0129641`；
- 输运后 `.00168660`，为原误差的 `.128475`，8/8 replicate 改善；
- 但换 walker batch 的 sampling action error 中位 `.0626408`；
- sampling / parameter-staleness error 中位 `4.66512x`；
- 面向真实下一 batch 的总 action error 只中位改善 `2.2517%`。

因此门控失败：curvature transport 在局部数学上有效，但它修正的参数陈旧性不是
当前 WSSR 的主导误差源；sampling/operator noise 占主导且总 action 收益很小，
不据此进入 E5000。完整报告：
`experiments/C_curvature_transport_audit_20260808/REPORT.md`。
