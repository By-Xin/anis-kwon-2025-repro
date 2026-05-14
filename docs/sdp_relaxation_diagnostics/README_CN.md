# SDP / SOCP / Big-M 松弛为什么在 E2E cardinality portfolio 中不一定 work：诊断实验 bundle

这个 bundle 是为导师提出的核心问题设计的：

> 对 cardinality-constrained portfolio 这类整数/混合整数问题，到底有没有必要用 SDP/SOCP 这种更紧的连续松弛做端到端训练？为什么 Anis & Kwon (2025) 里 SDP 这样更紧的 relaxation 反而不如 Big-M？是他们没做好、SDP 本身不适合、还是这个问题 LP/Big-M 松弛已经够了？

它不是简单重复论文主实验，而是把问题拆成几组可验证假设，并提供可运行脚本：

1. **松弛质量实验**：SDP/SOCP 的 lower bound 是否真的更紧？更紧是否带来更好的 rounded/test-time integer portfolio？
2. **relaxation-to-integer mismatch 实验**：训练用连续 relaxation 的 `w_relax`，测试用 MIQP 的 `w_int`。二者支持集、权重、loss 是否对齐？
3. **梯度对齐实验**：连续松弛层产生的梯度方向，是否和“真实整数 forward loss”的有限差分方向一致？
4. **样本量/epoch scaling 实验**：SDP/SOCP 是否只是因为样本、epoch、solver tolerance 不够而 under-train？
5. **数值稳定性实验**：SCS 求解 SDP/SOCP 时的 residual、状态、NaN 梯度、梯度范数是否异常？
6. **直接整数 forward + relaxed backward 的 STE 实验**：测试“正向直接整数求解，反向用 relaxation 近似”是否优于纯 relaxation forward。

## 快速开始

### 1. 建环境

```bash
conda env create -f environment.yml
conda activate anis-kwon-e2e
pip install -e .
```

本补充实验已经接入主复现仓库，复用同一个 Python 环境、同一个
`data/prices.csv` 和 `data/factors_daily.csv`。若只想跑 smoke test，不安装
Gurobi 也可以用小规模 enumeration；SOCP/SDP 和梯度实验需要
`cvxpy + cvxpylayers + torch`。

### 2. 先用合成数据跑 smoke test

```bash
python scripts/sdp_diagnostics/make_synthetic_data.py --n-assets 12 --n-factors 5
python scripts/sdp_diagnostics/validate_data.py --config configs/sdp_diagnostics/smoke.yaml
python scripts/sdp_diagnostics/run_relaxation_quality.py --config configs/sdp_diagnostics/smoke.yaml --methods bigm exact --max-windows 1
python scripts/sdp_diagnostics/aggregate_results.py --results-dir results/sdp_diagnostics/smoke
```

如果你装了 CVXPY，可以加上 SOCP/SDP：

```bash
python scripts/sdp_diagnostics/run_relaxation_quality.py --config configs/sdp_diagnostics/smoke.yaml --methods bigm socp sdp exact --max-windows 1
```

如果你装了 CVXPYLayers，可以跑梯度对齐：

```bash
python scripts/sdp_diagnostics/run_gradient_alignment.py --config configs/sdp_diagnostics/smoke.yaml --methods bigm socp --max-windows 1 --n-bootstrap 8 --n-directions 8
```

### 3. 用论文数据跑诊断

把数据放到：

```text
data/prices.csv
data/factors_daily.csv
```

然后：

```bash
python scripts/sdp_diagnostics/validate_data.py --config configs/sdp_diagnostics/paper50.yaml

# 核心：relaxation lower bound / rounding / integer mismatch
python scripts/sdp_diagnostics/run_relaxation_quality.py \
  --config configs/sdp_diagnostics/paper50.yaml \
  --methods exact bigm socp sdp \
  --max-windows 24

# 核心：梯度是否和最终整数目标对齐。N=50 时建议有 Gurobi。
python scripts/sdp_diagnostics/run_gradient_alignment.py \
  --config configs/sdp_diagnostics/paper50.yaml \
  --methods bigm socp sdp \
  --max-windows 3 \
  --n-bootstrap 32 \
  --n-directions 16

# 样本量/epoch scaling，只建议先从 k=10 和 1-3 个窗口开始
python scripts/sdp_diagnostics/run_train_eval_sweep.py \
  --config configs/sdp_diagnostics/paper50.yaml \
  --methods bigm socp sdp \
  --sample-grid 128 512 2000 \
  --epoch-grid 1 4 16 \
  --max-windows 1

python scripts/sdp_diagnostics/aggregate_results.py --results-dir results/sdp_diagnostics/paper50
```

直接整数 forward + relaxed backward 的 STE 诊断：

```bash
python scripts/sdp_diagnostics/run_ste_integer_forward.py \
  --config configs/sdp_diagnostics/paper50.yaml \
  --methods bigm socp sdp \
  --cardinalities 10 \
  --n-samples 32 \
  --epochs 4 \
  --max-windows 1
```

## 输出

主要 CSV：

```text
results/sdp_diagnostics/<run>/relaxation_quality.csv
results/sdp_diagnostics/<run>/gradient_alignment.csv
results/sdp_diagnostics/<run>/train_eval_sweep.csv
results/sdp_diagnostics/<run>/summary_*.csv
results/sdp_diagnostics/<run>/fig_*.png
```

## 读结果的核心规则

### 结论 A：SDP 作为 relaxation 有效，但 E2E 不 work

你会看到：

- `sdp_gap_to_exact` 小于 `socp_gap_to_exact`，远小于 `bigm_gap_to_exact`；
- `bound_violation` 接近 0；如果它明显为正，先提高 SCS 精度再解释 gap；
- 但 `sdp_topk_overlap`、`sdp_rounded_sharpe`、`sdp_grad_fd_corr` 不好；
- 说明 SDP lower bound 紧，不等价于“对最终整数决策有好梯度”。

### 结论 B：LP/Big-M 确实足够

你会看到：

- Big-M 虽然 lower bound 很松，但 rounded/integer portfolio 的 realized Sharpe 不差；
- Big-M 梯度与 finite-difference integer loss 更一致；
- 随样本量/epoch 增加，SOCP/SDP 仍不追上。

### 结论 C：文章可能没给 SDP 足够训练资源

你会看到：

- SDP 在 `J=128/512` 不好，但 `J=2000/8000` 或 `epoch=16/32` 后改善明显；
- 梯度稳定性随 solver tolerance 提高改善；
- 则可以说原文的结论更像 sample/compute constrained，而不是 SDP 本质无效。

### 结论 D：是 solver/implicit differentiation 问题

你会看到：

- SDP/SOCP 的 solver status 差、residual 大、NaN/Inf 梯度多；
- 同一个 relaxation 用更精 solver/tolerance 后梯度对齐改善；
- 则问题在 CvxPyLayers + SCS 的数值链条，而不是数学 relaxation 本身。
