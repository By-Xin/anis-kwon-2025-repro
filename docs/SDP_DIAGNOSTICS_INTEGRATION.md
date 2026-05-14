# SDP/SOCP/Big-M 补充诊断实验接入说明

这组实验已经作为补充模块接入本仓库，不替换主复现实验。

## 目录

```text
src/sdp_relax_diag/                       # 补充实验 Python 包
scripts/sdp_diagnostics/                  # 补充实验入口脚本
configs/sdp_diagnostics/smoke.yaml        # 合成数据 smoke test
configs/sdp_diagnostics/paper50.yaml      # 论文 50 资产诊断配置
docs/sdp_relaxation_diagnostics/          # 原 bundle 文档
```

补充实验共享主仓库环境和真实数据：

```text
data/prices.csv
data/factors_daily.csv
```

输出统一写到：

```text
results/sdp_diagnostics/
```

## 环境

继续使用主项目环境：

```bash
conda activate anis-kwon-e2e
pip install -e .
```

当前 WSL 环境中已经具备主诊断需要的依赖：`cvxpy`、`cvxpylayers`、`torch`、
`scs`、`gurobipy`、`matplotlib`。项目依赖现在接受 SCS `3.2.x`，因为本机可安装
版本是 `3.2.2`，而论文报告版本是 `3.2.1`。

## Smoke Test

这一步只使用合成 N=12 数据，不会覆盖主实验真实数据。

```bash
python scripts/sdp_diagnostics/make_synthetic_data.py
python scripts/sdp_diagnostics/validate_data.py --config configs/sdp_diagnostics/smoke.yaml
python scripts/sdp_diagnostics/run_relaxation_quality.py \
  --config configs/sdp_diagnostics/smoke.yaml \
  --methods exact bigm \
  --max-windows 1
python scripts/sdp_diagnostics/aggregate_results.py --results-dir results/sdp_diagnostics/smoke
```

## Paper-50 诊断入口

先检查共享数据：

```bash
python scripts/sdp_diagnostics/validate_data.py --config configs/sdp_diagnostics/paper50.yaml
```

核心 relaxation quality：

```bash
python scripts/sdp_diagnostics/run_relaxation_quality.py \
  --config configs/sdp_diagnostics/paper50.yaml \
  --methods exact bigm socp sdp \
  --max-windows 24
```

梯度对齐诊断建议先用小切片：

```bash
python scripts/sdp_diagnostics/run_gradient_alignment.py \
  --config configs/sdp_diagnostics/paper50.yaml \
  --methods bigm socp sdp \
  --cardinalities 10 \
  --max-windows 1 \
  --n-bootstrap 8 \
  --n-directions 4
```

## 注意

N=50 的 exact MIQP 和 finite-difference gradient alignment 会频繁调用 Gurobi。
在当前 WSL/sandbox 环境里，涉及 Gurobi license 网络校验的命令需要用提升权限运行。
SDP 相关实验仍应按长任务处理，优先跑小窗口、小 bootstrap、小 cardinality 切片。
如果 `relaxation_quality.csv` 中的 `bound_violation` 明显大于 0，优先用
`--override solver.cvx_eps=1e-6 --override solver.cvx_max_iters=20000` 重跑对应切片，
避免把 SCS 精度误差误读成 relaxation 本身的问题。
