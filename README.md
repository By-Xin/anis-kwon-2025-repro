# Anis & Kwon (2025) 复现实验 bundle

这是用于复现论文 **End-to-end, decision-based, cardinality-constrained portfolio optimization** 的工程化脚本包。

论文目标：比较 end-to-end decision-based learning 在 cardinality-constrained minimum-variance portfolio optimization 中相对两个 decoupled baseline 的表现。核心 E2E 方法把 Big-M、SOCP、SDP 三种连续松弛作为可微优化层，用 realized Sharpe ratio 作为外层 decision-based loss。

## 目录结构

```text
anis_kwon_2025_repro/
  configs/                  # 复现实验配置
  data/                     # 你放入 prices.csv 和 factors_daily.csv
  docs/                     # 数据申请单、算力需求、复现说明
  scripts/                  # 一键检查/运行脚本
  src/e2e_cardinality_portfolio/
                            # 核心 Python 包
  results/                  # 运行输出
```

## 1. 安装

推荐使用 conda：

```bash
conda env create -f environment.yml
conda activate anis-kwon-e2e
pip install -e .
```

Gurobi 是论文测试时求解 Big-M MIQP 的关键依赖；没有 Gurobi license 时只能跑非忠实 smoke test。

## 2. 准备数据

请按 `docs/DATA_REQUEST.md` 准备：

```text
data/prices.csv
data/factors_daily.csv
```

最重要的要求：

- `prices.csv`：50 个论文 ticker 的日度 adjusted close，2010-01-01 到 2021-12-31。
- `factors_daily.csv`：Fama-French 5 因子的日收益：`Mkt-RF, SMB, HML, RMW, CMA`。

检查：

```bash
python scripts/check_data.py --config configs/reproduce_2015_2020.yaml
```

## 3. 快速 smoke test

这个命令会生成合成数据并跑 `nominal + linreg`，用于确认工程可以启动。它不是论文复现结果。

```bash
python scripts/run_smoke_test.py
```

## 4. 渐进式复现

先跑一个小切片：

```bash
python scripts/run_reproduction.py \
  --config configs/reproduce_2015_2020.yaml \
  --methods nominal linreg \
  --cardinalities 10 \
  --max-rebalances 1 \
  --verbose
```

跑 E2E_M 的一个小切片：

```bash
python scripts/run_reproduction.py \
  --config configs/reproduce_2015_2020.yaml \
  --methods e2e_m \
  --cardinalities 10 \
  --max-rebalances 1 \
  --verbose
```

## 5. 全量复现

默认口径采用 2015--2020 out-of-sample，因为论文 Tables 3--4 的标题是 2015--2020，且文中 in-sample 表格报告 24 个季度；同时 bundle 保留 2015--2021 配置用于正文口径。

```bash
python scripts/run_reproduction.py --config configs/reproduce_2015_2020.yaml --verbose
```

结果输出：

```text
results/reproduce_2015_2020/metrics.csv
results/reproduce_2015_2020/weights.csv
results/reproduce_2015_2020/daily_returns.csv
results/reproduce_2015_2020/run_log.csv
results/reproduce_2015_2020/models/
```

## 6. 重要配置项

`configs/reproduce_2015_2020.yaml`：

```yaml
bootstrap:
  n_samples: 2000
  block_size: 20
train:
  epochs: 4
  batch_size: 20       # 论文未报告；默认 20，可改
  lr: 0.01
  final_epoch_lr: 0.001
solver:
  test_solver: GUROBI
```

## 7. 常见问题

### 没有 Gurobi 怎么办？

可以把 config 中的：

```yaml
solver:
  allow_heuristic_without_gurobi: true
```

这样脚本会用 top-k heuristic 做 smoke/debug。但论文测试时使用 Gurobi v9 求 exact Big-M formulation，因此正式复现不要用 heuristic。

### 为什么 SDP 非常慢？

论文报告 E2E_SDP 单个 2000 样本 epoch 超过 6 小时，而且使用的是 30 个 compute-optimized vCPU。建议把 SDP 单独排队，并按 method × k × rebalance date 拆成并行任务。

### batch size 为什么是 20？

论文 Algorithm 2 列出 batch size，但没有给出数值。bundle 默认 20，是工程上稳定的默认值；请在敏感性实验中记录它。
