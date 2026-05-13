# 复现工作记录

## 2026-05-13

- 从 `C:/Users/xinby/Downloads/anis_kwon_2025_repro_bundle.zip` 解包到当前工作区。
- 创建本地 Git 仓库并推送到 GitHub 私有仓库：`https://github.com/By-Xin/anis-kwon-2025-repro`。
- 初始 bundle 已提交：`d02b6ca Add initial Anis Kwon reproduction bundle`。
- 增加 `.gitattributes`，固定文本文件使用 LF，避免 Windows/WSL 混用时产生行尾噪声。
- 修改 `scripts/run_smoke_test.py`：如果 `data/smoke/prices.csv` 和 `data/smoke/factors_daily.csv` 已存在，则复用 fixture，不再每次运行时重写 CSV。

## 环境检查

- Windows base Python 是 3.12.7，不满足项目声明的 `>=3.10,<3.12`。
- WSL Ubuntu 当前只有 Python 3.12.3，未发现 conda/mamba/micromamba 或 Python 3.10/3.11。
- 已创建 Windows conda 环境 `anis-kwon-e2e`，Python 3.10.20。
- `scs==3.2.1` 在当前 Windows/Python 3.10 pip 索引中不可用；为本机验证安装了 `scs==3.2.2`。这不是论文严格版本，需要在正式复现报告中标注。
- 已安装 `cvxpy==1.4.4`、`cvxpylayers==0.1.6`、`gurobipy==13.0.2`、`torch==2.10.0`。
- 本机检测到 Gurobi restricted license：non-production use，expires 2027-11-29。
- `cvxpy` 启动时会报告 DIFFCP solver import warning，但 `diffcp` 本身可 import，三类 CVXPYLayer 的小维度构建均通过。

## 已通过验证

- Base 环境单元测试：`2 passed`。
- `anis-kwon-e2e` 环境单元测试：`2 passed`。
- `scripts/run_smoke_test.py` 在 `anis-kwon-e2e` 中跑通，`nominal` 和 `linreg` 的测试 MIQP 状态均为 `optimal`。
- Synthetic `e2e_m` 小切片跑通：1 epoch，20 bootstrap samples，测试 MIQP 状态 `optimal`。
- Synthetic `e2e_socp` 小切片跑通：1 epoch，20 bootstrap samples，测试 MIQP 状态 `optimal`。
- `e2e_sdp` 已做小维度 layer 编译检查；未在 N=50 smoke 上完整训练，预计耗时明显更高。

## 下一步

- 准备真实 `data/prices.csv` 和 `data/factors_daily.csv`，再运行 `scripts/check_data.py`。
- 用真实数据先跑 `nominal`/`linreg` 的 `k=10`、`max_rebalances=1` 小切片。
- 再跑真实数据 `e2e_m` 小切片，确认训练时间、solver warning 和 Gurobi 状态。
- 正式跑全量前，需要决定是否接受 Windows 上 `scs==3.2.2` 的偏差，或另建 Linux/Python 3.10 环境以追求 `scs==3.2.1`。
