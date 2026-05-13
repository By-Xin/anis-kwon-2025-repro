# 算力与软件需求

## 1. 软件环境

论文报告的核心环境：Python、CVXPYLayers 的 PyTorch 接口、SCS v3.2.1；测试时用 Gurobi v9 求 Big-M MIQP。复现实验建议：

- Python 3.10 或 3.11
- PyTorch 2.x
- CVXPY + CVXPYLayers
- SCS == 3.2.1
- Gurobi + 有效 license（强烈建议；否则只能跑非忠实 heuristic smoke test）

安装：

```bash
conda env create -f environment.yml
conda activate anis-kwon-e2e
pip install -e .
```

Windows 注意：截至 2026-05-13，在本机 Windows/Python 3.10 环境中，`scs==3.2.1` 没有可用 pip wheel；本机 smoke/E2E 小切片使用 `scs==3.2.2` 跑通。论文严格复现应优先使用能安装 `scs==3.2.1` 的 Linux/Python 3.10 环境，或者在结果报告中明确记录 SCS 版本偏差。

## 2. 推荐硬件

### 最小可用配置

用于 `nominal`、`linreg`、`e2e_m` 小规模调试：

- 8 CPU cores
- 16--32 GB RAM
- 无需 GPU

### 论文级 E2E_M / E2E_SOCP

- 8--16 CPU cores
- 32 GB RAM
- 每个 `(method, k, rebalance_date)` 可独立并行

论文中一轮 2000 样本 epoch 的大致时间：E2E_M 约 15 秒，E2E_SOCP 约 1 分钟。

### 论文级 E2E_SDP

- 建议 30 vCPU 或以上
- 64 GB RAM 起步
- 强烈建议用 SLURM / 云服务器并行拆分 quarter × k

论文报告 E2E_SDP 单个 2000 样本 epoch 超过 6 小时，而且是在 30 个 compute-optimized vCPU 上运行；4 个 epoch、多个季度和多个 k 会非常贵。建议先完成 `nominal + linreg + e2e_m`，再扩展到 SOCP，最后单独安排 SDP。

## 3. 任务拆分建议

全量默认配置：

```bash
python scripts/run_reproduction.py --config configs/reproduce_2015_2020.yaml
```

调试/渐进式复现：

```bash
# 只跑第一个季度、k=10、两个 decoupled baseline
python scripts/run_reproduction.py --config configs/reproduce_2015_2020.yaml --methods nominal linreg --cardinalities 10 --max-rebalances 1

# 跑 E2E_M 的 k=10
python scripts/run_reproduction.py --config configs/reproduce_2015_2020.yaml --methods e2e_m --cardinalities 10

# 跑 SOCP / SDP 建议用 array job
bash scripts/slurm_run_array.sh
```
