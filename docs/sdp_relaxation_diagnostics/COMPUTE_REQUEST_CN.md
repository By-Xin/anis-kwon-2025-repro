# 算力需求单

## Smoke test

- CPU: 4 cores
- RAM: 8 GB
- 数据：合成 N=12
- 求解：enumeration exact + Big-M；可选 CVXPY 的 SOCP/SDP

## 主诊断实验

### Relaxation quality

- N=50, k=10/15/20, 24 windows
- exact MIQP：建议 Gurobi
- SOCP/SDP：CVXPY + SCS/MOSEK/CLARABEL（可用哪个用哪个；可微训练仍建议 CVXPYLayers/SCS）
- CPU: 16 cores
- RAM: 32-64 GB

### Gradient alignment

- N=50 需要 Gurobi，否则 finite-difference exact forward 很慢；建议先 N=20 子集。
- CPU: 16-32 cores
- RAM: 64 GB

### Training sweep

- Big-M：8-16 cores 足够
- SOCP：16-32 cores
- SDP：建议 30+ compute-optimized vCPU，64-128 GB RAM
- 强烈建议把 method/k/window/J/epoch 拆成 array jobs。

## 软件

- Python 3.10/3.11
- numpy, pandas, scipy, pyyaml, matplotlib, tqdm
- cvxpy：SOCP/SDP relaxation quality
- torch + cvxpylayers：gradient alignment 和 train sweep
- gurobipy：N=50 exact MIQP finite difference 强烈建议
