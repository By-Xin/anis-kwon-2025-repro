# 复现说明与已知不确定点

## 已对齐论文的部分

1. 资产集合：论文 Table 1 的 50 个 ticker。
2. 因子：Fama-French 五因子 `Mkt-RF, SMB, HML, RMW, CMA`。
3. 数据：2010-01-01 到 2021-12-31 的日度股票价格，转换为日收益和周收益。
4. 滚动设计：首次 5 年训练窗口；每 3 个月再平衡；每次用前 5 年周收益重训。
5. Baseline：
   - `nominal`：直接用资产周收益样本协方差。
   - `linreg`：OLS 线性因子模型，构造 `Sigma = B Sigma_f B' + Psi`。
6. E2E：
   - `e2e_m`：Big-M 连续松弛 / full-set QP 型 layer。
   - `e2e_socp`：Cui et al. perspective/SOCP 松弛。
   - `e2e_sdp`：Wiegele & Zhao SDP 松弛。
7. 训练实例：CBB，`J=2000`，block size `20`。
8. 训练 schedule：4 epochs，前三轮学习率 0.01，最后一轮 0.001。
9. 测试时：用训练出的 `B, psi` 和真实历史 5 年因子协方差构造 covariance，再用 Big-M MIQP 求 cardinality-constrained minimum variance portfolio。

## 论文中没有唯一给出的细节

1. **batch size**：Algorithm 2 有 batch size 输入，但正文没有报告具体数值。本 bundle 默认 `batch_size=20`，可在 config 中修改。
2. **out-of-sample 年份**：数据段说 rolling OOS 跨 2015--2021，但 Tables 3--4 标题写 2015--2020，Table 2 写 24 个训练期，正好对应 2015--2020 的 24 个季度。默认配置采用 `2015-01-01` 到 `2020-12-31`，另附 `reproduce_2015_2021.yaml`。
3. **年化公式细节**：论文附录给出年化收益、年化波动和 Sharpe 公式，使用 365 与 256。bundle 默认使用这些常数；若想以周收益训练 loss，可在 `train_loss_ann_return_periods` 和 `train_loss_ann_vol_periods` 改为 52。
4. **CVXPY/SCS 数值误差**：隐式优化层的梯度对 solver tolerance 和版本较敏感，尤其 SOCP/SDP。为论文级复现，应固定 SCS 3.2.1 并记录随机种子、solver log 和运行机器。
5. **原始数据供应商差异**：Tiingo adjusted close 与其他供应商的 total return/adjusted close 可能略有差异，结果不会逐位相同。

## 输出文件

默认输出目录：`results/reproduce_2015_2020/`

- `metrics.csv`：复现 Table 3/4 所需的 risk/return 指标。
- `weights.csv`：每个 rebalance date 的组合权重。
- `daily_returns.csv`：各方法、各 cardinality 的日度组合收益。
- `run_log.csv`：训练/求解时间、求解状态等。
- `models/*.npy` 和 `models/*training_log.json`：E2E 模型参数与训练日志。
