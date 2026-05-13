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

## 当前 WSL 会话复核

- GitHub remote 已确认可访问：`git@github.com:By-Xin/anis-kwon-2025-repro.git`，本地 `main` 与 `origin/main` 同步且工作树干净。
- 当前 WSL 系统只有 `/usr/bin/python3`，版本为 Python 3.12.3；缺少 `numpy`、`pytest`、CVXPYLayers 等依赖，也不满足项目声明的 `>=3.10,<3.12`。
- 当前 WSL 会话没有 `conda`、`mamba`、`micromamba`、`uv` 或 `gh` 命令。
- Windows 侧存在 `C:\Users\xinby\.conda\envs\anis-kwon-e2e\python.exe`，但从当前 WSL 会话调用 Windows `python.exe` 会失败并报 `WSL (2) ERROR: UtilBindVsockAnyPort:287: socket failed 1`。
- 现有 `results/smoke/` 是忽略文件；当前内容对应最近一次 `e2e_socp` synthetic smoke 输出，`solve_status=optimal`，但不是本轮 WSL 重新运行所得。
- 已在 `/tmp/anis_pydeps` 临时安装项目兼容范围内的基础依赖：`numpy==1.26.4`、`pandas==2.2.3`、`scipy==1.13.1`、`PyYAML==6.0.3`、`pytest==9.0.3`。这是 WSL smoke 环境，不是论文级环境。
- WSL smoke 复核通过：`PYTHONPATH=/tmp/anis_pydeps:src python3 -m pytest -q` 当前得到 `5 passed`。
- WSL synthetic data check 通过：50 个资产、5 个因子、1367 个日样本、274 个周样本、首个训练窗口 260 个周样本。
- WSL `scripts/run_smoke_test.py` 跑通 `nominal` 与 `linreg`，但因当前 WSL 无 CVXPY/Gurobi，测试求解状态是 `heuristic_topk`，只可作为工程 smoke，不可作为论文级验证。
- 正式配置此前阻塞在缺少真实 `data/prices.csv` 和 `data/factors_daily.csv`。
- 加强了 Fama-French 因子百分比自动识别：从单一 median 阈值改为 median/95 分位组合规则，并补充 raw-percent 与 decimal 两个单元测试。
- 修正了资产收益与因子收益的缺失值对齐逻辑：日度和周度数据都会先取共同日期，再按两侧任一缺失行一起删除，避免周聚合时资产/因子使用不同日集合。
- Kenneth French 五因子已生成到 `data/factors_daily.csv`，3021 条数据，日期 2010-01-04 到 2021-12-31，原始百分比口径。
- Choice 导出的价格文件实际在 `../snp500_50top.xlsx`，已清洗到 `data/prices.csv`，3021 行、51 列，日期 2010-01-04 到 2021-12-31；但源数据有两个关键缺口：`ABC` 只来自可疑代码 `ABC.LD`，缺 2259 天；`ORCL.O` 只到 2017-11-20，缺 1040 天。
- 因 `ABC` 与 `ORCL` 的缺失区间没有完整重叠，正式 `check_data` 当前失败：没有任何 50 个资产与 5 个因子同时非缺失的日度观测。
- 修正了 `prices_to_returns`，明确使用 `pct_change(fill_method=None)`，避免 pandas 默认前向填充缺失价格并掩盖数据问题；当前 WSL 单元测试为 `6 passed`。
- 补充文件 `../2more.xlsx` 包含完整 `COR.N` 和 `ORCL.N`，均为 3021 条，日期 2010-01-04 到 2021-12-31。已用 `COR.N` 覆盖 `data/prices.csv` 的 `ABC` 列，用 `ORCL.N` 覆盖 `ORCL` 列。
- 正式数据检查已通过：50 个资产、5 个因子、3020 个日收益样本、626 个周收益样本、24 个再平衡日期，首个再平衡日 2015-01-02，首个训练窗口 260 周。

## 第一轮代码审计待核验点

- `data.factor_returns_are_percent: auto` 已加强，但正式数据到位后仍必须检查 `max_abs_daily_factor_return`，必要时显式设置 `true/false`。
- `train_start` 配置目前不参与回测窗口生成；真实数据检查时应确认首个 5 年窗口确实覆盖 2010-01-01 至首个再平衡日前。
- `E2E_M` 的 Big-M 连续松弛在 `k>=1` 时理论上可能很松，`sum(z)<=k` 对 long-only `sum(w)=1` 的层解约束力有限；这与论文把 Big-M relaxation 作为较松层的设定一致，但后续要用真实切片确认不同 `k` 的训练参数和测试组合是否合理。
- `results/` 被 `.gitignore` 忽略，正式实验输出不会自动进 git。需要另行决定是否只提交 summary/metadata，还是用外部存储保存大结果文件。
