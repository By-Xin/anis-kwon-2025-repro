# 完整实验设计：验证 SDP/SOCP 为什么在 E2E cardinality portfolio 中不一定 work

## 0. 背景判断

导师录音稿里的问题可以整理成四个科学问题：

1. **更紧的 relaxation 是否真的有必要？** 传统 MIP 观点是更紧的 relaxation 对 branch-and-bound 好，但 E2E 训练关心的是梯度质量，而不只是 lower bound。
2. **直接整数 forward 是否可以？** 如果 forward 直接解 MIQP，backward 再用某种 surrogate/STE，会不会比“forward 也用 relaxation”更接近最终部署？
3. **SDP 不 work 是文章没做好，还是问题结构导致？** 需要区分 relaxation quality、gradient quality、sample complexity、solver numerical issue。
4. **能否迁移到更复杂的 multi-period QP/control？** 先在 portfolio 上弄清楚“紧 relaxation 是否帮 E2E”，再扩展到多阶段。

论文自己也承认了这一点：Big-M 更简单，但实证上 outperform 更紧的 SOCP/SDP；作者认为这可能与问题规模、结构、样本量、decision layer 的变量和约束数量有关。我们要把这个未来工作变成可执行实验。

## 1. 假设

### H1：SDP 的 lower bound 很好，但对最终整数 portfolio 没有好映射

SDP 优化的是 lifted relaxation 的 lower bound。E2E 训练 loss 用 relaxation 输出的 `w_relax`，但最终部署用 MIQP 输出的 `w_int`。如果 `w_relax` 与 `w_int` 支持集差异大，训练就会优化一个和部署不一致的 surrogate。

可观测指标：

- `gap_to_exact = (obj_exact - obj_relax) / obj_exact`
- `bound_violation = max(obj_relax - obj_exact, 0) / obj_exact`
- `relax_portfolio_gap_to_exact = (w_relax' Sigma w_relax - obj_exact) / obj_exact`
- `topk_overlap = |TopK(w_relax) ∩ supp(w_int)| / k`
- `l1_to_exact = ||w_relax - w_int||_1`
- `rounded_gap = (obj_rounded - obj_exact) / obj_exact`
- `rounded_sharpe` vs `exact_sharpe`

判定：若 SDP gap 小但 overlap/rounded_sharpe 差，则 SDP 是好 bound，但不是好 learning surrogate。

### H2：Big-M 虽松，但产生更平滑、更有用的梯度

Big-M continuous relaxation 对 long-only budget cardinality 问题几乎退化为 full-set min-variance，因此它忽略 cardinality，但也因此可微映射更平滑。E2E 外层优化 Sharpe，不一定需要 tight lower bound；反而需要稳定、有信息量的梯度。

可观测指标：

- gradient norm 是否爆炸/消失；
- gradient cosine/correlation 与 finite-difference integer loss 是否一致；
- loss curve 是否稳定下降；
- 相同训练预算下 exact-deployment Sharpe 是否更高。

### H3：SDP/SOCP 不是本质不行，而是样本/epoch 不够

论文中三种方法学习的 factor model 参数数量相同，但 decision layer 大小不同。SDP 的变量/约束远多于 Big-M，可能需要更多 CBB samples 和 epochs。

可观测指标：

- `method × J × epochs` 网格下的 validation Sharpe；
- SDP 是否随 `J` 或 `epochs` 呈系统性改善；
- 学习曲线在原文 4 epoch 是否还没收敛。

### H4：SDP/SOCP 的隐式微分被 SCS 数值误差污染

论文用 CVXPYLayers/SCS 做 SOCP/SDP。SDP 的 cone size 大，KKT 系统可能病态，导致反向传播梯度噪声大。

可观测指标：

- solver status、solve time、primal/dual residual；
- 梯度 NaN/Inf rate；
- 不同 tolerance 下 gradient alignment 是否变化；
- 同一参数多次求解的梯度 variance。

### H5：直接整数 forward + relaxed backward 可能更适合

把 forward 换成 `w_int = MIQP(Sigma(theta))`，loss 用整数组合；backward 用 relaxation 的 Jacobian 作为 straight-through estimator。若这比 pure relaxation forward 好，说明问题主要是 train/deploy mismatch。

## 2. 实验组

### 实验 1：Relaxation Quality & Rounding Quality

对每个 rebalance window、每个 cardinality：

1. 用 OLS factor model 构造 `Sigma = B Sigma_f B' + diag(psi^2)`。
2. 解 exact MIQP 得到 `w_int,obj_int`。
3. 解 Big-M continuous relaxation、SOCP、SDP 得到 `w_relax,obj_relax`。
4. 把 `w_relax` top-k rounding 后，在 support 上解 long-only QP 得到 `w_round,obj_round`。
5. 在训练窗口/验证窗口上比较 variance、Sharpe、support overlap。

输出：`relaxation_quality.csv`。

### 实验 2：Gradient Alignment with Integer Forward Loss

在每个 window 上生成 CBB bootstrap instances。对每个 relaxation：

1. 在 OLS 初始化点计算 autograd gradient `g_relax = ∂ L_relax / ∂ theta`。
2. 随机采样方向 `d_1,...,d_m`。
3. 用 exact MIQP forward 的 finite difference 估计真实整数目标方向导数：
   `fd(d) = [L_int(theta + eps d) - L_int(theta - eps d)] / (2 eps)`。
4. 比较 `g_relax·d` 与 `fd(d)` 的相关系数、符号一致率、均方误差。

输出：`gradient_alignment.csv`。

### 实验 3：Training Budget Sweep

网格：

- method: Big-M, SOCP, SDP
- `J`: 128, 512, 2000, 8000
- epochs: 1, 4, 16, 32
- tolerance: 1e-3, 1e-4, 1e-5
- cardinality: 10, 15, 20

每个组合训练 E2E factor model；最终一律用 exact MIQP 部署，比较 in-sample/out-of-sample Sharpe。

输出：`train_eval_sweep.csv`。

### 实验 4：Train/Deploy Mismatch over Epochs

训练每个 epoch 后记录：

- relaxed layer portfolio loss；
- exact MIQP deployment portfolio loss；
- `||w_relax - w_int||_1`；
- support overlap；
- covariance parameter drift `||B-B0||`, `||psi-psi0||`。

如果 relaxation loss 下降但 exact deployment loss 不降，就是 surrogate mismatch。

### 实验 5：Direct Integer Forward + STE

比较三种训练方式：

1. `relax_forward_relax_backward`：论文式；
2. `integer_forward_relax_backward_ste`：forward 用 MIQP，backward 用 relaxed layer；
3. `integer_forward_zero_order`：小 N 上用 SPSA/finite-difference 直接优化整数 forward loss。

若 2/3 优于 1，说明端到端确实有价值，但 relaxation forward 不是最合适的训练 surrogate。

## 3. 推荐最小实验路线

第一周建议只跑：

1. `run_relaxation_quality.py`：N=50, k=10/15/20, 24 windows。
2. `run_gradient_alignment.py`：N=20 子集, k=5/10, 3 windows。
3. `run_train_eval_sweep.py`：N=20 子集, k=5, J={128,512,2000}, epoch={1,4,16}。

拿到这三张表，就基本能回答导师最关心的“SDP 有没有必要”。

## 4. 预期可发表/可汇报结论模板

### 模板 1：SDP 数学上紧，但学习上不优

> SDP relaxation produces the tightest lower bounds, but its relaxed solutions have poor alignment with the final integer portfolios and its implicit gradients show weaker agreement with finite-difference integer-loss directions. Hence relaxation tightness alone is insufficient for decision-focused learning in nonlinear MIPs.

### 模板 2：Big-M 是一种有益 smoothing

> The Big-M layer degenerates toward the full-set continuous min-variance portfolio, yielding a smoother and better-conditioned solution map. Although loose as a MIP relaxation, it provides more useful gradients for learning covariance parameters under limited samples.

### 模板 3：文章 SDP 可能 under-trained

> Increasing CBB sample size and training epochs improves SDP alignment/performance, indicating the original experiment may have been compute-constrained. The relevant complexity is not only the number of learned parameters but also the size and curvature of the optimization layer.

### 模板 4：solver 是瓶颈

> SDP performance is sensitive to SCS tolerance and exhibits unstable gradients/residuals. This suggests that the bottleneck is not only the SDP formulation but also the differentiable cone-solver implementation.
