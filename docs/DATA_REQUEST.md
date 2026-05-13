# 数据申请需求单：Anis & Kwon (2025) 复现实验

## 1. 必备数据文件

请准备两个 CSV 文件，并放在 bundle 根目录的 `data/` 下：

```text
data/prices.csv
data/factors_daily.csv
```

所有日期列统一命名为 `date`，日期格式建议为 `YYYY-MM-DD`，编码 UTF-8。

---

## 2. `data/prices.csv`：50 只股票的日度调整收盘价

论文使用 50 只 S&P500 成分股，日度价格来自 Tiingo，时间范围为 **2010-01-01 到 2021-12-31**。请提供 **adjusted close / total-return adjusted close**，即已调整拆股、分红、合并等公司行为后的价格。

### 列格式

```csv
date,CVX,HES,OXY,SO,BALL,ECL,VMC,FDX,LMT,MMM,RHI,UPS,AMZN,AZO,BBWI,F,HAS,YUM,CPB,EL,MKC,PEP,PM,A,ABC,BIIB,CVS,DGX,JNJ,SYK,AIG,BAC,PGR,SCHW,WFC,AAPL,AKAM,CRM,CTSH,MA,ORCL,DIS,EA,T,AES,CMS,DUK,EQR,PLD,SPG
2010-01-04,....
```

### 必须包含的 ticker

| GICS | Tickers |
|---|---|
| Energy | CVX, HES, OXY, SO |
| Materials | BALL, ECL, VMC |
| Industrials | FDX, LMT, MMM, RHI, UPS |
| Consumer Discretionary | AMZN, AZO, BBWI, F, HAS, YUM |
| Consumer Staples | CPB, EL, MKC, PEP, PM |
| Health Care | A, ABC, BIIB, CVS, DGX, JNJ, SYK |
| Financials | AIG, BAC, PGR, SCHW, WFC |
| Information Technology | AAPL, AKAM, CRM, CTSH, MA, ORCL |
| Communication Services | DIS, EA, T |
| Utilities | AES, CMS, DUK |
| Real Estate | EQR, PLD, SPG |

### 数据质量要求

1. 每个交易日一行；建议覆盖 NYSE/Nasdaq 交易日。
2. 所有 50 个 ticker 的价格列必须为正数。
3. 最好无缺失值。若供应商因历史 ticker 变更导致列名不同，请在交付前映射回论文 ticker 列名。例如历史数据供应商可能对 `ABC`、`BBWI` 等发生过更名或映射差异；最终 CSV 仍必须使用上面的列名。
4. 脚本会从价格自动计算日收益，并按周复合为周收益。

---

## 3. `data/factors_daily.csv`：Fama-French 五因子日收益

论文使用 Fama and French (2015) 五因子：`Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`，来源是 Kenneth French Data Library。请提供与股票日期尽量重合的日度因子收益，时间范围同样覆盖 **2010-01-01 到 2021-12-31**。

### 列格式

```csv
date,Mkt-RF,SMB,HML,RMW,CMA
2010-01-04,0.0169,0.0028,0.0116,-0.0023,0.0034
```

### 百分比 / 小数

Kenneth French 原始日度文件通常以“百分比数值”给出，例如 `0.34` 表示 0.34%。本 bundle 默认 `factor_returns_are_percent: auto`，会自动判断并转换为小数收益。为了避免歧义，建议在交付时注明：

- 百分比口径：`0.34` 表示 `0.0034`；或
- 小数口径：`0.0034` 表示 `0.34%`。

---

## 4. 推荐数据检查命令

```bash
python scripts/check_data.py --config configs/reproduce_2015_2020.yaml
```

预期输出会报告：资产数量 50、因子数量 5、日度和周度样本量、首末日期、再平衡日期数量、首个 5 年训练窗口的周度样本数等。

---

## 5. 论文期望实验口径

- 股票数量：`N = 50`
- 因子数量：`P = 5`
- 训练窗口：每次再平衡前 5 年周收益
- 再平衡频率：每 3 个月
- out-of-sample：bundle 默认使用 `2015-01-01` 到 `2020-12-31`，对应 24 个季度；另附 `configs/reproduce_2015_2021.yaml`，用于论文正文提到的 2015--2021 口径。
- cardinality：`k = 10, 15, 20`
