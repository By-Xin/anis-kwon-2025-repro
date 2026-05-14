# 数据需求单

## 1. 资产价格 `data/prices.csv`

宽表，第一列 `date`，后续是 adjusted close。论文 50 个 ticker：

```text
CVX,HES,OXY,SO,BALL,ECL,VMC,FDX,LMT,MMM,RHI,UPS,AMZN,AZO,BBWI,F,HAS,YUM,CPB,EL,MKC,PEP,PM,A,ABC,BIIB,CVS,DGX,JNJ,SYK,AIG,BAC,PGR,SCHW,WFC,AAPL,AKAM,CRM,CTSH,MA,ORCL,DIS,EA,T,AES,CMS,DUK,EQR,PLD,SPG
```

建议日期范围：2010-01-01 至 2020-12-31；如果要复现论文图注口径，可扩展到 2021-12-31。

要求：

- adjusted close；
- 日期 ISO 格式；
- ticker 变更请映射回论文 ticker 名，例如 `ABC`, `BBWI`；
- 少量缺失可 forward fill；连续缺失不建议超过 5 个交易日。

## 2. Fama-French daily factors `data/factors_daily.csv`

```text
date,Mkt-RF,SMB,HML,RMW,CMA
```

可用小数或百分比。脚本会自动检测：若绝对值中位数大于 0.05，默认视为百分比并乘 0.01。

## 3. 子集实验

如果先跑 N=20 或 N=12，只需在 config 里改 `data.tickers`。脚本会按 config 中的 ticker 列读取。
