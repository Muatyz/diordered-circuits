# Paper Card: oreilly2026_neocortex_learning

## Bibliography

- Author: Randall C. O'Reilly
- Date/version: 2026-06-07, Version 1
- URL: https://compcogneuro.org/oreilly-2026-cortlearn
- Local PDF: `paper.pdf`
- Search cache: `paper.txt`
- Status: digested / conceptual reference

## One-sentence summary

文章主张 neocortical learning 可由 prediction 与 outcome 两个时间状态之差隐式表示 error gradient，并通过 corticothalamic 回路和快慢 kinase 积分在局部突触上实现 temporal-derivative learning。

## Why it matters

它提供一种“局部可塑性可携带误差信号”的机制候选，但属于 error-driven predictive learning，不等同于无监督 self-organization；本项目若采用，必须明确 teacher/outcome 信号从哪里来。

## Core claims

1. plus/minus phase 的 activity difference 可近似 backpropagated error。
2. corticothalamic prediction/outcome 交替可在约 200 ms 周期内产生该差异。
3. fast-minus-slow biochemical integration（例如 CaMKII/DAPK1 竞争）可把 temporal derivative 转成 LTP/LTD。

## Project boundary

- 可作为带 modulatory/local error proxy 的对照规则。
- 不能在没有 prediction/outcome 或 teaching phase 的实验中称为 purely unsupervised。
- 文章的强结论具有立场性，项目应把它当作候选框架而非已确立事实。
