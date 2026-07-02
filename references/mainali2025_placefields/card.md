# Paper Card: mainali2025_placefields

## Bibliography

- Authors: Nischal Mainali, Rava Azeredo da Silveira, Yoram Burak
- Venue: Neuron 113 (2025), 1110–1120
- DOI: https://doi.org/10.1016/j.neuron.2025.01.017
- Local PDF: `paper.pdf`
- Search cache: `paper.txt`
- Status: partially implemented

## One-sentence summary

将 CA1 place fields 建模为平移不变 Gaussian process 的阈值化/整流 excursion sets，用两个主要参数统一解释多物种、1D/2D/3D 和不同环境尺度下的场大小、间隔、形状与拓扑统计。

## Project relevance

它说明“异质、多峰调谐”可以来自简单随机输入统计，而无需为每个 field 设计独立结构；可作为 Clark ring generator 向 line/place manifold 扩展的候选，但不应在 ring learning baseline 稳定前进入主线。

## Core claims

1. 高阈值下，field-size 与 gap 统计具有对相关函数细节不敏感的 universal form。
2. normalized threshold `q` 与 correlation length `s` 可共同预测 field arrangement、shape 和 Euler characteristic。
3. 多个随机空间输入经随机权重求和自然产生近 Gaussian 的 CA1 输入，支持 predominantly random projections 的解释。

## Implementation notes

- 论文模型是 threshold + rectification，不等同于 Clark 的 normalized softplus。
- 高阈值近似与有限阈值模拟需区分。
- 从 1D toy 扩展到 2D/3D 时，field topology 与 connected components 不能只靠逐轴切片替代。
