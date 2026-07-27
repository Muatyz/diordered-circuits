# Paper Card: sagodi2024_back_to_continuous_attractor

## Bibliography

- Title: Back to the Continuous Attractor
- Authors: Ábel Ságodi, Guillermo Martín-Sánchez, Piotr Sokół, Il Memming Park
- Year / venue: NeurIPS 2024; arXiv v3 dated 2025-01-17
- arXiv: 2408.00109
- Local PDF: `paper.pdf`
- Search cache: `paper.txt`
- Status: digested / mapped to the current learning code / analysis not yet implemented

## One-sentence summary

精确的连续固定点族虽然结构不稳定，但只要原流形是紧致、吸引且 normally hyperbolic，小的动力学扰动仍会保留同胚的吸引 slow invariant manifold；有限时间记忆质量主要由流形内最大漂移速度控制，而长期误差由固定点、鞍点或极限环等流形内拓扑控制。

## Why this paper matters for `learning`

它改变的是项目的验收标准和分析层，而不是 Vafidis 局部学习规则本身：学习结果不应只按“是否得到精确零漂移 ring”或“权重是否看起来 circulant”判定，而应检查是否得到一个与角变量同拓扑、法向快速吸引、切向缓慢漂移的 autonomous invariant ring，并明确它在行为相关时间窗内的误差界。

## Core claims

1. 对紧致 normally hyperbolic continuous attractor 的足够小光滑扰动，会保留同胚、吸引且 invariant 的 slow manifold。
2. 流形内向量场的 uniform norm `eta` 同时刻画近似系统到理想 continuous attractor 的动力学距离，并给出短时间记忆误差的线性上界。
3. 一维 ring 上的长期行为不能仅由有限时间 gain 判断；固定点/鞍点对的数量、位置、basin 大小或 limit cycle 决定 asymptotic generalization。
4. 正常双曲性需要沿整个 slow manifold 检查：每个参考点应有一个接近零的切向 mode，并与其余收缩 mode 保持一致的 spectral gap。
5. 状态扰动（S-type）与动力学/权重扰动（D-type）是不同的鲁棒性问题，必须分开实验。

## Four conditions for an approximate continuous attractor

- C1: neural state 与被记忆角度之间存在足够光滑的近似一一对应，因此 neural manifold 与角变量同拓扑。
- C2: decoded memory 在流形上的漂移速度有界且足够慢。
- C3: 对 state perturbation/noise 的流动非扩张，并能返回 slow manifold。
- C4: 对 vector field 或 synaptic-weight perturbation 的动力学结构保持稳健。

## Direct consequences for the repository

- `timescale_separation_history.npz` 目前使用 visual-teacher steady states 的 `v_hd_distal` 曲线作为参考。它是有用的 operational assay，但不是 autonomous slow invariant manifold 的直接证据。
- `weight_eigenvalues.npz` 是连接矩阵谱，不是 frozen nonlinear dynamics 的 Jacobian 谱，不能据此声称 normal hyperbolicity。
- `phase_flow.py` 已实现输出角上的离散流、fixed-point reversal 与 basin 边界，是最接近论文方法的现有模块；应补充 `eta = max |d theta / dt|`、误差界验证和 basin entropy。
- neuron-count/noise 比较不应只报告 PI error 和 weight ringness；还应比较 topology、normal spectral gap、tangential drift 与 finite-horizon bound。
- 论文中的 backprop-trained RNN 只支持“slow manifold 是常见解”这一分析结论，不授权本项目在局部学习主线中使用 backprop/autograd。

## Implementation guardrails

- 先定义 frozen autonomous state/map，再计算 Jacobian；不要对 `W_HD->HD` 本身求谱后称其为动力学稳定性。
- manifold distance 必须在完整 Markov state 中计算；PCA 只用于展示，不能作为距离或 invariance 判据。
- 先从无输入、无噪声、冻结权重的轨迹识别 slow set，再独立做 state perturbation recovery；不能用 visual target curve 同时定义和验证 autonomous manifold。
- 切向 `eta` 使用明确的输出角速度单位（rad/s），并记录覆盖度与采样分辨率。
- state noise、input noise、training cue noise 和 weight perturbation 分别命名，避免把不同鲁棒性混成一个 `noise_std`。

## Priority recommendation

1. 抽出 frozen autonomous map 和最小 Markov state 的 pack/unpack 接口。
2. 从长时间 darkness trajectories 识别并保存 periodic slow manifold。
3. 沿流形计算 finite-difference Jacobian、切向 mode、法向 spectral gap。
4. 在现有 phase-flow 输出上增加 `eta`、有限时间误差界与 fixed-point basin entropy。
5. 最后再做 PCA 三维图、neuron-count scaling 和独立的 S-type/D-type perturbation sweeps。

