# Paper Card: bell2024_plasticity_rules

## Bibliography

- Authors: David Bell, Alison Duffy, Adrienne Fairhall
- Venue: NeurIPS 2024
- Local PDF: `paper.pdf`
- Search cache: `paper.txt`
- Status: digested / not implemented

## One-sentence summary

论文用 meta-learning 在由 pre/post activity、时间滤波 trace 和 synaptic weight 构成的 basis 中搜索局部规则，使随机 E/I 网络自组织出 sequence dynamics，并在 synaptic turnover 下学到额外 homeostasis。

## Why it matters

它为本项目提供“搜索局部 plasticity motif”的具体方法，而不是直接提供 ring attractor 规则；最值得迁移的是规则参数化、term dropout/精简和在结构扰动下优化鲁棒性的实验设计。

## Core claims

1. 无 turnover 时，关键三项构成 temporally asymmetric Oja-like rule。
2. 有 turnover 时，规则加入 constant potentiation 与 postsynaptic homeostatic bound。
3. 同时允许 excitatory/inhibitory plasticity 比只调整 E→E 更能维持扰动后的 sequence representation。

## Guardrails for this project

- Meta-objective 是全局的，但部署后的 synaptic update 必须只依赖局部变量。
- sequence decoding loss 不能原样用于 ring；应替换为 manifold reconstruction、normal stability 和 phase drift。
- 先用小型可解释 basis，做 term ablation，再扩大搜索空间。
