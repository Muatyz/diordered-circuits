# References Index

## 使用顺序

1. 先根据任务类型定位论文。
2. 阅读该论文的 `card.md`。
3. 实现公式时读 `equation_map.md`；复现图时读 `figure_map.md`。
4. 修改代码前后核对 `code_map.md`。
5. 仅在摘要不足或有冲突时搜索 `paper.txt`/查阅 `paper.pdf`。

## Core papers

### clark2025_symmetries

- Title: Symmetries and Continuous Attractors in Disordered Neural Circuits
- Local path: `references/clark2025_symmetries/`
- Role: 项目核心理论、head-direction 数据复现、optimized RNN、Gaussian generative process 与 large-N/DMFT。
- Implementation: `reproduction/src/compute_hd_tuning.py`、`utils.py`、`plot_figure2_reproduction.py`、`plot_figure3_*.py`、`plot_figure4_reproduction.py`、`plot_figure5_*.py`、`plot_figure6_*.py`
- Status: Figure 2–6 已有完整功能性实现与稳定性诊断；后续重点转向 learning rule。

### mainali2025_placefields

- Title: Universal statistics of hippocampal place fields across species and dimensionalities
- Local path: `references/mainali2025_placefields/`
- Role: 以阈值化 Gaussian process 统一解释 1D/2D/3D place-field 统计，为 ring 框架向空间场扩展提供参考。
- Implementation: `reproduction/src/compute_place_cell_optimized_weights.py`、`reproduction/src/utils.py::gaussian_process_place_fields_1d`
- Status: toy implementation；尚未进入 learning 主线。

## Learning-rule papers

### bell2024_plasticity_rules

- Title: Discovering plasticity rules that organize and maintain neural circuits
- Local path: `references/bell2024_plasticity_rules/`
- Role: 用 basis-function + meta-learning 搜索局部可塑性规则，并研究 synaptic turnover 下的鲁棒自组织。
- Status: 已摄取精华；尚未实现。

### oreilly2026_neocortex_learning

- Title: This is how the Neocortex Learns
- Local path: `references/oreilly2026_neocortex_learning/`
- Role: temporal-derivative/error-driven predictive learning 的生物实现参考。
- Status: 已摄取精华；需与“无监督局部规则”严格区分。
