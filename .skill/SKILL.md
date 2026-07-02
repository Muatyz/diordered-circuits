# Literature-Code Workflow Skill for Random RNN

本 Skill 用于管理 Random RNN 项目中的“文献 → 公式 → 代码 → 数据 → 图表 → 研究结论”工作流。目标不是让 Codex 每次临时阅读完整 PDF，而是把论文信息压缩成可检索、可追踪、可执行的本地知识层，使代码实现始终能追溯到论文中的公式、图、方法和假设。

## 0. 总原则

本项目的核心工作流是：

文献 PDF
→ 结构化文献笔记
→ 公式与图表映射
→ 代码模块
→ 数据产物
→ 复现实验与诊断报告
→ 新研究问题

Codex 在写代码前，必须先明确当前任务属于以下哪一类：

1. reproduction：复现 Clark 等论文的图、公式、数据处理和动力学结果。
2. learning：探索 local / unsupervised / biologically plausible learning rule 是否能逼近 optimized weight matrix。
3. meta-learning：探索可学习的 plasticity rule 或学习规则搜索。
4. literature：整理新论文、建立文献与代码之间的映射。
5. diagnostics：诊断数值稳定性、Jacobian、activation inverse、smoothing、regularization、simulation dynamics 等问题。

不要把“论文阅读”和“代码实现”混成一次性聊天内容。每个重要解释都应该落盘到 `references/` 或对应任务目录下的 markdown 文件中。

## 1. 推荐目录结构

在当前项目基础上，建议逐步整理为：

```text
.
├── .skills/
│   └── literature/
│       └── SKILL.md
├── references/
│   ├── index.md
│   ├── index.yml
│   ├── clark2025_symmetries/
│   │   ├── paper.pdf
│   │   ├── paper.md
│   │   ├── card.md
│   │   ├── equation_map.md
│   │   ├── figure_map.md
│   │   ├── code_map.md
│   │   └── open_questions.md
│   ├── mainali2025_placefields/
│   │   ├── paper.pdf
│   │   ├── card.md
│   │   ├── equation_map.md
│   │   ├── figure_map.md
│   │   └── code_map.md
│   ├── nair2023_line_attractor/
│   ├── zhao2026_history_biased_decisions/
│   ├── bell2024_plasticity_rules/
│   └── oreilly2026_neocortex_learning/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── reproduction/
│   ├── src/
│   ├── config/
│   ├── reports/
│   ├── reports/figures/
│   ├── tests/
│   └── .skills/
├── learning/
│   ├── src/
│   ├── experiments/
│   ├── reports/
│   └── .todo/
└── reports/
    └── figures/
```

如果暂时不想移动 PDF，也可以保留 `references/*.pdf` 的扁平结构，但至少要为每篇论文建立一个同名 `.md` 文件。例如：

```text
references/
├── Symmetries and Continuous Attractors in Disordered Neural Circuits.pdf
├── Symmetries and Continuous Attractors in Disordered Neural Circuits.card.md
├── Universal statistics of hippocampal place fields across species and dimensionalities.pdf
└── Universal statistics of hippocampal place fields across species and dimensionalities.card.md
```

长期更推荐用 `paper_id/` 子文件夹，因为一篇论文最终会对应多个笔记、公式映射、图表映射、代码映射和未解决问题。

## 2. paper_id 命名规则

每篇论文必须有唯一 `paper_id`。格式为：

```text
firstauthorYYYY_shorttopic
```

本项目建议使用：

```text
clark2025_symmetries
mainali2025_placefields
nair2023_line_attractor
zhao2026_history_biased_decisions
bell2024_plasticity_rules
oreilly2026_neocortex_learning
```

所有代码注释、报告、实验配置中引用论文时，优先使用 `paper_id`，不要反复使用完整论文标题。

例如：

```python
# Implements clark2025_symmetries Eq. 2: manifold flow-matching loss.
```

## 3. Codex 启动任务时必须读取的文件

每次处理本项目任务时，Codex 应按照以下顺序读取上下文：

1. 当前任务目录下的 Skill 文件，例如：

   * `.skills/literature/SKILL.md`
   * `reproduction/.skills/SKILL.md`
   * `learning/.SKILL.md`

2. 文献索引：

   * `references/index.md`
   * `references/index.yml`

3. 与任务相关的文献卡片：

   * `references/<paper_id>/card.md`
   * `references/<paper_id>/equation_map.md`
   * `references/<paper_id>/figure_map.md`
   * `references/<paper_id>/code_map.md`

4. 与任务相关的代码：

   * `reproduction/src/*.py`
   * `learning/src/*.py`
   * `reproduction/config/*.json`
   * `tests/*.py`

5. 与任务相关的数据产物和诊断报告：

   * `data/processed/*.npz`
   * `data/processed/*.json`
   * `reproduction/reports/*.json`
   * `reproduction/reports/*.csv`
   * `reproduction/reports/figures/*.png`

如果文献卡片和代码冲突，以论文原文为最高优先级；如果论文原文、文献卡片和已有代码三者不一致，必须先报告冲突，再修改代码。

## 4. references/index.md 模板

`references/index.md` 是文献入口，不写长篇综述，只写“任务导航”。

模板如下：

```markdown
# References Index

## Core papers

### clark2025_symmetries
- Title: Symmetries and Continuous Attractors in Disordered Neural Circuits
- Local path: references/clark2025_symmetries/
- Role in project: 核心理论与复现对象；定义 data-derived RNN、optimized weight matrix、Gaussian process generative process、large-N / DMFT 分析。
- Current implementation:
  - reproduction/src/compute_hd_tuning.py
  - reproduction/src/gaussian_generative_process.py
  - reproduction/src/generate_synthetic_hd_dataset.py
  - reproduction/src/plot_figure3_abcd.py
  - reproduction/src/plot_figure5_abc.py
  - reproduction/src/plot_figure6_*.py
- Status: Figure 2 到 Figure 6 基本复现；后续重点是 synthetic data、learning rule、local update。

### mainali2025_placefields
- Title: Universal statistics of hippocampal place fields across species and dimensionalities
- Local path: references/mainali2025_placefields/
- Role in project: 提供 place cell 的 Gaussian process 生成模型；用于考虑将 Clark 框架从 ring/head direction 推广到 place fields。
- Current implementation:
  - reproduction/src/compute_place_cell_optimized_weights.py
- Status: 初步 toy implementation；需要明确是否值得纳入 learning rule 主线。

### bell2024_plasticity_rules
- Role in project: local plasticity rule / meta-learning learning rule 的重要参考。
- Status: 用于构造 learning/meta-learning 方向的候选规则空间。

### nair2023_line_attractor
- Role in project: aggression line attractor 参考；用于理解近似线吸引子、积分维度、行为状态变量。
- Status: 概念分析阶段。

### zhao2026_history_biased_decisions
- Role in project: attractor-to-integrator hierarchy 参考；用于比较 discrete attractor、integrator 和 history-biased decision。
- Status: 概念分析阶段。

### oreilly2026_neocortex_learning
- Role in project: error-driven / temporal derivative / biologically plausible learning 参考。
- Status: 与 local unsupervised learning rule 的关系需要谨慎区分。
```

## 5. card.md 模板

每篇论文都要有 `card.md`。这是 Codex 最常读取的文件，要求短、准、可执行。

模板如下：

```markdown
# Paper Card: <paper_id>

## Bibliography
- Title:
- Authors:
- Year:
- Venue:
- DOI / URL:
- Local PDF:
- Extracted text:
- Status: unread / skimmed / partially implemented / reproduced / used for new hypothesis

## One-sentence summary
用一句话概括这篇论文真正做了什么。

## Why this paper matters for Random RNN
说明它和本项目的关系。不要写泛泛而谈的背景介绍，要写它能改变哪些代码、实验或研究判断。

## Core claims
1.
2.
3.

## Mathematical objects
列出核心变量、函数、矩阵、动力学方程、损失函数、统计量。

例如 Clark:
- target firing-rate manifold: phi_star_i(theta)
- input-current manifold: x_star_i(theta)
- recurrent dynamics: tau dx_i/dt = -x_i + sum_j J_ij phi_j + b
- optimized weights: least-squares / pseudoinverse solution
- covariance function: Gamma^x(Delta theta)
- order parameter: overlap m(theta)

## Figures relevant to code
| Figure | Meaning | Code target | Current status |
|---|---|---|---|
| Fig. 2 | tuning heterogeneity | plot_figure2_reproduction.py | done / partial |
| Fig. 3 | data-derived weight matrix and dynamics | plot_figure3_*.py | partial |
| Fig. 5 | generative process | plot_figure5_*.py | partial |
| Fig. 6 | synthetic population statistics | plot_figure6_*.py | partial |

## Equations relevant to code
| Equation | Meaning | Code target | Risk |
|---|---|---|---|
| Eq. 1 | RNN dynamics | utils.py / dynamics.py | activation convention |
| Eq. 2 | optimization loss | weight_solver.py | normalization and lambda |
| Eq. 3 | optimized J | weight_solver.py | kernel inversion |
| Eq. 5 | velocity input | not implemented | later |

## Implementation notes
写出工程实现时最容易出错的地方。

例如：
- firing rate 为 0 时，softplus inverse 会产生非常大的负输入电流，需要 floor 或 clipping。
- tuning curve smoothing 会影响 Fourier cutoff 和 Jacobian stability。
- c inhibition 与 lambda regularization 会共同影响 mean mode stability。
- 论文中连续积分实现为离散 theta grid 时，需要显式检查归一化因子和网格权重。

## Open questions
1.
2.
3.

## Next coding actions
1.
2.
3.
```

## 6. equation_map.md 模板

`equation_map.md` 负责把论文公式翻译成代码接口。它不是推导笔记，而是实现契约。

模板如下：

```markdown
# Equation Map: <paper_id>

## Eq. <number>: <short name>

### Paper form
只写必要公式，不粘贴大段原文。

### Discrete implementation
说明连续变量如何离散化。

例如：
- theta in [0, 2pi) is discretized into n_theta bins.
- Integral dtheta / 2pi becomes mean over theta grid.
- Phi has shape (N, n_theta).
- X has shape (N, n_theta).
- J has shape (N, N).

### Code target
- File:
- Function:
- Inputs:
- Outputs:
- Shape conventions:

### Numerical risks
- singular covariance
- ill-conditioned kernel
- rate floor
- diagonal constraint J_ii = 0
- non-normal Jacobian
- mismatch between normalized and unnormalized rates

### Validation
- Unit test:
- Diagnostic plot:
- Expected scale:
- Failure signature:
```

对 Clark 项目，至少建立以下条目：

```markdown
## Eq. 1: rate RNN dynamics
## Eq. 2: flow-matching loss
## Eq. 3: optimal recurrent weights
## Eq. 4: firing-rate correlation C_phi
## Eq. 5: velocity-modulated weights
## Appendix B activation functions
## Appendix B double normalization
## Jacobian on target manifold
```

## 7. figure_map.md 模板

`figure_map.md` 负责把论文图和项目中的脚本、数据文件、输出图片对应起来。

模板如下：

```markdown
# Figure Map: <paper_id>

## Figure <number><panel>

### Paper meaning
该图验证的科学命题是什么？

### Required inputs
- raw data:
- interim data:
- processed data:
- config:

### Code path
- script:
- helper functions:
- output figure:
- output diagnostics:

### Reproduction criteria
写出“什么程度算复现成功”。

例如：
- 不是要求像素级一致。
- 要求趋势、数量级、统计检验、动力学行为一致。
- 若存在随机种子或数据筛选差异，必须记录。

### Current status
- not started / running / partial / reproduced / failed

### Known deviations from paper
1.
2.
3.

### Next action
```

对 Clark 复现，建议至少维护：

```text
Fig. 2C-F: tuning curves and heterogeneity
Fig. 3A-D: optimized / sorted / circulant / noise-corrupted weights
Fig. 3E-F: trajectory convergence to target manifold
Fig. 3G-I: bump states
Fig. 3J-L: overlap order parameter
Fig. 4: circular distributional symmetry
Fig. 5A-C: Gaussian process generative process
Fig. 5D-E: parameter fitting
Fig. 5F-G: correlation and synthetic samples
Fig. 6A-F: synthetic data statistics
```

## 8. code_map.md 模板

`code_map.md` 负责让 Codex 知道“哪些代码实现了哪些论文概念”。

模板如下：

```markdown
# Code Map: <paper_id>

## Concept-to-code map

| Paper concept | Code file | Function / class | Data output | Test / diagnostic |
|---|---|---|---|---|
| HD tuning curve | reproduction/src/compute_hd_tuning.py | ... | data/processed/*hd_tuning*.npz | tests/test_hd_tuning_pipeline.py |
| GP generator | reproduction/src/gaussian_generative_process.py | ... | figure5_abc_generative_process.npz | tests/test_gaussian_generative_process.py |
| optimal J | reproduction/src/... | ... | figure3_abcd_weight_matrices.npz | diagnose_weight_stability.py |
| Jacobian stability | reproduction/src/diagnostics/... | ... | figure3_jacobian_*.json | diagnostics |

## File ownership rules
- 不要把文献解析代码、数据处理代码、图表绘制代码混在一个脚本里。
- `src/` 放可复用函数。
- `plot_figure*.py` 只负责读取 processed/interim 数据并画图。
- `diagnostics/` 只负责定位数值问题，不作为主 pipeline。
- `tests/` 负责最小可重复验证，不依赖超大 raw data。

## When editing code
每次修改代码前，先回答：

1. 修改对应哪篇论文、哪一个公式、哪一个图？
2. 输入和输出 shape 是什么？
3. 修改是否改变已有 processed 数据格式？
4. 是否需要更新 config？
5. 是否需要更新 tests？
6. 是否需要更新 `code_map.md`？

## After editing code
每次修改代码后，必须更新：

- 对应 `code_map.md`
- 必要时更新 `figure_map.md`
- 若产生新诊断，写入 `reports/*.json` 或 `reports/*.md`
- 若发现论文解释变化，写入 `open_questions.md` 或 `card.md`
```

## 9. PDF 提取与文献摄取流程

Codex 不应该依赖临时打开 PDF。每篇新论文进入项目时，执行以下流程：

### Step 1: 建立文献目录

```bash
mkdir references/<paper_id>
```

放入：

```text
references/<paper_id>/paper.pdf
```

### Step 2: 提取文本

优先使用 PyMuPDF：

```bash
python -m pip install pymupdf
```

建议建立脚本：

```text
scripts/lit_extract.py
```

脚本目标：

```bash
python scripts/lit_extract.py references/<paper_id>/paper.pdf --out references/<paper_id>/paper.md
```

如果有 `pdftotext`，也可以使用：

```bash
pdftotext -layout references/<paper_id>/paper.pdf references/<paper_id>/paper.md
```

不要把 OCR 作为默认方案。OCR 只在扫描版 PDF 且无文本层时使用。

### Step 3: 建立 paper card

```text
references/<paper_id>/card.md
references/<paper_id>/equation_map.md
references/<paper_id>/figure_map.md
references/<paper_id>/code_map.md
references/<paper_id>/open_questions.md
```

### Step 4: 建立索引

更新：

```text
references/index.md
references/index.yml
```

### Step 5: 搜索而不是通读

Codex 需要查找公式或图表时，优先使用：

```bash
rg -n "Figure 3|Fig. 3|Eq. 3|Equation 3|Jacobian|activation|softplus|Gaussian process" references/<paper_id>/
```

不要每次从头阅读整篇论文。

## 10. ChatGPT 与 Codex 的分工

ChatGPT 项目侧适合做：

* 读多篇论文，比较理论路线。
* 判断研究方向优先级。
* 解释公式、动力学、计算神经科学背景。
* 将文献观点转成研究计划。
* 生成 `card.md`、`equation_map.md`、`figure_map.md` 初稿。
* 帮助判断某个复现结果是否足以进入下一阶段。

Codex 侧适合做：

* 读取本地代码。
* 修改 Python 脚本。
* 跑实验、生成数据、生成图。
* 检查 shape、dtype、路径、配置。
* 写测试。
* 根据 `equation_map.md` 和 `figure_map.md` 把公式落地为代码。
* 更新 `code_map.md` 和诊断报告。

重要原则：

ChatGPT 的解释不能只停留在聊天记录里。凡是会影响代码实现的解释，都必须同步写入 `references/<paper_id>/*.md` 或当前任务目录的 `.todo/TODO.md`。

## 11. 文献到代码的最小闭环

每个实现任务必须形成闭环：

```text
paper claim
→ equation / figure
→ code function
→ config
→ processed output
→ plot / diagnostic
→ report
→ next question
```

示例：

```text
Clark Fig. 3F claims perturbed states converge rapidly back to target manifold.
→ figure_map.md: Fig. 3F
→ code: plot_figure3_ef.py
→ processed: figure3_ef_dynamics.npz
→ diagnostic: distance_to_manifold(t)
→ report: figure3_dynamics_diagnostics.json
→ next question: why convergence only reaches 1e-2 not 1e-3?
```

## 12. 数据层规则

项目已有 `data/raw`、`data/interim`、`data/processed`，必须保持三层语义清晰。

### data/raw

只放原始数据，例如 DANDI NWB 文件。不要修改。不要把 raw data 作为普通 git 内容提交。

### data/interim

放从 raw data 中抽取但尚未成为最终分析对象的数据，例如：

* wake square 行为数据
* spike matrix
* units metadata
* session index

### data/processed

放可直接用于复现论文图、训练模型、诊断稳定性的中间结果，例如：

* tuning curves
* optimized weights
* synthetic datasets
* figure-specific `.npz`
* diagnostics `.json`
* summary `.csv`

### reports

放面向阅读和汇报的结果：

* figures
* diagnostics summaries
* reproduction notes
* meeting reports

不要让代码直接依赖 `reports/figures/*.png`。图片是结果，不是输入。

## 13. 公式实现的 shape convention

默认约定：

```text
N: number of neurons
T: number of time steps
n_theta: number of theta bins
Phi: firing rate tuning curves, shape (N, n_theta)
X: input current tuning curves, shape (N, n_theta)
J: recurrent weight matrix, shape (N, N)
theta_grid: shape (n_theta,)
trajectory_x: shape (T, N) or (n_init, T, N)
trajectory_phi: shape (T, N) or (n_init, T, N)
```

如果某个脚本使用转置约定，例如 `(n_theta, N)`，必须在函数 docstring 和输出 `.npz` metadata 中明确说明。

所有 `.npz` 输出建议包含：

```python
metadata = {
    "paper_id": "...",
    "figure": "...",
    "script": "...",
    "date": "...",
    "shape_convention": "...",
    "config_path": "...",
    "random_seed": ...
}
```

## 14. Clark 项目的特殊注意事项

### 14.1 softplus inverse 与 rate floor

如果使用 softplus activation 的反函数，firing rate 为 0 或接近 0 会导致输入电流出现很大的负值，从而造成 weight solver 病态。处理方式必须显式记录：

* rate floor 的数值
* floor 是在 normalization 前还是后使用
* 是否改变 unit mean normalization
* 对 Jacobian stability 的影响
* 对 Fig. 3F convergence 的影响

任何修改 floor、smoothing、activation 参数的实验，都应该写入：

```text
reproduction/reports/figure3_rate_floor_stability.json
reproduction/reports/figure3_jacobian_*.json
```

### 14.2 lambda 与 c inhibition

Clark optimized weight 的稳定性不仅由 least-squares residual 决定，还受 regularization `lambda` 和 uniform inhibition `c` 影响。

每次修改 `lambda` 或 `c`，必须检查：

* flow residual on target manifold
* distance-to-manifold convergence
* tangent drift
* Jacobian leading eigenvalues
* mean activity mode stability

### 14.3 “复现成功”的判据

不要要求像素级复现。对本项目而言，复现成功应满足：

1. 数据处理 pipeline 与论文描述一致。
2. 主要统计量趋势一致。
3. 关键图的数量级一致。
4. 动力学行为一致，例如扰动后能回到 target manifold。
5. 已知偏差有诊断解释，例如 smoothing、floor、lambda、c、time step、theta discretization。

若结果从 1e-1 收敛到 1e-2，而论文达到 1e-3，不能简单判定失败；需要检查是否保持了准连续吸引子的时间尺度分离和稳定回归趋势。

## 15. Learning rule 方向的文献-代码桥接

当前核心研究问题是：

是否存在 local / unsupervised / biologically plausible learning rule，使得网络从初始随机或部分结构化连接出发，逐步逼近 Clark optimized weight matrix 所描述的 attractor dynamics？

这个问题需要把 Clark 的 optimized weight 看成 target 或 teacher，而不是直接把伪逆解当成 biological rule。

学习规则候选应该写成：

```math
\Delta J_{ij} = \eta F(\text{pre}_j, \text{post}_i, J_{ij}, \text{local traces}, \text{possibly modulatory variables})
```

优先考虑的局部变量：

```text
presynaptic rate: phi_j(t)
postsynaptic current: x_i(t)
postsynaptic rate: phi_i(t)
filtered pre trace: \bar{phi}_j(t)
filtered post trace: \bar{x}_i(t), \bar{phi}_i(t)
current synapse: J_ij
homeostatic term: target mean activity
local error proxy if available
```

禁止直接使用全局目标：

```text
full target manifold distance
global gradient of loss
complete pseudoinverse solution
future trajectory information
all-neuron covariance unless explicitly treated as nonlocal baseline
```

但允许设置对照组：

```text
global gradient baseline
offline optimized J baseline
nonlocal covariance rule baseline
teacher-student regression baseline
```

## 16. 每个 learning 实验的记录模板

每个 learning 实验应建立：

```text
learning/experiments/<date>_<short_name>/
├── config.yml
├── hypothesis.md
├── results.md
├── figures/
├── checkpoints/
└── diagnostics.json
```

`hypothesis.md` 模板：

```markdown
# Hypothesis

## Question
这个实验要回答什么？

## Literature source
- clark2025_symmetries:
- bell2024_plasticity_rules:
- oreilly2026_neocortex_learning:
- other:

## Proposed rule
写出 Delta J_ij 或 dJ_ij/dt。

## Locality
说明该规则使用哪些变量。逐项判断是否 local。

## Expected outcome
- weight similarity to J_opt:
- attractor stability:
- manifold recovery:
- spectral degeneracy:
- bump / ring structure:

## Baselines
1.
2.
3.

## Failure modes
1.
2.
3.
```

## 17. 报告与导师汇报规则

用于导师汇报的内容不要从聊天记录临时整理，而应从以下文件生成：

```text
references/<paper_id>/card.md
references/<paper_id>/open_questions.md
reproduction/reports/*.json
learning/experiments/*/results.md
reports/figures/*.png
```

每次 meeting 后，建议新增：

```text
reports/meetings/YYYY-MM-DD.md
```

模板：

```markdown
# Meeting YYYY-MM-DD

## What was shown
1.
2.

## Advisor feedback
1.
2.

## Decisions
1.
2.

## New questions
1.
2.

## Next coding tasks
1.
2.

## Next reading tasks
1.
2.
```

## 18. Codex 回答格式要求

Codex 在执行任务前，应先给出简短计划：

```markdown
I will:
1. Read ...
2. Check ...
3. Modify ...
4. Run ...
5. Update ...
```

执行完任务后，必须报告：

```markdown
Changed files:
- ...

Generated outputs:
- ...

Validation:
- ...

Remaining issues:
- ...

Updated literature/code maps:
- ...
```

如果没有更新 `references/<paper_id>/code_map.md`，必须说明原因。

## 19. 禁止事项

不要：

* 在不知道论文出处的情况下修改核心公式。
* 把 ChatGPT 的解释当作论文原文。
* 在代码中硬编码绝对路径。
* 把 raw data 和大体积缓存提交进 git。
* 把 PDF 全文复制进 markdown。
* 让 figure plotting script 同时承担数据清洗、模型训练、诊断和画图。
* 只看最终图片，不检查 `.npz`、`.json`、`.csv` 中的中间量。
* 在未检查 shape convention 的情况下转置矩阵。
* 在未记录参数的情况下改变 smoothing、floor、lambda、c、random seed。

## 20. 推荐的日常工作流

每天开始时：

1. 打开当前任务目录。
2. 阅读对应 Skill。
3. 查看 `.todo/TODO.md`。
4. 查看相关 `paper_card` 和 `code_map`。
5. 只选择一个最小闭环任务。

每完成一个闭环后：

1. 运行最小测试。
2. 生成或更新诊断。
3. 更新文献-代码映射。
4. 写一句当前结论。
5. 写下一步问题。

最理想的工作节奏是：

```text
上午：读 paper card + 修改小段代码
下午：跑实验 + 看 diagnostics
晚上：更新 notes + 明确下一步
```

本项目不要追求一次性完成大系统，而要追求每周稳定增加几个可追踪、可复现、可向导师解释的闭环。
