# LearnPI：Vafidis 论文原始代码

本目录保存 Vafidis、Owald、D'Albis 与 Kempter 为论文 [*Learning accurate path integration in ring attractor models of the head direction system*](https://doi.org/10.7554/eLife.69841)（eLife，2022）公开的原始 Python 代码，作为本项目 `learning/` 子项目中 Vafidis-style predictive local plasticity 的基线。上游代码见 [panvaf/LearnPI](https://github.com/panvaf/LearnPI)，复现图所需的预训练网络和数据见 [GIN 数据仓库](https://gin.g-node.org/pavaf/LearnPI)。

这里的文件应尽量保持为上游原貌。若要开发、重构或扩展模型，请使用 `learning/src/learning/`，不要直接把本目录改造成新实现。

## 代码结构

| 文件 | 作用 | 是否需要外部数据 |
|---|---|---|
| `fly_rec.py` | 主网络动力学与局部学习规则；提供 `simulate()` 和单步 `network()` | 否 |
| `utilities.py` | visual input、sigmoid、OU 角速度、PVA/COM 解码、velocity-gain 测试及文件名工具 | 否 |
| `run_simulation.py` | 生成 OU 训练轨迹、训练网络并保存权重和误差历史 | 只要求输出目录已存在 |
| `generate_plots.py` | 分析单个训练网络，绘制权重、活动、PI error、velocity gain 等图 | 是：标准网络和 `PI_example*.npz` |
| `stability.py` | bump 稳定位置测试；可选噪声扩散系数估计 | 是：标准网络 |
| `basin_test.py` | 从均匀分布的预设 bump 出发，映射冻结网络的 zero-input 离散 basin | 是：标准网络 |
| `multinet_plots.py` | 跨网络、突触延迟、目标 gain、扩散系数等汇总图 | 是：完整 `Parallel/` 和汇总数据 |
| `EB_synapses.py` | 果蝇 EB connectome 突触位置与 SVM 可分性分析 | 是：connectome CSV；可选缓存的 SVM 分数 |
| `math_appendix.py` | 数学附录中的连续约化模型及其作图 | 否 |

主模型中有 `n_neu=60` 个 HD 单元和同样数量的 HR 单元。60 个 HD 单元按 odd/even 成对表示 30 个角位置；HR 的前、后各 30 个单元分别构成左右旋转通路。核心权重约定如下：

```text
w.shape == (n_hd, n_hr + n_hd)
w[:, :n_neu]       = HR -> HD，可塑
w[:, n_neu:]       = HD -> HD，可塑
w_rot              = HD -> HR，结构预设、默认不可塑
```

每个时间步先生成视觉和角速度输入，再更新 HR、HD dendritic current/voltage、somatic firing rate、局部预测误差和 PSP trace，最后在训练阶段用
`outer(error, PSP)` 更新 `w`。测试调用将 `train=False`，因此不会改变权重。这正是 `learning/` 新实现需要复现和比较的 local-learning baseline；代码不使用 backpropagation 或全局 loss optimizer。

## 1. 建立兼容环境

这是 2021 年前后的研究代码，使用了新版 NumPy、Pandas、Matplotlib 和 SciPy 已移除的 API，例如 `np.int`、`DataFrame.append`、`matplotlib.colors.DivergingNorm` 和 `scipy.stats.t.interval(alpha=...)`。为了在不修改原始代码的前提下运行，推荐单独建立旧版环境：

```powershell
conda create -n learnpi-original -c conda-forge `
  python=3.8 numpy=1.21.6 scipy=1.7.3 matplotlib=3.3.4 `
  pandas=1.3.5 scikit-learn=1.0.2 astropy=4.3.1
conda activate learnpi-original
```

所有后续命令都应从本目录启动，因为原代码使用 `Path(os.getcwd()).parent` 寻找数据，并用 Windows 反斜杠拼接路径：

```powershell
cd D:\codefiles\python\diordered-circuits\learning\original
```

按原样运行时，本代码实际是 Windows 路径限定的。在 Linux/macOS 上需要先把各脚本中的 `"\\savefiles\\..."` 路径改为 `pathlib.Path` 或平台无关的 `os.path.join`。

## 2. 下载并放置论文数据

推荐使用 [GIN 客户端](https://github.com/G-Node/gin-cli) 下载，因为普通 ZIP 对大于 10 MB 的 git-annex 文件只包含指针，不包含真实数据：

```powershell
# 在任意临时下载目录执行；命令会创建 LearnPI 文件夹
gin get pavaf/LearnPI
cd LearnPI
gin get-content .
```

然后将下载结果中的 `savefiles/` 和 `Connectome Synapses/` 放到本目录的父目录 `learning/` 下。最终结构必须是：

```text
learning/
|-- original/
|   |-- README.md
|   |-- fly_rec.py
|   |-- run_simulation.py
|   `-- ...
|-- savefiles/
|   |-- PI_example.npz
|   |-- PI_example1.npz
|   |-- PI_example_360_max.npz
|   |-- Diff1000tr.npz
|   |-- Diff1000trNoise.npz
|   |-- PIerr.npy
|   |-- PIerrNoise.npy
|   |-- PIerrPerturb.npy
|   |-- SVM3Drbf.npz
|   `-- trained_networks/
|       |-- fly_rec2Enough...npz
|       `-- Parallel/
|           |-- Main_Net/
|           |-- Theor_Lim/
|           |-- Adapt_Gain/
|           `-- Perturb_Conn/
`-- Connectome Synapses/
    |-- ER2 to E-PG (...).csv
    |-- ER4d to E-PG (...).csv
    |-- PEN1 to E-PG (...).csv
    `-- PEN2 to E-PG (...).csv
```

可在 `learning/original` 中执行以下检查：

```powershell
Test-Path ..\savefiles\trained_networks
Test-Path ..\savefiles\PI_example.npz
Test-Path "..\Connectome Synapses"
```

三个结果均应为 `True`。只运行 `math_appendix.py` 时不需要下载这些数据；只从头训练主网络时，仅需先创建输出目录：

```powershell
New-Item -ItemType Directory -Force ..\savefiles\trained_networks
```

## 3. 最快验证方式

### 3.1 运行无需数据的约化模型

```powershell
python -i math_appendix.py
```

该脚本直接训练数学附录中的连续约化模型并生成 Appendix 5 Figure 1–4。上游说明给出的典型运行时间约为 3 分钟。脚本执行完成进入 `>>>` 后，如图窗没有自动显示，执行：

```python
pl.show()
```

关闭图窗后用 `exit()` 离开交互解释器。

### 3.2 使用预训练网络生成主图

确认 GIN 数据已放好后：

```powershell
python -i generate_plots.py
```

脚本会在若干处调用 `plt.show()`；每次需关闭当前图窗，脚本才会继续。脚本最后回到 `>>>` 后再执行一次：

```python
plt.show()
```

默认配置会：

1. 从 `..\savefiles\trained_networks\` 加载由参数编码出的标准 `.npz`；
2. 绘制视觉/速度输入、训练误差、三类权重和权重演化；
3. 因 `perturb_conn=True`，在内存中给最终连接加入噪声；
4. 使用 `..\savefiles\PI_example.npz` 模拟 light–dark–light 路径积分；
5. 使用缓存或重新计算 velocity gain 和 darkness PI error。

注意：代码没有调用 `savefig()`，图默认只显示、不写入文件。需要保存时请在交互提示符中对当前 figure 调用 `plt.savefig(...)`，或在相应绘图段后自行加入 `fig.savefig(...)`。顶部的 `save=True` 保存的是计算结果 `.npz`，不是图片；它还可能覆盖同名分析文件，因此首次运行建议保留 `save=False`。

若只想分析未经连接噪声扰动的网络，将 `generate_plots.py` 顶部改为：

```python
perturb_conn = False
```

## 4. 从头训练主网络

`run_simulation.py` 没有 CLI 参数；所有设置都在文件顶部的 `sim_run`、`store_f`、`train`、`adapt_gain` 和 `params` 中修改。

默认训练命令为：

```powershell
python run_simulation.py
```

默认 `sim_run="2Enough"` 表示 80,000 秒模型时间，`dt=0.0005 s`，约有 1.6 亿个 Euler step。上游作者报告在中档游戏本上约需 8 小时；实际内存和耗时取决于机器。训练轨迹和权重都没有固定随机种子，因此重新训练不会逐元素复现发布的网络。

首次检查安装和路径时，建议暂时把：

```python
sim_run = "Look"       # 5 s 模型时间，仅作 smoke test
store_f = False         # 不保存每个时间步的全部 firing rate
train = True
adapt_gain = False
```

改完后运行 `python run_simulation.py`。成功时会打印完成百分比和总运行时间，并在以下目录生成网络文件：

```text
learning/savefiles/trained_networks/fly_rec<参数编码>.npz
```

`utilities.sim_time()` 支持的时长标签为：

| 标签 | 模型时间 |
|---|---:|
| `Look` | 5 s |
| `Short` | 200 s |
| `Medium` | 4,000 s |
| `2Medium` | 8,000 s |
| `4Medium` | 16,000 s |
| `Enough` | 40,000 s |
| `2Enough` | 80,000 s |
| `Long` | 200,000 s |

训练输出的主要数组为：

| key | 内容 |
|---|---|
| `w` | 训练期间按百分比采样的可塑权重；最终权重为 `w[:, :, -1]` |
| `w_rot` | 固定的 HD→HR 权重 |
| `f`, `f_rot` | `store_f=True` 时的完整 firing-rate history；否则为标量占位值 |
| `err` | `store_f=True` 时的逐时刻局部误差，否则为 100 个训练进度点的平均误差 |
| `params` | 参数字典（object array，读取时通常需要 `allow_pickle=True`） |

训练后若要让 `generate_plots.py` 找到结果，其顶部 `sim_run` 与 `params` 必须和训练时一致，因为 `utilities.filename(params)` 会把参数编码进文件名。

### Gain adaptation

`adapt_gain=False` 表示从随机可塑权重开始正常训练。若要在已有标准网络上继续做目标 gain 适配，将它改为数值，例如：

```python
adapt_gain = 0.5
```

脚本会先按原参数文件名载入标准网络，再把 `gain` 改为目标值，并用 `4Medium` 时长继续训练。所需基准网络必须已经位于 `..\savefiles\trained_networks\`。

## 5. 各分析脚本的使用方法

### `stability.py`

```powershell
python -i stability.py
```

默认加载标准网络，对论文 Figure 2D 使用的 6 个位置分别给予 2 秒 visual stimulation，再在 darkness 中检查 bump 是否保持。控制台会打印稳定位置数量；最后执行 `plt.show()` 查看图。

若将顶部 `d_coeff=True`，脚本还会对 `n_levels=np.arange(0, 1.1, 0.1)` 的每个噪声水平运行 1,000 次 darkness simulation。该设置非常耗时。结果只保留在交互变量 `D_coeff` 中，不会自动保存；例如可在运行结束后执行：

```python
np.savez(r"..\savefiles\Diff1000tr.npz", D=D_coeff, n_levels=n_levels)
```

### `multinet_plots.py`

```powershell
python -i multinet_plots.py
```

它不是单网络入口，会一次性读取 `Parallel/Main_Net` 的 12 个 seed、`Theor_Lim/run0..run5`、`Adapt_Gain` 的多组网络，以及根 `savefiles/` 下的扩散和 PI-error 汇总文件。缺少任一预期文件时会直接报 `FileNotFoundError`。脚本中途有一次阻塞式 `plt.show()`；关闭图窗让其继续，最终回到提示符后再次执行 `plt.show()`。

### `EB_synapses.py`

```powershell
python -i EB_synapses.py
```

脚本读取 `..\Connectome Synapses\` 中 16 个 E-PG 神经元的四类 CSV。若 `..\savefiles\SVM3Drbf.npz` 存在，则直接使用缓存的 nested-CV 分数；否则会现场运行每个神经元 30 次 nested cross-validation，耗时明显更长。执行完成后输入 `plt.show()`。

### `basin_test.py`：均匀初态的离散 basin 测试

这是添加在原始代码外围的诊断入口；它不修改 `fly_rec.py` 或学习规则。每个 trial 都调用原始的 `fly_rec.simulate()`，并固定：

```text
train=False   # 冻结权重
day=False     # 无视觉输入
stab=True     # 无速度输入
```

测试以等宽角区间的中点在 360 度上均匀设置初始 heading（例如 36 个初态为 `5°, 15°, ..., 355°`）。初态直接使用 release code 在 `simulate()` 内部已有的矩形 bump 初始化，不额外使用 visual cue，因此测量的是同一组权重下的 autonomous zero-input dynamics。使用半个间距的相位偏移，是为了绕开原发布矩形初始化器在恰好 `0°` 时的跨圈切片边界特例，而不修改该函数本身。

标准网络已经放在默认位置时可直接运行：

```powershell
python basin_test.py `
  --initial-conditions 36 `
  --duration 30 `
  --sample-interval 0.25 `
  --basin-tolerance-deg 5
```

也可明确指定任意训练网络和输出目录：

```powershell
python basin_test.py `
  --network "..\savefiles\trained_networks\<network>.npz" `
  --output-dir "..\savefiles\basin_tests\my_test" `
  --initial-conditions 120 `
  --duration 120
```

输出包括：

```text
bump_basin_history.npz       # PVA/peak 轨迹、强度、contrast、末段漂移、终态 firing rate
bump_basin_summary.json      # candidate basin 中心、occupancy、末段漂移与质量指标
bump_basin_diagnostics.png   # trajectory、endpoint map、终点直方图、PVA strength
```

主要判读方式：

- endpoint map 接近对角线且每个起点保持在不同终点：接近 continuous ring；
- 多段初始角收缩到少数水平带：存在 pinned/discrete basins；
- `candidate_basin_count / valid_endpoint_count` 越小，终点压缩越明显；
- `valid_endpoint_count` 很少或 final PVA strength 很低：bump 已消失，不能解释为 basin 收敛；
- `late_abs_pva_drift_*_deg_s` 仍较大：末段仍在漂移，当前 duration 下不能声称已经收敛；
- 30 s 时轨迹仍在缓慢移动时，应增加 `--duration`；增加 `--initial-conditions` 只提高 basin boundary 的角度分辨率。

`--basin-tolerance-deg` 表示一个候选 basin 内终点允许的最大角直径。省略时自动取初始角间距的 0.45 倍，避免仅因增加初态采样密度而降低 compression ratio；显式比较多次运行时则应固定同一个角度值。聚类采用 circular complete-link 约束，不会因相邻点链式连接而把均匀覆盖整圈的连续终点误并为一个 basin。自动聚类只是 endpoint map 的摘要，科学判断仍应结合完整 trajectory、PVA strength、末段漂移与不同 duration 的结果。

默认 `--test-noise-std 0`，所以诊断是确定性的；若显式加入 test noise，应同时记录 `--seed`。脚本将数据保存在 `learning/savefiles/basin_tests/`，不会覆盖训练网络。

## 6. 论文图与脚本对应关系

下表按作者在 GIN 仓库给出的原始复现说明整理。修改选项后，应在下一组实验前恢复默认值。

| 论文图 | 脚本与设置 |
|---|---|
| Fig. 1B,C；2A–C；3A–C,E,F；Fig. 3 Supplement 2 | `generate_plots.py`，设 `vel_hist=True` |
| Fig. 1 Supplement 1（其中 A 左上也用于 Fig. 1E） | `EB_synapses.py` |
| Fig. 2D | `stability.py` |
| Fig. 3D；Fig. 4；Fig. 4 Supplement 1A；Appendix 1 Fig. 1E；Appendix 2 Fig. 1A–C | `multinet_plots.py` |
| Fig. 3 Supplement 1 | `generate_plots.py`，设 `cut_exc=True` |
| Fig. 3 Supplement 3 | `data_dir='\\savefiles\\trained_networks\\Parallel\\Perturb_Conn\\'`，并将 `params['run']` 设为 `0..11` |
| Fig. 4 Supplement 1B | `PI_err=False`，`params['gain']=0.125` |
| Fig. 4 Supplement 1C | `PI_err=False`，`params['gain']=10`；同时按图调整 gain plot 的边距 |
| Fig. 4 Supplement 1D–F | `sim_run='Long'`，`PI_err=False`，`params['gain']=-1` |
| Appendix 1 Fig. 1A–D | `params['n_sigma']=0.7` |
| Appendix 2 Fig. 1D,E | `sim_run='4Medium'`，`PI_err=False`，`params['tau_s']=1` |
| Appendix 3 Fig. 1 | `params['vary_w_rot']=True`，`params['adj']=True` |
| Appendix 3 Fig. 2 | `PI_example_dir='\\savefiles\\PI_example1.npz'`，`vary_w_rot=True`，并在构造 `filename` 后追加 `'NoLearn'`；将 PI 图的 `err_lim` 设为 180 |
| Appendix 3 Fig. 3 | `PI_example_dir='\\savefiles\\PI_example_360_max.npz'`，`PI_err=False`，`rand_w_rot=True` |
| Appendix 5 Fig. 1–4 | `math_appendix.py` |

`multinet_plots.py` 使用的扩散系数来自对相应网络运行 `stability.py` 且设置 `d_coeff=True` 的结果。

## 7. 常见问题

**`FileNotFoundError` 指向 `learning\savefiles`**

通常是启动目录错误或 GIN 数据层级错误。先 `cd learning\original`，再确认 `..\savefiles` 与 `..\Connectome Synapses` 是 `original` 的同级目录。

**`AttributeError: module 'numpy' has no attribute 'int'`**

当前环境中的 NumPy 太新。使用上面的兼容环境；若只是临时移植，可将 `np.int` 改成内置 `int`，但这会改变本目录的原始基线代码。

**`DataFrame` 没有 `append`，或 Matplotlib 没有 `DivergingNorm`**

分别说明 Pandas 或 Matplotlib 太新。使用推荐的固定版本。

**脚本似乎卡住**

先检查是否有 Matplotlib 图窗等待关闭；`plt.show()` 是阻塞式的。若控制台仍在输出百分比，则模型还在模拟。默认主训练、velocity scan、1,000-trial PI error 和 diffusion scan 都不是快速任务。

**只下载 ZIP 后 `.npz` 无法读取**

GIN 的普通 ZIP 对 annex 大文件可能只给出指针文件。改用 `gin get pavaf/LearnPI`，或在 GIN 网页逐个下载对应大文件的真实内容。

**结果每次不同**

原始脚本没有统一设置随机种子；OU trajectory、初始权重、噪声和部分重复实验都会变化。需要精确重复自行训练结果时，应在入口脚本生成轨迹和初始化网络之前显式调用 `np.random.seed(<seed>)`，并记录参数与 seed。

## 8. 与本项目新实现的关系

本目录用于回答“论文原始实现实际上如何更新状态、学习权重和评估 PI”这一基线问题。新实现位于 `learning/src/learning/`，应继续遵守训练/测试分离、测试冻结权重、circular error、PVA/COM decode 和局部变量学习等约束。比较两者时尤其要核对：

- HD odd/even 配对与左右 HR wing 的索引约定；
- `w` 的 `[HR | HD]` presynaptic column 排列；
- firing rate 的单位是 kHz，图中通常乘 1,000 转成 spikes/s；
- 主动力学使用 `dt=0.5 ms`，而部分时间常数在 `params` 中以 ms 表示；
- visual teacher 只应出现在训练或明确的 cue/light 阶段；
- darkness/测试阶段必须使用 `train=False`。

这些约定比逐像素复现论文图片更重要，也是后续检验 bump maintenance、velocity-driven path integration 和学习后权重结构时的对照基础。
