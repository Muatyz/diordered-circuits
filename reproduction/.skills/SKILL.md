---
name: reproduction-of-reference-disordered-circuits
description: 参照文献 Symmetries and Continuous Attractors in Disordered Neural Circuits, 对其依赖的数据集进行读取和处理, 复现出文献中的 Figures.
---

该子项目文件被放置于 /reproduction 目录下. 

# Background

- 使用者被假定具有一定的计算神经科学基础, 但是对于代码实现不熟练, 经常需要借助 AI agent 来完成代码的实现
- 目前代码库被设定在 Windows 11 Pro 上执行, 因此暂时不考虑仅 Linux 系统才可使用的数学库. 

# Reference

- /references/Symmetries and Continuous Attractors in Disordered Neural Circuits.pdf

    是参考的主要文献, 需要学习其提出的 Gaussian generative process 框架并且理解其插图是如何绘制得到的

- /references/Universal statistics of hippocampal place fields across species and dimensionalities.pdf

    是一篇相关的文献, 通过类似的 Gaussian generative process 提出了 place cell 的异质性神经场是如何产生的

# Data

- /data/raw/dandi_000939

    是参考文献使用过的数据集, 将其作为复现的 Baseline. 后续有两个方向: 
    
    1. 将该文献的 symmetry/generative-process 框架迁移到新的数据集;
    2. 将 Clark 等未在 000939 中重点分析的 sleep/opto 等部分提出新问题

# Guide

- /.chatgpt/chatgpt.md

    向 ChatGPT Plus 咨询的有关该项目的指导, 未来的研究需要在其设计的框架下进行. 


# Code generating criterion

0. 代码思路

    所有生成的代码均需要按照 Clark 原文中的思想(包括正文和 appendix 部分)来进行复现, 不可以因为了仅仅追求结果上的复现而引入一些原文中不存在的处理手法. 

1. 生成的 .py 文件中, 新定义函数需要有注释说明, 如

```python
def circular_smooth(values, sigma_bins):
    """
    对环形角度调谐曲线做高斯平滑。

    `values` 的最后一维被视为 0 到 2π 的环形方向 bin，因此平滑时使用
    `np.roll` 让首尾相接，避免 0 度和 360 度边界处出现断裂。
    `sigma_bins <= 0` 时不平滑，只返回浮点副本。
    """
    if sigma_bins <= 0:
        return values.astype(float, copy=True)

    radius = int(np.ceil(4 * sigma_bins))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel = kernel / kernel.sum()

    out = np.zeros_like(values, dtype=float)
    for offset, weight in zip(offsets, kernel):
        out += weight * np.roll(values, offset, axis=-1)
    return out
```

2. 实现目标

按照 /.todo/ 中指定的任务来对代码库进行实现.

3. 代码库结构
    - reproduction: 用于复现 Clark 这篇文章中的一些计算和插图
        - /reports: 存放生成的图, 表或者文字性报告
        - /src: 存放代码实现
    - /data
        - /raw: 存放 dandi000939 等原始数据集
        - /data/interim: 存放中间处理结果, 避免反复读取原始的(/data/raw)中的 .nwb 文件造成速度较低
        - /data/processed: 存放数据预处理结果, 用于后续函数直接读取进行绘图
    - /references: 项目相关的文献资料
    
    
    

    - 虽然实现目标是对各插图进行实现, 但是最好不要直接写作 `plot_figure_x.py` 的结构, 而是将可复用的新定义函数/工具放在 `src/utils.py` 中, 以便后续在其他插图中复用. 例如上面定义的 `circular_smooth` 就是一个可以复用的工具函数, 可以放在 `src/utils.py` 中, 而不是写在 `plot_figure2_reproduction.py` 中. 

4. 代码依赖环境

    通过本地的 Python 3.11(random) 环境进行代码执行. 

    ```python
    conda activate random
    python -m pip install -U dandi pynwb h5py pandas numpy scipy matplotlib tqdm pyarrow ipykernel
    ```

5. 代码执行

    由于数据集较大, 使用 CPU 计算通常较为耗时, 因此若要测试代码的正确性, 建议可以忽略常规的 Codex 时间限制, 运行出完整结果再下判断, 不可为了节省时间引入一些错误的加速算法(通常在做出可观结果之后再进行运行速度优化)