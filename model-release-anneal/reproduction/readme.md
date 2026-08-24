# 环境配置

项目使用 Python 3.11。首次在新主机上配置环境时运行：

```bash
conda env create -f environment.yml
conda activate random
python -m ipykernel install --user --name random --display-name "Python (random)"
```

依赖用途：

- 数值计算与数据处理：`numpy`、`pandas`、`scipy`
- 论文文本缓存：`pymupdf`（缺失时 `scripts/lit_extract.py` 会回退到系统 `pdftotext`）
- NWB/HDF5 数据读取：`pynwb`、`h5py`
- Parquet 文件读写：`pyarrow`
- 绘图与进度显示：`matplotlib`、`tqdm`
- DANDI 数据下载：`dandi`
- Notebook kernel：`ipykernel`

# 数据处理流程

1. 下载数据集

```bash
dandi download "https://dandiarchive.org/dandiset/000939/0.240528.1542" -o data\raw\dandi_000939
```

2. 列出数据集中的文件

```bash
dandi ls -r data\raw\dandi_000939 > reports\dandi_000939_filelist.txt
```

3. 生成 NWB 文件的 inventory 报告

```bash
python src\inspect_nwb.py
```

查看 `reports/nwb_inventory.csv`，了解 NWB 文件的内容和结构。

4. 更深层次的 NWB 结构检查: 确认 head direction 在哪个 data inteface 里

```bash
python src/deep_inspect_nwb.py
```

5. 提取 wake_square 阶段的行为和 spike 数据

```bash
python src\extract_wake_square.py
```

输出：

- `data/interim/session_index.csv`: 每个 NWB/session 的提取状态和 unit 数量
- `data/interim/*_behavior_wake_square.parquet`: `time_s`, `head_direction_rad`, `x_cm`, `y_cm`
- `data/interim/*_units_meta.parquet`: unit 元数据、初筛和 QC 标记
- `data/interim/*_spikes_wake_square.npz`: 每个 unit 在 wake_square 内的 spike times

6. 计算 head-direction tuning curves

```bash
python src\compute_hd_tuning.py
```

输出：

- `data/processed/hd_tuning_index.csv`
- `data/processed/*_hd_tuning_100bins.npz`

每个 `.npz` 包含 occupancy、spike counts、raw firing rate、CV Gaussian smoothing 后的 firing rate、unit-mean-normalized tuning matrix、split-half reliability。

7. 画快速检查图

```bash
python src\plot_hd_tuning_examples.py
```

输出到 `reports/figures/`。
