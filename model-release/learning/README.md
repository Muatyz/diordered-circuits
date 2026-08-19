# Learning Release 速查

`model-release/learning` 是长时间训练使用的冻结快照。训练过程中不要修改这里的 Python 或 YAML；日常开发应在 `model-dev/learning` 完成。

文献阅读与研究笔记统一放在仓库根目录 `notebooks/`，由 dev 和 release 共享，但不属于冻结快照，也不是训练运行时依赖。需要从 notebook 检查当前 release 时使用 `pro` 环境，并先确认 `learning.__file__` 指向本目录。

## 环境部署

首次创建：

```bat
cd model-release\learning
conda env create -n pro -f environment.yml
conda activate pro
python -m pip install -e .
```

已有环境更新：

```bat
cd model-release\learning
conda env update -n pro -f environment.yml
conda activate pro
python -m pip install -e .
```

验证当前环境没有误指向 dev：

```bat
python -c "import learning; print(learning.__file__)"
```

输出应包含 `model-release\learning\src\learning`。若提示缺少 pip：

```bat
conda install -n pro -c conda-forge pip setuptools wheel
python -m pip install -e .
```

## `work\release.bat` 模板

```bat
@echo off
cd /d "%~dp0..\model-release\learning"
call conda activate pro
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_toy.yaml
```

这段可以直接复制到 BAT 文件，通常只需替换最后一行。

## 三个正式配置

```bat
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_toy.yaml
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_von_mises.yaml
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_heterogeneous.yaml
```

## Block multirate 加速

将下面一行复制为 `work\release.bat` 的最后一行：

```bat
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_toy.yaml --profile configs\profiles\block_multirate.yaml
```

正式启动前先检查最终配置：

```bat
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_toy.yaml --profile configs\profiles\block_multirate.yaml --print-config
```

该 profile 保持神经状态使用 `simulation.dt`，只把局部可塑性改为每 0.01 秒 block 累积更新。它属于 opt-in 数值近似；默认 `single_clock` 仍是 release-aligned baseline。训练其他网络时只需替换 `--config` 后的 YAML。

## 长训练前检查

确认导入目录并打印最终配置，不会开始训练：

```bat
python -c "import learning; print(learning.__file__)"
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_toy.yaml --print-config
```

确认无误后，再从最后一条命令删除 `--print-config`。

## 诊断与已有结果

训练并运行正式诊断：

```bat
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_toy.yaml --diagnostics-config configs\diagnostics\vafidis_diagnostics.yaml
```

重新诊断或绘图：

```bat
python -m learning.experiments.test_vafidis_toy --run-dir runs\vafidis_baseline\YOUR_RUN_ID --diagnostics-config configs\diagnostics\diagnostics.yaml
python -m learning.analysis.make_vafidis_figures --run-dir runs\vafidis_baseline\YOUR_RUN_ID
python scripts\inspect_run.py runs\vafidis_baseline\YOUR_RUN_ID
```

核心训练文件会在诊断前保存；实际参数以 run 内的 `config_resolved.yaml` 为准。

## 测试说明

```bat
python -m pytest -q
```

当前若仅 `tests\test_config_defaults.py` 因旧配置预期而失败，应先核对 YAML 与测试意图；不要为了全绿而任意修改正式训练参数。

## 从 dev 更新冻结快照

必须先结束所有 release 训练，再从仓库根目录执行：

```bat
python model-dev\scripts\promote_release.py
python model-dev\scripts\promote_release.py --apply
```

更新后重新运行 `conda env update`、`python -m pip install -e .` 和 `--print-config` 检查。
