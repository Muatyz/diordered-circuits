# Learning Dev 速查

`model-dev/learning` 用于日常开发、修改配置和短测试。命令以 Windows CMD 为准；核心代码约束见 `.SKILL.md`，诊断结构见 `docs/architecture/diagnostics.md`。

文献阅读与研究笔记统一放在仓库根目录 `notebooks/`，由 dev 和 release 共享。它不是训练入口或配置来源；需要在 notebook 中调用开发代码时使用 `dev` 环境，并先确认 `learning.__file__` 指向本目录。

## 环境部署

首次创建：

```bat
cd model-dev\learning
conda env create -n dev -f environment.yml
conda activate dev
python -m pip install -e .
```

已有环境更新：

```bat
cd model-dev\learning
conda env update -n dev -f environment.yml
conda activate dev
python -m pip install -e .
```

验证当前环境没有误指向 release：

```bat
python -c "import learning; print(learning.__file__)"
```

输出应包含 `model-dev\learning\src\learning`。若提示缺少 pip：

```bat
conda install -n dev -c conda-forge pip setuptools wheel
```

## `work\dev.bat` 模板

```bat
@echo off
cd /d "%~dp0..\model-dev\learning"
call conda activate dev
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_toy.yaml
```

这段可以直接复制到 BAT 文件，通常只需替换最后一行。

## 三个正式配置

```bat
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_toy.yaml
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_von_mises.yaml
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_heterogeneous.yaml
```

## 变视觉刺激协议（visual-annealing / "夜视"）

协议配置单独放在 `configs\protocols\`，与常规 experiment config 区分。当前
提供 `visual_anneal_vafidis.yaml`：训练期把 visual teacher 幅值从强到弱衰减，
让递归权重在训练末期必须在近 darkness 下维持 bump，观察 frozen-weight
darkness 性能（bump maintenance、低速 PI、pinning）是否改善。

三段调度（可用 `--set` 临时覆盖，必须严格递增且末段为 1.0）：

```yaml
visual:
  amplitude: 4.0        # 诊断/冻结权重使用；也是调度上限
  training_amplitude_schedule:
    - end_fraction: 0.50
      amplitude: 4.0
    - end_fraction: 0.75
      amplitude: 1.5
    - end_fraction: 1.00
      amplitude: 0.5
```

从零训练：

```bat
python -m learning.experiments.run_vafidis_toy --config configs\protocols\visual_anneal_vafidis.yaml --diagnostics-config configs\diagnostics\vafidis_diagnostics.yaml
```

从预训练权重继续（`--init-weights` 接受任意 run 的 `trained_weights.npz` /
`best_weights.npz` / `final_weights.npz`；只覆盖权重与静态几何，动态状态按
当前 seed 重新初始化，保证可复现）：

```bat
python -m learning.experiments.run_vafidis_toy --config configs\protocols\visual_anneal_vafidis.yaml --diagnostics-config configs\diagnostics\vafidis_diagnostics.yaml --init-weights runs\vafidis_baseline\YOUR_RUN_ID\best_weights.npz
```

其它 experiment config 也可组合该调度与 `--init-weights`：

```bat
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_von_mises.yaml --set visual.training_amplitude_schedule=... --init-weights ...
```

说明：调度只作用于训练期 teacher；冻结权重诊断仍使用 `visual.amplitude` /
`bump_attractor_cue_amplitude` 的配置值，不受影响。`light_excitation` 是常数
项不随调度衰减（只有调制分量衰减），这与论文 Eq. 4 的电流结构一致。

## 训练前检查

只解析最终配置，不训练：

```bat
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_toy.yaml --print-config
```

快速 smoke test：

```bat
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_toy.yaml --profile configs\profiles\code_smoke.yaml --run-id smoke_current --no-figures
```

## 常用替换

### Block multirate 加速

将下面一行复制为 `work\dev.bat` 的最后一行：

```bat
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_toy.yaml --profile configs\profiles\block_multirate.yaml
```

训练前可追加 `--print-config` 检查合并结果：

```bat
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_toy.yaml --profile configs\profiles\block_multirate.yaml --print-config
```

### Vafidis-only PI robustness profile

This profile keeps `predictive_local.py` and the Vafidis update equation
unchanged. It adds a broad-to-low-speed OU curriculum plus accepted-first,
moving-cue frozen PI checkpoint selection over headings and low velocities:

```bat
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_toy.yaml --profile configs\profiles\pi_robust_vafidis.yaml --print-config
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_toy.yaml --profile configs\profiles\pi_robust_vafidis.yaml --profile configs\profiles\pi_robust_n120.yaml --print-config
```

The first command retains 60 HD/HR cells. The second is the optional 120-cell
finite-size control. A selected checkpoint is behaviorally accepted only when
all configured bias, zero-drift, depinning, PVA-strength, and bump-contrast
thresholds pass. Otherwise the run records an explicit fallback rather than
claiming that its lowest score is a successful PI solution.

该 profile 设置：

```yaml
simulation:
  training_integration_method: block_multirate
  plasticity_update_interval_duration: 0.01
```

神经状态仍按 `simulation.dt` 逐步更新，只有较慢的局部可塑性按 0.01 秒 block 累积更新。它是显式选择的数值近似；默认 `single_clock` 仍是 release-aligned baseline。训练其他网络时只需把 `--config` 后的 YAML 换成 `vafidis_von_mises.yaml` 或 `vafidis_heterogeneous.yaml`。

正式诊断或全部诊断：

```bat
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_toy.yaml --diagnostics-config configs\diagnostics\vafidis_diagnostics.yaml
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_toy.yaml --diagnostics-config configs\diagnostics\diagnostics_all.yaml
```

临时覆盖参数：

```bat
python -m learning.experiments.run_vafidis_toy --config configs\experiments\vafidis_toy.yaml --set simulation.seed=7 --set simulation.train_duration=1000.0 --run-id seed7 --no-figures
```

## 测试与已有结果

```bat
python -m pytest -q
python -m learning.experiments.test_vafidis_toy --run-dir runs\vafidis_baseline\YOUR_RUN_ID --diagnostics-config configs\diagnostics\diagnostics.yaml
python -m learning.analysis.make_vafidis_figures --run-dir runs\vafidis_baseline\YOUR_RUN_ID
python scripts\inspect_run.py runs\vafidis_baseline\YOUR_RUN_ID
```

每个 run 的 `config_resolved.yaml` 才是实际训练参数。训练、重测和绘图产物均保存在配置指定的 `runs_root` 下。

## 更新 release

从仓库根目录执行；第一条只预览，第二条才写入：

```bat
python model-dev\scripts\promote_release.py
python model-dev\scripts\promote_release.py --apply
```

release 正在训练时不要执行 `--apply`。
