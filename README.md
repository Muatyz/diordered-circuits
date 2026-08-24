# Disordered Circuits 工作区

本仓库把日常开发与长时间训练分开，避免训练过程中修改到实际运行的代码。

## 目录约定

- `model-dev/`：唯一的日常开发目录；代码、配置和测试默认只在这里修改。
- `model-release/`：从已验证的开发版本生成的冻结快照，用于长时间训练。除非用户明确要求发布新快照，否则不要修改，也不要让它导入 `model-dev` 中的代码。
- `model-release-<系列名>/`（如 `model-release-anneal/`）：可选的其他冻结快照，用于在默认 release 被占用时并行跑长训练。每个快照有独立的 `.release-manifest.json` 与训练锁，通过 `promote_release.py --release-root` 创建/更新。
- `runs/`：**所有分支共享的 run 目录**（仓库根下）。`paths.runs_root` 相对仓库根解析，因此 `model-dev` / `model-release` / `model-release-anneal` 三棵树的训练与诊断读写同一份 run 文件；根 `.gitignore` 已忽略该目录。旧 run 可按需用 `--run-dir` 显式指定路径读取。
- `notebooks/`：dev 与 release 共用的文献阅读笔记和研究知识库；它可以在长训练期间继续更新，但不属于任何冻结版本，也不能成为训练代码的运行时依赖。
- `data/`：两套代码共用的数据目录；reproduction 代码通过 `workspace_paths.py` 定位它，不依赖当前工作目录。
- `references/`：共享参考资料。
- `paper/`：论文与汇报材料。

## 后续 agent 的读取顺序

先读本文件，再按任务读取局部说明：

- learning 任务：`model-dev/learning/.SKILL.md`
- reproduction 任务：`model-dev/reproduction/.skills/SKILL.md`
- 文献背景：按当前任务读取 `notebooks/` 中相关笔记，不需要每次遍历整个知识库。
- **研究现状与下一步方向**：`notebooks/toy_model_status.md`（阶段性总结，随实验进展更新）
- 慢流形诊断原理：`notebooks/slow_manifold_diagnostics.md`

默认只检查和编辑 `model-dev/`。不要改动正在运行的 `model-release/`、`model-release-anneal/`、训练输出或权重文件。开发版本通过测试后，再由用户明确决定是否更新冻结快照。

## 更新冻结版本

发布脚本默认只预览差异；确认测试通过后才执行同步：

```powershell
python model-dev\scripts\promote_release.py
python model-dev\scripts\promote_release.py --apply
```

脚本同步 `model-dev/` 内的代码、配置与说明文件，并记录 `.release-manifest.json`；根目录 `notebooks/` 天然由两边共享，无需复制。脚本不会同步或删除 `runs/`、`reports/`、数据和缓存。默认要求 `model-dev` 已提交；只有确实需要发布未提交快照时才使用 `--allow-dirty`。

长训练开始前创建锁，结束后再解锁。锁存在时 `--apply` 会直接拒绝更新：

```powershell
python model-dev\scripts\promote_release.py --lock-training
# 在 model-release 中运行训练
python model-dev\scripts\promote_release.py --unlock-training
```

### 多 release-root（并行长训练）

当默认 `model-release` 被正在运行的实验占用、而 dev 又需要保持可编辑时，可以用
`--release-root` 创建另一个冻结快照（独立 manifest 与训练锁）：

```powershell
# 预览 → 应用（dev 有未提交修改时需 --allow-dirty）
python model-dev\scripts\promote_release.py --release-root model-release-anneal
python model-dev\scripts\promote_release.py --release-root model-release-anneal --apply --allow-dirty

# 独立 conda 环境（与 dev/pro 隔离）
conda env create -n pro-anneal -f model-release-anneal\learning\environment.yml
cd model-release-anneal\learning && python -m pip install -e .

# 训练前锁住该快照
python model-dev\scripts\promote_release.py --release-root model-release-anneal --lock-training
```

之后在该快照目录、对应环境（如 `pro-anneal`）下按标准流程跑长训练。实验结束后
`--unlock-training --release-root ...` 再更新快照。

## 环境与运行

[`work/`](work/) 只保留可直接阅读和手工修改的 `dev.bat`、`release.bat` 两个快捷脚本；可复制的 CMD 速查分别见 `model-dev/learning/README.md` 与 `model-release/learning/README.md`。

`model-dev/learning` 和 `model-release/learning` 暴露相同的 Python 包名 `learning`，必须使用两个独立环境，不能在同一环境中同时执行 editable install：

```powershell
cd model-dev\learning
conda env create -n dev -f environment.yml
conda activate dev
python -m pip install -e .
python -m pytest -q
```

冻结版本在独立的 `pro` 环境中安装并运行：

```powershell
cd model-release\learning
conda env create -n pro -f environment.yml
conda activate pro
python -m pip install -e .
```

具体训练、诊断和 reproduction 命令以各子目录的 README 与配置文件为准。每次实验应保留 resolved config、代码版本/快照标识和输出目录，以保证结果可追溯。
