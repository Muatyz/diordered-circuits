# 共享研究笔记

本目录是 `model-dev` 与 `model-release` 共用的文献阅读笔记和研究知识库。这里的内容可以在 release 长时间训练期间继续整理，不需要发布或复制到两个模型目录。

## 边界

- 适合保存论文阅读笔记、公式推导、研究想法以及跨版本都需要查阅的背景材料。
- 核心模型、训练、诊断和绘图实现必须放在 `model-dev/learning/src/learning/`，验证后再发布到 release。
- notebook 不得成为训练入口、配置来源或运行时依赖；移动、编辑笔记不应改变已冻结网络的行为。
- notebook 若需要调用模型代码，应先激活明确的 `dev` 或 `pro` 环境，并用 `learning.__file__` 确认导入来源。
- 会影响实验结论的参数、代码版本和结果必须落入 resolved config、run 目录或正式报告，不能只保存在可变的 notebook 中。

论文 PDF 和结构化参考资料仍放在 `references/`；本目录主要保存日常阅读和思考过程。
