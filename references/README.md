# 文献知识层

日常任务先读 `index.md`，再读相关论文目录中的 `card.md`、`equation_map.md`、
`figure_map.md` 和 `code_map.md`。只有在这些摘要不足或存在冲突时，才搜索
`paper.txt` 或打开 `paper.pdf`。

每篇论文采用唯一 `paper_id`：

```text
references/<paper_id>/
├── paper.pdf          # 原始论文
├── paper.txt          # 可重建的全文搜索缓存，不提交 Git
├── extraction.json    # 缓存指纹和提取元数据，不提交 Git
├── card.md            # 论文精华与项目意义
├── equation_map.md    # 公式到代码接口
├── figure_map.md      # 图到数据、脚本和输出
├── code_map.md        # 概念到现有代码
└── open_questions.md  # 未决问题
```

提取或刷新全文缓存：

```powershell
python scripts\lit_extract.py references\<paper_id>\paper.pdf
```

查找内容时优先定向搜索：

```powershell
rg -n "Eq\. 3|Figure 5|Jacobian|plasticity" references\<paper_id>
```
