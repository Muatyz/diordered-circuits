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