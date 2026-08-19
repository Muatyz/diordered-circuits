# src/inspect_nwb.py
# 生成 reports/nwb_inventory.csv。这个 CSV 会告诉你：有多少 NWB 文件、每个文件有多少 units、有哪些 acquisition/processing/interval 字段、head direction 可能藏在哪个 processing module 里、是否有 trials/epochs/invalid_times/stimulus intervals
from pathlib import Path
import json
import pandas as pd
from pynwb import NWBHDF5IO

RAW = Path("data/raw/dandi_000939")

def summarize_nwb(path: Path) -> dict:
    """
    汇总单个 NWB 文件的顶层结构信息。

    返回一个字典，包含文件路径、session 信息、subject 信息、acquisition /
    processing / interval 键、units 表是否存在以及 unit 数量等。这个摘要会被
    写入 CSV，作为了解数据集结构的清单。
    """
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()

        out = {
            "file": str(path),
            "identifier": nwb.identifier,
            "session_description": nwb.session_description,
            "session_start_time": str(nwb.session_start_time),
            "subject": None,
            "acquisition_keys": list(nwb.acquisition.keys()),
            "processing_keys": list(nwb.processing.keys()),
            "interval_keys": list(nwb.intervals.keys()),
            "has_units": nwb.units is not None,
            "unit_columns": list(nwb.units.colnames) if nwb.units is not None else [],
            "n_units": len(nwb.units.id[:]) if nwb.units is not None else 0,
        }

        if nwb.subject is not None:
            out["subject"] = {
                "subject_id": nwb.subject.subject_id,
                "species": nwb.subject.species,
                "sex": nwb.subject.sex,
                "age": nwb.subject.age,
            }

        return out

def main():
    """
    命令行入口：扫描原始 NWB 目录并生成结构清单 CSV。

    每个 NWB 文件会调用 `summarize_nwb`；单个文件失败时记录错误并继续处理其余
    文件。输出保存到 `reports/nwb_inventory.csv`。
    """
    files = sorted(RAW.rglob("*.nwb"))
    print(f"Found {len(files)} NWB files")

    rows = []
    for f in files:
        print(f"Inspecting {f}")
        try:
            rows.append(summarize_nwb(f))
        except Exception as e:
            rows.append({"file": str(f), "error": repr(e)})

    df = pd.DataFrame(rows)
    Path("reports").mkdir(exist_ok=True)
    df.to_csv("reports/nwb_inventory.csv", index=False)
    print(df[["file", "n_units", "acquisition_keys", "processing_keys", "interval_keys"]].head())

if __name__ == "__main__":
    main()
