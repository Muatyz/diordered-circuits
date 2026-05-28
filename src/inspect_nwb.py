# src/inspect_nwb.py
# 生成 reports/nwb_inventory.csv。这个 CSV 会告诉你：有多少 NWB 文件、每个文件有多少 units、有哪些 acquisition/processing/interval 字段、head direction 可能藏在哪个 processing module 里、是否有 trials/epochs/invalid_times/stimulus intervals
from pathlib import Path
import json
import pandas as pd
from pynwb import NWBHDF5IO

RAW = Path("data/raw/dandi_000939")

def summarize_nwb(path: Path) -> dict:
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