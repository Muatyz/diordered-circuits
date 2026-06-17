# src/extract_wake_square.py
from pathlib import Path
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO
from tqdm import tqdm

RAW = Path("data/raw/dandi_000939/000939")
OUT = Path("data/interim")
OUT.mkdir(parents=True, exist_ok=True)


def tags_contain(tags, target):
    """
    判断 NWB epoch 的 tags 中是否包含目标标签。

    NWB 读出的 tags 可能不是普通 Python 字符串列表，因此这里先统一转成
    `str` 再比较，避免类型差异影响 `wake_square` 的筛选。
    """
    return target in [str(x) for x in list(tags)]


def object_to_text(value):
    """
    把 NWB 对象或其他复杂对象转成适合写入 parquet 的短文本。

    parquet 对 object 列比较挑剔；如果对象有 `name` 属性就优先保留名字，
    否则退化为 `str(value)`。`None` 保持为空值。
    """
    if value is None:
        return None

    name = getattr(value, "name", None)
    if name is not None:
        return str(name)

    return str(value)


def make_parquet_safe(df):
    """
    清理 DataFrame 中的 object 列，使其更容易写成 parquet。

    对每个 object 类型列应用 `object_to_text`，把 NWB 对象、数组式对象等复杂值
    压成字符串，避免 parquet writer 因无法推断类型而失败。
    """
    safe = df.copy()

    for col in safe.columns:
        if safe[col].dtype == "object":
            safe[col] = safe[col].map(object_to_text)

    return safe


def get_wake_square_interval(nwb):
    """
    从 NWB 的 epochs 表中读取唯一的 `wake_square` 时间区间。

    如果没有找到或找到多个 `wake_square` epoch，就抛出异常，避免后续行为和
    spike 裁剪使用含糊的时间范围。
    """
    epochs = nwb.intervals["epochs"].to_dataframe()
    mask = epochs["tags"].apply(lambda tags: tags_contain(tags, "wake_square"))
    rows = epochs.loc[mask]

    if len(rows) != 1:
        raise ValueError(f"Expected exactly one wake_square epoch, got {len(rows)}")

    row = rows.iloc[0]
    return float(row["start_time"]), float(row["stop_time"])


def load_behavior(nwb, start, stop):
    """
    读取并裁剪 wake_square 阶段的行为数据。

    从 behavior processing module 中取 head direction 和 position。若二者时间戳
    完全一致，直接合并；否则把 position 插值到 head-direction 的时间轴上。
    返回包含时间、头方向弧度、x/y 位置的 DataFrame。
    """
    hd_ts = nwb.processing["behavior"]["CompassDirection"].spatial_series["head-direction"]
    pos_ts = nwb.processing["behavior"]["Position"].spatial_series["position"]

    t_hd = np.asarray(hd_ts.timestamps[:], dtype=float)
    hd = np.asarray(hd_ts.data[:], dtype=float).squeeze()

    t_pos = np.asarray(pos_ts.timestamps[:], dtype=float)
    pos = np.asarray(pos_ts.data[:], dtype=float)

    # 这个数据集中 head direction 和 position 的 timestamps 通常应当一致。
    # 保险起见，如果完全一致就直接合并；否则把 position 插值到 head-direction 时间轴。
    if len(t_hd) == len(t_pos) and np.allclose(t_hd, t_pos, equal_nan=True):
        x = pos[:, 0]
        y = pos[:, 1]
    else:
        x = np.interp(t_hd, t_pos, pos[:, 0])
        y = np.interp(t_hd, t_pos, pos[:, 1])

    keep = (
        np.isfinite(t_hd)
        & np.isfinite(hd)
        & np.isfinite(x)
        & np.isfinite(y)
        & (t_hd >= start)
        & (t_hd < stop)
    )

    behavior = pd.DataFrame(
        {
            "time_s": t_hd[keep],
            "head_direction_rad": hd[keep] % (2 * np.pi),
            "x_cm": x[keep],
            "y_cm": y[keep],
        }
    )

    return behavior


def load_units_and_spikes(nwb, start, stop):
    """
    读取 unit 元数据并裁剪每个 unit 在 wake_square 阶段的 spike times。

    返回两个对象：适合写入 parquet 的 unit 元数据表，以及以 `unit_<id>` 命名的
    spike 字典。函数还会添加初筛/QC 标记、spike 数量、wake_square 平均放电率和
    排除原因，方便后续分析追踪样本来源。
    """
    units = nwb.units.to_dataframe().reset_index(names="unit_id")

    meta = units.drop(columns=["spike_times", "waveform_mean"], errors="ignore").copy()
    meta = make_parquet_safe(meta)

    # 防止 bool 列被读成 object
    for col in ["is_excitatory", "is_fast_spiking", "is_head_direction"]:
        if col in meta.columns:
            meta[col] = meta[col].astype(bool)

    meta["included_initial"] = (
        meta["is_excitatory"].astype(bool)
        & meta["is_head_direction"].astype(bool)
    )

    spike_dict = {}
    n_spikes_total = []
    n_spikes_wake_square = []

    for _, row in units.iterrows():
        unit_id = int(row["unit_id"])
        spikes = np.asarray(row["spike_times"], dtype=float)
        spikes = spikes[np.isfinite(spikes)]

        spikes_valid = spikes[(spikes >= start) & (spikes < stop)]

        spike_dict[f"unit_{unit_id}"] = spikes_valid
        n_spikes_total.append(len(spikes))
        n_spikes_wake_square.append(len(spikes_valid))

    meta["n_spikes_total"] = n_spikes_total
    meta["n_spikes_wake_square"] = n_spikes_wake_square
    meta["mean_rate_wake_square_hz"] = meta["n_spikes_wake_square"] / max(stop - start, 1e-12)

    # 第一版 QC：先不要太激进
    meta["included_qc"] = meta["included_initial"] & (meta["n_spikes_wake_square"] >= 50)

    meta["exclude_reason"] = ""
    meta.loc[~meta["is_excitatory"].astype(bool), "exclude_reason"] += "not_excitatory;"
    meta.loc[~meta["is_head_direction"].astype(bool), "exclude_reason"] += "not_head_direction;"
    meta.loc[meta["n_spikes_wake_square"] < 50, "exclude_reason"] += "too_few_spikes;"

    return meta, spike_dict


def process_one_file(path):
    """
    处理单个 NWB 文件并写出中间数据。

    对一个 session 提取 wake_square 行为表、unit 元数据和裁剪后的 spike times，
    分别保存为 parquet/npz 文件。返回该 session 的汇总信息，用于生成
    `session_index.csv`。
    """
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()

        subject_id = nwb.subject.subject_id if nwb.subject is not None else path.parent.name
        session_id = path.stem

        start, stop = get_wake_square_interval(nwb)

        behavior = load_behavior(nwb, start, stop)
        units_meta, spike_dict = load_units_and_spikes(nwb, start, stop)

        behavior["subject_id"] = subject_id
        behavior["session_id"] = session_id

        units_meta["subject_id"] = subject_id
        units_meta["session_id"] = session_id
        units_meta["wake_square_start_s"] = start
        units_meta["wake_square_stop_s"] = stop
        units_meta["wake_square_duration_s"] = stop - start

        behavior_path = OUT / f"{subject_id}_behavior_wake_square.parquet"
        units_path = OUT / f"{subject_id}_units_meta.parquet"
        spikes_path = OUT / f"{subject_id}_spikes_wake_square.npz"

        behavior.to_parquet(behavior_path, index=False)
        units_meta.to_parquet(units_path, index=False)
        np.savez_compressed(spikes_path, **spike_dict)

        return {
            "subject_id": subject_id,
            "session_id": session_id,
            "file": str(path),
            "wake_square_start_s": start,
            "wake_square_stop_s": stop,
            "wake_square_duration_s": stop - start,
            "n_behavior_samples": len(behavior),
            "n_units": len(units_meta),
            "n_included_initial": int(units_meta["included_initial"].sum()),
            "n_included_qc": int(units_meta["included_qc"].sum()),
            "behavior_path": str(behavior_path),
            "units_path": str(units_path),
            "spikes_path": str(spikes_path),
        }


def main():
    """
    命令行入口：批量抽取所有 NWB 文件的 wake_square 数据。

    每个文件独立处理；失败的文件不会中断整个批次，而是把错误信息写入
    `session_index.csv`，便于之后排查。
    """
    files = sorted(RAW.rglob("*.nwb"))
    if not files:
        raise FileNotFoundError(f"No NWB files found under {RAW}")

    rows = []
    for path in tqdm(files, desc="Extracting wake_square"):
        try:
            rows.append(process_one_file(path))
        except Exception as e:
            rows.append(
                {
                    "file": str(path),
                    "status": "error",
                    "n_units": 0,
                    "n_included_initial": 0,
                    "n_included_qc": 0,
                    "error": repr(e),
                }
            )

    session_index = pd.DataFrame(rows)
    if "status" not in session_index.columns:
        session_index["status"] = "ok"
    session_index["status"] = session_index["status"].fillna("ok")

    for col in ["n_units", "n_included_initial", "n_included_qc"]:
        if col not in session_index.columns:
            session_index[col] = 0
        session_index[col] = pd.to_numeric(session_index[col], errors="coerce").fillna(0).astype(int)

    session_index.to_csv(OUT / "session_index.csv", index=False)

    print(session_index)
    ok = session_index["status"].eq("ok")
    print("\nFiles processed:", int(ok.sum()), "/", len(session_index))
    print("Files failed:", int((~ok).sum()))
    print("Total units:", int(session_index.loc[ok, "n_units"].sum()))
    print("Total included_initial:", int(session_index.loc[ok, "n_included_initial"].sum()))
    print("Total included_qc:", int(session_index.loc[ok, "n_included_qc"].sum()))
    print("Saved:", OUT / "session_index.csv")


if __name__ == "__main__":
    main()
