# src/inspect_spatial_series.py
# 1. CompassDirection 下面哪个 key 是 head direction
# 2. head direction 的 data shape 是 (T,) 还是 (T,1)
# 3. 有没有 timestamps；如果没有，starting_time 和 rate 是多少
# 4. unit 是 radians、degrees，还是 unknown
# 5. Position 的 data shape 是 (T,2) 还是 (T,3)
from pathlib import Path
from pynwb import NWBHDF5IO

RAW = Path("data/raw/dandi_000939/000939")

def describe_timeseries(ts):
    """
    打印一个 NWB TimeSeries / SpatialSeries 的关键元数据。

    重点查看名称、类型、单位、描述、数据 shape/dtype，以及时间信息来自
    `timestamps` 还是 `starting_time + rate`。这个输出用于确认后续抽取脚本
    应该怎样读取 head direction 和 position。
    """
    print(f"    name: {ts.name}")
    print(f"    type: {type(ts).__name__}")
    print(f"    unit: {getattr(ts, 'unit', None)}")
    print(f"    description: {getattr(ts, 'description', None)}")
    print(f"    comments: {getattr(ts, 'comments', None)}")
    print(f"    conversion: {getattr(ts, 'conversion', None)}")
    print(f"    offset: {getattr(ts, 'offset', None)}")

    data = getattr(ts, "data", None)
    if data is not None:
        print(f"    data shape: {getattr(data, 'shape', None)}")
        print(f"    data dtype: {getattr(data, 'dtype', None)}")

    if getattr(ts, "timestamps", None) is not None:
        print(f"    timestamps shape: {getattr(ts.timestamps, 'shape', None)}")
        print(f"    timestamps dtype: {getattr(ts.timestamps, 'dtype', None)}")
    else:
        print(f"    starting_time: {getattr(ts, 'starting_time', None)}")
        print(f"    rate: {getattr(ts, 'rate', None)}")

def main():
    """
    命令行入口：抽样检查行为 spatial series 和 epochs 表。

    脚本会查看前几个 NWB 文件中 `CompassDirection` 与 `Position` 下的
    spatial_series key，并打印每个序列的结构，辅助确认数据字段名称和维度。
    """
    nwb_files = sorted(RAW.rglob("*.nwb"))

    for path in nwb_files[:5]:
        print("\n" + "=" * 100)
        print(path)

        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            nwb = io.read()
            behavior = nwb.processing["behavior"]

            for interface_name in ["CompassDirection", "Position"]:
                interface = behavior[interface_name]
                print(f"\n[{interface_name}] spatial_series keys:")
                print(list(interface.spatial_series.keys()))

                for key, ts in interface.spatial_series.items():
                    print(f"\n  spatial_series key: {key}")
                    describe_timeseries(ts)

            print("\n[epochs]")
            print(nwb.intervals["epochs"].to_dataframe())

if __name__ == "__main__":
    main()
