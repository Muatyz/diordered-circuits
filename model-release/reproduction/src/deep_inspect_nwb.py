# src/deep_inspect_nwb.py
# head direction 的数据接口名
# behavior timestamp 的获取方式
# epochs 里 open field / square exploration 的开始结束时间
from pathlib import Path
from pynwb import NWBHDF5IO

RAW = Path("data/raw/dandi_000939")

def describe_obj(obj, indent=0, max_depth=3):
    """
    递归打印 NWB 对象的层级结构和关键字段。

    `indent` 控制缩进层级，`max_depth` 限制递归深度，避免大型 NWB 对象展开过深。
    对 `data` 和 `timestamps` 只打印 shape，方便快速判断数据规模。
    """
    pad = "  " * indent
    if indent > max_depth:
        return

    print(f"{pad}{type(obj).__name__}: {getattr(obj, 'name', '')}")

    if hasattr(obj, "data_interfaces"):
        for k, v in obj.data_interfaces.items():
            print(f"{pad}- data_interface: {k} ({type(v).__name__})")
            describe_obj(v, indent + 1, max_depth)

    if hasattr(obj, "fields"):
        for k, v in obj.fields.items():
            if k in {"data", "timestamps"}:
                shape = getattr(v, "shape", None)
                print(f"{pad}- {k}: shape={shape}")
            else:
                print(f"{pad}- field: {k} ({type(v).__name__})")

def main():
    """
    命令行入口：抽样检查前几个 NWB 文件的 processing、interval 和 units 结构。

    这个脚本用于探索数据集内部命名和表结构，帮助后续确定 head direction、
    behavior timestamp、epochs 等字段应该从哪里读取。
    """
    nwb_files = sorted(RAW.rglob("*.nwb"))
    for path in nwb_files[:3]:
        print("\n" + "=" * 100)
        print(path)

        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            nwb = io.read()

            print("\nPROCESSING MODULES")
            for name, module in nwb.processing.items():
                print(f"\n[{name}]")
                describe_obj(module)

            print("\nINTERVAL TABLES")
            for name, table in nwb.intervals.items():
                print(f"\n[{name}] columns:", table.colnames)
                try:
                    print(table.to_dataframe().head())
                except Exception as e:
                    print("Could not convert to dataframe:", repr(e))

            print("\nUNITS")
            print(nwb.units.colnames)
            print(nwb.units.to_dataframe().head())

if __name__ == "__main__":
    main()
