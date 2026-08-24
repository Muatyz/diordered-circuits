try:
    from ._path import ensure_src_on_path
except ImportError:
    from _path import ensure_src_on_path

ensure_src_on_path()

from pynwb import NWBHDF5IO
from workspace_paths import DATA_ROOT


RAW = DATA_ROOT / "raw" / "dandi_000939" / "000939"


def main():
    """
    Print epoch tags and durations for a few NWB files.
    """
    for path in sorted(RAW.rglob("*.nwb"))[:5]:
        print(f"\n{path}")
        with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
            nwb = io.read()
            epochs = nwb.intervals["epochs"].to_dataframe()
            cols = [col for col in ["start_time", "stop_time", "tags"] if col in epochs.columns]
            print(epochs[cols].to_string())


if __name__ == "__main__":
    main()
