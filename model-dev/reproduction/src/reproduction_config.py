import json
from functools import lru_cache
from pathlib import Path


REPRODUCTION_ROOT = Path(__file__).resolve().parent.parent
FIGURE3_CONFIG_PATH = REPRODUCTION_ROOT / "config" / "figure3.json"


@lru_cache(maxsize=1)
def _load_figure3_config_cached():
    """
    Load and minimally validate the shared Figure 3 reproduction parameters.

    The private cached object is copied by `load_figure3_config`.
    """
    raw = json.loads(FIGURE3_CONFIG_PATH.read_text(encoding="utf-8"))
    required_sections = {
        "schema_version",
        "network",
        "panels_abcd",
        "panels_ef",
        "panels_ghi",
        "panels_jkl",
    }
    missing = required_sections.difference(raw)
    if missing:
        raise ValueError(f"Figure 3 config is missing sections: {sorted(missing)}")

    network = raw["network"]
    required_network = {
        "regularization",
        "activation_beta",
        "alpha_floor",
        "do_double_normalize",
        "inhibition_c",
        "tau_s",
        "dt_s",
        "circulant_gain",
    }
    missing_network = required_network.difference(network)
    if missing_network:
        raise ValueError(
            f"Figure 3 network config is missing fields: {sorted(missing_network)}"
        )
    if network["regularization"] <= 0 or network["tau_s"] <= 0 or network["dt_s"] <= 0:
        raise ValueError("Figure 3 regularization, tau_s, and dt_s must be positive")
    if not 0 < network["alpha_floor"] < 1:
        raise ValueError("Figure 3 alpha_floor must lie between zero and one")

    return raw


def load_figure3_config():
    """
    Return a fresh copy of the validated shared Figure 3 configuration.
    """
    return json.loads(json.dumps(_load_figure3_config_cached()))


def figure3_config_version():
    """
    Return the schema version stored alongside generated Figure 3 caches.
    """
    return int(load_figure3_config()["schema_version"])
