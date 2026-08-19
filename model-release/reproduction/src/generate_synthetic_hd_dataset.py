import argparse
import json
from pathlib import Path

import numpy as np

from gaussian_generative_process import generate_heterogeneous_tuning_dataset


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_FIT_PATH = WORKSPACE_ROOT / "data/processed/figure5_de_parameter_fit.json"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "data/processed/synthetic_hd_tuning_optimal_n2048.npz"


def load_fitted_parameters(path=DEFAULT_FIT_PATH):
    """
    读取 Figure 5D-E 得到的最优 Gaussian generative process 参数。

    Args:
        path: ``figure5_de_parameter_fit.json`` 的路径。

    Returns:
        包含 ``sigma``、``beta`` 和 ``bias`` 的字典。
    """
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "sigma": float(values["best_sigma"]),
        "beta": float(values["best_beta"]),
        "bias": float(values["best_bias"]),
    }


def main(argv=None):
    """
    生成可供 learning 子项目使用的假设性头朝向 teacher dataset。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-neurons", type=int, default=2048)
    parser.add_argument("--n-angles", type=int, default=100)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--bias", type=float)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--fit-path", type=Path, default=DEFAULT_FIT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    fitted = load_fitted_parameters(args.fit_path)
    sigma = fitted["sigma"] if args.sigma is None else args.sigma
    beta = fitted["beta"] if args.beta is None else args.beta
    bias = fitted["bias"] if args.bias is None else args.bias
    dataset = generate_heterogeneous_tuning_dataset(
        n_neurons=args.n_neurons,
        n_angles=args.n_angles,
        sigma=sigma,
        beta=beta,
        bias=bias,
        seed=args.seed,
        dtype=np.dtype(args.dtype),
    )
    output = dataset.save_npz(args.output)
    print("Saved:", output)
    print("Shape:", dataset.firing_rates.shape)
    print(
        "Parameters:",
        f"sigma={dataset.sigma:.6g},",
        f"beta={dataset.beta:.6g},",
        f"b={dataset.bias:.6g}",
    )
    print(
        "Angular-mean range:",
        f"{np.min(np.mean(dataset.firing_rates, axis=1)):.8f}",
        "to",
        f"{np.max(np.mean(dataset.firing_rates, axis=1)):.8f}",
    )


if __name__ == "__main__":
    main()
