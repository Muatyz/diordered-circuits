"""Test the M2 equilibrium width law across beta and random seeds."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from prospective.config import load_config
from prospective.experiments.run_feedforward import train_feedforward
from prospective.io.run_dir import write_json
from prospective.theory.equilibrium import equilibrium_widths


def run_beta_sweep(config_path: str | Path) -> Path:
    """Train independent local-learning runs and compare widths with Eq. 13."""

    config_path = Path(config_path)
    values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base_path = Path(values["base_config"])
    if not base_path.is_absolute():
        base_path = config_path.parents[2] / base_path
    base = load_config(base_path)
    betas = [float(value) for value in values["betas"]]
    seeds = [int(value) for value in values["seeds"]]
    duration = float(values.get("duration", base.simulation.duration))
    output_root = Path(values.get("output_root", "reports/beta_sweep"))
    if not output_root.is_absolute():
        output_root = config_path.parents[2] / output_root
    output_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir.mkdir(parents=True)
    rows = []
    for beta in betas:
        for seed in seeds:
            config = replace(
                base,
                experiment=replace(base.experiment, name=f"beta_{beta:g}", seed=seed, output_root=str(output_dir / "runs")),
                feedforward_learning=replace(base.feedforward_learning, beta=beta),
                simulation=replace(base.simulation, duration=duration, progress=False),
                animation=replace(base.animation, enabled=False),
            )
            config.validate()
            result = train_feedforward(config, output_cwd=config_path.parents[2], make_figures=False)
            rows.append({"beta": beta, "seed": seed, "run_dir": str(result.run_dir), **result.metrics})
    with (output_dir / "beta_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "beta_sweep.json", {"conditions": rows})
    beta_axis = np.linspace(min(betas), max(betas), 300)
    theory = [equilibrium_widths(beta, base.tutor.sigma)[0] for beta in beta_axis]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(beta_axis, theory, "k--", label="Eq. 13")
    for beta in betas:
        observed = np.asarray([row["learned_width"] for row in rows if row["beta"] == beta], dtype=float)
        ax.errorbar(beta, np.nanmedian(observed), yerr=[[np.nanmedian(observed)-np.nanpercentile(observed, 25)], [np.nanpercentile(observed, 75)-np.nanmedian(observed)]], fmt="o", color="tab:red")
    ax.set(xlabel="decay exponent beta", ylabel="learned sigma_J", title="Feedforward width law")
    ax.legend()
    fig.savefig(output_dir / "beta_width_test.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/analysis/beta_sweep.yaml")
    args = parser.parse_args()
    print(run_beta_sweep(args.config))


if __name__ == "__main__":
    main()
