"""Plot fixed-BV recovery in globally shared pairwise C-alpha RMSD intervals."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from jaxent.examples.ATLAS_BV.analysis.common import HERE
from jaxent.examples.ATLAS_BV.analysis.conditional_likelihood_comparison import aggregate
from jaxent.examples.ATLAS_BV.analysis.global_w1_recovery_plot import MODELS


def main() -> None:
    output = HERE / "outputs" / "analysis" / "pairwise_geometry" / "checkpoint16_global_rmsd"
    summary = pd.read_parquet(output / "global_rmsd_assignment_summary.parquet")
    system, population = aggregate(summary)
    selected = population[(population.target == "rmsd") & (population.calibration == "marginal")
                          & (population.stratum == "common_support") & population.model.isin(MODELS)][
        ["model", "band", "distribution_recovery"]
    ].copy()
    selected["label"] = selected.model.map(MODELS)
    selected["recovery_percent"] = 100 * selected.distribution_recovery
    system_selected = system[(system.target == "rmsd") & (system.calibration == "marginal")
                             & (system.stratum == "common_support") & system.model.isin(MODELS)]
    sd = (system_selected.groupby(["model", "band"], as_index=False).distribution_recovery.std()
          .rename(columns={"distribution_recovery": "system_sd_percent"}))
    sd.system_sd_percent *= 100
    selected = selected.merge(sd, on=["model", "band"], validate="one_to_one")
    contributors = system_selected.groupby("band").system_id.nunique()
    with open(output / "global_rmsd_edges.yaml") as handle: report = yaml.safe_load(handle)
    edges = np.asarray(report["edges_angstrom"]); order = [f"q{i}" for i in range(6)]
    labels = [
        f"q{i}\n{edges[i]:.2f}–{edges[i + 1]:.2f} Å\nn={contributors.get(f'q{i}', 0)} systems"
        for i in range(6)
    ]
    selected["band_low_angstrom"] = selected.band.map({b: edges[i] for i, b in enumerate(order)})
    selected["band_high_angstrom"] = selected.band.map({b: edges[i + 1] for i, b in enumerate(order)})
    selected["contributing_systems"] = selected.band.map(contributors)
    selected.to_parquet(output / "global_rmsd_recovery.parquet", index=False)

    x = np.arange(6); fig, axis = plt.subplots(figsize=(13.0, 6.2))
    for model, label in MODELS.items():
        block = selected[selected.model == model].set_index("band")
        values = np.array([block.recovery_percent.get(b, np.nan) for b in order])
        spread = np.array([block.system_sd_percent.get(b, np.nan) for b in order])
        line = axis.plot(x, values, marker="o", linewidth=2, label=label)[0]
        axis.fill_between(x, np.maximum(0, values - spread), np.minimum(100, values + spread),
                          color=line.get_color(), alpha=.12, linewidth=0)
    axis.set_xticks(x, labels); axis.set_xlabel("Global pairwise Cα RMSD band")
    axis.set_ylabel(r"Distribution recovery, $100(1-\sqrt{JSD})$ (%)")
    axis.set_title("Fixed BV log-PF: globally shared RMSD bands\nLines: system median; shading: ±1 system SD")
    axis.grid(alpha=.25); axis.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(output / "global_rmsd_recovery.png", dpi=180); plt.close(fig)
    print("Global RMSD edges (Å):", np.round(edges, 6).tolist())
    print(selected.pivot(index="label", columns="band", values="recovery_percent").round(2).to_string())


if __name__ == "__main__":
    main()
