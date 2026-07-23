#!/usr/bin/env python3
"""Pair conditional- and marginal-variance recovery diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("conditional_dir", type=Path)
    parser.add_argument("marginal_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    conditional = pd.read_csv(args.conditional_dir / "cluster_recovery_diagnostics.csv")
    marginal = pd.read_csv(args.marginal_dir / "cluster_recovery_diagnostics.csv")
    keys = ["ensemble", "target_mode", "split_type", "split_index"]
    paired = conditional.merge(
        marginal,
        on=keys,
        suffixes=("_conditional", "_marginal"),
        validate="one_to_one",
    )
    paired["conditional_minus_marginal_recovery_pp"] = (
        paired.candidate_recovery_pct_conditional - paired.candidate_recovery_pct_marginal
    )
    paired["conditional_minus_marginal_open_ess"] = (
        paired.candidate_open_ess_conditional - paired.candidate_open_ess_marginal
    )
    paired["recovery_winner"] = paired.conditional_minus_marginal_recovery_pp.map(
        lambda difference: "conditional" if difference > 0 else "marginal"
    )

    groups = []
    for keys_value, group in paired.groupby(["ensemble", "target_mode"]):
        groups.append(
            {
                "ensemble": keys_value[0],
                "target_mode": keys_value[1],
                "conditional_wins": int(
                    (group.conditional_minus_marginal_recovery_pp > 0).sum()
                ),
                "comparison_count": len(group),
                "conditional_recovery_median": float(
                    group.candidate_recovery_pct_conditional.median()
                ),
                "marginal_recovery_median": float(
                    group.candidate_recovery_pct_marginal.median()
                ),
                "conditional_minus_marginal_median_pp": float(
                    group.conditional_minus_marginal_recovery_pp.median()
                ),
                "conditional_open_ess_median": float(group.candidate_open_ess_conditional.median()),
                "marginal_open_ess_median": float(group.candidate_open_ess_marginal.median()),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.output_dir / "paired_results.csv", index=False)
    summary = {
        "comparison_count": len(paired),
        "conditional_recovery_wins": int((paired.recovery_winner == "conditional").sum()),
        "marginal_recovery_wins": int((paired.recovery_winner == "marginal").sum()),
        "groups": groups,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
