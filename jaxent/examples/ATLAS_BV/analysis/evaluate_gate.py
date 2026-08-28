from __future__ import annotations

import argparse

import pandas as pd

from jaxent.examples.ATLAS_BV.analysis.common import HERE, atomic_yaml, load_config


def evaluate_census(census: pd.DataFrame, config: dict) -> dict:
    settings = config["analysis"]["stage1"]
    informative = census[census["usable_basins"] >= settings["min_usable_basins"]]
    passed = len(informative) >= settings["min_informative_systems"]
    return {
        "stage1_pass": False,
        "stage2_authorized": False,
        "decision": "proceed_to_stage1" if passed else "redesign_required",
        "blocking_gate": None if passed else "structural_basin_census",
        "criteria": {
            "minimum_informative_systems": settings["min_informative_systems"],
            "minimum_usable_basins_per_system": settings["min_usable_basins"],
            "minimum_delta_f_range_kcal_mol": settings["min_delta_f_range_kcal_mol"],
        },
        "observed": {
            "systems": int(len(census)),
            "informative_systems_before_convergence_and_delta_f": int(len(informative)),
            "usable_basin_distribution": {
                int(key): int(value)
                for key, value in census["usable_basins"].value_counts().sort_index().items()
            },
            "raw_basin_distribution": {
                int(key): int(value)
                for key, value in census["basins"].value_counts().sort_index().items()
            },
        },
        "reason": (
            "No systems meet the predeclared minimum of three basins with at least "
            "50 frames in every replica; held-out basin ordering is therefore not identifiable."
            if not passed
            else "The basin census supports continuing to convergence-filtered Stage 1 analysis."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    census_path = HERE / "outputs" / "analysis" / "basin_census.csv"
    if not census_path.is_file():
        raise FileNotFoundError("run basin-census before evaluating the Stage 1 gate")
    decision = evaluate_census(pd.read_csv(census_path), load_config())
    output = HERE / "outputs" / "analysis" / "stage1_decision.yaml"
    atomic_yaml(output, decision)
    print(f"{decision['decision']}: {decision['reason']}")
    print(f"Decision report: {output}")
    if decision["decision"] != "proceed_to_stage1":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
