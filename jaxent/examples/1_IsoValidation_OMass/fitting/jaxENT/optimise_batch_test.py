"""Disposable batched equivalent of the Example 1 rate-averaged sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

from jaxent.examples.common.batch_test_optimization import BatchTestSpec, run_batch_test


HERE = Path(__file__).resolve().parent


def _csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _csv_floats(value: str) -> tuple[float, ...]:
    return tuple(float(item) for item in _csv_strings(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=7)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--ensembles", default="ISO_TRI,ISO_BI")
    parser.add_argument("--split-types", default="sequence_cluster,spatial")
    parser.add_argument("--maxent-values", default="1,5,10,50,100,500,1000")
    args = parser.parse_args()

    ensembles = _csv_strings(args.ensembles)
    feature_names = {"ISO_TRI": "iso_tri", "ISO_BI": "iso_bi"}
    unknown = set(ensembles) - feature_names.keys()
    if unknown:
        parser.error(f"unknown ensembles: {', '.join(sorted(unknown))}")

    run_batch_test(
        BatchTestSpec(
            experiment=1,
            fitting_dir=HERE,
            ensembles=ensembles,
            feature_names=feature_names,
            timepoints=(0.167, 1.0, 10.0, 60.0, 120.0),
            covariance_path=HERE / "_covariance_matrices_sigma/ISO_BI_Sigma_weighted.npz",
            primary_loss="hdx_uptake_MSE_loss",
            kint_unit="s^-1",
        ),
        args.output_dir,
        maxent_values=_csv_floats(args.maxent_values),
        split_types=_csv_strings(args.split_types),
        n_splits=args.n_splits,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
