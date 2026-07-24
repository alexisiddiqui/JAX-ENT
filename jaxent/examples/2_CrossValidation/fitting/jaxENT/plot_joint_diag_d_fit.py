#!/usr/bin/env python3
"""Plot persisted joint-BV diagnostic artifacts without re-fitting."""

from __future__ import annotations

import argparse
from pathlib import Path

from joint_diag_d_diagnostics import plot_joint_diag_d_fit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("diag_d_absolute", "diag_d_scalefree"))
    args = parser.parse_args()
    for path in plot_joint_diag_d_fit(args.input_dir, args.arm):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
