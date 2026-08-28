#!/usr/bin/env bash
set -euo pipefail

example_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${example_dir}/../../.."

case "${1:-}" in
  benchmark)
    shift
    exec uv run --no-sync python "${example_dir}/benchmark_featurisation.py" "$@"
    ;;
  featurise)
    shift
    exec uv run --no-sync python "${example_dir}/featurise_batch.py" "$@"
    ;;
  convergence)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.convergence "$@"
    ;;
  basin-census)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.basin_census "$@"
    ;;
  stage1)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.within_basin_stage1 "$@"
    ;;
  stage2)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.stage2_gate "$@"
    ;;
  geometry-stage1)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.pairwise_geometry_stage1 "$@"
    ;;
  geometry-support-audit)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.support_w1_checkpoint1 "$@"
    ;;
  geometry-boundary-audit)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.boundary_checkpoint2 "$@"
    ;;
  geometry-vector-audit)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.vector_checkpoint3 "$@"
    ;;
  geometry-vector-ridge)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.vector_ridge_checkpoint3b "$@"
    ;;
  geometry-vector-knn)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.vector_knn_checkpoint3c "$@"
    ;;
  geometry-vector-compare)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.vector_final_comparison "$@"
    ;;
  geometry-vector-likelihood)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.vector_likelihood_checkpoint4 "$@"
    ;;
  geometry-likelihood-compare)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.vector_likelihood_comparison "$@"
    ;;
  geometry-scale-calibration)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.vector_scale_calibration_checkpoint5 "$@"
    ;;
  geometry-scale-compare)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.vector_scale_calibration_comparison "$@"
    ;;
  geometry-novelty-calibration)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.vector_scale_calibration_checkpoint5 --scale-coordinate pf_novelty "$@"
    ;;
  geometry-novelty-compare)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.vector_novelty_comparison "$@"
    ;;
  geometry-nearest-calibration)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.vector_nearest_support_checkpoint7 "$@"
    ;;
  geometry-nearest-compare)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.vector_nearest_support_comparison "$@"
    ;;
  geometry-conformal-strict)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.strict_conformal_checkpoint8 "$@"
    ;;
  geometry-conformal-compare)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.strict_conformal_comparison "$@"
    ;;
  geometry-opening-baseline)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.strict_likelihood_checkpoint9 "$@"
    ;;
  geometry-opening-baseline-report)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.strict_likelihood_comparison "$@"
    ;;
  all-analysis)
    shift
    uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.basin_census "$@"
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.within_basin_stage1
    ;;
  *)
    echo "usage: $0 {benchmark|featurise|convergence|basin-census|stage1|stage2|geometry-stage1|geometry-support-audit|geometry-boundary-audit|geometry-vector-audit|geometry-vector-ridge|geometry-vector-knn|geometry-vector-compare|geometry-vector-likelihood|geometry-likelihood-compare|geometry-scale-calibration|geometry-scale-compare|geometry-novelty-calibration|geometry-novelty-compare|geometry-nearest-calibration|geometry-nearest-compare|geometry-conformal-strict|geometry-conformal-compare|geometry-opening-baseline|geometry-opening-baseline-report|all-analysis} [options]" >&2
    exit 2
    ;;
esac
