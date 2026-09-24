#!/usr/bin/env bash
set -euo pipefail

example_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${example_dir}/../../.."

case "${1:-}" in
  geometry-graph-representation)
    shift
    exec .venv/bin/python -m jaxent.examples.ATLAS_BV.analysis.graph_representation_audit "$@"
    ;;
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
  geometry-opening-screen)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.opening_distance_checkpoint10 "$@"
    ;;
  geometry-opening-screen-report)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.opening_distance_comparison "$@"
    ;;
  geometry-fixed-metric-plot)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.fixed_metric_recovery_plot "$@"
    ;;
  geometry-fixed-metrics)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.fixed_metric_checkpoint14 "$@"
    ;;
  geometry-global-w1)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.global_w1_checkpoint15 "$@"
    ;;
  geometry-global-w1-plot)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.global_w1_recovery_plot "$@"
    ;;
  geometry-global-rmsd)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.global_w1_checkpoint15 --target rmsd "$@"
    ;;
  geometry-global-rmsd-plot)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.global_rmsd_recovery_plot "$@"
    ;;
  geometry-opening-likelihood)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.conditional_likelihood_checkpoint11 "$@"
    ;;
  geometry-opening-likelihood-report)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.conditional_likelihood_comparison "$@"
    ;;
  geometry-opening-joint)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.joint_lowrank_checkpoint12 "$@"
    ;;
  geometry-opening-joint-report)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.joint_lowrank_comparison "$@"
    ;;
  geometry-bv-refit)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.bv_refit_checkpoint13 "$@"
    ;;
  geometry-bv-refit-report)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.bv_refit_comparison "$@"
    ;;
  geometry-kde-population)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.kde_population_checkpoint17 "$@"
    ;;
  geometry-kde-population-report)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.kde_population_report_checkpoint17 "$@"
    ;;
  geometry-thermodynamic-population)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.thermodynamic_population_checkpoint18 "$@"
    ;;
  geometry-thermodynamic-population-report)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.thermodynamic_population_report_checkpoint18 "$@"
    ;;
  geometry-thermodynamic-combination-pilot)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.thermodynamic_combination_pilot_checkpoint19 "$@"
    ;;
  geometry-contact-difference-pilot)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.contact_difference_pilot_checkpoint20 "$@"
    ;;
  geometry-pf-information-pilot)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.pf_information_pilot_checkpoint21 "$@"
    ;;
  geometry-cluster-stratified)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.cluster_stratified_checkpoint22 "$@"
    ;;
  openmm-env)
    shift
    exec "${example_dir}/openmm_vacuum_env.sh" create "$@"
    ;;
  openmm-audit)
    shift
    exec "${example_dir}/openmm_vacuum_env.sh" run python "${example_dir}/analysis/openmm_vacuum_score_checkpoint23.py" audit "$@"
    ;;
  openmm-score)
    shift
    exec "${example_dir}/openmm_vacuum_env.sh" run python "${example_dir}/analysis/openmm_vacuum_score_checkpoint23.py" score "$@"
    ;;
  openmm-score-parallel)
    shift
    exec "${example_dir}/openmm_score_parallel.sh" "$@"
    ;;
  geometry-openmm-energy)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.openmm_energy_population_checkpoint23 "$@"
    ;;
  pyrosetta-audit)
    shift
    exec "${example_dir}/pyrosetta_score_env.sh" "${example_dir}/analysis/pyrosetta_energy_score_checkpoint24.py" audit "$@"
    ;;
  pyrosetta-score)
    shift
    exec "${example_dir}/pyrosetta_score_env.sh" "${example_dir}/analysis/pyrosetta_energy_score_checkpoint24.py" score "$@"
    ;;
  pyrosetta-score-parallel)
    shift
    exec "${example_dir}/pyrosetta_score_parallel.sh" "$@"
    ;;
  geometry-pyrosetta-energy)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.pyrosetta_energy_population_checkpoint24 "$@"
    ;;
  geometry-alpha-variance)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.alpha_variance_checkpoint25 "$@"
    ;;
  geometry-pyrosetta-graph)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.pyrosetta_graph_checkpoint26 "$@"
    ;;
  geometry-work-graph)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.work_graph_checkpoint27 "$@"
    ;;
  geometry-local-variance)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.local_variance_checkpoint28 "$@"
    ;;
  geometry-corrected-variance-recovery)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.corrected_variance_recovery_checkpoint36 "$@"
    ;;
  geometry-variance-graph)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.variance_graph_checkpoint29 "$@"
    ;;
  geometry-laplacian-prior)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.laplacian_prior_validation "$@"
    ;;
  geometry-laplacian-metrics)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.laplacian_metric_comparison_checkpoint31 "$@"
    ;;
  geometry-laplacian-topology)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.laplacian_topology_checkpoint32 "$@"
    ;;
  geometry-laplacian-residue-scaling)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.laplacian_residue_scaling_checkpoint33 "$@"
    ;;
  geometry-basin-difficulty)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.basin_difficulty_reanalysis_checkpoint34 "$@"
    ;;
  geometry-original-omc)
    shift
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.original_omc_iso_validation_checkpoint35 "$@"
    ;;
  geometry-cluster-filtering-omc)
    shift
    exec env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.cluster_filtering_omc "$@"
    ;;
  geometry-omc-bandwidth-control)
    shift
    exec env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.omc_bandwidth_control "$@"
    ;;
  geometry-omc-bandwidth-diagnostics)
    shift
    exec env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/omc-diagnostics-mpl uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.omc_bandwidth_diagnostics "$@"
    ;;
  geometry-omc-coverage-control)
    shift
    exec env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/omc-coverage-mpl uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.omc_coverage_control "$@"
    ;;
  geometry-omc-coupling-control)
    shift
    exec env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/omc-coupling-mpl uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.omc_coupling_control "$@"
    ;;
  geometry-omc-decoy-control)
    shift
    exec env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/omc-decoy-mpl uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.omc_decoy_control "$@"
    ;;
  geometry-omc-graph-control)
    shift
    exec env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/omc-graph-mpl uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.omc_graph_control "$@"
    ;;
  all-analysis)
    shift
    uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.basin_census "$@"
    exec uv run --no-sync python -m jaxent.examples.ATLAS_BV.analysis.within_basin_stage1
    ;;
  *)
    echo "usage: $0 {...|geometry-local-variance|geometry-corrected-variance-recovery|geometry-laplacian-prior|geometry-laplacian-metrics|geometry-laplacian-topology|geometry-laplacian-residue-scaling|geometry-basin-difficulty|geometry-original-omc|geometry-cluster-filtering-omc|geometry-omc-bandwidth-control|geometry-omc-bandwidth-diagnostics|geometry-omc-coverage-control|geometry-omc-coupling-control|geometry-omc-decoy-control|geometry-omc-graph-control|all-analysis} [options]" >&2
    exit 2
    ;;
esac
