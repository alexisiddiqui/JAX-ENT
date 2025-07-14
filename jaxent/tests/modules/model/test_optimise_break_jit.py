import os
import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

import pytest

# JAX setup
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import jax

jax.config.update("jax_platform_name", "cpu")
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax.numpy as jnp
from MDAnalysis import Universe

from jaxent.src.custom_types.config import FeaturiserSettings, OptimiserSettings
from jaxent.src.custom_types.HDX import HDX_protection_factor

# Import all the necessary modules
from jaxent.src.data.loader import Dataset, ExpD_Dataloader
from jaxent.src.data.splitting.sparse_map import create_sparse_map
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.featurise import run_featurise
from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.forwardmodel import BV_model
from jaxent.src.opt.losses import hdx_pf_l2_loss
from jaxent.src.opt.optimiser import OptaxOptimizer
from jaxent.src.opt.run import _optimise, run_optimise


@contextmanager
def timeout_context(seconds=60):
    """Context manager to timeout operations that might hang."""
    import platform
    import signal

    if platform.system() != "Windows":

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)

        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        yield


class OptimizationTestEnvironment:
    """Class to manage the complete test environment for optimization testing."""

    def __init__(self, topology_path: str, trajectory_path: str):
        self.topology_path = topology_path
        self.trajectory_path = trajectory_path
        self.universes = None
        self.models = None
        self.features = None
        self.feature_topology = None
        self.exp_data = None
        self.dataset = None
        self.base_params = None
        self.base_simulation = None

    def setup_environment(self):
        """Set up the complete environment for testing."""
        try:
            # Load MD data
            test_universe = Universe(self.topology_path, self.trajectory_path)
            self.universes = [test_universe]

            # Set up models
            bv_config = BV_model_Config()
            self.models = [BV_model(bv_config)]

            # Featurize
            featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)
            ensemble = Experiment_Builder(self.universes, self.models)
            self.features, self.feature_topology = run_featurise(ensemble, featuriser_settings)

            # Create base parameters
            BV_features = self.features[0]
            trajectory_length = BV_features.features_shape[2]

            self.base_params = Simulation_Parameters(
                frame_weights=jnp.ones(trajectory_length) / trajectory_length,
                frame_mask=jnp.ones(trajectory_length),
                model_parameters=[bv_config.forward_parameters],
                forward_model_weights=jnp.ones(1),
                forward_model_scaling=jnp.ones(1),
                normalise_loss_functions=jnp.ones(1),
            )

            # Create experimental data
            top_segments = Partial_Topology.find_common_residues(
                self.universes, ignore_mda_selection="(resname PRO or resid 1) "
            )[0]
            top_segments = sorted(top_segments, key=lambda x: x.residue_start)

            self.exp_data = [
                HDX_protection_factor(protection_factor=10.0 + i * 0.5, top=top)
                for i, top in enumerate(self.feature_topology[0], start=1)
            ]

            # Set up dataset
            self.dataset = ExpD_Dataloader(data=self.exp_data)
            self._setup_data_splits()

            return True

        except Exception as e:
            print(f"Environment setup failed: {e}")
            return False

    def _setup_data_splits(self):
        """Set up train/val/test splits."""
        splitter = DataSplitter(
            self.dataset,
            random_seed=42,
            ensemble=self.universes,
            common_residues=set(self.feature_topology[0]),
        )
        train_data, val_data = splitter.random_split()

        # Create sparse maps
        train_sparse_map = create_sparse_map(self.features[0], self.feature_topology[0], train_data)
        val_sparse_map = create_sparse_map(self.features[0], self.feature_topology[0], val_data)
        test_sparse_map = create_sparse_map(
            self.features[0], self.feature_topology[0], self.exp_data
        )

        # Set up dataset splits
        self.dataset.train = Dataset(
            data=train_data,
            y_true=jnp.array([data.extract_features() for data in train_data]),
            residue_feature_ouput_mapping=train_sparse_map,
        )
        self.dataset.val = Dataset(
            data=val_data,
            y_true=jnp.array([data.extract_features() for data in val_data]),
            residue_feature_ouput_mapping=val_sparse_map,
        )
        self.dataset.test = Dataset(
            data=self.exp_data,
            y_true=jnp.array([data.extract_features() for data in self.exp_data]),
            residue_feature_ouput_mapping=test_sparse_map,
        )

    def create_simulation(self, params: Optional[Simulation_Parameters] = None) -> Simulation:
        """Create a new simulation instance."""
        if params is None:
            params = self.base_params
        return Simulation(forward_models=self.models, input_features=self.features, params=params)

    def create_parameter_variants(self, num_variants: int = 5) -> List[Simulation_Parameters]:
        """Create parameter variants for testing."""
        variants = []
        key = jax.random.PRNGKey(123)

        for i in range(num_variants):
            key, *subkeys = jax.random.split(key, 6)

            # Vary frame weights
            frame_weights = jax.random.uniform(subkeys[0], self.base_params.frame_weights.shape)
            frame_weights = frame_weights / jnp.sum(frame_weights)

            # Vary frame mask
            frame_mask = jax.random.choice(
                subkeys[1], 2, self.base_params.frame_mask.shape, p=jnp.array([0.1, 0.9])
            )

            # Scale model parameters
            scale_factor = 0.5 + jax.random.uniform(subkeys[2]) * 1.5
            scaled_model_params = []
            for param_set in self.base_params.model_parameters:
                scaled_params = jax.tree.map(
                    lambda x: x * scale_factor if jnp.isscalar(x) or x.ndim == 0 else x, param_set
                )
                scaled_model_params.append(scaled_params)

            # Vary other parameters
            fw_weights = jax.random.uniform(
                subkeys[3], self.base_params.forward_model_weights.shape
            )
            fw_weights = fw_weights / jnp.sum(fw_weights)
            fw_scaling = (
                0.1
                + jax.random.uniform(subkeys[4], self.base_params.forward_model_scaling.shape) * 2.0
            )

            variant = Simulation_Parameters(
                frame_weights=frame_weights,
                frame_mask=frame_mask,
                model_parameters=scaled_model_params,
                forward_model_weights=fw_weights,
                forward_model_scaling=fw_scaling,
                normalise_loss_functions=self.base_params.normalise_loss_functions,
            )
            variants.append(variant)

        return variants

    def cleanup(self):
        """Clean up resources."""
        try:
            if hasattr(self, "base_simulation") and self.base_simulation:
                del self.base_simulation
            jax.clear_caches()
        except Exception:
            pass


def run_optimization_step(
    simulation: Simulation,
    dataset: ExpD_Dataloader,
    n_steps: int = 10,
    use_run_optimise: bool = True,
) -> Tuple[bool, Any, str]:
    """Run a single optimization step and return success status."""
    try:
        if use_run_optimise:
            # Use high-level interface
            opt_settings = OptimiserSettings(name="test")
            result = run_optimise(
                simulation,
                data_to_fit=(dataset,),
                config=opt_settings,
                forward_models=simulation.forward_models,
                indexes=[0],
                loss_functions=[hdx_pf_l2_loss],
            )
            return True, result, "run_optimise"
        else:
            # Use low-level interface
            optimizer = OptaxOptimizer(optimizer="adam")
            opt_state = optimizer.initialise(simulation, [True], _jit_test_args=None)

            result_simulation, result_optimizer = _optimise(
                simulation,
                [dataset],
                n_steps,
                tolerance=1e-6,
                convergence=1e-8,
                indexes=[0],
                loss_functions=[hdx_pf_l2_loss],
                opt_state=opt_state,
                optimizer=optimizer,
            )
            return True, (result_simulation, result_optimizer), "_optimise"
    except Exception as e:
        return False, str(e), "error"


def test_jit_optimization_stress_comprehensive():
    """
    Comprehensive stress test combining JIT compilation with optimization.
    Tests various permutations of initialization, optimization, and parameter changes.
    """
    # File paths - adjust these to match your environment
    topology_path = "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_overall_combined_stripped.pdb"
    trajectory_path = (
        "/home/alexi/Documents/JAX-ENT/jaxent/tests/inst/clean/BPTI/BPTI_sampled_500.xtc"
    )

    # Check if files exist
    if not (os.path.exists(topology_path) and os.path.exists(trajectory_path)):
        pytest.skip(f"Test files not found: {topology_path}, {trajectory_path}")
        return

    # Initialize test environment
    test_env = OptimizationTestEnvironment(topology_path, trajectory_path)

    try:
        with timeout_context(120):  # 2 minute timeout for setup
            setup_success = test_env.setup_environment()

        if not setup_success:
            pytest.fail("Failed to set up test environment")
            return

        # Create parameter variants
        param_variants = test_env.create_parameter_variants(4)

        # Define test scenarios
        test_scenarios = [
            "optimize_before_jit_stress",
            "jit_stress_then_optimize",
            "interleaved_jit_optimization",
            "rapid_param_switching_with_opt",
            "optimization_state_persistence",
            "multi_simulation_optimization",
        ]

        results = {}

        for scenario in test_scenarios:
            print(f"\n=== Testing scenario: {scenario} ===")
            scenario_results = []

            try:
                if scenario == "optimize_before_jit_stress":
                    scenario_results = _test_optimize_before_jit_stress(test_env, param_variants)

                elif scenario == "jit_stress_then_optimize":
                    scenario_results = _test_jit_stress_then_optimize(test_env, param_variants)

                elif scenario == "interleaved_jit_optimization":
                    scenario_results = _test_interleaved_jit_optimization(test_env, param_variants)

                elif scenario == "rapid_param_switching_with_opt":
                    scenario_results = _test_rapid_param_switching_with_opt(
                        test_env, param_variants
                    )

                elif scenario == "optimization_state_persistence":
                    scenario_results = _test_optimization_state_persistence(
                        test_env, param_variants
                    )

                elif scenario == "multi_simulation_optimization":
                    scenario_results = _test_multi_simulation_optimization(test_env, param_variants)

            except Exception as e:
                scenario_results = [
                    {"error": str(e), "scenario": scenario, "traceback": traceback.format_exc()}
                ]

            results[scenario] = scenario_results

        # Analyze and display results
        _analyze_optimization_stress_results(results)

        return results

    finally:
        test_env.cleanup()


def _test_optimize_before_jit_stress(
    test_env: OptimizationTestEnvironment, param_variants: List[Simulation_Parameters]
) -> List[Dict]:
    """Test optimization before JIT stress testing."""
    results = []

    # Create simulation and optimize first
    simulation = test_env.create_simulation()

    try:
        with timeout_context(90):
            simulation.initialise()
        results.append({"operation": "initial_init", "success": True})

        # Run optimization first
        with timeout_context(60):
            opt_success, opt_result, opt_method = run_optimization_step(
                simulation, test_env.dataset, n_steps=5, use_run_optimise=True
            )
        results.append(
            {"operation": "pre_optimization", "success": opt_success, "method": opt_method}
        )

        if not opt_success:
            return results

        # Now stress test with different parameters
        for i, params in enumerate(param_variants):
            try:
                with timeout_context(30):
                    simulation.forward(params)
                results.append(
                    {"operation": f"post_opt_forward_{i}", "success": True, "param_variant": i}
                )
            except TimeoutError:
                results.append(
                    {"operation": f"post_opt_forward_{i}", "success": False, "error": "timeout"}
                )
            except Exception as e:
                results.append(
                    {"operation": f"post_opt_forward_{i}", "success": False, "error": str(e)}
                )

    except Exception as e:
        results.append({"operation": "optimize_before_jit", "success": False, "error": str(e)})

    return results


def _test_jit_stress_then_optimize(
    test_env: OptimizationTestEnvironment, param_variants: List[Simulation_Parameters]
) -> List[Dict]:
    """Test JIT stress first, then optimization."""
    results = []

    simulation = test_env.create_simulation()

    try:
        with timeout_context(90):
            simulation.initialise()
        results.append({"operation": "initial_init", "success": True})

        # JIT stress test first
        for i, params in enumerate(param_variants[:3]):
            try:
                with timeout_context(20):
                    simulation.forward(params)
                results.append(
                    {"operation": f"pre_opt_forward_{i}", "success": True, "param_variant": i}
                )
            except Exception as e:
                results.append(
                    {"operation": f"pre_opt_forward_{i}", "success": False, "error": str(e)}
                )

        # Now try optimization after JIT stress
        with timeout_context(60):
            opt_success, opt_result, opt_method = run_optimization_step(
                simulation, test_env.dataset, n_steps=5, use_run_optimise=False
            )
        results.append(
            {"operation": "post_jit_optimization", "success": opt_success, "method": opt_method}
        )

    except Exception as e:
        results.append({"operation": "jit_then_optimize", "success": False, "error": str(e)})

    return results


def _test_interleaved_jit_optimization(
    test_env: OptimizationTestEnvironment, param_variants: List[Simulation_Parameters]
) -> List[Dict]:
    """Test interleaved JIT and optimization steps."""
    results = []

    simulation = test_env.create_simulation()

    try:
        with timeout_context(90):
            simulation.initialise()
        results.append({"operation": "initial_init", "success": True})

        # Interleave forward passes and optimization
        for i in range(min(3, len(param_variants))):
            # Forward pass
            try:
                with timeout_context(20):
                    simulation.forward(param_variants[i])
                results.append(
                    {"operation": f"interleaved_forward_{i}", "success": True, "param_variant": i}
                )
            except Exception as e:
                results.append(
                    {"operation": f"interleaved_forward_{i}", "success": False, "error": str(e)}
                )
                continue

            # Optimization step
            try:
                with timeout_context(30):
                    opt_success, opt_result, opt_method = run_optimization_step(
                        simulation, test_env.dataset, n_steps=3, use_run_optimise=(i % 2 == 0)
                    )
                results.append(
                    {
                        "operation": f"interleaved_opt_{i}",
                        "success": opt_success,
                        "method": opt_method,
                    }
                )
            except Exception as e:
                results.append(
                    {"operation": f"interleaved_opt_{i}", "success": False, "error": str(e)}
                )

    except Exception as e:
        results.append({"operation": "interleaved_jit_opt", "success": False, "error": str(e)})

    return results


def _test_rapid_param_switching_with_opt(
    test_env: OptimizationTestEnvironment, param_variants: List[Simulation_Parameters]
) -> List[Dict]:
    """Test rapid parameter switching combined with optimization."""
    results = []

    simulation = test_env.create_simulation()

    try:
        with timeout_context(90):
            simulation.initialise()
        results.append({"operation": "initial_init", "success": True})

        # Rapid parameter switching
        key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
        for iteration in range(10):
            try:
                # Random parameter selection
                key, subkey = jax.random.split(key)
                idx = jax.random.randint(subkey, (), 0, len(param_variants))
                params = param_variants[int(idx)]

                with timeout_context(10):
                    simulation.forward(params)

                # Occasional optimization
                if iteration % 3 == 0:
                    with timeout_context(20):
                        opt_success, _, opt_method = run_optimization_step(
                            simulation, test_env.dataset, n_steps=2, use_run_optimise=True
                        )
                    results.append(
                        {
                            "operation": f"rapid_opt_{iteration}",
                            "success": opt_success,
                            "method": opt_method,
                        }
                    )

                results.append(
                    {
                        "operation": f"rapid_forward_{iteration}",
                        "success": True,
                        "param_idx": int(idx),
                    }
                )

            except Exception as e:
                results.append(
                    {"operation": f"rapid_forward_{iteration}", "success": False, "error": str(e)}
                )
                break

    except Exception as e:
        results.append({"operation": "rapid_switching_opt", "success": False, "error": str(e)})

    return results


def _test_optimization_state_persistence(
    test_env: OptimizationTestEnvironment, param_variants: List[Simulation_Parameters]
) -> List[Dict]:
    """Test optimization state persistence across parameter changes."""
    results = []

    simulation = test_env.create_simulation()

    try:
        with timeout_context(90):
            simulation.initialise()
        results.append({"operation": "initial_init", "success": True})

        # Initialize optimizer state
        optimizer = OptaxOptimizer(optimizer="adam")
        opt_state = optimizer.initialise(simulation, [True], _jit_test_args=None)
        results.append({"operation": "optimizer_init", "success": True})

        # Test optimization with different parameters while maintaining state
        for i, params in enumerate(param_variants[:3]):
            try:
                # Update simulation parameters
                simulation.params = params

                # Run optimization step with persistent state
                with timeout_context(30):
                    result_simulation, result_optimizer = _optimise(
                        simulation,
                        [test_env.dataset],
                        n_steps=3,
                        tolerance=1e-6,
                        convergence=1e-8,
                        indexes=[0],
                        loss_functions=[hdx_pf_l2_loss],
                        opt_state=opt_state,
                        optimizer=optimizer,
                    )
                    # Update state for next iteration
                    opt_state = result_optimizer

                results.append(
                    {"operation": f"persistent_opt_{i}", "success": True, "param_variant": i}
                )

            except Exception as e:
                results.append(
                    {"operation": f"persistent_opt_{i}", "success": False, "error": str(e)}
                )

    except Exception as e:
        results.append({"operation": "state_persistence", "success": False, "error": str(e)})

    return results


def _test_multi_simulation_optimization(
    test_env: OptimizationTestEnvironment, param_variants: List[Simulation_Parameters]
) -> List[Dict]:
    """Test multiple simulations with cross-optimization."""
    results = []

    try:
        # Create multiple simulations
        simulations = []
        for i, params in enumerate(param_variants[:3]):
            sim = test_env.create_simulation(params)
            with timeout_context(60):
                sim.initialise()
            simulations.append(sim)
            results.append({"operation": f"multi_sim_init_{i}", "success": True})

        # Cross-optimize simulations
        for i, simulation in enumerate(simulations):
            try:
                with timeout_context(30):
                    opt_success, opt_result, opt_method = run_optimization_step(
                        simulation, test_env.dataset, n_steps=5, use_run_optimise=(i % 2 == 0)
                    )
                results.append(
                    {
                        "operation": f"multi_sim_opt_{i}",
                        "success": opt_success,
                        "method": opt_method,
                    }
                )

                # Test cross-parameter application
                if i > 0:
                    with timeout_context(15):
                        simulation.forward(param_variants[(i - 1) % len(param_variants)])
                    results.append({"operation": f"multi_sim_cross_forward_{i}", "success": True})

            except Exception as e:
                results.append(
                    {"operation": f"multi_sim_opt_{i}", "success": False, "error": str(e)}
                )

    except Exception as e:
        results.append({"operation": "multi_simulation", "success": False, "error": str(e)})

    return results


def _analyze_optimization_stress_results(results: Dict[str, List[Dict]]):
    """Analyze and display optimization stress test results."""
    print("\n" + "=" * 60)
    print("JIT + OPTIMIZATION STRESS TEST ANALYSIS")
    print("=" * 60)

    total_operations = 0
    successful_operations = 0
    timeouts = 0
    errors = 0
    optimization_failures = 0

    for scenario, scenario_results in results.items():
        print(f"\nScenario: {scenario}")
        scenario_success = 0
        scenario_total = len(scenario_results)
        scenario_timeouts = 0
        scenario_opt_failures = 0

        for result in scenario_results:
            total_operations += 1
            if result.get("success", False):
                successful_operations += 1
                scenario_success += 1
            elif result.get("error") == "timeout":
                timeouts += 1
                scenario_timeouts += 1
            else:
                errors += 1
                if "opt" in result.get("operation", "").lower():
                    optimization_failures += 1
                    scenario_opt_failures += 1

        print(f"  Success: {scenario_success}/{scenario_total}")
        if scenario_timeouts > 0:
            print(f"  Timeouts: {scenario_timeouts}")
        if scenario_opt_failures > 0:
            print(f"  Optimization failures: {scenario_opt_failures}")
        if scenario_total - scenario_success - scenario_timeouts > 0:
            print(f"  Other errors: {scenario_total - scenario_success - scenario_timeouts}")

    print("\nOVERALL SUMMARY:")
    print(f"Total operations: {total_operations}")
    print(
        f"Successful: {successful_operations} ({successful_operations / total_operations * 100:.1f}%)"
    )
    print(f"Timeouts: {timeouts} ({timeouts / total_operations * 100:.1f}%)")
    print(
        f"Optimization failures: {optimization_failures} ({optimization_failures / total_operations * 100:.1f}%)"
    )
    print(f"Other errors: {errors} ({errors / total_operations * 100:.1f}%)")

    # Report critical issues
    critical_issues = []
    for scenario, scenario_results in results.items():
        failures = [r for r in scenario_results if not r.get("success", False)]
        if len(failures) > len(scenario_results) * 0.5:  # >50% failure rate
            critical_issues.append(f"{scenario}: {len(failures)}/{len(scenario_results)} failures")

    if critical_issues:
        print("\nCRITICAL ISSUES (>50% failure rate):")
        for issue in critical_issues:
            print(f"  - {issue}")
    else:
        print("\nNo critical issues detected (all scenarios <50% failure rate)")


# Direct execution for testing
if __name__ == "__main__":
    print("Running JIT + Optimization stress test...")
    results = test_jit_optimization_stress_comprehensive()
    print("\nStress test completed!")
