import time
from contextlib import contextmanager
from typing import List

import jax
import jax.numpy as jnp
import pytest

from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.forwardmodel import BV_input_features, BV_model

# --- Test Fixtures ---


@pytest.fixture
def real_inputs_random_data():
    """
    Provides a set of real input objects populated with random data.
    This avoids file I/O while still testing the integration of real classes.
    """
    key = jax.random.PRNGKey(42)
    num_residues = 20
    num_frames = 100

    # 1. Create a real model instance
    bv_config = BV_model_Config()
    forward_models = [BV_model(bv_config)]

    # 2. Create real input features with random data
    key, *subkeys = jax.random.split(key, 4)
    input_features = [
        BV_input_features(
            heavy_contacts=jax.random.uniform(subkeys[0], (num_residues, num_frames)),
            acceptor_contacts=jax.random.uniform(subkeys[1], (num_residues, num_frames)),
            k_ints=jax.random.uniform(subkeys[2], (num_residues,)),
        )
    ]

    # 3. Create real simulation parameters with random weights
    key, subkey = jax.random.split(key)
    frame_weights = jax.random.uniform(subkey, (num_frames,))
    frame_weights /= jnp.sum(frame_weights)  # Normalize

    params = Simulation_Parameters(
        frame_weights=frame_weights,
        frame_mask=jnp.ones(num_frames),
        model_parameters=[model.config.forward_parameters for model in forward_models],
        forward_model_weights=jnp.array([1.0]),
        forward_model_scaling=jnp.array([1.0]),
        normalise_loss_functions=jnp.array([0.0]),
    )

    return input_features, forward_models, params


@contextmanager
def timeout_context(seconds=30):
    """Context manager to timeout operations that might hang."""
    import platform
    import signal

    # Only use signal on Unix-like systems
    if platform.system() != "Windows":

        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {seconds} seconds")

        # Set up the timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)

        try:
            yield
        finally:
            signal.alarm(0)  # Cancel the alarm
            signal.signal(signal.SIGALRM, old_handler)  # Restore old handler
    else:
        # On Windows, just yield without timeout (or implement threading-based timeout)
        print("Warning: Timeout not implemented on Windows, proceeding without timeout")
        yield


def create_parameter_variants(
    base_params: Simulation_Parameters, num_variants: int = 5
) -> List[Simulation_Parameters]:
    """Create multiple parameter variants for testing."""
    variants = []
    key = jax.random.PRNGKey(123)

    for i in range(num_variants):
        key, *subkeys = jax.random.split(key, 6)  # Split into 6 total: key + 5 subkeys

        # Vary frame weights
        frame_weights = jax.random.uniform(subkeys[0], base_params.frame_weights.shape)
        frame_weights = frame_weights / jnp.sum(frame_weights)

        # Vary frame mask (some framesz on/off)
        frame_mask = jax.random.choice(
            subkeys[1], 2, base_params.frame_mask.shape, p=jnp.array([0.2, 0.8])
        )

        # Vary model parameters by scaling
        scale_factor = 0.5 + jax.random.uniform(subkeys[2]) * 1.5  # Scale between 0.5-2.0
        scaled_model_params = []
        for param_set in base_params.model_parameters:
            # Assuming param_set has numeric fields that can be scaled
            scaled_params = jax.tree.map(
                lambda x: x * scale_factor if jnp.isscalar(x) or x.ndim == 0 else x, param_set
            )
            scaled_model_params.append(scaled_params)

        # Vary forward model weights and scaling
        fw_weights = jax.random.uniform(subkeys[3], base_params.forward_model_weights.shape)
        fw_weights = fw_weights / jnp.sum(fw_weights)

        fw_scaling = (
            0.1 + jax.random.uniform(subkeys[4], base_params.forward_model_scaling.shape) * 2.0
        )

        variant = Simulation_Parameters(
            frame_weights=frame_weights,
            frame_mask=frame_mask,
            model_parameters=scaled_model_params,
            forward_model_weights=fw_weights,
            forward_model_scaling=fw_scaling,
            normalise_loss_functions=base_params.normalise_loss_functions,
        )
        variants.append(variant)

    return variants


@pytest.mark.parametrize("raise_jit_failure", [True, False])
def test_jit_permutations_comprehensive(real_inputs_random_data, raise_jit_failure):
    """
    Comprehensive test for JIT failures across different permutations of:
    - Parameter variations
    - Initialization timing
    - Forward call patterns
    - JIT recompilation scenarios
    """
    input_features, forward_models, base_params = real_inputs_random_data

    # Create parameter variants
    param_variants = create_parameter_variants(base_params, num_variants=6)

    # Test scenarios
    test_scenarios = [
        "init_once_multiple_params",
        "reinit_each_param",
        "init_forward_reinit_forward",
        "multiple_forwards_same_param",
        "param_cycling",
        "jit_cache_invalidation",
    ]

    results = {}

    for scenario in test_scenarios:
        print(f"\n=== Testing scenario: {scenario} ===")
        scenario_results = []

        try:
            if scenario == "init_once_multiple_params":
                scenario_results = _test_init_once_multiple_params(
                    input_features, forward_models, param_variants, raise_jit_failure
                )

            elif scenario == "reinit_each_param":
                scenario_results = _test_reinit_each_param(
                    input_features, forward_models, param_variants, raise_jit_failure
                )

            elif scenario == "init_forward_reinit_forward":
                scenario_results = _test_init_forward_reinit_forward(
                    input_features, forward_models, param_variants, raise_jit_failure
                )

            elif scenario == "multiple_forwards_same_param":
                scenario_results = _test_multiple_forwards_same_param(
                    input_features, forward_models, param_variants, raise_jit_failure
                )

            elif scenario == "param_cycling":
                scenario_results = _test_param_cycling(
                    input_features, forward_models, param_variants, raise_jit_failure
                )

            elif scenario == "jit_cache_invalidation":
                scenario_results = _test_jit_cache_invalidation(
                    input_features, forward_models, param_variants, raise_jit_failure
                )

        except Exception as e:
            scenario_results = [{"error": str(e), "scenario": scenario}]

        results[scenario] = scenario_results

    # Analyze results
    _analyze_results(results)

    return results


def _test_init_once_multiple_params(
    input_features, forward_models, param_variants, raise_jit_failure
):
    """Initialize once, then try multiple different parameters."""
    results = []

    simulation = Simulation(
        input_features, forward_models, param_variants[0], raise_jit_failure=raise_jit_failure
    )

    try:
        with timeout_context(30):
            init_success = simulation.initialise()
        results.append(
            {
                "operation": "initial_init",
                "success": init_success,
                "jit_active": hasattr(simulation, "_jit_forward_pure"),
            }
        )
    except Exception as e:
        results.append({"operation": "initial_init", "success": False, "error": str(e)})
        return results

    for i, params in enumerate(param_variants):
        try:
            with timeout_context(15):
                simulation.forward(params)
            results.append({"operation": f"forward_{i}", "success": True, "param_variant": i})
        except TimeoutError:
            results.append(
                {
                    "operation": f"forward_{i}",
                    "success": False,
                    "error": "timeout",
                    "param_variant": i,
                }
            )
        except Exception as e:
            results.append(
                {"operation": f"forward_{i}", "success": False, "error": str(e), "param_variant": i}
            )

    return results


def _test_reinit_each_param(input_features, forward_models, param_variants, raise_jit_failure):
    """Reinitialize simulation for each parameter set."""
    results = []

    for i, params in enumerate(param_variants):
        try:
            simulation = Simulation(
                input_features, forward_models, params, raise_jit_failure=raise_jit_failure
            )

            with timeout_context(30):
                init_success = simulation.initialise()

            with timeout_context(15):
                simulation.forward(params)

            results.append(
                {"operation": f"reinit_forward_{i}", "success": True, "param_variant": i}
            )

        except TimeoutError:
            results.append(
                {
                    "operation": f"reinit_forward_{i}",
                    "success": False,
                    "error": "timeout",
                    "param_variant": i,
                }
            )
        except Exception as e:
            results.append(
                {
                    "operation": f"reinit_forward_{i}",
                    "success": False,
                    "error": str(e),
                    "param_variant": i,
                }
            )

    return results


def _test_init_forward_reinit_forward(
    input_features, forward_models, param_variants, raise_jit_failure
):
    """Alternating pattern: init->forward->reinit->forward."""
    results = []

    for i in range(0, len(param_variants) - 1, 2):
        try:
            # First init and forward
            simulation = Simulation(
                input_features,
                forward_models,
                param_variants[i],
                raise_jit_failure=raise_jit_failure,
            )

            with timeout_context(30):
                simulation.initialise()

            with timeout_context(15):
                simulation.forward(param_variants[i])

            results.append({"operation": f"init_forward_{i}", "success": True, "param_variant": i})

            # Reinit with different params and forward
            simulation.params = param_variants[i + 1]

            with timeout_context(30):
                simulation.initialise()

            with timeout_context(15):
                simulation.forward(param_variants[i + 1])

            results.append(
                {"operation": f"reinit_forward_{i + 1}", "success": True, "param_variant": i + 1}
            )

        except TimeoutError:
            results.append({"operation": f"alternating_{i}", "success": False, "error": "timeout"})
        except Exception as e:
            results.append({"operation": f"alternating_{i}", "success": False, "error": str(e)})

    return results


def _test_multiple_forwards_same_param(
    input_features, forward_models, param_variants, raise_jit_failure
):
    """Multiple forward calls with the same parameters to test JIT caching."""
    results = []

    simulation = Simulation(
        input_features, forward_models, param_variants[0], raise_jit_failure=raise_jit_failure
    )

    try:
        with timeout_context(30):
            simulation.initialise()

        # Multiple forwards with same params
        for call_num in range(5):
            try:
                with timeout_context(10):
                    simulation.forward(param_variants[0])
                results.append({"operation": f"repeat_forward_{call_num}", "success": True})
            except TimeoutError:
                results.append(
                    {
                        "operation": f"repeat_forward_{call_num}",
                        "success": False,
                        "error": "timeout",
                    }
                )
            except Exception as e:
                results.append(
                    {"operation": f"repeat_forward_{call_num}", "success": False, "error": str(e)}
                )

    except Exception as e:
        results.append({"operation": "repeat_forwards_init", "success": False, "error": str(e)})

    return results


def _test_param_cycling(input_features, forward_models, param_variants, raise_jit_failure):
    """Cycle through parameters multiple times to test for accumulating issues."""
    results = []

    simulation = Simulation(
        input_features, forward_models, param_variants[0], raise_jit_failure=raise_jit_failure
    )

    try:
        with timeout_context(30):
            simulation.initialise()

        # Cycle through parameters 3 times
        for cycle in range(3):
            for i, params in enumerate(param_variants[:3]):  # Use first 3 variants
                try:
                    with timeout_context(10):
                        simulation.forward(params)
                    results.append(
                        {
                            "operation": f"cycle_{cycle}_param_{i}",
                            "success": True,
                            "cycle": cycle,
                            "param_variant": i,
                        }
                    )
                except TimeoutError:
                    results.append(
                        {
                            "operation": f"cycle_{cycle}_param_{i}",
                            "success": False,
                            "error": "timeout",
                            "cycle": cycle,
                        }
                    )
                except Exception as e:
                    results.append(
                        {
                            "operation": f"cycle_{cycle}_param_{i}",
                            "success": False,
                            "error": str(e),
                            "cycle": cycle,
                        }
                    )

    except Exception as e:
        results.append({"operation": "param_cycling_init", "success": False, "error": str(e)})

    return results


def _test_jit_cache_invalidation(input_features, forward_models, param_variants, raise_jit_failure):
    """Test manual JIT cache clearing and recompilation."""
    results = []

    simulation = Simulation(
        input_features, forward_models, param_variants[0], raise_jit_failure=raise_jit_failure
    )

    try:
        with timeout_context(30):
            simulation.initialise()

        # Forward with first params
        with timeout_context(15):
            simulation.forward(param_variants[0])
        results.append({"operation": "initial_forward", "success": True})

        # Manually clear JIT function and force recompilation
        simulation._jit_forward_pure = simulation.forward_pure

        # Forward with different params (should trigger recompilation)
        with timeout_context(30):
            simulation.forward(param_variants[1])
        results.append({"operation": "post_clear_forward", "success": True})

        # Re-JIT and test again
        from jax import jit

        simulation._jit_forward_pure = jit(
            simulation.forward_pure, static_argnames=("forwardpass",)
        )

        with timeout_context(30):
            simulation.forward(param_variants[2])
        results.append({"operation": "post_rejit_forward", "success": True})

    except TimeoutError:
        results.append({"operation": "jit_invalidation", "success": False, "error": "timeout"})
    except Exception as e:
        results.append({"operation": "jit_invalidation", "success": False, "error": str(e)})

    return results


def _analyze_results(results):
    """Analyze test results and print summary."""
    print("\n" + "=" * 50)
    print("JIT PERMUTATION TEST ANALYSIS")
    print("=" * 50)

    total_operations = 0
    successful_operations = 0
    timeouts = 0
    errors = 0

    for scenario, scenario_results in results.items():
        print(f"\nScenario: {scenario}")
        scenario_success = 0
        scenario_total = len(scenario_results)
        scenario_timeouts = 0

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

        print(f"  Success: {scenario_success}/{scenario_total}")
        if scenario_timeouts > 0:
            print(f"  Timeouts: {scenario_timeouts}")
        if scenario_total - scenario_success - scenario_timeouts > 0:
            print(f"  Errors: {scenario_total - scenario_success - scenario_timeouts}")

    print("\nOVERALL SUMMARY:")
    print(f"Total operations: {total_operations}")
    print(
        f"Successful: {successful_operations} ({successful_operations / total_operations * 100:.1f}%)"
    )
    print(f"Timeouts: {timeouts} ({timeouts / total_operations * 100:.1f}%)")
    print(f"Errors: {errors} ({errors / total_operations * 100:.1f}%)")

    # Identify problematic scenarios
    if timeouts > 0 or errors > 0:
        print("\nPROBLEMATIC SCENARIOS:")
        for scenario, scenario_results in results.items():
            scenario_issues = [r for r in scenario_results if not r.get("success", False)]
            if scenario_issues:
                print(f"  {scenario}: {len(scenario_issues)} issues")
                for issue in scenario_issues[:3]:  # Show first 3 issues
                    print(
                        f"    - {issue.get('operation', 'unknown')}: {issue.get('error', 'failed')}"
                    )


# Additional helper test for edge cases
@pytest.mark.parametrize("raise_jit_failure", [True, False])
def test_extreme_jit_edge_cases(real_inputs_random_data, raise_jit_failure):
    """Test specific edge cases that are known to cause JIT issues."""
    input_features, forward_models, base_params = real_inputs_random_data

    # Test rapid parameter switching
    simulation = Simulation(
        input_features, forward_models, base_params, raise_jit_failure=raise_jit_failure
    )
    simulation.initialise()

    param_variants = create_parameter_variants(base_params, 10)

    # Rapid switching test
    key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)  # Use current time as seed
    for i in range(20):
        try:
            key, subkey = jax.random.split(key)
            idx = jax.random.randint(subkey, (), 0, len(param_variants))
            random_params = param_variants[int(idx)]
            with timeout_context(5):
                simulation.forward(random_params)
        except Exception as e:
            print(f"Rapid switching failed at iteration {i}: {e}")
            break

    print("Rapid switching test completed")


# Direct execution for testing outside of pytest
if __name__ == "__main__":
    print("Setting up test data...")

    # Create test data manually (equivalent to the fixture)
    key = jax.random.PRNGKey(42)
    num_residues = 20
    num_frames = 100

    # Create a real model instance
    bv_config = BV_model_Config()
    forward_models = [BV_model(bv_config)]

    # Create real input features with random data
    key, *subkeys = jax.random.split(key, 4)
    input_features = [
        BV_input_features(
            heavy_contacts=jax.random.uniform(subkeys[0], (num_residues, num_frames)),
            acceptor_contacts=jax.random.uniform(subkeys[1], (num_residues, num_frames)),
            k_ints=jax.random.uniform(subkeys[2], (num_residues,)),
        )
    ]

    # Create real simulation parameters with random weights
    key, subkey = jax.random.split(key)
    frame_weights = jax.random.uniform(subkey, (num_frames,))
    frame_weights /= jnp.sum(frame_weights)  # Normalize

    params = Simulation_Parameters(
        frame_weights=frame_weights,
        frame_mask=jnp.ones(num_frames),
        model_parameters=[model.config.forward_parameters for model in forward_models],
        forward_model_weights=jnp.array([1.0]),
        forward_model_scaling=jnp.array([1.0]),
        normalise_loss_functions=jnp.array([0.0]),
    )

    test_data = (input_features, forward_models, params)

    print("Running comprehensive JIT permutation tests...")
    results = test_jit_permutations_comprehensive(test_data)

    print("\nRunning extreme edge case tests...")
    test_extreme_jit_edge_cases(test_data)

    print("\nAll tests completed!")
