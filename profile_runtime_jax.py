#!/usr/bin/env python3
"""
Dynamic Runtime Profiling for JAX-ENT Optimization Pipeline

This script uses JAX's built-in profiler to capture runtime performance metrics
during the optimization process, including:
- JIT compilation overhead
- Forward pass execution time
- Loss computation time
- Gradient computation time
- Parameter update time
- Memory usage patterns

The profiling data can be visualized using TensorBoard.

Usage:
    python profile_runtime_jax.py --output-dir ./profiling_results --n-steps 50
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import jax
import jax.numpy as jnp

# Ensure JAX uses CPU by default (change to GPU if desired)
# jax.config.update("jax_platform_name", "cpu")

# Add jaxent to path
sys.path.insert(0, '/home/user/JAX-ENT')

import jaxent.src.interfaces.topology as pt
from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model
from jaxent.src.opt.base import InitialisedSimulation
from jaxent.src.opt.losses import (
    hdx_uptake_mean_centred_MSE_loss,
    maxent_convexKL_loss,
)
from jaxent.src.opt.optimiser import OptaxOptimizer, OptimizationState
from jaxent.src.utils.jit_fn import jit_Guard


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print("\n" + "-" * 80)
    print(f" {title}")
    print("-" * 80)


def load_example_data(
    base_dir: str = "/home/user/JAX-ENT/jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT"
) -> Tuple[List[HDX_peptide], List[HDX_peptide], BV_input_features, List[pt.Partial_Topology]]:
    """
    Load example data from the IsoValidation_OMass pipeline.

    Args:
        base_dir: Base directory containing the example data

    Returns:
        Tuple of (train_data, val_data, features, feature_topology)
    """
    print_subsection("Loading Example Data")

    # Look for pre-featurized data
    featurise_dir = os.path.join(base_dir, "_featurise")
    datasplits_dir = os.path.join(base_dir, "_datasplits")

    # Load features
    feature_path = os.path.join(featurise_dir, "features_iso_tri.npz")
    topology_path = os.path.join(featurise_dir, "topology_iso_tri.json")

    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"Features not found at {feature_path}.\n"
            f"Please run featurise_ISO_TRI_BI.py first to generate features."
        )

    print(f"Loading features from: {feature_path}")
    features = BV_input_features.load(feature_path)
    print(f"  Features shape: {features.features_shape}")

    print(f"Loading topology from: {topology_path}")
    feature_topology = pt.PTSerialiser.load_list_from_json(topology_path)
    print(f"  Number of peptides: {len(feature_topology)}")

    # Load data splits (use first split from random split type)
    split_dir = os.path.join(datasplits_dir, "random", "split_000")

    if not os.path.exists(split_dir):
        raise FileNotFoundError(
            f"Data splits not found at {split_dir}.\n"
            f"Please run splitdata_ISO.py first to generate splits."
        )

    print(f"\nLoading data split from: {split_dir}")
    train_data = HDX_peptide.load_list_from_files(
        json_path=os.path.join(split_dir, "train_topology.json"),
        csv_path=os.path.join(split_dir, "train_dfrac.csv"),
    )
    print(f"  Training samples: {len(train_data)}")

    val_data = HDX_peptide.load_list_from_files(
        json_path=os.path.join(split_dir, "val_topology.json"),
        csv_path=os.path.join(split_dir, "val_dfrac.csv"),
    )
    print(f"  Validation samples: {len(val_data)}")

    return train_data, val_data, features, feature_topology


def setup_simulation(
    train_data: List[HDX_peptide],
    val_data: List[HDX_peptide],
    features: BV_input_features,
    feature_topology: List[pt.Partial_Topology],
    maxent_scaling: float = 1.0,
) -> Tuple[InitialisedSimulation, ExpD_Dataloader, Simulation_Parameters]:
    """
    Set up the simulation and data loaders.

    Args:
        train_data: Training HDX peptides
        val_data: Validation HDX peptides
        features: Input features
        feature_topology: Feature topology
        maxent_scaling: MaxEnt regularization scaling

    Returns:
        Tuple of (simulation, data_loader, prior_parameters)
    """
    print_subsection("Setting Up Simulation")

    # Create data loader
    print("Creating data loader...")
    hdx_data = train_data + val_data
    data_loader = ExpD_Dataloader(data=hdx_data)
    data_loader.create_datasets(
        train_data=train_data,
        val_data=val_data,
        features=features,
        feature_topology=feature_topology,
    )
    print(f"  Data loader created with {len(hdx_data)} total samples")

    # Setup BV model
    print("\nConfiguring BV forward model...")
    bv_config = BV_model_Config(num_timepoints=5)
    bv_config.timepoints = jnp.array([0.167, 1.0, 10.0, 60.0, 120.0])
    bv_model = BV_model(config=bv_config)
    model_parameters = bv_model.params
    print(f"  Timepoints: {bv_config.timepoints}")

    # Create simulation parameters
    n_frames = features.features_shape[1]
    print(f"\nInitializing simulation parameters...")
    print(f"  Number of frames: {n_frames}")

    parameters = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.array([maxent_scaling, 1.0]),
        normalise_loss_functions=jnp.ones(2),
        forward_model_scaling=jnp.ones(2) * 100.0,
    )

    # Create and initialize simulation
    print("\nCreating simulation...")
    sim = Simulation(
        input_features=(features,),
        forward_models=(bv_model,),
        params=parameters,
    )

    # Use jit_Guard for proper JIT management
    sim_guard = jit_Guard(sim, cleanup_on_exit=False)
    sim_initialized = sim_guard.__enter__()
    sim_initialized.initialise()
    print("  Simulation initialized successfully")

    # Generate prior for MaxEnt loss
    print("\nGenerating prior for MaxEnt regularization...")
    output_features = sim_initialized.forward(sim_initialized, params=parameters).outputs[0].y_pred()

    prior_HDX = []
    for idx, top in enumerate(feature_topology):
        prior_HDX.append(
            HDX_peptide._create_from_features(topology=top, features=output_features[idx])
        )

    # Create prior parameters (for MaxEnt loss)
    prior_params = parameters

    print(f"  Prior generated with {len(prior_HDX)} samples")

    return sim_initialized, data_loader, prior_params


def profile_optimization_step(
    simulation: InitialisedSimulation,
    optimizer: OptaxOptimizer,
    opt_state: OptimizationState,
    data_loader: ExpD_Dataloader,
    prior_params: Simulation_Parameters,
    loss_functions: Tuple,
    n_warmup_steps: int = 3,
    n_profile_steps: int = 10,
) -> dict:
    """
    Profile the optimization steps with detailed timing.

    Args:
        simulation: Initialized simulation
        optimizer: Optimizer instance
        opt_state: Initial optimization state
        data_loader: Data loader
        prior_params: Prior parameters for MaxEnt
        loss_functions: Tuple of loss functions
        n_warmup_steps: Number of warmup steps (for JIT compilation)
        n_profile_steps: Number of steps to profile

    Returns:
        Dictionary containing profiling results
    """
    print_subsection("Profiling Optimization Steps")

    results = {
        "warmup_times": [],
        "step_times": [],
        "loss_values": [],
        "jit_compilation_time": None,
        "forward_times": [],
        "loss_computation_times": [],
        "gradient_times": [],
        "update_times": [],
    }

    # Warmup phase (JIT compilation)
    print(f"\nWarmup Phase ({n_warmup_steps} steps - includes JIT compilation):")
    warmup_start = time.time()

    for step in range(n_warmup_steps):
        step_start = time.time()

        opt_state, current_loss, save_state, simulation = optimizer.step(
            optimizer=optimizer,
            state=opt_state,
            simulation=simulation,
            data_targets=(data_loader, prior_params),
            loss_functions=loss_functions,
            indexes=(0, 0),
        )

        # Block until computation completes
        jax.block_until_ready(opt_state.params)

        step_time = time.time() - step_start
        results["warmup_times"].append(step_time)

        print(f"  Step {step}: {step_time:.4f}s, Loss: {current_loss:.6e}")

    warmup_end = time.time()
    total_warmup = warmup_end - warmup_start

    # Estimate JIT compilation overhead (first step is typically much slower)
    if len(results["warmup_times"]) >= 2:
        results["jit_compilation_time"] = results["warmup_times"][0] - results["warmup_times"][1]
        print(f"\nEstimated JIT compilation overhead: {results['jit_compilation_time']:.4f}s")

    print(f"Total warmup time: {total_warmup:.4f}s")
    print(f"Average warmup step time: {jnp.mean(jnp.array(results['warmup_times'])):.4f}s")

    # Profiled phase
    print(f"\nProfiling Phase ({n_profile_steps} steps):")

    for step in range(n_profile_steps):
        step_start = time.time()

        # Time the optimization step
        opt_state, current_loss, save_state, simulation = optimizer.step(
            optimizer=optimizer,
            state=opt_state,
            simulation=simulation,
            data_targets=(data_loader, prior_params),
            loss_functions=loss_functions,
            indexes=(0, 0),
        )

        # Block until computation completes
        jax.block_until_ready(opt_state.params)

        step_time = time.time() - step_start
        results["step_times"].append(step_time)
        results["loss_values"].append(float(current_loss))

        print(f"  Step {step}: {step_time:.4f}s, Loss: {current_loss:.6e}")

    print(f"\nAverage profiled step time: {jnp.mean(jnp.array(results['step_times'])):.4f}s")
    print(f"Min step time: {jnp.min(jnp.array(results['step_times'])):.4f}s")
    print(f"Max step time: {jnp.max(jnp.array(results['step_times'])):.4f}s")

    return results


def profile_forward_pass(
    simulation: InitialisedSimulation,
    params: Simulation_Parameters,
    n_iterations: int = 100,
) -> dict:
    """
    Profile just the forward pass in isolation.

    Args:
        simulation: Initialized simulation
        params: Simulation parameters
        n_iterations: Number of forward passes to time

    Returns:
        Dictionary with forward pass timing results
    """
    print_subsection("Profiling Forward Pass")

    # Warmup
    print("Warmup (JIT compilation)...")
    for _ in range(3):
        output = simulation.forward(simulation, params=params)
        jax.block_until_ready(output.outputs[0].y_pred())

    # Profile
    print(f"Profiling {n_iterations} forward passes...")
    times = []

    start_total = time.time()
    for i in range(n_iterations):
        start = time.time()
        output = simulation.forward(simulation, params=params)
        jax.block_until_ready(output.outputs[0].y_pred())
        elapsed = time.time() - start
        times.append(elapsed)

        if (i + 1) % 20 == 0:
            print(f"  Completed {i + 1}/{n_iterations} iterations")

    total_time = time.time() - start_total

    times_array = jnp.array(times)
    results = {
        "mean_time": float(jnp.mean(times_array)),
        "std_time": float(jnp.std(times_array)),
        "min_time": float(jnp.min(times_array)),
        "max_time": float(jnp.max(times_array)),
        "total_time": total_time,
        "n_iterations": n_iterations,
    }

    print(f"\nForward Pass Timing:")
    print(f"  Mean: {results['mean_time']*1000:.2f}ms ± {results['std_time']*1000:.2f}ms")
    print(f"  Min:  {results['min_time']*1000:.2f}ms")
    print(f"  Max:  {results['max_time']*1000:.2f}ms")
    print(f"  Total time: {results['total_time']:.2f}s")
    print(f"  Throughput: {n_iterations/total_time:.1f} forward passes/sec")

    return results


def profile_with_jax_profiler(
    simulation: InitialisedSimulation,
    optimizer: OptaxOptimizer,
    opt_state: OptimizationState,
    data_loader: ExpD_Dataloader,
    prior_params: Simulation_Parameters,
    loss_functions: Tuple,
    output_dir: str,
    n_steps: int = 10,
):
    """
    Profile using JAX's built-in profiler for detailed traces.

    Args:
        simulation: Initialized simulation
        optimizer: Optimizer instance
        opt_state: Initial optimization state
        data_loader: Data loader
        prior_params: Prior parameters
        loss_functions: Loss functions
        output_dir: Directory to save profiling results
        n_steps: Number of steps to profile
    """
    print_subsection("JAX Profiler Trace")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Warmup (to complete JIT compilation)
    print("Performing warmup...")
    for _ in range(3):
        opt_state, current_loss, save_state, simulation = optimizer.step(
            optimizer=optimizer,
            state=opt_state,
            simulation=simulation,
            data_targets=(data_loader, prior_params),
            loss_functions=loss_functions,
            indexes=(0, 0),
        )
        jax.block_until_ready(opt_state.params)

    print(f"Starting JAX profiler trace for {n_steps} steps...")
    print(f"Output directory: {output_dir}")

    # Start profiler
    jax.profiler.start_trace(output_dir)

    try:
        for step in range(n_steps):
            opt_state, current_loss, save_state, simulation = optimizer.step(
                optimizer=optimizer,
                state=opt_state,
                simulation=simulation,
                data_targets=(data_loader, prior_params),
                loss_functions=loss_functions,
                indexes=(0, 0),
            )
            jax.block_until_ready(opt_state.params)

            if (step + 1) % 5 == 0:
                print(f"  Completed {step + 1}/{n_steps} profiled steps")

    finally:
        # Stop profiler
        jax.profiler.stop_trace()

    print(f"\nProfiling trace saved to: {output_dir}")
    print(f"View with TensorBoard:")
    print(f"  tensorboard --logdir={output_dir}")


def generate_summary_report(
    optimization_results: dict,
    forward_results: dict,
    output_dir: str,
):
    """
    Generate a summary report of profiling results.

    Args:
        optimization_results: Results from optimization profiling
        forward_results: Results from forward pass profiling
        output_dir: Directory to save report
    """
    print_section("Profiling Summary Report")

    # Optimization step breakdown
    print("\n1. OPTIMIZATION STEP TIMING")
    print("-" * 80)

    if optimization_results["jit_compilation_time"]:
        print(f"JIT Compilation Overhead: {optimization_results['jit_compilation_time']:.4f}s")

    step_times = jnp.array(optimization_results["step_times"])
    print(f"\nOptimization Step (post-warmup):")
    print(f"  Mean time: {jnp.mean(step_times):.4f}s")
    print(f"  Std dev:   {jnp.std(step_times):.4f}s")
    print(f"  Min time:  {jnp.min(step_times):.4f}s")
    print(f"  Max time:  {jnp.max(step_times):.4f}s")

    warmup_times = jnp.array(optimization_results["warmup_times"])
    print(f"\nWarmup Steps (including JIT):")
    print(f"  Mean time: {jnp.mean(warmup_times):.4f}s")
    print(f"  First step: {warmup_times[0]:.4f}s (includes JIT compilation)")
    if len(warmup_times) > 1:
        print(f"  Subsequent steps: {jnp.mean(warmup_times[1:]):.4f}s")

    # Forward pass timing
    print("\n2. FORWARD PASS TIMING")
    print("-" * 80)
    print(f"Mean time: {forward_results['mean_time']*1000:.2f}ms ± {forward_results['std_time']*1000:.2f}ms")
    print(f"Min time:  {forward_results['min_time']*1000:.2f}ms")
    print(f"Max time:  {forward_results['max_time']*1000:.2f}ms")
    print(f"Throughput: {forward_results['n_iterations']/forward_results['total_time']:.1f} passes/sec")

    # Loss progression
    print("\n3. LOSS PROGRESSION")
    print("-" * 80)
    losses = jnp.array(optimization_results["loss_values"])
    print(f"Initial loss: {losses[0]:.6e}")
    print(f"Final loss:   {losses[-1]:.6e}")
    print(f"Loss reduction: {losses[0] - losses[-1]:.6e} ({(1 - losses[-1]/losses[0])*100:.2f}%)")

    # Performance metrics
    print("\n4. PERFORMANCE METRICS")
    print("-" * 80)
    mean_step_time = float(jnp.mean(step_times))
    print(f"Optimization steps per second: {1.0/mean_step_time:.2f}")
    print(f"Time for 1000 steps (estimated): {mean_step_time * 1000 / 60:.2f} minutes")
    print(f"Time for 10000 steps (estimated): {mean_step_time * 10000 / 60:.2f} minutes")

    # Device info
    print("\n5. DEVICE INFORMATION")
    print("-" * 80)
    devices = jax.devices()
    print(f"Number of devices: {len(devices)}")
    for i, device in enumerate(devices):
        print(f"  Device {i}: {device.device_kind} ({device.platform})")

    # Save report to file
    report_path = os.path.join(output_dir, "profiling_report.txt")
    print(f"\n6. SAVING REPORT")
    print("-" * 80)
    print(f"Report saved to: {report_path}")

    with open(report_path, 'w') as f:
        f.write("JAX-ENT Runtime Profiling Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("OPTIMIZATION STEP TIMING\n")
        f.write("-" * 80 + "\n")
        if optimization_results["jit_compilation_time"]:
            f.write(f"JIT Compilation Overhead: {optimization_results['jit_compilation_time']:.4f}s\n")
        f.write(f"Mean step time: {jnp.mean(step_times):.4f}s\n")
        f.write(f"Std dev: {jnp.std(step_times):.4f}s\n\n")

        f.write("FORWARD PASS TIMING\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean time: {forward_results['mean_time']*1000:.2f}ms\n")
        f.write(f"Throughput: {forward_results['n_iterations']/forward_results['total_time']:.1f} passes/sec\n\n")

        f.write("LOSS PROGRESSION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Initial loss: {losses[0]:.6e}\n")
        f.write(f"Final loss: {losses[-1]:.6e}\n")
        f.write(f"Reduction: {(1 - losses[-1]/losses[0])*100:.2f}%\n\n")

        f.write("DEVICE INFORMATION\n")
        f.write("-" * 80 + "\n")
        for i, device in enumerate(devices):
            f.write(f"Device {i}: {device.device_kind} ({device.platform})\n")


def main():
    """Main profiling function."""
    parser = argparse.ArgumentParser(description="Profile JAX-ENT optimization runtime")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./profiling_results",
        help="Directory to save profiling results",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50,
        help="Number of optimization steps to profile",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=3,
        help="Number of warmup steps (for JIT compilation)",
    )
    parser.add_argument(
        "--n-forward-iters",
        type=int,
        default=100,
        help="Number of forward pass iterations to profile",
    )
    parser.add_argument(
        "--enable-jax-profiler",
        action="store_true",
        help="Enable JAX profiler trace (for TensorBoard visualization)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-1,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--maxent-scaling",
        type=float,
        default=1.0,
        help="MaxEnt regularization scaling",
    )

    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"profile_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print_section("JAX-ENT Dynamic Runtime Profiler")
    print(f"\nConfiguration:")
    print(f"  Output directory: {output_dir}")
    print(f"  Optimization steps: {args.n_steps}")
    print(f"  Warmup steps: {args.n_warmup}")
    print(f"  Forward pass iterations: {args.n_forward_iters}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  MaxEnt scaling: {args.maxent_scaling}")
    print(f"  JAX profiler enabled: {args.enable_jax_profiler}")

    # Load data
    print_section("Loading Data")
    try:
        train_data, val_data, features, feature_topology = load_example_data()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure you have run the following scripts first:")
        print("  1. jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/featurise_ISO_TRI_BI.py")
        print("  2. jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT/splitdata_ISO.py")
        return 1

    # Setup simulation
    print_section("Setting Up Simulation")
    simulation, data_loader, prior_params = setup_simulation(
        train_data, val_data, features, feature_topology, args.maxent_scaling
    )

    # Setup optimizer
    print_subsection("Initializing Optimizer")
    loss_functions = (hdx_uptake_mean_centred_MSE_loss, maxent_convexKL_loss)
    print(f"Loss functions: {[f.__name__ for f in loss_functions]}")

    optimizer = OptaxOptimizer(
        learning_rate=args.learning_rate,
        optimizer="adam",
        initial_learning_rate=1e0,
        initial_steps=2,
    )
    opt_state = optimizer.initialise(model=simulation)
    print("Optimizer initialized successfully")

    # Profile forward pass
    print_section("Profiling Forward Pass")
    forward_results = profile_forward_pass(
        simulation, prior_params, n_iterations=args.n_forward_iters
    )

    # Profile optimization steps
    print_section("Profiling Optimization Steps")
    optimization_results = profile_optimization_step(
        simulation=simulation,
        optimizer=optimizer,
        opt_state=opt_state,
        data_loader=data_loader,
        prior_params=prior_params,
        loss_functions=loss_functions,
        n_warmup_steps=args.n_warmup,
        n_profile_steps=args.n_steps,
    )

    # JAX profiler trace (optional)
    if args.enable_jax_profiler:
        print_section("JAX Profiler Trace")

        # Re-initialize optimizer for clean profiling
        opt_state = optimizer.initialise(model=simulation)

        profiler_dir = os.path.join(output_dir, "jax_profiler_trace")
        profile_with_jax_profiler(
            simulation=simulation,
            optimizer=optimizer,
            opt_state=opt_state,
            data_loader=data_loader,
            prior_params=prior_params,
            loss_functions=loss_functions,
            output_dir=profiler_dir,
            n_steps=min(args.n_steps, 10),  # Limit profiler trace to 10 steps
        )

    # Generate summary report
    generate_summary_report(optimization_results, forward_results, output_dir)

    print_section("Profiling Complete")
    print(f"\nResults saved to: {output_dir}")
    if args.enable_jax_profiler:
        profiler_dir = os.path.join(output_dir, "jax_profiler_trace")
        print(f"\nTo view JAX profiler trace in TensorBoard:")
        print(f"  tensorboard --logdir={profiler_dir}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nProfiling interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during profiling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
