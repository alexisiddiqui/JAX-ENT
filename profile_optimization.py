#!/usr/bin/env python3
"""
JAX Profiler Script for JAX-ENT Optimization Process

This script runs a simplified version of the IsoValidation optimization
with JAX profiling enabled to generate computational graph reports.

Usage:
    python profile_optimization.py
"""

import os
import sys
import jax
import jax.numpy as jnp
import jax.profiler

# Add jaxent to path
sys.path.insert(0, '/home/user/JAX-ENT')

from jaxent.src.custom_types.HDX import HDX_peptide
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.interfaces.simulation import Simulation_Parameters
from jaxent.src.interfaces.topology import Partial_Topology
from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.core import Simulation
from jaxent.src.models.HDX.BV.features import BV_input_features
from jaxent.src.models.HDX.BV.forwardmodel import BV_model
from jaxent.src.opt.losses import hdx_uptake_mean_centred_MSE_loss
from jaxent.src.opt.optimiser import OptaxOptimizer


def create_synthetic_small_dataset():
    """Create a small synthetic dataset for profiling"""
    print("Creating synthetic dataset...")

    # Create synthetic features (small size for quick profiling)
    n_residues = 50
    n_frames = 100

    heavy_contacts = jnp.ones((n_residues, n_frames)) * 5.0
    acceptor_contacts = jnp.ones((n_residues, n_frames)) * 2.0
    k_ints = jnp.ones(n_residues) * 0.1

    features = BV_input_features(
        heavy_contacts=heavy_contacts,
        acceptor_contacts=acceptor_contacts,
        k_ints=k_ints
    )

    # Create synthetic topology
    feature_topology = [
        Partial_Topology(
            chain="A",
            residues=[i],
            fragment_index=i,
            fragment_name=f"res_{i}"
        )
        for i in range(n_residues)
    ]

    # Create synthetic experimental data (10 peptides)
    n_peptides = 10
    n_timepoints = 5

    hdx_peptides = []
    for i in range(n_peptides):
        # Random peptide spanning 5 residues
        start_res = i * 5
        residues = list(range(start_res, start_res + 5))

        top = Partial_Topology(
            chain="A",
            residues=residues,
            fragment_index=i,
            fragment_name=f"pep_{i}",
            peptide=True,
            peptide_trim=2
        )

        # Synthetic dfrac data
        dfrac = jnp.linspace(0.1, 0.9, n_timepoints).tolist()

        peptide = HDX_peptide(dfrac=dfrac, top=top)
        hdx_peptides.append(peptide)

    return features, feature_topology, hdx_peptides


def profile_optimization():
    """Run optimization with JAX profiling enabled"""

    print("=" * 80)
    print("JAX-ENT Optimization Profiling")
    print("=" * 80)

    # Configure JAX
    print("\nConfiguring JAX...")
    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Available devices: {jax.devices()}")

    # Create synthetic data
    features, feature_topology, hdx_peptides = create_synthetic_small_dataset()

    print(f"\nDataset info:")
    print(f"  Features shape: {features.features_shape}")
    print(f"  Number of peptides: {len(hdx_peptides)}")

    # Split data (80/20 train/val)
    split_idx = int(len(hdx_peptides) * 0.8)
    train_data = hdx_peptides[:split_idx]
    val_data = hdx_peptides[split_idx:]

    # Create dataloader
    print("\nCreating dataloader...")
    loader = ExpD_Dataloader(data=hdx_peptides)
    loader.create_datasets(
        train_data=train_data,
        val_data=val_data,
        features=features,
        feature_topology=feature_topology,
    )

    # Setup BV model
    print("\nSetting up BV model...")
    bv_config = BV_model_Config(num_timepoints=5)
    bv_config.timepoints = jnp.array([0.167, 1.0, 10.0, 60.0, 120.0])
    bv_model = BV_model(config=bv_config)
    model_parameters = bv_model.params

    # Create simulation
    print("\nInitializing simulation...")
    n_frames = features.features_shape[1]

    parameters = Simulation_Parameters(
        frame_weights=jnp.ones(n_frames) / n_frames,
        frame_mask=jnp.ones(n_frames),
        model_parameters=(model_parameters,),
        forward_model_weights=jnp.array([1.0]),
        normalise_loss_functions=jnp.ones(1),
        forward_model_scaling=jnp.ones(1),
    )

    sim = Simulation(
        input_features=(features,),
        forward_models=(bv_model,),
        params=parameters
    )
    sim.initialise()

    # Create optimizer
    print("\nCreating optimizer...")
    optimizer = OptaxOptimizer(
        learning_rate=1e-3,
        optimizer="adam",
    )

    opt_state = optimizer.initialise(
        model=sim,
        optimisable_funcs=None,
    )

    # Create profiler output directory
    profile_dir = "/home/user/JAX-ENT/profiler_output"
    os.makedirs(profile_dir, exist_ok=True)

    print(f"\nProfile output directory: {profile_dir}")
    print("\n" + "=" * 80)
    print("PROFILING PHASE 1: JIT Compilation (Cold Start)")
    print("=" * 80)

    # Profile JIT compilation (first call)
    with jax.profiler.trace(profile_dir + "/jit_compilation"):
        print("\nExecuting first optimization step (JIT compilation)...")
        opt_state, loss, history = optimizer.step(
            optimizer=optimizer,
            state=opt_state,
            simulation=sim,
            data_targets=(loader,),
            loss_functions=(hdx_uptake_mean_centred_MSE_loss,),
            indexes=(0,),
            history=optimizer.history,
        )
        # Wait for computation to complete
        loss.block_until_ready()
        print(f"  Step 0 - Loss: {loss:.6f}")

    print("\n" + "=" * 80)
    print("PROFILING PHASE 2: Optimized Execution (Warm Start)")
    print("=" * 80)

    # Profile optimized execution (subsequent calls)
    with jax.profiler.trace(profile_dir + "/optimized_execution"):
        print("\nExecuting optimization steps 1-5...")
        for step in range(1, 6):
            opt_state, loss, history = optimizer.step(
                optimizer=optimizer,
                state=opt_state,
                simulation=sim,
                data_targets=(loader,),
                loss_functions=(hdx_uptake_mean_centred_MSE_loss,),
                indexes=(0,),
                history=optimizer.history,
            )
            loss.block_until_ready()
            print(f"  Step {step} - Loss: {loss:.6f}")

    print("\n" + "=" * 80)
    print("PROFILING PHASE 3: Forward Pass Only")
    print("=" * 80)

    # Profile just the forward pass
    with jax.profiler.trace(profile_dir + "/forward_pass"):
        print("\nExecuting forward passes...")
        for i in range(10):
            sim.forward(parameters)
            # Wait for computation to complete
            sim.outputs[0].log_Pf.block_until_ready()
        print(f"  Completed 10 forward passes")

    print("\n" + "=" * 80)
    print("PROFILING PHASE 4: Loss Computation Only")
    print("=" * 80)

    # Profile just loss computation
    with jax.profiler.trace(profile_dir + "/loss_computation"):
        print("\nExecuting loss computations...")
        for i in range(10):
            train_loss, val_loss = hdx_uptake_mean_centred_MSE_loss(
                sim, loader, 0
            )
            train_loss.block_until_ready()
            val_loss.block_until_ready()
        print(f"  Completed 10 loss computations")
        print(f"  Final train loss: {train_loss:.6f}, val loss: {val_loss:.6f}")

    print("\n" + "=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)
    print(f"\nProfile traces saved to: {profile_dir}")
    print("\nView profiles:")
    print(f"  1. Start TensorBoard: tensorboard --logdir={profile_dir}")
    print(f"  2. Open browser: http://localhost:6006")
    print(f"  3. Navigate to the 'Profile' tab")
    print("\nAlternatively, use Chrome tracing:")
    print(f"  1. Open Chrome and navigate to: chrome://tracing")
    print(f"  2. Load the trace files from: {profile_dir}")

    # Generate text-based analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Print compilation statistics
    print("\nJAX Compilation Statistics:")
    print(f"  Number of compiled functions: {len(jax._src.dispatch.xla_callable_cache)}")

    # Print final optimization state
    print(f"\nFinal Optimization State:")
    print(f"  Train loss: {opt_state.losses.total_train_loss:.6f}")
    print(f"  Val loss: {opt_state.losses.total_val_loss:.6f}")
    print(f"  Frame weights (first 5): {opt_state.params.frame_weights[:5]}")
    print(f"  Frame weights sum: {jnp.sum(opt_state.params.frame_weights):.6f}")

    return profile_dir


if __name__ == "__main__":
    try:
        profile_dir = profile_optimization()
        print("\n✓ Profiling completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Profiling failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
