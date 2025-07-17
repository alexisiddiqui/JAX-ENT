import copy
import os
from pathlib import Path

import matplotlib.pyplot as plt
from MDAnalysis import Universe

from jaxent.src.custom_types.config import FeaturiserSettings
from jaxent.src.custom_types.HDX import HDX_protection_factor
from jaxent.src.data.loader import ExpD_Dataloader
from jaxent.src.data.splitting.split import DataSplitter
from jaxent.src.featurise import run_featurise
from jaxent.src.interfaces.builder import Experiment_Builder
from jaxent.src.interfaces.topology import Partial_Topology
from jaxent.src.models.config import BV_model_Config
from jaxent.src.models.HDX.BV.forwardmodel import BV_model
from jaxent.tests.plots.datasplitting import plot_split_visualization
from jaxent.tests.test_utils import get_inst_path


def ensure_output_dir():
    """Create the output directory if it doesn't exist."""
    output_dir = Path(__file__).parents[3] / "_plots" / "datasplitting"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def test_split_with_feature_topology():
    """
    Test data splitting using feature topology as common residues instead of MDAnalysis universe.
    This approach uses the topology information directly from the featurization process.
    """
    print("=" * 60)
    print("Testing data splitting with feature topology as common residues")
    print("=" * 60)

    # Setup model and featurizer
    bv_config = BV_model_Config()
    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)

    # Load topology and trajectory
    base_dir = Path(__file__).parents[4]
    inst_path = get_inst_path(base_dir)
    topology_path = inst_path / "clean" / "BPTI" / "BPTI_overall_combined_stripped.pdb"
    test_universe = Universe(str(topology_path))
    universes = [test_universe]
    models = [BV_model(bv_config)]

    # Create ensemble and run featurization
    ensemble = Experiment_Builder(universes, models)
    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    print(f"Features shape: {features[0].features_shape}")
    print(f"Feature topology length: {len(feature_topology[0])}")

    # Use feature topology directly as common residues
    # The feature_topology contains Partial_Topology objects that correspond to the features
    common_residues_from_features = set(feature_topology[0])

    print(f"Common residues from feature topology: {len(common_residues_from_features)}")
    print("Sample feature topologies:")
    for i, topo in enumerate(list(common_residues_from_features)[:5]):
        print(f"  {i + 1}: {topo}")

    # Create fake experimental dataset using feature topology
    # This simulates having experimental data for all residues that were featurized
    exp_data = [
        HDX_protection_factor(protection_factor=10 + i * 0.1, top=top)
        for i, top in enumerate(common_residues_from_features)
    ]

    print(f"Created experimental dataset with {len(exp_data)} datapoints")

    # Create dataset loader
    dataset = ExpD_Dataloader(data=exp_data)
    print(f"Dataset size: {len(dataset.data)}")

    # Extract topology objects from the experimental dataset for splitting
    exp_top_segments = [datapoint.top for datapoint in dataset.data]

    # Create splitter using feature topology as common residues
    # Note: We don't provide an ensemble since we're using pre-computed common residues
    splitter = DataSplitter(
        dataset=dataset,
        random_seed=42,
        ensemble=None,  # Not using MDAnalysis universes
        common_residues=common_residues_from_features,  # Use feature topology directly
        check_trim=False,  # Since these aren't peptides
        train_size=0.6,
        centrality=True,
    )

    print("\nPerforming random split...")
    train_data, val_data = splitter.random_split(remove_overlap=True)

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(
        f"Total split coverage: {len(train_data) + len(val_data)}/{len(dataset.data)} "
        f"({100 * (len(train_data) + len(val_data)) / len(dataset.data):.1f}%)"
    )

    # Verify no overlap between train and validation sets
    train_residues = set()
    val_residues = set()

    for datapoint in train_data:
        train_residues.update(datapoint.top.residues)

    for datapoint in val_data:
        val_residues.update(datapoint.top.residues)

    overlap = train_residues.intersection(val_residues)
    print(f"Overlapping residues between train/val: {len(overlap)}")
    if overlap:
        print(f"Overlap residues: {sorted(list(overlap))[:10]}...")  # Show first 10

    # Analyze feature coverage
    feature_residues = set()
    for topo in feature_topology[0]:
        feature_residues.update(topo.residues)

    split_residues = train_residues.union(val_residues)
    coverage = len(split_residues.intersection(feature_residues)) / len(feature_residues)
    print(f"Feature coverage by split: {coverage:.2%}")

    # Visualize split and save plot
    print("\nGenerating split visualization...")
    fig = plot_split_visualization(train_data, val_data, dataset.data)
    output_dir = ensure_output_dir()
    output_path = output_dir / "feature_topology_split.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)

    return train_data, val_data, features, feature_topology


def test_compare_splitting_approaches():
    """
    Compare splitting using feature topology vs MDAnalysis universe approach.
    This helps validate that the feature topology approach gives reasonable results.
    """
    print("\n" + "=" * 60)
    print("Comparing feature topology vs universe-based splitting")
    print("=" * 60)

    # Setup
    bv_config = BV_model_Config()
    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)
    base_dir = Path(__file__).parents[4]
    inst_path = get_inst_path(base_dir)
    topology_path = inst_path / "clean" / "BPTI" / "BPTI_overall_combined_stripped.pdb"
    test_universe = Universe(str(topology_path))
    universes = [test_universe]
    models = [BV_model(bv_config)]

    # Run featurization
    ensemble = Experiment_Builder(universes, models)
    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    # Method 1: Feature topology approach
    common_residues_features = set(feature_topology[0])

    # Method 2: Universe approach (original)
    common_residues_universe = Partial_Topology.find_common_residues(
        universes, exclude_selection="(resname PRO or resid 1) "
    )[0]

    print(f"Feature topology common residues: {len(common_residues_features)}")
    print(f"Universe-based common residues: {len(common_residues_universe)}")

    # Compare residue coverage
    feature_residue_ids = set()
    universe_residue_ids = set()

    for topo in common_residues_features:
        feature_residue_ids.update(topo.residues)

    for topo in common_residues_universe:
        universe_residue_ids.update(topo.residues)

    print(f"Feature approach covers residues: {sorted(list(feature_residue_ids))[:10]}...")
    print(f"Universe approach covers residues: {sorted(list(universe_residue_ids))[:10]}...")

    overlap = feature_residue_ids.intersection(universe_residue_ids)
    print(
        f"Residue overlap between approaches: {len(overlap)}/{len(feature_residue_ids.union(universe_residue_ids))}"
    )

    # Test both splitting approaches with the same experimental data
    # Use intersection of both approaches to ensure fair comparison
    common_intersection = common_residues_features.intersection(common_residues_universe)
    print(f"Common residues in both approaches: {len(common_intersection)}")

    if len(common_intersection) > 10:  # Only proceed if we have enough residues
        exp_data = [
            HDX_protection_factor(protection_factor=10 + i * 0.1, top=top)
            for i, top in enumerate(common_intersection)
        ]
        dataset = ExpD_Dataloader(data=exp_data)

        # Split using feature topology
        splitter_features = DataSplitter(
            dataset=copy.deepcopy(dataset),
            random_seed=42,
            ensemble=None,
            common_residues=common_residues_features,
            train_size=0.6,
        )
        train_feat, val_feat = splitter_features.random_split()

        # Split using universe approach
        splitter_universe = DataSplitter(
            dataset=copy.deepcopy(dataset),
            random_seed=42,
            ensemble=universes,
            common_residues=None,  # Will be computed from ensemble
            train_size=0.6,
        )
        train_univ, val_univ = splitter_universe.random_split()

        print(f"Feature approach - Train: {len(train_feat)}, Val: {len(val_feat)}")
        print(f"Universe approach - Train: {len(train_univ)}, Val: {len(val_univ)}")
    else:
        print("Not enough common residues for comparison split")


def test_feature_topology_with_peptides():
    """
    Test feature topology splitting with peptide data.
    This demonstrates how to handle peptide trimming in the splitting process.
    """
    print("\n" + "=" * 60)
    print("Testing feature topology splitting with peptide trimming")
    print("=" * 60)

    # Setup
    bv_config = BV_model_Config()
    featuriser_settings = FeaturiserSettings(name="BV", batch_size=None)
    base_dir = Path(__file__).parents[4]
    inst_path = get_inst_path(base_dir)
    topology_path = inst_path / "clean" / "BPTI" / "BPTI_overall_combined_stripped.pdb"
    test_universe = Universe(str(topology_path))
    universes = [test_universe]
    models = [BV_model(bv_config)]

    # Run featurization
    ensemble = Experiment_Builder(universes, models)
    features, feature_topology = run_featurise(ensemble, featuriser_settings)

    # Convert some feature topologies to peptides for testing
    peptide_feature_topology = []
    for i, topo in enumerate(feature_topology[0]):
        if i % 3 == 0 and topo.length > 4:  # Make every 3rd topology a peptide if long enough
            peptide_topo = copy.deepcopy(topo)
            peptide_topo.set_peptide(True, trim=2)
            peptide_feature_topology.append(peptide_topo)
        else:
            peptide_feature_topology.append(topo)

    print(f"Created {sum(1 for t in peptide_feature_topology if t.peptide)} peptide topologies")

    # Create experimental data
    exp_data = [
        HDX_protection_factor(protection_factor=10 + i * 0.1, top=top)
        for i, top in enumerate(peptide_feature_topology)
    ]
    dataset = ExpD_Dataloader(data=exp_data)

    # Test splitting with peptide trimming
    splitter = DataSplitter(
        dataset=dataset,
        random_seed=42,
        ensemble=None,
        common_residues=set(peptide_feature_topology),
        check_trim=True,  # Enable peptide trimming
        train_size=0.6,
    )

    train_data, val_data = splitter.random_split()

    print(f"Peptide-aware split - Train: {len(train_data)}, Val: {len(val_data)}")

    # Analyze peptide coverage
    train_peptides = sum(1 for d in train_data if d.top.peptide)
    val_peptides = sum(1 for d in val_data if d.top.peptide)
    total_peptides = sum(1 for d in dataset.data if d.top.peptide)

    print(
        f"Peptide distribution - Train: {train_peptides}, Val: {val_peptides}, Total: {total_peptides}"
    )


if __name__ == "__main__":
    import jax

    print("Local devices:", jax.local_devices())
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    # Ensure output directory exists
    ensure_output_dir()

    try:
        # Test basic feature topology splitting
        train_data, val_data, features, feature_topology = test_split_with_feature_topology()

        # Compare approaches
        test_compare_splitting_approaches()

        # Test with peptides
        test_feature_topology_with_peptides()

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()