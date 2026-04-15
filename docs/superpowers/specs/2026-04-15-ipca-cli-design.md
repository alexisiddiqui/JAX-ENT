# iPCA CLI Design

**Date:** 2026-04-15  
**Status:** Approved

---

## Context

`efficient_k_cluster.py` already performs PCA on single MD trajectories using `IncrementalPCA`, but has no support for multiple trajectories with condition/replicate metadata. The PCA computation logic is also embedded inline in the CLI rather than being importable.

This design introduces:
1. A shared `src/analysis/PCA/` module — extracts PCA compute and plot functions from kcluster so both CLIs use the same code
2. A new `jaxent-ipca` CLI — multi-trajectory PCA visualization with condition/replicate-aware plots

No clustering is performed; this is a visualization and analysis tool.

---

## Module Architecture

### New: `jaxent/src/analysis/PCA/`

```
jaxent/src/analysis/PCA/
├── __init__.py
├── core.py       # Computation: pairwise distances, IncrementalPCA fit/transform
└── plots.py      # All PCA plots: single-traj (for kcluster) + multi-traj (for iPCA)
```

**`core.py`** — extracted and cleaned up from `jaxent/cli/efficient_k_cluster.py`:
- `calculate_pairwise_rmsd(universe, atom_selection, chunk_size)` → `np.ndarray (n_frames, n_features)`
- `perform_pca_on_distances(distance_matrix, n_components, chunk_size)` → `(pca_coords, explained_variance, pca_object)`
- `calculate_distances_and_perform_pca(universe, atom_selection, n_components, chunk_size)` → single-trajectory entry point (used by kcluster; signature unchanged)
- `calculate_multi_traj_pca(universes, atom_selection, n_components, chunk_size)` → new multi-trajectory entry point for iPCA; validates equal feature dimensions, stacks frames, fits IncrementalPCA jointly, returns `(pca_coords, explained_variance, metadata_dict)`

**`plots.py`** — new multi-trajectory plots + existing kcluster plots relocated here:
- `plot_combined_scatter(pca_coords, metadata, condition_colormap_map, output_path)`
- `plot_combined_density(pca_coords, metadata, condition_colormap_map, output_path)`
- `plot_condition_replicates(pca_coords, metadata, condition, output_path)`
- Existing kcluster plot functions (`create_publication_plots`) relocated here with no behaviour change

### Modified: `jaxent/cli/efficient_k_cluster.py`

Inline compute functions replaced with imports from `analysis/PCA/core.py`. Plotting calls updated to use `analysis/PCA/plots.py`. No change to CLI interface or output.

### New: `jaxent/cli/ipca.py`

New CLI file. Imports compute from `analysis/PCA/core.py`, plots from `analysis/PCA/plots.py`.

### Modified: `pyproject.toml`

```toml
jaxent-ipca = "jaxent.cli.ipca:main"
```

---

## CLI Interface

```bash
jaxent-ipca \
  --topology   wt.pdb       --topology   mut.pdb      --topology   mut.pdb \
  --trajectory wt_r1.xtc    --trajectory mut_r1.xtc   --trajectory mut_r2.xtc \
  --condition  WT            --condition  MUT           --condition  MUT \
  --replicate  1             --replicate  1             --replicate  2 \
  --output_dir ./ipca_out \
  --name       my_analysis \
  --atom_selection "name CA" \
  --num_components 10 \
  --chunk_size 100
```

**Arguments:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--topology` | str (repeated) | required | Topology file per trajectory |
| `--trajectory` | str (repeated) | required | Trajectory file |
| `--condition` | str (repeated) | required | Condition label (e.g. `WT`, `MUT`) |
| `--replicate` | int (repeated) | required | Replicate number |
| `--output_dir` | str | required | Root output directory |
| `--name` | str | required | Analysis name used in filenames |
| `--atom_selection` | str | `"name CA"` | MDAnalysis selection string |
| `--num_components` | int | `10` | Number of PCA components |
| `--chunk_size` | int | `100` | Frames per IncrementalPCA chunk |

**Validation (hard errors):**
- `len(topology) == len(trajectory) == len(condition) == len(replicate)` — must all be equal
- Selected atom count must match across all topologies — error message identifies which topology differs

---

## Computation Pipeline

1. **Load** — create one MDAnalysis `Universe(topology, trajectory)` per input pair
2. **Feature extraction** — call `calculate_pairwise_rmsd()` per trajectory → `(n_frames, n_features)` distance vectors; validate all have the same `n_features`
3. **Stack** — concatenate all frames into `(total_frames, n_features)`; build a parallel metadata array recording `(condition, replicate, traj_idx, local_frame_idx)` per row
4. **Fit IncrementalPCA** — call `perform_pca_on_distances()` on the combined matrix (chunked for memory efficiency)
5. **Transform** — project all frames → `(total_frames, n_components)` PCA coordinates
6. **Save data**:
   - `data/pca_coordinates.npz` — arrays: `coords (total_frames, n_components)`, `conditions (total_frames,)`, `replicates (total_frames,)`, `traj_indices (total_frames,)`, `frame_indices (total_frames,)`
   - `data/pca_variance.npz` — explained variance ratios per component
7. **Plot** — call functions from `analysis/PCA/plots.py`

PCA is fit jointly across all trajectories so all conditions/replicates share the same PCA space.

---

## Plots

### Figure 1: Combined scatter (`plots/ipca_combined_scatter.png`)

- PC1 vs PC2, all frames from all trajectories
- Color scheme: each unique condition mapped to a matplotlib sequential colormap family (Blues → Oranges → Greens → Purples → Reds, cycling if needed)
- Each replicate within a condition gets a distinct shade sampled evenly from the colormap range `[0.3, 0.9]` (avoids illegible extremes)
- Legend entries labelled `{condition} rep{replicate}`

### Figure 2: Combined density (`plots/ipca_combined_density.png`)

- One KDE contour per condition using `sns.kdeplot`
- Filled contours hued by local density, overlaid for all conditions
- Each condition uses its assigned color family
- Shows how condition populations overlap in the shared PCA space

### Figure 3: Per-condition replicate panels (`plots/ipca_{condition}_replicates.png`, one per condition)

- Grid layout: **one column per replicate, two rows**
  - **Top row**: scatter of that replicate's frames, hued by frame index (viridis, showing temporal progression)
  - **Bottom row**: KDE density plot for that replicate, hued by density
- Shared PC1/PC2 axis limits across all panels within the figure for direct comparison between replicates

---

## Output Structure

```
{output_dir}/
├── plots/
│   ├── ipca_combined_scatter.png
│   ├── ipca_combined_density.png
│   └── ipca_{condition}_replicates.png   # one per unique condition
└── data/
    ├── pca_coordinates.npz
    └── pca_variance.npz
```

---

## Verification

1. Run `jaxent-ipca` with 2 conditions × 2 replicates (4 trajectories total) against a test system
2. Confirm `pca_coordinates.npz` contains correct shape `(total_frames, n_components)` and matching metadata arrays
3. Confirm `pca_variance.npz` sums explained variance ≤ 1.0
4. Confirm `2 + N_conditions` plot files are produced: combined scatter, combined density, and one per-condition figure per unique condition
5. Verify colors: each condition uses a distinct colormap family; replicates show distinct shades
6. Verify per-condition figure has 2 rows × N_replicates columns with shared axis limits
7. Run `jaxent-kCluster` on the same test system and confirm it still produces identical output to pre-refactor (no regression)
8. Pass mismatched atom counts → confirm informative error message
9. Pass unequal list lengths → confirm informative error message
