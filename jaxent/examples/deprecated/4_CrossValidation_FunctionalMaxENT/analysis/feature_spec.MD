# Feature Specification (`--json-feature-spec`)

This document outlines how to use a JSON file to specify which structural features to calculate in the `analyse_cluster_LocalFeatures_PUFs.py` script. This method provides a flexible way to define features, regions, and plotting options without modifying the script itself.

## Command-Line Arguments

To use this functionality, you need to provide two arguments:

1.  `--json_data_path`: Path to a JSON file containing the definitions of structural regions (e.g., helices, strands). This file maps human-readable names (like "α1 helix") to residue ranges.
2.  `--json-feature-spec`: Path to the JSON file that specifies which features to compute, as described in this document.

### Example Usage

```bash
python jaxent/examples/2_CrossValidation/analysis/analyse_cluster_LocalFeatures_PUFs.py \
    --ensembles "path/to/your.pdb,path/to/your.xtc" \
    --names "MyEnsemble" \
    --json_data_path /path/to/key_residues.json \
    --json-feature-spec /path/to/feature_spec.json \
    --output_dir /path/to/output \
    --reference_pdb /path/to/reference.pdb
```

## JSON Structure for Feature Specification

The feature specification JSON is organized by feature type. The top-level keys correspond to the types of features you can calculate:

-   `region_dihedrals`: For calculating phi and psi dihedral angles within a specific region. This directly measures secondary structure precisely but exact thresholds are forcefield specific.
-   `internal_rmsd`: For calculating the internal root-mean-square deviation (RMSD) of a region. Measures the general amount of disorder. When used with a reference it describes a change in shape.
-   `region_rmsd`: For calculating the pairwise RMSD between the centers of mass of two different regions. Measure the displacement between regions.

Each key holds a list of "spec" objects, where each object defines a specific feature to be calculated.

---

### 1. `region_dihedrals`

Calculates the mean phi and psi dihedral angles for a specified structural region. This will generate two features per spec object: one for phi and one for psi.

#### Parameters:

-   `spec_name` (string, required): A unique name for this feature specification. This will be used as the base name for the output features (e.g., `dih_a3_phi`, `dih_a3_psi`).
-   `region_name` (string, required): The name of the structural region to analyze. This name **must** correspond to a region defined in the `--json_data_path` file.
-   `reference` (boolean, optional): If `true`, calculates the change in dihedral angles relative to the reference structure. If `false`, calculates the absolute dihedral angles. Defaults to the global `--reference_pdb` setting if not provided.
-   `centre` (string, optional): If set to `"mean"` or `"median"`, the feature data for each ensemble will be centered independently by subtracting the ensemble's mean or median value, respectively. This is useful for comparing relative changes across ensembles with different baselines.
-   `use_median` (boolean, optional): For plotting, determines whether to use the median (`true`) or mean (`false`) as the center for highlighting. Defaults to `false` (mean).
-   `plot_highlight_range` (float, optional): If provided, the distribution plot for this feature will include a highlighted region. The range is `center ± plot_highlight_range`. For example, a value of `1` with `use_median: true` will highlight the range from `median - 1` to `median + 1`.

#### Example:

```json
{
  "region_dihedrals": [
    {
      "spec_name": "dih_α3",
      "region_name": "α3 helix",
      "reference": true,
      "use_median": true,
      "plot_highlight_range": 1.5
    }
  ]
}
```

---

### 2. `internal_rmsd`

Calculates the mean internal pairwise RMSD of C-alpha atoms within a single structural region. This measures the compactness or expansion of a region.

#### Parameters:

-   `spec_name` (string, required): A unique name for the output feature.
-   `region_name` (string, required): The name of the structural region to analyze, which must be defined in the `--json_data_path` file.
-   `reference` (boolean, optional): If `true`, calculates the difference from the reference structure's internal RMSD. If `false`, calculates the absolute internal RMSD. Defaults to the global `--reference_pdb` setting.
-   `centre` (string, optional): If set to `"mean"` or `"median"`, the feature data for each ensemble will be centered independently.
-   `use_median` (boolean, optional): For plotting, use median (`true`) or mean (`false`) for highlighting. Defaults to `false`.
-   `plot_highlight_range` (float, optional): The range to highlight on the distribution plot.

#### Example:

```json
{
  "internal_rmsd": [
    {
      "spec_name": "compactness_α2",
      "region_name": "α2 helix",
      "reference": true,
      "use_median": false,
      "plot_highlight_range": 0.5
    }
  ]
}
```

---

### 3. `region_rmsd`

Calculates the pairwise RMSD between the C-alpha atoms of two different structural regions. This is useful for measuring the displacement or change in distance between regions.

#### Parameters:

-   `spec_name` (string, required): A unique name for the output feature.
-   `region_names` (string, required): A comma-separated string of the two region names to compare. Both names must be defined in the `--json_data_path` file.
-   `reference` (boolean, optional): If `true`, calculates the difference from the reference structure's pairwise RMSD. If `false`, calculates the absolute pairwise RMSD. Defaults to the global `--reference_pdb` setting.
-   `centre` (string, optional): If set to `"mean"` or `"median"`, the feature data for each ensemble will be centered independently.
-   `use_median` (boolean, optional): For plotting, use median (`true`) or mean (`false`) for highlighting. Defaults to `false`.
-   `plot_highlight_range` (float, optional): The range to highlight on the distribution plot.

#### Example:

```json
{
  "region_rmsd": [
    {
      "spec_name": "displacement_α2_vs_α3",
      "region_names": "α2 helix, α3 helix",
      "reference": true
    }
  ]
}
```

---

### Full Example `feature_spec.json`

You can combine all feature types in a single file.

```json
{
  "region_dihedrals": [
    {
      "spec_name": "dih_α3",
      "region_name": "α3 helix",
      "reference": true,
      "use_median": true,
      "plot_highlight_range": 1.5
    },
    {
      "spec_name": "dih_β1",
      "region_name": "β1 strand",
      "reference": true
    }
  ],
  "internal_rmsd": [
    {
      "spec_name": "compactness_α2",
      "region_name": "α2 helix",
      "reference": true,
      "use_median": false,
      "plot_highlight_range": 0.5
    }
  ],
  "region_rmsd": [
    {
      "spec_name": "displacement_α2_vs_α3",
      "region_names": "α2 helix, α3 helix",
      "reference": true
    }
  ]
}
```
