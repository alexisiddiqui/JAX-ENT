# IsoValidation Workflow

The Iso Validaiton experiment is a self-consistent and model-agnostic workflow that can be used to benchmark models using a known ground truth.

## Essential Outline

The basic procedure for this workflow involves creating synthetic data using reference states, inverting the natural population and then asking the model to fit against this. As the reference state populations are set in advance, this can be used to assess the quality of the fit.

1. Cluster trajectory into states using reference states. (Ca RMSD).

2. Calculate artificial biophysical data from identified clusters. (Uptake curves).

3. Combine artificial data across clusters, setting the relative populations to a plausible, yet unatural distribution. (Inversion)

4. Fit trajectory against artificial data generated.

5. Measure population recovery using identified clustering.

#### Notes

This example uses the TeaA membrane transporter with two major conformations: Open and Closed.

## Actual Implementation

While the IsoValidaiton outline is straightforward, we make some minor modifications in order to make this a more challenging target.

In addition to the original trajectory we create an additional trajectory that only contains structures that the artificial HDX uptake curves are composed from.

#### Modifications:

- Artificial HDX uptake curves are computed from protection factors using the Persson-Halle model. Details are provided in the _Bradshaw folder.
    - Contacts were featurised using a switch cutoff.

- Featurisation for fitting uses the Best-Vendruscolo (BV) structure-uptake model and uses hard cutoffs
    - Trajectories used for fitting are sliced by an interval of 100 by default.

- Clusters to compute uptake curves were obtained using DBScan for clustering using the RMSD (eps=1.0 Angstrom)
     - In jaxENT, clusters are assigned using a hard RMSD cutoff to each reference state (1.0 Angstrom)

### Procedure

#### Artificial Uptake Curve Generation (from HDXer tutorial)

- Cluster trajectory by DBScan using Open and Closed RMSD as features (eps=1.0).
- Compute uptake curves using Persson-Halle and switch contacts.
- Combine uptake curves to desired state ratios (Open:Closed = 0.6:0.4).

#### Ensemble Generation

- Tri-Modal ensemble: The sliced (interval = 100) trajectory used to compute the artifical uptake curves.

- Bi-Modal ensemble: Clustered Tri-Modal ensemble by hard RMSD cutoff to both Open and Closed reference states.

#### Fitting and Optimisation

- Featurise ensembles using the BV model and hard contacts.
    - Uses HDXer computed intrinsic rates
- Fit data using training/validation splits and replicates.

#### Analysis

- Reference state clustering: Is performed by computing aligned RMSD to each reference state by Ca atoms. Frames < 1.0 Angstrom are assigned to each cluster. If RMSD > 1.0 then frame is left unassigned.


- Open state recovery% is defined as:
```python
    open_target_population = 0.6

    open_state_weights = normalised_weights[open_state_assignments]

    open_sate_population = sum(open_state_weights)

    open_state_recovery% = (open_sate_population/open_target_population) * 100
```

- KL Divergence is computed against a uniform prior.


#### Data Splitting

- Data split to 0.5 training fraction
- Split types: random, sequence, sequence_cluster, spatial, and stratified