# ATLAS data provenance

This directory acquires the ATLAS analysis-tier molecular-dynamics archives used by the
ATLAS_BV experiment. The raw catalogue, archives, trajectories, and simulation-parameter bundle
are local generated inputs and are intentionally ignored by Git. `systems.csv`,
`selection_provenance.yaml`, `download_manifest.csv`, and `acquisition_report.csv` are the small
reproducibility records retained with the experiment.

ATLAS is described by Vander Meersche et al., *Nucleic Acids Research* 52 (2024), D384-D392,
DOI: 10.1093/nar/gkad1084. The dataset is distributed under CC-BY-NC 4.0 and must only be used for
non-commercial work with attribution.

From the repository root:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python \
  jaxent/examples/ATLAS_BV/data/select_systems.py
bash jaxent/examples/ATLAS_BV/data/fetch_atlas.sh --pilot 1
bash jaxent/examples/ATLAS_BV/data/fetch_atlas.sh --all
bash jaxent/examples/ATLAS_BV/data/fetch_atlas.sh --verify-only
```

Downloads are sequential, use at most two connections to the ATLAS server, and resume when rerun.
Use `--repair` only when an existing extracted system fails validation; it moves the old directory
to `data/quarantine/` before reacquiring it.

