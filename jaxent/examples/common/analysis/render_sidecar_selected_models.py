"""Top-N PyMOL renders of the models shown in the sidecar model-selection bar plots.

For each bar of all_pairs_knn20_parameter_selected (panel x model [x selector]),
the selected saved states of the six split replicates are re-read from their
optimisation histories with each experiment's own candidate enumeration, stacked
as (n_replicates, n_frames) weights and rendered with
jaxent/src/analysis/pymol/Top_N_structures.py using the example panel configs.

Run with the jaxENT venv; rendering calls the PyMOL from the PYMOL_ANA env:
    .venv/bin/python jaxent/examples/common/analysis/render_sidecar_selected_models.py
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from jaxent.src.utils.hdf import load_optimization_history_from_file

REPO = Path(__file__).resolve().parents[4]
EXAMPLES = REPO/"jaxent/examples"
TOP_N = REPO/"jaxent/src/analysis/pymol/Top_N_structures.py"
PYMOL = Path.home()/"anaconda3/envs/PYMOL_ANA/bin/pymol"
SELECTION = "all_pairs_knn20_parameter_selected"
SELECTOR_LABELS = {"val_mse": "MSE selection", "val_closed_sigma_mse": "Closed-Sigma selection"}
MOPRP_TRAJECTORY = "_archived_MoPrP101/_cluster_MoPrP/clusters/all_clusters.xtc,MoPrP_max_plddt_4334.pdb"


def experiment(name, fitting, module, function, analysis, template, models, panels, trajectory=None, viewport="1600,1200"):
    return dict(name=name, fitting=EXAMPLES/fitting, module=module, function=function, analysis=analysis,
        template=EXAMPLES/template, models=models, panels=panels, trajectory=trajectory, viewport=viewport)


EXPERIMENTS = {
    "iso": experiment("ISO TRI · MSE fitting", "1_IsoValidation_OMass/fitting/jaxENT",
        "analyze_within_trajectory_selection", "candidates", REPO/"artifacts/iso_policy_sidecars",
        "1_IsoValidation_OMass/analysis/pymol_viz_configs/panels/top_n_TeaA_RMSD_ISO_TRI_MSE.yaml",
        ("MaxEnt", "OMC Work Scale", "OMC RMSD", "OMC PyRosetta"), ("All pairs", "k=20"),
        viewport="2100,1200"),  # the TeaA panel view was framed for a wide window
    "moprp": experiment("MoPrP AF2-MSAss · MSE fitting", "2_CrossValidation/fitting/jaxENT",
        "moprp_sidecar_support", "saved_candidates", EXAMPLES/"2_CrossValidation/analysis/_moprp_policy_sidecars",
        "2_CrossValidation/analysis/pymol_viz_configs/panels/top_n_MoPrP_AF2_MSAss_MSE.yaml",
        ("MaxEnt", "OMC Work Scale", "OMC Dihedral RMSD"), ("All pairs", "k=20 · L2-logPF neighbours"),
        MOPRP_TRAJECTORY),
    "moprp_bv": experiment("MoPrP AF2-MSAss · MSE fitting · BV fitted", "3_CrossValidationBV/fitting/jaxENT",
        "run_moprp_bv_policy_sidecars", "saved_candidates", EXAMPLES/"3_CrossValidationBV/analysis/_moprp_policy_sidecars",
        "2_CrossValidation/analysis/pymol_viz_configs/panels/top_n_MoPrP_AF2_MSAss_MSE.yaml",
        ("MaxEnt", "OMC Work Scale", "OMC Dihedral RMSD"), ("All pairs", "k=20 · L2-logPF neighbours"),
        MOPRP_TRAJECTORY),
}


def slug(*parts):
    parts = ("-".join("".join(c if c.isalnum() else " " for c in str(p).split("·")[0]).split()) for p in parts)
    return "_".join(p for p in parts if p)


def candidate_function(spec):
    sys.path.insert(0, str(spec["fitting"]))
    try:
        return getattr(importlib.import_module(spec["module"]), spec["function"])
    finally:
        sys.path.remove(str(spec["fitting"]))


def selected_weights(rows, candidates):
    """Weights of each selected saved state, checked against the selection table."""
    stack = []
    for _, row in rows.iterrows():
        kind, state, weights = list(candidates(load_optimization_history_from_file(row.history_path)))[int(row.candidate_index)]
        ess = 100/np.sum(weights**2)/len(weights)
        if kind != row.candidate_kind or int(state.step) != row.step or not np.isclose(ess, row.ess_percent, rtol=1e-6):
            raise ValueError(f"Saved state does not reproduce selection: {row.run_id}/{row.candidate_index}")
        stack.append(weights)
    return np.stack(stack)


def groups(spec, selected):
    keys = ["panel", "model"] + (["selection_metric"] if "selection_metric" in selected else [])
    seen = {}
    for key, rows in selected.groupby(keys, sort=False):
        key = dict(zip(keys, key))
        identity = tuple(sorted(zip(rows.history_path, rows.candidate_index)))
        # MaxEnt has no graph, so both panels select the same states.
        if identity in seen:
            seen[identity]["panels"].append(key["panel"])
            continue
        seen[identity] = dict(key, panels=[key["panel"]], rows=rows.sort_values(["split_type", "split_idx"]))
    return list(seen.values())


def write_config(spec, weights_path, prefix, title):
    config = yaml.safe_load(spec["template"].read_text())
    working_dir = Path(config["general"]["working_dir"])
    if not working_dir.is_dir():
        raise FileNotFoundError(working_dir)
    config["general"]["trajectory_label"] = "ensemble"
    config["render"]["ray_trace_on_save"] = True
    config["output"] = dict(save_png=True, save_session=False, output_prefix=str(prefix))
    config["top_n"]["weights"] = str(weights_path)
    if spec["trajectory"]:
        config["top_n"]["trajectory"] = spec["trajectory"]
    path = prefix.with_suffix(".yaml")
    path.write_text(f"# {title}\n"+yaml.safe_dump(config, sort_keys=False))
    png = Path(f"{prefix}_Top_{config['top_n']['top_n']}_colour_{config['top_n']['colour_metric']}.png")
    return path, png


def export(key, spec):
    selected = pd.read_csv(spec["analysis"]/f"{SELECTION}.csv")
    output = spec["analysis"]/"renders"
    output.mkdir(exist_ok=True)
    candidates = candidate_function(spec)
    manifest = []
    for group in groups(spec, selected):
        rows = group.pop("rows")
        panel = group["panels"][0] if len(group["panels"]) == 1 else "both-panels"
        name = slug(key, panel, group["model"], group.get("selection_metric", ""))
        weights = selected_weights(rows, candidates)
        weights_path = output/f"{name}_weights.npz"
        np.savez(weights_path, weights=weights, run_id=rows.run_id.to_numpy(str),
            step=rows.step.to_numpy(), candidate_index=rows.candidate_index.to_numpy())
        title = f"{spec['name']} · {' / '.join(group['panels'])} · {group['model']}"
        config, png = write_config(spec, weights_path, output/name, title)
        mean = weights.mean(0)
        manifest.append(dict(group, panels=" | ".join(group["panels"]), name=name, config=str(config),
            png=str(png),
            n_replicates=len(rows), recovery_mean=rows.recovery_percent.mean(), recovery_sd=rows.recovery_percent.std(),
            ess_mean=rows.ess_percent.mean(), averaged_ess_percent=100/np.sum(mean**2)/len(mean)))
    manifest = pd.DataFrame(manifest)
    manifest.to_csv(output/"manifest.csv", index=False)
    return manifest


def render(config, viewport, pymol):
    result = subprocess.run([str(pymol), "-cq", "-d", f"viewport {viewport}", "-r", str(TOP_N), "--", "--config", str(config)],
        capture_output=True, text=True, cwd=REPO)
    return config, result.returncode, result.stdout[-2000:]+result.stderr[-2000:]


def grid(key, spec, manifest):
    selectors = [s for s in SELECTOR_LABELS if "selection_metric" in manifest and s in set(manifest.selection_metric)] or [None]
    rows = [(panel, selector) for selector in selectors for panel in spec["panels"]]
    fig, axes = plt.subplots(len(rows), len(spec["models"]), figsize=(4*len(spec["models"]), 3.4*len(rows)), squeeze=False)
    for (panel, selector), line in zip(rows, axes, strict=True):
        for model, axis in zip(spec["models"], line, strict=True):
            match = manifest[(manifest.model == model) & manifest.panels.str.split(" | ", regex=False).map(lambda p: panel in p)]
            if selector:
                match = match[match.selection_metric == selector]
            axis.axis("off")
            if len(match) != 1 or not Path(match.png.iloc[0]).exists():
                axis.text(.5, .5, "missing render", ha="center", transform=axis.transAxes)
                continue
            item = match.iloc[0]
            axis.imshow(plt.imread(item.png))
            axis.set_title(f"{model}\nrecovery {item.recovery_mean:.1f} ± {item.recovery_sd:.1f}% · ESS {item.ess_mean:.1f}%", fontsize=9)
        label = panel+(f"\n{SELECTOR_LABELS[selector]}" if selector else "")
        line[0].text(-.04, .5, label, rotation=90, ha="right", va="center", fontsize=11, transform=line[0].transAxes)
    fig.suptitle(f"{spec['name']} · selected models (top-N frames by replicate-averaged weight)")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(spec["analysis"]/"renders"/f"{key}_selected_models_grid.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS), default=list(EXPERIMENTS))
    parser.add_argument("--pymol", type=Path, default=PYMOL)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()
    manifests = {key: export(key, EXPERIMENTS[key]) for key in args.experiments}
    if not args.skip_render:
        jobs = [(c, EXPERIMENTS[key]["viewport"]) for key, m in manifests.items() for c in m.config]
        with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
            for config, code, log in pool.map(lambda job: render(*job, args.pymol), jobs):
                print(("ok  " if code == 0 else "FAIL")+f" {Path(config).name}")
                Path(config).with_suffix(".log").write_text(log)
    for key, manifest in manifests.items():
        missing = [p for p in manifest.png if not Path(p).exists()]
        if missing:
            print(f"{key}: {len(missing)} renders missing, e.g. {missing[0]}")
        grid(key, EXPERIMENTS[key], manifest)
        print(manifest[["name", "panels", "recovery_mean", "recovery_sd", "averaged_ess_percent"]].to_string(index=False))
    print(json.dumps({k: str(EXPERIMENTS[k]["analysis"]/"renders") for k in manifests}, indent=2))


if __name__ == "__main__":
    main()
