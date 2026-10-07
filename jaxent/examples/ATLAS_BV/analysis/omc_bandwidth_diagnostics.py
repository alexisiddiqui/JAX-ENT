"""Read-only mechanistic analysis of frozen OMC fits; never imports the optimiser.

All vector identities use float64, simplex-normalised saved weights. No fitting,
interpolation, alternate targets, or bandwidth selection occurs in this module.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import html
import json
import multiprocessing
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from .common import HERE, load_config, load_contact_coordinates, load_systems

INPUT = HERE / "outputs/analysis/pairwise_geometry/omc_bandwidth_control"
OUTPUT = INPUT.parent / "omc_bandwidth_diagnostics"
TIMES = (0.1, 1.0, 10.0)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def table(path, rows):
    data = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    data.to_parquet(path.with_suffix(".parquet"), index=False)
    data.to_csv(path.with_suffix(".csv"), index=False)
    return data


def basin_stats(w, labels):
    k = int(labels.max()) + 1
    p = np.bincount(labels, weights=w, minlength=k)
    sums = np.bincount(labels, weights=w * w, minlength=k)
    conditional = np.divide(p * p, sums, out=np.ones(k), where=sums > 0)
    return p, conditional


def inverse_ess_change(w1, w2, labels):
    p1, e1 = basin_stats(w1, labels)
    p2, e2 = basin_stats(w2, labels)
    population = np.sum((p2 * p2 - p1 * p1) * (1 / e1 + 1 / e2) / 2)
    within = np.sum((1 / e2 - 1 / e1) * (p1 * p1 + p2 * p2) / 2)
    delta = np.dot(w2, w2) - np.dot(w1, w1)
    np.testing.assert_allclose(population + within, delta, atol=2e-15)
    return dict(
        delta_inverse_ess=delta,
        population_contribution=population,
        within_contribution=within,
    )


def balanced_weights(labels, truth):
    counts = np.bincount(labels, minlength=len(truth))
    if np.any(counts == 0):
        raise ValueError("A target basin is absent from the candidate")
    w = truth[labels] / counts[labels]
    return w, 1 / np.sum(truth**2 / counts)


def residual_decomposition(x, labels, keep, w):
    """Population + lost coverage + conditional reweighting, observation-wise."""
    k = int(labels.max()) + 1
    labs = labels[keep]
    truth = np.bincount(labels, minlength=k) / len(labels)
    p, _ = basin_stats(w, labs)
    source = np.stack([x[:, labels == c].mean(axis=1) for c in range(k)])
    retained = np.stack([x[:, keep[labs == c]].mean(axis=1) for c in range(k)])
    conditional = np.stack(
        [
            x[:, keep[labs == c]] @ w[labs == c] / p[c] if p[c] > 0 else retained[c]
            for c in range(k)
        ]
    )
    a = (p - truth) @ source
    b = np.sum(p[:, None] * (retained - source), axis=0)
    c = np.sum(p[:, None] * (conditional - retained), axis=0)
    residual = x[:, keep] @ w - x.mean(axis=1)
    np.testing.assert_allclose(a + b + c, residual, atol=2e-14)
    terms = dict(
        population_sq=a * a,
        coverage_sq=b * b,
        reweight_sq=c * c,
        population_coverage_cross=2 * a * b,
        population_reweight_cross=2 * a * c,
        coverage_reweight_cross=2 * b * c,
    )
    np.testing.assert_allclose(sum(terms.values()), residual**2, atol=2e-14)
    return residual, terms


def graph_stats(kernel, w, labels):
    """Exact original OMC energy, with ordered edges; diagonal energy is zero."""
    n = len(w)
    between = labels[:, None] != labels[None, :]
    off = ~np.eye(n, dtype=bool)
    edge_energy = (
        0.5 * n * n * w[:, None] * w[None, :] * kernel * (w[:, None] - w[None, :]) ** 2
    )
    within_energy = edge_energy[~between].sum()
    between_energy = edge_energy[between].sum()
    c = w - 1 / n
    original = n * n * (np.dot(w * c * c, kernel @ w) - np.dot(w * c, kernel @ (w * c)))
    np.testing.assert_allclose(
        within_energy + between_energy, original, atol=2e-12, rtol=1e-9
    )
    return dict(
        graph_energy=within_energy + between_energy,
        within_energy=within_energy,
        between_energy=between_energy,
        mean_offdiagonal_kernel=kernel[off].mean(),
        within_kernel_mean=kernel[off & ~between].mean(),
        between_kernel_mean=kernel[between].mean(),
        between_kernel_fraction=kernel[between].sum() / kernel[off].sum(),
    )


def observability(x, labels):
    """Projection of basin contrasts on local weak/simplex-null directions."""
    centred = x - x.mean(axis=1, keepdims=True)
    _, s, vt = np.linalg.svd(centred, full_matrices=True)
    all_s = np.zeros(len(labels))
    all_s[: len(s)] = s
    numerical_null = all_s <= s[0] * max(centred.shape) * np.finfo(float).eps
    rows = []
    for c in np.unique(labels):
        contrast = (labels == c).astype(float)
        contrast -= contrast.mean()
        projection = vt @ contrast
        for cutoff in (1e-6, 1e-4, 1e-2):
            weak = all_s <= (s[0] * cutoff if s[0] else 0)
            rows.append(
                dict(
                    cluster=int(c),
                    relative_cutoff=cutoff,
                    weak_population_fraction=float(
                        np.sum(projection[weak] ** 2) / np.dot(contrast, contrast)
                    ),
                    numerical_null_population_fraction=float(
                        np.sum(projection[numerical_null] ** 2)
                        / np.dot(contrast, contrast)
                    ),
                    strong_rank=int((~weak).sum()),
                    n_candidate=len(labels),
                )
            )
    return rows


def aligned(xyz):
    x = xyz - xyz.mean(axis=1, keepdims=True)
    u, _, vt = np.linalg.svd(np.einsum("fai,aj->fij", x, x[0]))
    negative = np.linalg.det(u @ vt) < 0
    u[negative, :, -1] *= -1
    return x @ (u @ vt)


def load_structure(sid, source, out):
    import MDAnalysis as mda

    config = load_config()
    feature = load_contact_coordinates(sid, 1, config)
    idx = source["indices"]
    heavy, acceptor = feature["heavy"][:, idx], feature["acceptor"][:, idx]
    z = config["protocol"]["bv_bc"] * heavy + config["protocol"]["bv_bh"] * acceptor

    # The original experiment centres each residue on these 512 source frames.
    def uptake(values):
        rate = np.exp(
            np.clip(-(values - np.median(values, axis=1, keepdims=True)), -20, 20)
        )
        return 1 - np.exp(-np.asarray(TIMES)[:, None, None] * rate[None, :, :])

    rebuilt = uptake(z).reshape(-1, len(idx))
    np.testing.assert_allclose(rebuilt, source["flat"], atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(
        abs(z.mean(axis=0)[:, None] - z.mean(axis=0)[None, :]),
        source["work"],
        atol=2e-5,
    )
    topology_path = HERE / f"outputs/stage1/{sid}/R1/topology.json"
    topologies = json.loads(topology_path.read_text())["topologies"]
    residues = np.array([t["residues"][0] for t in topologies])
    assert len(residues) * len(TIMES) == source["flat"].shape[0]
    row = next(r for r in load_systems() if r["system_id"] == sid)
    universe = mda.Universe(
        str(HERE / row["pdb_path"]), str(HERE / row["replica_paths"].split(";")[0])
    )
    ca = universe.select_atoms("protein and name CA")
    assert len(np.unique(ca.resids)) == len(ca)
    lookup = {int(r): i for i, r in enumerate(ca.resids)}
    mapping = np.array([lookup[int(r)] for r in residues])
    frames = feature["frame"][idx]
    xyz = np.stack([ca.positions.copy() for _ in universe.trajectory[frames]])
    signatures = np.stack(
        [np.quantile(pdist(frame), np.linspace(0, 1, 256)) for frame in xyz]
    )
    np.testing.assert_allclose(
        cdist(signatures, signatures, metric="cityblock") / 256,
        source["structural"],
        atol=2e-5,
    )
    xyz = aligned(xyz)
    gyr = np.einsum("fai,faj->fij", xyz, xyz) / xyz.shape[1]
    eig = np.linalg.eigvalsh(gyr)
    rg = np.sqrt(eig.sum(axis=1))
    anisotropy = 1.5 * (eig**2).sum(axis=1) / eig.sum(axis=1) ** 2 - 0.5
    contacts = np.stack([squareform(pdist(frame)) < 8 for frame in xyz])
    contacts &= (
        abs(np.arange(len(ca))[:, None] - np.arange(len(ca))[None, :])[None, :, :] > 3
    )
    residue_contacts = contacts.sum(axis=2)
    rmsf = np.sqrt(np.mean(np.sum((xyz - xyz.mean(axis=0)) ** 2, axis=2), axis=0))
    correspondence = pd.read_csv(HERE / f"data/raw/{sid}/{sid}_corresp.tsv", sep="\t")
    from MDAnalysis.lib.util import convert_aa_code

    assert len(correspondence) == len(ca)
    assert "".join(correspondence.UnP_seq) == "".join(
        convert_aa_code(r) for r in ca.resnames
    )
    crystal_contacts = pd.read_csv(
        HERE / f"data/raw/{sid}/{sid}_contacts.tsv", sep="\t"
    )
    assert len(crystal_contacts) == len(ca)
    table(
        out / "residue_mapping",
        [
            dict(
                feature_row=i,
                simulated_residue=int(r),
                deposited_residue=str(correspondence.PDB_num.iloc[mapping[i]]),
                crystal_partner_contact_count=int(
                    crystal_contacts.nb_chain.iloc[mapping[i]]
                ),
                resname=str(ca.resnames[mapping[i]]),
                ca_index=int(mapping[i]),
                rmsf_angstrom=rmsf[mapping[i]],
            )
            for i, r in enumerate(residues)
        ],
    )
    table(
        out / "frames",
        [
            dict(
                source_position=i,
                feature_column=int(idx[i]),
                trajectory_frame=int(frames[i]),
                cluster=int(source["labels"][i]),
                rg_angstrom=rg[i],
                shape_anisotropy=anisotropy[i],
                mean_logpf=z[:, i].mean(),
                ca_contacts=contacts[i].sum() / 2,
            )
            for i in range(len(idx))
        ],
    )
    return dict(
        xyz=xyz,
        rg=rg,
        anisotropy=anisotropy,
        contacts=contacts,
        residue_contacts=residue_contacts,
        rmsf=rmsf,
        residues=residues,
        mapping=mapping,
        heavy=heavy,
        acceptor=acceptor,
        z=z,
        row=row,
        ca_resids=ca.resids,
        ca_resnames=ca.resnames,
        crystal_contacts=crystal_contacts.nb_chain.to_numpy(),
        topology_sha256=digest(topology_path),
    )


def analyse_system(spec, input_root, output_root):
    threadpool_limits(1)
    sid = spec["system_id"]
    srcdir = Path(input_root) / "systems" / sid
    out = Path(output_root) / "systems" / sid
    out.mkdir(parents=True, exist_ok=True)
    assert digest(srcdir / "source.npz") == spec["source_sha256"]
    assert digest(srcdir / "clustering.parquet") == spec["clustering_sha256"]
    for name, expected in spec["inputs"].items():
        if "/R1/" in name or name.endswith(("_R1.xtc", ".pdb")):
            assert digest(HERE / name) == expected, name
    with np.load(srcdir / "source.npz") as data:
        source = {k: data[k] for k in data}
    x, labels = source["flat"].astype(float), source["labels"]
    np.testing.assert_allclose(source["target"], x.mean(axis=1), atol=2e-7)
    structure = load_structure(sid, source, out)
    k = int(labels.max()) + 1
    truth = np.bincount(labels) / len(labels)
    (
        rows,
        residues,
        transitions,
        graphs,
        svds,
        matches,
        match_residues,
        basins,
        coverage,
        residue_coverage,
    ) = ([] for _ in range(10))
    saved_hashes = {}
    max_weight_correction = 0.0
    for case in spec["cases"]:
        keep = np.asarray(case["keep"])
        labs = labels[keep]
        directory = srcdir / case["case"]
        fits = pd.read_parquet(directory / "fits.parquet")
        for filename in ("weights.npz", "fits.parquet", "populations.parquet"):
            saved_hashes[str(directory / filename)] = digest(directory / filename)
        with np.load(directory / "weights.npz") as data:
            np.testing.assert_array_equal(
                data["candidate_indices"], source["indices"][keep]
            )
            np.testing.assert_array_equal(data["labels"], labels)
            np.testing.assert_array_equal(data["indices"], source["indices"])
            allweights = data["weights"].astype(float)
        assert allweights.shape == (len(fits), len(labels))
        assert np.all(allweights >= 0) and np.isfinite(allweights).all()
        assert np.all(allweights[:, np.setdiff1d(np.arange(len(labels)), keep)] == 0)
        max_weight_correction = max(
            max_weight_correction, float(abs(allweights.sum(axis=1) - 1).max())
        )
        weights = allweights[:, keep] / allweights[:, keep].sum(axis=1, keepdims=True)
        balanced, ceiling = balanced_weights(labs, truth)
        meta = dict(
            system_id=sid,
            role=spec["role"],
            case=case["case"],
            retention=case["retention"],
        )
        for d in observability(x[:, keep], labs):
            svds.append(dict(**meta, **d))
        for c in range(k):
            selected = keep[labs == c]
            removed = np.setdiff1d(np.flatnonzero(labels == c), selected)
            for variable in ("rg", "anisotropy"):
                values = structure[variable]
                coverage.append(
                    dict(
                        **meta,
                        cluster=c,
                        variable=variable,
                        source_mean=values[labels == c].mean(),
                        retained_mean=values[selected].mean(),
                        removed_mean=values[removed].mean() if len(removed) else np.nan,
                        source_sd=values[labels == c].std(),
                        retained_sd=values[selected].std(),
                    )
                )
            for r, resid in enumerate(structure["residues"]):
                ci = structure["mapping"][r]
                residue_coverage.append(
                    dict(
                        **meta,
                        cluster=c,
                        residue=int(resid),
                        heavy_retained_shift=structure["heavy"][r, selected].mean()
                        - structure["heavy"][r, labels == c].mean(),
                        acceptor_retained_shift=structure["acceptor"][
                            r, selected
                        ].mean()
                        - structure["acceptor"][r, labels == c].mean(),
                        ca_contact_retained_shift=structure["residue_contacts"][
                            selected, ci
                        ].mean()
                        - structure["residue_contacts"][labels == c, ci].mean(),
                    )
                )
        records = fits.to_dict("records") + [
            dict(
                arm="balanced_truth",
                family="deterministic",
                converged=True,
                quantile=np.nan,
            )
        ]
        weight_rows = list(weights) + [balanced]
        calculated = {}
        for index, (fit, w) in enumerate(zip(records, weight_rows)):
            arm = fit["arm"]
            residual, terms = residual_decomposition(x, labels, keep, w)
            p, ce = basin_stats(w, labs)
            mse = np.mean(residual**2)
            if arm != "balanced_truth":
                np.testing.assert_allclose(mse, fit["mse"], rtol=2e-3, atol=2e-9)
                np.testing.assert_allclose(
                    1 / np.dot(w, w) / len(keep), fit["ess_fraction"], atol=2e-6
                )
                np.testing.assert_allclose(
                    0.5 * np.abs(p - truth).sum(), fit["population_tv"], atol=2e-6
                )
            record = dict(
                **meta,
                arm=arm,
                family=fit["family"],
                converged=bool(fit["converged"]),
                quantile=fit["quantile"],
                mse=mse,
                scaled_mse=mse / (np.var(source["target"]) + 1e-8),
                population_tv=0.5 * np.abs(p - truth).sum(),
                ess_fraction=1 / np.dot(w, w) / len(keep),
                correct_population_ess_ceiling=ceiling / len(keep),
                above_population_ceiling=1 / np.dot(w, w) > ceiling + 1e-7,
                rg_mean=w @ structure["rg"][keep],
                shape_mean=w @ structure["anisotropy"][keep],
                **{name: value.mean() for name, value in terms.items()},
            )
            record["cancellation_fraction"] = 1 - mse / max(
                sum(record[t] for t in ("population_sq", "coverage_sq", "reweight_sq")),
                1e-30,
            )
            calculated[arm] = (record, residual, w)
            rows.append(record)
            for c in range(k):
                selected = keep[labs == c]
                basins.append(
                    dict(
                        **meta,
                        arm=arm,
                        cluster=c,
                        target_population=truth[c],
                        candidate_population=(labs == c).mean(),
                        population=p[c],
                        conditional_ess=ce[c],
                        conditional_ess_fraction=ce[c] / len(selected),
                        rg_mean=w[labs == c] @ structure["rg"][selected] / p[c],
                    )
                )
            shaped = {name: value.reshape(3, -1) for name, value in terms.items()}
            for t, time in enumerate(TIMES):
                for r, resid in enumerate(structure["residues"]):
                    residues.append(
                        dict(
                            **meta,
                            arm=arm,
                            converged=bool(fit["converged"]),
                            time=time,
                            residue=int(resid),
                            residual=residual.reshape(3, -1)[t, r],
                            **{name: value[t, r] for name, value in shaped.items()},
                        )
                    )
            if fit["family"] == "omc":
                distance = source["work"][np.ix_(keep, keep)]
                kernel = np.exp(-np.minimum(0.5 * (distance / fit["sigma"]) ** 2, 80))
                for name, ww in ((arm, w), ("balanced_truth", balanced)):
                    graphs.append(
                        dict(
                            **meta,
                            arm=name,
                            quantile=fit["quantile"],
                            sigma=fit["sigma"],
                            **graph_stats(kernel, ww, labs),
                        )
                    )
        omc = fits[fits.family == "omc"].sort_values("quantile")
        for (_, one), (_, two) in zip(
            omc.iloc[:-1].iterrows(), omc.iloc[1:].iterrows()
        ):
            r1, _, w1 = calculated[one.arm]
            r2, _, w2 = calculated[two.arm]
            transitions.append(
                dict(
                    **meta,
                    q_from=one["quantile"],
                    q_to=two["quantile"],
                    both_converged=bool(one.converged and two.converged),
                    delta_ess_fraction=r2["ess_fraction"] - r1["ess_fraction"],
                    **inverse_ess_change(w1, w2, labs),
                )
            )
        maxent = fits[(fits.family == "maxent") & fits.converged]
        for _, fit in omc[omc.converged].iterrows():
            if maxent.empty:
                continue
            r1, residual1, w1 = calculated[fit.arm]
            nearest = (
                maxent.assign(gap=abs(maxent.ess_fraction - r1["ess_fraction"]))
                .sort_values(["gap", "strength"])
                .iloc[0]
            )
            r2, residual2, w2 = calculated[nearest.arm]
            distance = source["work"][np.ix_(keep, keep)]
            kernel = np.exp(-np.minimum(0.5 * (distance / fit["sigma"]) ** 2, 80))
            energy1 = graph_stats(kernel, w1, labs)["graph_energy"]
            energy2 = graph_stats(kernel, w2, labs)["graph_energy"]
            data_delta = r1["scaled_mse"] - r2["scaled_mse"]
            for tolerance in (0.02, 0.01, 0.005):
                matched = bool(nearest.gap <= tolerance)
                matches.append(
                    dict(
                        **meta,
                        arm=fit.arm,
                        maxent_arm=nearest.arm,
                        tolerance=tolerance,
                        matched=matched,
                        ess_gap=nearest.gap,
                        delta_population_tv=r1["population_tv"] - r2["population_tv"],
                        delta_mse=r1["mse"] - r2["mse"],
                        delta_rg=r1["rg_mean"] - r2["rg_mean"],
                        scaled_mse_difference=data_delta,
                        omc_energy_difference=energy1 - energy2,
                        omc_objective_difference=data_delta + 0.1 * (energy1 - energy2),
                        weight_l1=float(abs(w1 - w2).sum()),
                    )
                )
            if nearest.gap <= 0.02:
                delta = (residual1**2 - residual2**2).reshape(3, -1)
                for t, time in enumerate(TIMES):
                    for r, resid in enumerate(structure["residues"]):
                        match_residues.append(
                            dict(
                                **meta,
                                arm=fit.arm,
                                maxent_arm=nearest.arm,
                                ess_gap=nearest.gap,
                                residue=int(resid),
                                time=time,
                                delta_squared_error=delta[t, r],
                            )
                        )
        if case["retention"] < 1:
            placement_plot(
                source,
                structure,
                keep,
                calculated,
                fits,
                out / f"placement_{case['retention']:g}.png",
            )
    fit_table = table(out / "fit_diagnostics", rows)
    for name, data in (
        ("residual_terms", residues),
        ("ess_transitions", transitions),
        ("graph_diagnostics", graphs),
        ("observability", svds),
        ("matches", matches),
        ("matched_residues", match_residues),
        ("basin_weights", basins),
        ("coverage", coverage),
        ("residue_coverage", residue_coverage),
    ):
        table(out / name, data)
    structural_report(source, structure, out)
    fit_plots(fit_table, pd.DataFrame(graphs), out)
    for path, expected in saved_hashes.items():
        assert digest(path) == expected
    summary = dict(
        system_id=sid,
        role=spec["role"],
        cath_class=spec["cath_class"],
        rmsf_tercile=spec["rmsf_tercile"],
        length=len(structure["ca_resids"]),
        mean_rmsf=float(structure["rmsf"].mean()),
        source_population=truth.tolist(),
        topology_sha256=structure["topology_sha256"],
        max_weight_normalisation_correction=max_weight_correction,
        saved_hashes=saved_hashes,
    )
    write_json(out / "complete.json", summary)
    return summary


def placement_plot(source, structure, keep, calculated, fits, path):
    """Display existing weight placement; never choose a fit by population truth."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    candidates = [
        a
        for a in ("omc_q0.02", "omc_q0.64", "unregularised")
        if calculated[a][0]["converged"]
    ]
    widest = calculated["omc_q0.64"][0]
    me = fits[(fits.family == "maxent") & fits.converged]
    if widest["converged"] and len(me):
        nearest = me.iloc[np.argmin(abs(me.ess_fraction - widest["ess_fraction"]))]
        if abs(nearest.ess_fraction - widest["ess_fraction"]) <= 0.02:
            candidates.append(nearest.arm)
    for ax, values, label in zip(
        axes,
        (structure["rg"], structure["z"].mean(axis=0)),
        ("Cα Rg (Å)", "Mean log-PF"),
    ):
        bins = np.linspace(values.min(), values.max(), 25)
        ax.hist(
            values,
            bins=bins,
            weights=np.ones(len(values)) / len(values),
            histtype="step",
            color="black",
            label="source target",
        )
        ax.hist(
            values[keep],
            bins=bins,
            weights=np.ones(len(keep)) / len(keep),
            histtype="step",
            ls=":",
            color="grey",
            label="candidate uniform",
        )
        for arm in candidates:
            record, _, w = calculated[arm]
            ax.hist(values[keep], bins=bins, weights=w, histtype="step", label=arm)
        ax.set(xlabel=label, ylabel="Probability per bin")
    axes[0].legend(fontsize=8)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def structural_report(source, s, out):
    labels, x = source["labels"], source["flat"].astype(float)
    means = np.stack([x[:, labels == c].mean(axis=1) for c in np.unique(labels)])
    contrast = means[-1] - means[0]
    uptake_contrast = contrast.reshape(3, -1)
    rows = []
    for i, r in enumerate(s["residues"]):
        ci = s["mapping"][i]
        rows.append(
            dict(
                residue=int(r),
                resname=str(s["ca_resnames"][ci]),
                rmsf=s["rmsf"][ci],
                heavy_contrast=s["heavy"][i, labels == 1].mean()
                - s["heavy"][i, labels == 0].mean(),
                acceptor_contrast=s["acceptor"][i, labels == 1].mean()
                - s["acceptor"][i, labels == 0].mean(),
                uptake_contrast_rms=np.sqrt(np.mean(uptake_contrast[:, i] ** 2)),
                crystal_partner_contact_count=int(s["crystal_contacts"][ci]),
                saturation_fraction=np.mean(
                    (x.reshape(3, -1, len(labels))[:, i, :] < 0.01)
                    | (x.reshape(3, -1, len(labels))[:, i, :] > 0.99)
                ),
            )
        )
    table(out / "residue_structure", rows)
    upper = np.triu_indices(len(labels), 1)
    work = source["work"][upper]
    structural = source["structural"][upper]
    same = (labels[:, None] == labels[None, :])[upper]
    separation = {}
    for name, values in (("work", work), ("structural", structural)):
        separation[name + "_between_within_ratio"] = float(
            values[~same].mean() / values[same].mean()
        )
    separation["work_structural_spearman"] = float(
        spearmanr(work, structural).statistic
    )
    separation["rg_mean_logpf_spearman"] = float(
        spearmanr(s["rg"], s["z"].mean(axis=0)).statistic
    )
    separation["basin_rg_means"] = [
        float(s["rg"][labels == c].mean()) for c in np.unique(labels)
    ]
    separation["basin_rg_sd"] = [
        float(s["rg"][labels == c].std()) for c in np.unique(labels)
    ]
    separation["basin_shape_means"] = [
        float(s["anisotropy"][labels == c].mean()) for c in np.unique(labels)
    ]
    separation["basin_uptake_contrast_rms"] = float(np.sqrt(np.mean(contrast**2)))
    separation["crystal_partner_contact_residue_fraction"] = float(
        (s["crystal_contacts"] > 0).mean()
    )
    rmsf_sq = s["rmsf"] ** 2
    separation["rmsf_variance_at_crystal_partner_contacts"] = float(
        rmsf_sq[s["crystal_contacts"] > 0].sum() / rmsf_sq.sum()
    )
    shaped = x.reshape(3, -1, len(labels))
    separation["saturated_fraction_by_time"] = [
        float(np.mean((v < 0.01) | (v > 0.99))) for v in shaped
    ]
    write_json(out / "structural_summary.json", separation)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for c in np.unique(labels):
        mask = labels == c
        axs[0, 0].scatter(
            s["rg"][mask],
            s["z"].mean(axis=0)[mask],
            s=8,
            alpha=0.5,
            label=f"basin {c} ({mask.sum()})",
        )
    axs[0, 0].set(xlabel="Cα Rg (Å)", ylabel="Mean log-PF (graph coordinate)")
    axs[0, 0].legend()
    axs[0, 1].plot(s["ca_resids"], s["rmsf"])
    axs[0, 1].set(xlabel="Simulated residue", ylabel="Aligned Cα RMSF (Å)")
    for i, time in enumerate(TIMES):
        axs[0, 2].plot(s["residues"], uptake_contrast[i], label=f"t={time:g}")
    axs[0, 2].legend()
    axs[0, 2].set(xlabel="Simulated residue", ylabel="Basin 1 − 0 synthetic uptake")
    cm = [s["contacts"][labels == c].mean(axis=0) for c in np.unique(labels)]
    im = axs[1, 0].imshow(
        cm[1] - cm[0], cmap="coolwarm", vmin=-1, vmax=1, origin="lower"
    )
    axs[1, 0].set(
        xlabel="Cα index",
        ylabel="Cα index",
        title="Basin contact probability difference",
    )
    fig.colorbar(im, ax=axs[1, 0])
    axs[1, 1].scatter(structural[::50], work[::50], s=2, alpha=0.25)
    axs[1, 1].set(
        xlabel="Structural W1 distance",
        ylabel="Scalar work distance",
        title=f"Pair-distance Spearman {separation['work_structural_spearman']:.2f}",
    )
    axs[1, 2].plot(
        s["residues"], [r["heavy_contrast"] for r in rows], label="heavy contacts"
    )
    axs[1, 2].plot(
        s["residues"], [r["acceptor_contrast"] for r in rows], label="acceptor proxy"
    )
    axs[1, 2].legend()
    axs[1, 2].set(xlabel="Simulated residue", ylabel="Basin 1 − 0 feature count")
    fig.savefig(out / "structure.png", dpi=140)
    plt.close(fig)
    fig = plt.figure(figsize=(10, 5), layout="constrained")
    contact_maps = {}
    for c in np.unique(labels):
        members = np.flatnonzero(labels == c)
        medoid = members[
            np.argmin(source["structural"][np.ix_(members, members)].mean(axis=1))
        ]
        coords = s["xyz"][medoid]
        ax = fig.add_subplot(1, len(np.unique(labels)), c + 1, projection="3d")
        ax.plot(*coords.T, color="grey", lw=0.7)
        signal = np.full(len(coords), np.nan)
        signal[s["mapping"]] = np.sqrt(np.mean(uptake_contrast**2, axis=0))
        points = ax.scatter(
            *coords[s["mapping"]].T, c=signal[s["mapping"]], cmap="viridis", s=12
        )
        ax.set_title(f"Basin {c}, source frame {medoid}")
        fig.colorbar(points, ax=ax, shrink=0.5, label="Uptake contrast RMS")
        contact_maps[f"basin_{c}"] = cm[c]
    fig.savefig(out / "representatives.png", dpi=140)
    plt.close(fig)
    np.savez_compressed(
        out / "contact_maps.npz", **contact_maps, ca_resids=s["ca_resids"]
    )


def fit_plots(fits, graphs, out):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    for retention, g in fits[(fits.family == "omc") & (fits.retention < 1)].groupby(
        "retention"
    ):
        g = g.sort_values("quantile")
        for ax, metric in zip(
            axes.flat,
            (
                "ess_fraction",
                "population_tv",
                "mse",
                "population_sq",
                "reweight_sq",
                "cancellation_fraction",
            ),
        ):
            ax.plot(g["quantile"], g[metric], "o-", label=f"retain {retention:g}")
            bad = g[~g.converged]
            ax.scatter(bad["quantile"], bad[metric], marker="x", s=90, color="red")
            ax.set(xscale="log", xlabel="Bandwidth quantile", ylabel=metric)
    axes[0, 0].legend()
    fig.savefig(out / "mechanisms.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for retention, g in graphs[
        (graphs.arm != "balanced_truth") & (graphs.retention < 1)
    ].groupby("retention"):
        g = g.sort_values("quantile")
        for ax, metric in zip(
            axes, ("mean_offdiagonal_kernel", "between_kernel_fraction", "graph_energy")
        ):
            ax.plot(g["quantile"], g[metric], "o-", label=f"retain {retention:g}")
            ax.set(xscale="log", xlabel="Bandwidth quantile", ylabel=metric)
    axes[0].legend()
    fig.savefig(out / "graph.png", dpi=140)
    plt.close(fig)


def summarise(output, manifest):
    output = Path(output)
    names = (
        "fit_diagnostics",
        "ess_transitions",
        "graph_diagnostics",
        "observability",
        "matches",
        "coverage",
        "basin_weights",
    )
    combined = {
        name: table(
            output / name,
            pd.concat(
                [
                    pd.read_parquet(
                        output / "systems" / s["system_id"] / f"{name}.parquet"
                    )
                    for s in manifest["systems"]
                ],
                ignore_index=True,
            ),
        )
        for name in names
    }
    f = combined["fit_diagnostics"]
    trans = combined["ess_transitions"]
    matches = combined["matches"]
    case_rows = []
    for (sid, case), g in f[f.retention < 1].groupby(["system_id", "case"]):
        omc = g[g.family == "omc"].sort_values("quantile")
        valid = omc[omc.converged]
        base = g[(g.family == "unregularised") & g.converged]
        t = trans[
            (trans.system_id == sid) & (trans.case == case) & trans.both_converged
        ]
        matched = matches[
            (matches.system_id == sid)
            & (matches.case == case)
            & matches.matched
            & (matches.tolerance == 0.02)
        ]
        balanced = g[g.arm == "balanced_truth"].iloc[0]
        case_rows.append(
            dict(
                system_id=sid,
                case=case,
                role=g.iloc[0].role,
                retention=g.iloc[0].retention,
                complete_curve=bool(omc.converged.all()),
                n_converged=len(valid),
                ess_span=valid.ess_fraction.max() - valid.ess_fraction.min()
                if omc.converged.all()
                else np.nan,
                endpoint_delta=omc.ess_fraction.iloc[-1] - omc.ess_fraction.iloc[0]
                if omc.converged.all()
                else np.nan,
                reversals=int((t.delta_ess_fraction < -1e-8).sum()),
                min_step=t.delta_ess_fraction.min(),
                tv_vs_unreg=valid.population_tv.median() - base.population_tv.iloc[0]
                if len(base) and omc.converged.all()
                else np.nan,
                tv_vs_maxent=matched.delta_population_tv.median(),
                mse_vs_maxent=matched.delta_mse.median(),
                n_matches=len(matched),
                above_ceiling=int(valid.above_population_ceiling.sum()),
                ess_ceiling=balanced.correct_population_ess_ceiling,
                balanced_mse=balanced.mse,
                unreg_mse=base.mse.iloc[0] if len(base) else np.nan,
                balanced_vs_unreg=balanced.mse / base.mse.iloc[0]
                if len(base) and base.mse.iloc[0] > 1e-20
                else np.nan,
                unreg_tv=base.population_tv.iloc[0] if len(base) else np.nan,
                cancellation=valid.cancellation_fraction.median(),
            )
        )
    cases = pd.DataFrame(case_rows)
    cases["low_response"] = False
    for retention, g in cases[(cases.role == "cohort") & cases.complete_curve].groupby(
        "retention"
    ):
        threshold = g.ess_span.quantile(0.25)
        cases.loc[g.index, "low_response"] = g.ess_span <= threshold
    table(output / "case_overlap", cases)
    summaries = []
    for sid, g in cases.groupby("system_id"):
        structural = json.loads(
            (output / "systems" / sid / "structural_summary.json").read_text()
        )
        receipt = json.loads((output / "systems" / sid / "complete.json").read_text())
        summaries.append(
            dict(
                system_id=sid,
                role=g.iloc[0].role,
                length=receipt["length"],
                cath_class=receipt["cath_class"],
                rmsf_tercile=receipt["rmsf_tercile"],
                mean_rmsf=receipt["mean_rmsf"],
                reversals=int(g.reversals.sum()),
                low_response_cases=int(g.low_response.sum()),
                ess_span=g.ess_span.median(),
                tv_vs_unreg=g.tv_vs_unreg.median(),
                tv_vs_maxent=g.tv_vs_maxent.median(),
                mse_vs_maxent=g.mse_vs_maxent.median(),
                above_ceiling=int(g.above_ceiling.sum()),
                cancellation=g.cancellation.median(),
                balanced_vs_unreg=g.balanced_vs_unreg.median(),
                work_structural_spearman=structural["work_structural_spearman"],
                work_between_within_ratio=structural["work_between_within_ratio"],
                basin_rg_difference=structural["basin_rg_means"][1]
                - structural["basin_rg_means"][0],
            )
        )
    summary = table(output / "system_overlap", summaries)
    cohort = summary[summary.role == "cohort"]
    associations = []
    for feature in (
        "length",
        "mean_rmsf",
        "work_structural_spearman",
        "work_between_within_ratio",
    ):
        for outcome in ("ess_span", "tv_vs_unreg", "tv_vs_maxent", "reversals"):
            pair = cohort[[feature, outcome]].dropna()
            associations.append(
                dict(
                    feature=feature,
                    outcome=outcome,
                    n_systems=len(pair),
                    spearman=float(spearmanr(pair[feature], pair[outcome]).statistic)
                    if len(pair) > 2
                    else np.nan,
                    interpretation="descriptive association; not a causal or class-level conclusion",
                )
            )
    table(output / "associations", associations)
    flags = pd.DataFrame(
        {
            "ESS reversal": cohort.reversals > 0,
            "Low ESS response": cohort.low_response_cases > 0,
            "Worse TV vs unreg": cohort.tv_vs_unreg > 0,
            "Worse TV vs MaxEnt": cohort.tv_vs_maxent > 0,
            "Higher MSE vs MaxEnt": cohort.mse_vs_maxent > 0,
            "ESS exceeds truth ceiling": cohort.above_ceiling > 0,
        }
    )
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    ax.imshow(flags.to_numpy(int), cmap="Oranges", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(flags.columns)), flags.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(cohort)), cohort.system_id)
    for i, row in enumerate(flags.to_numpy()):
        for j, v in enumerate(row):
            ax.text(j, i, "yes" if v else "—", ha="center", va="center")
    fig.savefig(output / "overlap.png", dpi=150)
    plt.close(fig)
    from .omc_diagnostic_report import build_findings

    build_findings(output)
    report_html(output, summary, cases, combined)


def report_html(output, summary, cases, combined):
    annotations = json.loads(
        Path(__file__).with_name("omc_diagnostic_annotations.txt").read_text()
    )

    def tab(df):
        return df.to_html(index=False, float_format=lambda v: f"{v:.5g}", border=0)

    def image(name):
        return f'<a href="{name}"><img src="{name}" loading="lazy"></a>'

    parts = [
        '<!doctype html><html><head><meta charset="utf-8"><title>OMC mechanism diagnostics</title><style>body{font:16px system-ui;max-width:1400px;margin:35px auto;padding:0 20px;color:#17212b}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}img{max-width:100%}a{color:#0759aa}.note{background:#eef5fa;padding:16px}h2{margin-top:45px}</style></head><body><h1>OMC: population recovery and distribution control</h1>',
        '<p class="note">Analysis of saved fits only. All-residue MSE means all eligible feature residues × three synthetic times; it excludes residues omitted by featurisation. Targets are the uniform 512-frame source means, not experimental exchange measurements. No new fits or ISO runs.</p>',
        '<p><a href="findings.md">Mechanistic findings</a> · <a href="system_overlap.csv">System table</a> · <a href="case_overlap.csv">Severity table</a> · <a href="mechanism_verdicts.csv">Mechanism verdicts</a> · <a href="matching_sensitivity.csv">Matching sensitivity</a> · <a href="associations.csv">Descriptive associations</a> · <a href="audit.json">Verification</a></p>',
        image("overlap.png"),
        "<p>Flags are descriptive and can overlap. Low response is the bottom quartile of complete cohort curves within each retention. A local reversal is an adjacent ESS decline with both fits converged. Positive comparison deltas mean OMC is worse. Reference is excluded from cohort counts. A higher MSE does not by itself imply worse true populations.</p>",
        tab(summary),
        (output / "findings_fragment.html").read_text(),
        "<h2>Four questions and the evidence available</h2><ol><li>ESS changes: exact inverse-ESS population/within-basin decomposition at every adjacent bandwidth.</li><li>Limited response: target-population ESS ceiling, retained structural diversity, scalar graph separation and total kernel coupling.</li><li>Population recovery: exact observation-level population, coverage and reweighting terms, including cross-term cancellation, and deterministic true-population weights.</li><li>MaxEnt comparison: actual saved-fit matching at 0.02, 0.01 and 0.005 ESS-fraction tolerance, with per-residue error differences.</li></ol>",
        "<h2>Severity and convergence</h2>",
        tab(cases),
    ]
    for name in combined:
        parts.append(
            f'<p><a href="{name}.csv">{name}: CSV</a> · <a href="{name}.parquet">Parquet</a></p>'
        )
    for _, row in summary.iterrows():
        sid = row.system_id
        prefix = f"systems/{sid}/"
        a = annotations.get(sid, {})
        parts.extend(
            [
                f'<h2 id="{sid}">{sid}: {html.escape(a.get("name", "Identity unresolved"))}</h2>',
                "<p>"
                + html.escape(a.get("context", ""))
                + f' <a href="{a.get("url", "https://www.rcsb.org/structure/" + sid[:4].upper())}">Primary structural annotation</a></p>',
                "<p>Assembly context is a hypothesis for interpreting isolated-chain dynamics, not a causal test of why a fit fails.</p>",
            ]
        )
        for filename in (
            "structure.png",
            "representatives.png",
            "mechanisms.png",
            "graph.png",
            "residue_errors.png",
        ):
            parts.append(image(prefix + filename))
        for retention in (0.125, 0.25, 0.5):
            parts.append(image(prefix + f"placement_{retention:g}.png"))
        for filename in (
            "residue_mapping",
            "residue_structure",
            "residual_terms",
            "matched_residues",
            "basin_weights",
            "coverage",
            "residue_coverage",
            "frames",
            "observability",
        ):
            parts.append(f'<a href="{prefix}{filename}.csv">{filename}</a> · ')
    parts.append(
        "<h2>Interpretation limits</h2><p>Weak-direction projections describe local linear observability on the simplex tangent; positivity may restrict the allowed movement. They are not alternate fitted populations. Frame weights and ESS describe this retained ensemble, not kinetic rates or independent thermodynamic samples. Acceptor counts are a protection-factor proxy, not measured hydrogen-bond occupancy. Correlations across twelve systems do not establish a fold-class cause. Original failed convergence flags remain excluded from paired mechanistic inference.</p></body></html>"
    )
    (output / "index.html").write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--systems", nargs="+")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Rebuild reports from completed diagnostic tables; no trajectory or fit work",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 10:
        parser.error("workers must be between 1 and 10")
    if (
        args.output.resolve() == args.input.resolve()
        or args.input.resolve() in args.output.resolve().parents
    ):
        parser.error("diagnostics must use a separate directory outside the fit inputs")
    manifest = json.loads((args.input / "manifest.json").read_text())
    if args.systems:
        manifest["systems"] = [
            s for s in manifest["systems"] if s["system_id"] in args.systems
        ]
    if not manifest["systems"]:
        parser.error("no systems selected")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        summarise(args.output, manifest)
        verify_artifacts(args.input, args.output, manifest)
        print(args.output / "index.html", flush=True)
        return
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(analyse_system, s, args.input, args.output): s["system_id"]
            for s in manifest["systems"]
        }
        for future in as_completed(futures):
            future.result()
            print(f"Completed {futures[future]}", flush=True)
    summarise(args.output, manifest)
    verify_artifacts(args.input, args.output, manifest)
    print(args.output / "index.html", flush=True)


def verify_artifacts(input_root, output, manifest):
    """Recheck immutable inputs and original summary/matching parity, plus links."""
    from html.parser import HTMLParser

    original = pd.read_parquet(input_root / "system_summary.parquet").set_index(
        "system_id"
    )
    actual = pd.read_parquet(output / "system_overlap.parquet").set_index("system_id")
    for new, old in (
        ("ess_span", "ess_span"),
        ("tv_vs_unreg", "median_population_tv_change"),
        ("tv_vs_maxent", "population_tv_difference"),
        ("mse_vs_maxent", "mse_difference"),
    ):
        np.testing.assert_allclose(
            actual[new],
            original.loc[actual.index, old],
            atol=2e-6,
            rtol=2e-4,
            equal_nan=True,
        )
    for spec in manifest["systems"]:
        sid = spec["system_id"]
        assert (
            digest(input_root / "systems" / sid / "source.npz") == spec["source_sha256"]
        )
        receipt = json.loads((output / "systems" / sid / "complete.json").read_text())
        for path, expected in receipt["saved_hashes"].items():
            assert digest(path) == expected
    original_matches = pd.read_parquet(input_root / "ess_matches.parquet")
    new_matches = pd.read_parquet(output / "matches.parquet")
    checked = new_matches[new_matches.tolerance == 0.02].merge(
        original_matches,
        left_on=["system_id", "case", "arm"],
        right_on=["system_id", "case", "omc_arm"],
        suffixes=("_new", "_old"),
        validate="one_to_one",
    )
    assert len(checked) == len(new_matches[new_matches.tolerance == 0.02])
    assert (checked.matched_new == checked.matched_old).all()
    assert (checked.maxent_arm == checked.nearest_maxent_arm).all()

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []

        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if (
                    key in ("href", "src")
                    and value
                    and not value.startswith(("http", "#"))
                ):
                    self.links.append(value.split("#")[0])

    parser = Links()
    parser.feed((output / "index.html").read_text())
    for link in parser.links:
        assert (output / link).is_file(), link
    write_json(
        output / "audit.json",
        dict(
            systems=len(manifest["systems"]),
            checked_local_links=len(parser.links),
            png_count=len(list(output.rglob("*.png"))),
            matching_pairs_verified=len(checked),
            source_manifest_sha256=digest(input_root / "manifest.json"),
            code_sha256=digest(__file__),
            narrative_code_sha256=digest(
                Path(__file__).with_name("omc_diagnostic_report.py")
            ),
            checked=[
                "source and R1 input hashes",
                "feature/frame/residue mapping",
                "saved fit files unchanged",
                "saved MSE, population TV and ESS agreement",
                "residual and squared-error decompositions",
                "inverse ESS decomposition",
                "graph energy decomposition",
                "original system summary parity",
                "original ESS matching parity",
                "all local HTML links exist",
            ],
            new_fits=0,
        ),
    )


if __name__ == "__main__":
    main()
