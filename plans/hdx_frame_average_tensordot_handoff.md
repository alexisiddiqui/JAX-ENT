# HDX Frame Averaging: Reduction vs `tensordot`

Status: **benchmarked, not applied.** The tracked working tree is restored to
implementation 0. The CUDA dependency group was removed after the GPU runs and
the project is back on vanilla JAX.

Date: 2026-07-27. Commit: `71e71b9`.

## Summary

The dominant CPU `reduce-window` kernels come from frame averaging and its
reverse-mode derivative:

```python
jnp.sum(x * frame_weights.reshape(1, -1), axis=-1)
```

Expressing the same contraction as `jnp.tensordot` is a large CPU win:

- **22.56x** at the standard `(144 residues, 500 frames, 5 timepoints)` setup.
- **1.91x** at `corner_high` `(600 residues, 5000 frames, 10 timepoints)`.

The rewrite is neutral on the RTX 3090: both configurations differ by less than
0.1%. CUDA already handles the original multiply/reduction efficiently.

The candidate changes floating-point reduction order. Outputs and direct
frame-weight gradients agree to small float32 absolute error, but the 1000-step
optimizer parameter fingerprints are not identical. This is expected for a
reassociated reduction but means the full optimizer correctness suite should be
run before adoption.

## Implementations

Implementation 0 is the current code in
`jaxent/src/utils/jax_fn.py::frame_average_features`:

```python
weights = frame_weights.reshape(1, -1)

def average_feature(x):
    x = jnp.asarray(x)
    if x.ndim <= 1:
        return x
    return jnp.sum(x * weights, axis=-1)
```

Implementation 1 was applied temporarily for measurement:

```python
def average_feature(x):
    x = jnp.asarray(x)
    if x.ndim <= 1:
        return x
    return jnp.tensordot(x, frame_weights, axes=((-1,), (0,)))
```

No candidate implementation remains in the tracked source.

## Benchmark protocol

- Runner: `profiling/profile_hdx_cpu.py --mode timing --path pure`
- 1000 fixed steps
- One cold sample followed by five warm samples
- Warm median and MAD reported below
- Zero warm compilations and zero host materialisations in every cell
- Every report valid and deterministic within its implementation
- Python 3.13.1, JAX 0.4.35, jaxlib 0.4.34
- CPU: Intel Core i7-12700KF, 12 cores / 20 logical CPUs
- GPU: NVIDIA GeForce RTX 3090, driver 580.173.02

Configurations:

| name | residues | frames | timepoints |
|---|---:|---:|---:|
| standard | 144 | 500 | 5 |
| corner_high | 600 | 5000 | 10 |

## Warm timing results

| backend | configuration | impl 0 median | impl 1 median | impl 1 MAD | speedup |
|---|---|---:|---:|---:|---:|
| CPU | standard | 0.581185 s | **0.025767 s** | 0.000406 s | **22.56x** |
| CPU | corner_high | 2.728938 s | **1.429717 s** | 0.134121 s | **1.91x** |
| GPU | standard | 0.086549 s | 0.086472 s | 0.000378 s | 1.001x |
| GPU | corner_high | 0.159581 s | 0.159515 s | 0.000253 s | 1.000x |

Raw warm samples:

| backend/configuration | implementation 0 | implementation 1 |
|---|---|---|
| CPU standard | 0.602164, 0.581185, 0.502192, 0.468255, 0.609795 | 0.026173, 0.025927, 0.025767, 0.025329, 0.024367 |
| CPU corner_high | 2.707163, 2.728938, 2.759395, 2.707284, 2.807846 | 1.822483, 1.429717, 1.295596, 1.271248, 1.517716 |
| GPU standard | 0.097568, 0.093416, 0.085968, 0.086407, 0.086549 | 0.086851, 0.085259, 0.085660, 0.086472, 0.086674 |
| GPU corner_high | 0.161261, 0.160016, 0.159581, 0.158640, 0.159156 | 0.159594, 0.159515, 0.159145, 0.159238, 0.159768 |

The CPU `corner_high` implementation-1 samples have more spread than the other
cells. Even its slowest sample, 1.822 s, is substantially faster than the
fastest implementation-0 sample, 2.707 s, so the conclusion does not depend on
the median.

## Why this helps CPU

The optimized CPU HLO for implementation 0 contains four dominant reductions
originating at `jaxent/src/utils/jax_fn.py:36`:

- Two forward reductions from `[residues, frames]` to `[residues]`.
- Two transpose/VJP reductions from `[residues, frames]` to `[frames]`.

At the standard profile these account for roughly 15-16 ms of a 23-25 ms
100-step compiled execution. The transpose/VJP reductions are the larger pair.
CPU XLA lowers the explicit broadcast multiply plus sum into parallel
`reduce-window` operations. `tensordot` exposes a matrix-vector contraction,
allowing the backend to use a more suitable dot implementation for both the
forward pass and its transpose.

The effect is shape dependent. At the standard shape, reduction scheduling
overhead dominates and the dot form removes almost all of it. At
`corner_high`, the contraction is larger and more compute/memory bound, so the
gain remains large but falls to 1.91x.

On CUDA, implementation 0 and implementation 1 have indistinguishable warm
times. The CUDA compiler already produces an efficient implementation for the
explicit reduction, so changing its expression does not improve the result.

## Numerical checks

Direct float32 contraction checks used random arrays matching both benchmark
shapes:

| shape | output max abs difference | frame-weight gradient max abs difference |
|---|---:|---:|
| `(144, 500)` | `5.96e-08` | `2.38e-06` |
| `(600, 5000)` | `3.54e-08` | `3.34e-06` |

Outputs pass `rtol=1e-5, atol=1e-6`. Gradients pass with `atol=5e-6`; the
largest case does not pass `atol=1e-6` because values near zero make relative
tolerance ineffective.

Final optimizer losses after 1000 steps:

| backend/configuration | impl 0 | impl 1 | absolute difference |
|---|---:|---:|---:|
| CPU standard | `2.9152419e-08` | `2.9116878e-08` | `3.55e-11` |
| CPU corner_high | `2.4619403e-08` | `2.4662540e-08` | `4.31e-11` |
| GPU standard | `2.9138819e-08` | `2.9140216e-08` | `1.40e-12` |
| GPU corner_high | `2.4611033e-08` | `2.4616874e-08` | `5.84e-12` |

Within each cell, cold and all warm repetitions have identical loss and
parameter fingerprints. Across implementations, fingerprints differ because
the reduction order changes the float32 optimization trajectory.

## GPU harness note

Importing `jaxent` selects the development runtime, which forces CPU. In
addition, `jaxent/src/utils/hdf.py` currently contains a module-level
`jax.config.update("jax_platform_name", "cpu")`.

For the GPU measurements only:

1. The two CPU-forcing lines in `hdf.py` were removed temporarily.
2. The harness was run with `JAXENT_MODE=performance`,
   `CUDA_ROOT=/usr/local/cuda`, and `LD_LIBRARY_PATH` unset.
3. Every retained GPU report explicitly records `"backend": "gpu"` and device
   `"cuda:0"`.
4. `hdf.py` was restored after the runs.

An earlier attempted pair selected CPU and was overwritten; it is not included
in the table.

## Recommendation

Adopt implementation 1, subject to the full optimizer correctness suite and a
full CPU scaling sweep. It is a major CPU improvement and has no measurable GPU
regression.

Before applying:

1. Add focused `frame_average_features` output and gradient parity tests with
   explicit float32 tolerances.
2. Run the optimize unit/integration tests, especially freeze, convergence, and
   terminal-state tests.
3. Run the full CPU scaling sweep to check intermediate residue/frame shapes.
4. Re-run at least the GPU standard and `corner_high` sentinels.
5. Inspect optimized HLO to confirm the production runner contains dot
   contractions instead of the four dominant `reduce-window` groups.

Do not require identical parameter fingerprints across implementations; use
loss/output/gradient tolerances appropriate for float32 reassociation.

## Reproduction

For either source implementation:

```sh
# CPU standard
uv run --no-sync python profiling/profile_hdx_cpu.py \
  --mode timing --path pure --steps 1000 --warm-repeats 5

# CPU corner_high
uv run --no-sync python profiling/profile_hdx_cpu.py \
  --mode timing --path pure --steps 1000 --warm-repeats 5 \
  --residues 600 --frames 5000 --timepoints 10
```

GPU requires the `cuda12` dependency group and the runtime/import workaround
described above:

```sh
uv sync --group cuda12
env -u LD_LIBRARY_PATH \
  CUDA_ROOT=/usr/local/cuda JAXENT_MODE=performance \
  uv run --no-sync python profiling/profile_hdx_cpu.py \
  --mode timing --path pure --steps 1000 --warm-repeats 5
```

## Artifacts

All retained JSON reports are in:

`profiling/_output/hdx_frame_average_dot_20260727/`

- `cpu_impl0_standard.json`
- `cpu_impl0_corner_high.json`
- `cpu_impl1_standard.json`
- `cpu_impl1_corner_high.json`
- `gpu_impl0_standard.json`
- `gpu_impl0_corner_high.json`
- `gpu_impl1_standard.json`
- `gpu_impl1_corner_high.json`
