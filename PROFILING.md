# JAX-ENT Runtime Profiling

This document describes the dynamic runtime profiling tools for JAX-ENT optimization pipelines.

## Overview

The repository contains two complementary profiling tools:

1. **`analyze_computation_graph.py`** - Static profiler that analyzes code structure
2. **`profile_runtime_jax.py`** - Dynamic profiler that measures actual execution performance

## Dynamic Runtime Profiler (`profile_runtime_jax.py`)

### Purpose

The dynamic runtime profiler uses JAX's built-in profiler to capture real-world performance metrics during optimization, including:

- **JIT compilation overhead** - Time spent compiling functions on first execution
- **Forward pass timing** - Execution time for forward model evaluation
- **Optimization step timing** - Complete step including loss, gradients, and updates
- **Loss progression** - How loss decreases during optimization
- **Memory usage patterns** - Device memory allocation and usage
- **Throughput metrics** - Steps per second, forward passes per second

### Prerequisites

Before running the profiler, ensure you have prepared the example data:

```bash
cd jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT

# 1. Generate features (requires trajectory data)
python featurise_ISO_TRI_BI.py

# 2. Generate data splits
python splitdata_ISO.py
```

### Basic Usage

```bash
# Run with default settings (50 steps)
python profile_runtime_jax.py

# Specify output directory
python profile_runtime_jax.py --output-dir ./my_profiling_results

# Profile more steps for better statistics
python profile_runtime_jax.py --n-steps 100

# Enable JAX profiler for TensorBoard visualization
python profile_runtime_jax.py --enable-jax-profiler
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output-dir` | `./profiling_results` | Directory to save profiling results |
| `--n-steps` | `50` | Number of optimization steps to profile |
| `--n-warmup` | `3` | Number of warmup steps (for JIT compilation) |
| `--n-forward-iters` | `100` | Number of forward pass iterations to profile |
| `--enable-jax-profiler` | `False` | Enable JAX profiler trace for TensorBoard |
| `--learning-rate` | `1e-1` | Optimizer learning rate |
| `--maxent-scaling` | `1.0` | MaxEnt regularization scaling |

### Example Workflows

#### Quick Performance Check

```bash
# Fast profiling run (10 steps)
python profile_runtime_jax.py --n-steps 10 --n-forward-iters 50
```

#### Detailed Performance Analysis

```bash
# Comprehensive profiling with TensorBoard trace
python profile_runtime_jax.py \
    --n-steps 100 \
    --n-forward-iters 200 \
    --enable-jax-profiler \
    --output-dir ./detailed_profile
```

#### CPU vs GPU Comparison

```bash
# CPU profiling
JAX_PLATFORM_NAME=cpu python profile_runtime_jax.py --output-dir ./profile_cpu

# GPU profiling (if available)
JAX_PLATFORM_NAME=gpu python profile_runtime_jax.py --output-dir ./profile_gpu
```

### Understanding the Output

The profiler generates several outputs:

#### 1. Console Output

Real-time profiling information including:
- Data loading progress
- Simulation setup details
- Forward pass timing statistics
- Optimization step timing with loss values
- Summary statistics

#### 2. Profiling Report (`profiling_report.txt`)

A text file summarizing:
- JIT compilation overhead
- Mean/std/min/max timings for all operations
- Loss progression
- Performance metrics (steps/sec, estimated time for longer runs)
- Device information

#### 3. JAX Profiler Trace (if enabled)

TensorBoard-compatible trace files that can be visualized:

```bash
# View profiling trace in TensorBoard
tensorboard --logdir=./profiling_results/profile_TIMESTAMP/jax_profiler_trace
```

Then open http://localhost:6006 in your browser.

### Interpreting Results

#### JIT Compilation Overhead

The first execution of JAX functions is typically 10-100x slower due to JIT compilation. The profiler separates:
- **Warmup steps**: Include JIT compilation overhead
- **Profiled steps**: Post-compilation steady-state performance

Example output:
```
Estimated JIT compilation overhead: 2.3451s
Average warmup step time: 1.1234s
Average profiled step time: 0.0523s  # <- This is the steady-state performance
```

#### Forward Pass Timing

Critical for understanding model evaluation performance:
```
Forward Pass Timing:
  Mean: 12.45ms ± 0.32ms
  Min:  11.98ms
  Max:  15.23ms
  Throughput: 80.3 forward passes/sec
```

Lower variance indicates consistent performance. High variance may indicate:
- Memory allocation/deallocation
- XLA optimization issues
- Background processes

#### Optimization Step Timing

Complete optimization step including forward pass, loss computation, gradient computation, and parameter updates:
```
Optimization Step (post-warmup):
  Mean time: 0.0523s
  Steps per second: 19.12
  Time for 10000 steps (estimated): 8.72 minutes
```

#### Loss Progression

Verifies optimization is working correctly:
```
Initial loss: 1.234567e+02
Final loss:   8.765432e+01
Loss reduction: 36.90%
```

### Performance Optimization Tips

Based on profiling results, consider:

1. **If JIT compilation dominates**:
   - Use longer optimization runs to amortize compilation cost
   - Consider caching compiled functions

2. **If forward pass is slow**:
   - Check for unnecessary Python loops (use vmap)
   - Verify sparse operations aren't converting to dense
   - Profile individual operations with JAX profiler

3. **If step-to-step variance is high**:
   - Check for memory allocation issues
   - Verify batch sizes are consistent
   - Look for conditional logic in JIT-compiled code

4. **If GPU is not faster than CPU**:
   - Check batch sizes (may be too small for GPU)
   - Verify data transfer overhead
   - Check for synchronization points

### Integration with `analyze_computation_graph.py`

Use both tools together for comprehensive performance analysis:

1. **Static analysis** (`analyze_computation_graph.py`):
   - Identifies potential bottlenecks in code structure
   - Finds loops that could be vectorized
   - Locates sparse-to-dense conversions

2. **Runtime profiling** (`profile_runtime_jax.py`):
   - Measures actual impact of identified bottlenecks
   - Validates optimization hypotheses
   - Quantifies performance improvements

Example workflow:
```bash
# 1. Static analysis to identify issues
python analyze_computation_graph.py

# 2. Baseline profiling
python profile_runtime_jax.py --output-dir ./baseline_profile

# 3. Apply optimizations based on static analysis

# 4. Compare performance
python profile_runtime_jax.py --output-dir ./optimized_profile

# 5. Compare results in baseline_profile/ vs optimized_profile/
```

## TensorBoard Visualization

When `--enable-jax-profiler` is used, you can visualize:

- **Trace Viewer**: Timeline of operations on each device
- **Op Profile**: Time spent in each operation type
- **Memory Profile**: Memory allocation over time
- **Python Tracer**: Map back to Python source code

Navigate to the "Profile" tab in TensorBoard to explore these views.

### Key TensorBoard Views

1. **Overview Page**: High-level performance summary
2. **Trace Viewer**: Detailed timeline of all operations
3. **Memory Profile**: Memory usage patterns
4. **Op Profile**: Which operations consume the most time

## Troubleshooting

### Issue: "Features not found" error

**Solution**: Run the featurisation script first:
```bash
cd jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT
python featurise_ISO_TRI_BI.py
```

### Issue: "Data splits not found" error

**Solution**: Run the split generation script:
```bash
cd jaxent/examples/1_IsoValidation_OMass/fitting/jaxENT
python splitdata_ISO.py
```

### Issue: JAX profiler trace is empty

**Solution**: Ensure you're running enough steps (at least 5-10) and that JAX is properly installed.

### Issue: Very slow performance

**Possible causes**:
- First run includes JIT compilation (expected)
- Running on CPU instead of GPU (check with `jax.devices()`)
- Debug mode enabled (disable with `JAX_DISABLE_JIT=0`)

## Advanced Usage

### Custom Data

To profile with your own data, modify the `load_example_data()` function in `profile_runtime_jax.py` or create a similar script that loads your specific data format.

### Integration into CI/CD

```bash
# Performance regression testing
python profile_runtime_jax.py --n-steps 20 --output-dir ./ci_profile

# Compare against baseline
python compare_profiles.py ./baseline_profile ./ci_profile
```

## Related Documentation

- [JAX Profiling Guide](https://jax.readthedocs.io/en/latest/profiling.html)
- [TensorBoard Profiler](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras)
- Static profiler: `analyze_computation_graph.py`

## Contributing

When adding new profiling capabilities:
1. Add timing points with `time.time()` and `jax.block_until_ready()`
2. Include both mean and variance in statistics
3. Separate JIT compilation from steady-state performance
4. Document new command-line options in this file
