import time

import jax
import jax.numpy as jnp
from tqdm import tqdm

# Uncomment the following line to force JAX to use the GPU
# os.environ["JAX_PLATFORM_NAME"] = "gpu"


def test_gpu_available():
    """Test that a GPU is available and is the default backend."""
    print("\nTesting GPU availability...")
    available_platforms = [d.platform for d in jax.devices()]
    assert len(jax.devices()) > 0
    # Accept both 'gpu' and 'cuda' as valid GPU platforms
    assert jax.default_backend() in ("gpu", "cuda")
    assert any(p in ("gpu", "cuda") for p in available_platforms)
    print("GPU is available and is the default backend.")


def test_gpu_memory():
    """Test that GPU memory statistics are available and valid."""
    print("\nTesting GPU memory statistics...")
    device = jax.devices()[0]
    print("Device:", device)
    memory_stats = device.memory_stats()
    print("Memory stats:", memory_stats)
    # Updated memory checks for current JAX API
    # Note: exact keys may vary by platform/version
    assert any("bytes" in key for key in memory_stats.keys())
    for key, value in memory_stats.items():
        if isinstance(value, (int, float)):
            assert value >= 0
    print("GPU memory statistics are valid.")


def test_basic_computation():
    """Test basic computation on the GPU."""
    print("\nTesting basic computation on GPU...")
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    z = x + y
    assert jnp.allclose(z, jnp.array([5.0, 7.0, 9.0]))
    print("Basic computation on GPU succeeded.")


def test_gpu_operations():
    """Test more complex GPU operations."""
    print("\nTesting complex GPU operations...")
    # Matrix multiplication
    a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    b = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    c = jnp.dot(a, b)
    expected_result = jnp.array([[19.0, 22.0], [43.0, 50.0]])
    assert jnp.allclose(c, expected_result)
    print("Matrix multiplication on GPU succeeded.")

    # Element-wise operations
    d = jnp.sin(a)
    expected_sin = jnp.array([[0.84147098, 0.90929743], [0.14112001, -0.7568025]])
    assert jnp.allclose(d, expected_sin, atol=1e-6)
    print("Element-wise operations on GPU succeeded.")


def test_gpu_info():
    """Test that GPU information can be retrieved."""
    print("\nTesting GPU information retrieval...")
    device = jax.devices()[0]
    assert hasattr(device, "device_kind")
    assert isinstance(device.device_kind, str)
    assert hasattr(device, "platform")
    assert isinstance(device.platform, str)
    assert device.platform == "gpu"
    print(
        f"GPU information retrieved successfully. Device kind: {device.device_kind}, Platform: {device.platform}"
    )


# Optional: Add more detailed GPU tests
def test_gpu_performance():
    """Test GPU performance with multiple large matrix multiplications and report statistics."""
    print("\nTesting GPU performance with multiple large matrix multiplications...")

    # Parameters
    matrix_size = 4096  # Adjust based on GPU memory capacity
    num_iterations = 10  # 00  # Number of iterations for the test
    warm_up_iterations = 5  # Number of warm-up runs

    # Create large matrices filled with ones on the GPU
    a = jnp.ones((matrix_size, matrix_size), dtype=jnp.float32)
    b = jnp.ones((matrix_size, matrix_size), dtype=jnp.float32)

    # JIT compile the dot product function for better performance
    matmul = jax.jit(jnp.dot)

    # Warm-up runs
    print(f"Performing {warm_up_iterations} warm-up iterations...")
    for _ in range(warm_up_iterations):
        c = matmul(a, b).block_until_ready()

    # Start the performance test
    print(f"Running {num_iterations} iterations...")
    start_time = time.time()
    for _ in tqdm(range(num_iterations)):
        c = matmul(a, b).block_until_ready()
    end_time = time.time()

    # Calculate statistics
    total_time = end_time - start_time
    average_time = total_time / num_iterations
    flops_per_multiplication = 2 * matrix_size**3  # FLOPs for one matrix multiplication
    total_flops = flops_per_multiplication * num_iterations
    throughput_flops = total_flops / total_time  # FLOPs per second
    throughput_gflops = throughput_flops / 1e9  # GFLOPs per second
    memory_per_multiplication = 2 * matrix_size**2 * 4  # Bytes for two matrices (float32)
    total_memory = memory_per_multiplication * num_iterations
    memory_bandwidth = total_memory / total_time  # Bytes per second
    memory_bandwidth_gb = memory_bandwidth / 1e9  # GB/s

    # Print performance statistics
    print(f"Total time for {num_iterations} iterations: {total_time:.4f} seconds")
    print(f"Average time per multiplication: {average_time:.4f} seconds")
    print(f"Throughput: {throughput_gflops:.2f} GFLOPs")
    print(f"Memory bandwidth: {memory_bandwidth_gb:.2f} GB/s")

    # Verify the result for one iteration (optional)
    expected_result = jnp.full((matrix_size, matrix_size), matrix_size, dtype=jnp.float32)
    assert jnp.allclose(c, expected_result, atol=1e-3)
    print("Multiple large matrix multiplications on GPU succeeded.")


if __name__ == "__main__":
    test_gpu_available()
    test_gpu_memory()
    test_basic_computation()
    test_gpu_operations()
    test_gpu_info()
    test_gpu_performance()
    test_gpu_available()
    test_gpu_memory()
    test_basic_computation()
    test_gpu_operations()
    test_gpu_info()
    test_gpu_performance()
