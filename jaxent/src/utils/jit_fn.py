"""
# In your simulation's initialise method:
@jit_Guard.on_jit_error(retry_count=2)
def initialise(self):
    self._jit_forward_pure(self.simulation_parameters, self.input_data)
    return True

# Alternative using context manager:
def initialise(self):
    with jit_Guard(self) as guard:
        try:
            self._jit_forward_pure(self.simulation_parameters, self.input_data)
            return True
        except Exception as e:
            # Try with cache clearing
            result = jit_Guard.safe_jit_operation(
                self._jit_forward_pure,
                self.simulation_parameters,
                self.input_data
            )
            return True

# In your tests - complete isolation:
@jit_Guard.test_isolation()
def test_simulation_initialise_real_inputs():
    simulation = create_simulation(use_jit=True)
    assert simulation.initialise(), "Initialisation should return True."

# In tests - clear only after completion:
@jit_Guard.clear_caches_after()
def test_simulation_forward_jit_real_inputs():
    simulation = create_simulation(use_jit=True)
    simulation.initialise()
    result = simulation.forward()
    assert result is not None

# For functions that should clear caches only on success:
@jit_Guard.clear_caches_after(clear_on_error=False)
def successful_operation_only():
    # Caches only cleared if this succeeds
    pass

# For combining with context manager in tests:
@jit_Guard.test_isolation()
def test_simulation_with_context_manager():
    simulation = create_simulation()

    with jit_Guard(simulation, cleanup_on_exit=True):
        simulation.initialise()
        result = simulation.forward()
        assert result is not None
    # Both decorator and context manager ensure clean state

# Class method decoration:
class MySimulation:
    @jit_Guard.on_jit_error(retry_count=1)
    def compile_model(self):
        # Will retry once if JIT compilation fails
        return self._compile_internal()

    @jit_Guard.clear_caches_after(clear_on_success=True, clear_on_error=False)
    def run_experiment(self):
        # Only clear caches if experiment succeeds
        return self._run_internal()
"""

import gc
import weakref
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, Generic, TypeVar

import jax

# Type variable for decorator return type preservation
F = TypeVar("F", bound=Callable[..., Any])
# Type variable for simulation object
T = TypeVar("T")


class jit_Guard(Generic[T]):
    """
    Context manager for handling JAX JIT compilation issues by managing
    simulation objects and clearing JAX caches.

    Usage:
        # Type checker automatically infers simulation type
        with jit_Guard(simulation) as sim:
            sim.initialise()  # sim is correctly typed as your simulation object
            result = sim.forward()

        # For manual cache clearing, just use static method
        jit_Guard.clear_all_caches()
    """

    def __init__(self, simulation_obj: T, cleanup_on_exit: bool = False):
        """
        Initialize the jit_Guard.

        Args:
            simulation_obj: Simulation object to manage
            cleanup_on_exit: If True, delete simulation object on context exit
        """
        self.simulation_obj = simulation_obj
        self.simulation_ref = weakref.ref(simulation_obj) if simulation_obj is not None else None
        self.cleanup_on_exit = cleanup_on_exit

    def __enter__(self) -> T:
        """Enter the context manager and return the managed simulation object."""
        # Clear caches before starting
        self.clear_all_caches()
        return self.simulation_obj

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager with cleanup."""
        try:
            # Clear JAX caches
            self.clear_all_caches()

            # Optional simulation object cleanup
            if self.cleanup_on_exit and self.simulation_obj is not None:
                # Delete any JIT-compiled methods if they exist
                if hasattr(self.simulation_obj, "_jit_forward_pure"):
                    delattr(self.simulation_obj, "_jit_forward_pure")
                if hasattr(self.simulation_obj, "_compiled_forward"):
                    delattr(self.simulation_obj, "_compiled_forward")

                # Remove reference
                self.simulation_obj = None

            # Force garbage collection
            gc.collect()

        except Exception as cleanup_error:
            if exc_type is None:
                raise cleanup_error
            print(f"Warning: Cleanup error in jit_Guard: {cleanup_error}")

        return False

    @staticmethod
    def clear_all_caches():
        """Clear all JAX-related caches and backends."""
        try:
            # Clear JAX compilation cache
            jax.clear_caches()

            # # Clear backends (more aggressive cleanup)
            # clear_backends()

            # Force garbage collection
            gc.collect()

        except Exception as e:
            print(f"Warning: Error clearing JAX caches: {e}")

    @staticmethod
    def clear_caches_after(
        clear_on_success: bool = True,
        clear_on_error: bool = True,
        force_gc: bool = True,
        suppress_cleanup_errors: bool = True,
    ) -> Callable[[F], F]:
        """
        Decorator that clears JAX caches after function execution.

        Args:
            clear_on_success: Clear caches when function succeeds
            clear_on_error: Clear caches when function raises exception
            force_gc: Force garbage collection after clearing caches
            suppress_cleanup_errors: Don't raise cleanup errors (just log them)

        Returns:
            Decorated function

        Example:
            @jit_Guard.clear_caches_after()
            def test_simulation():
                # ... test code
                pass

            @jit_Guard.clear_caches_after(clear_on_error=False)
            def simulation_forward(self):
                # Only clear caches on success
                pass
        """

        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args, **kwargs):
                exception_occurred = False
                result = None
                original_exception = None

                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    exception_occurred = True
                    original_exception = e
                    raise
                finally:
                    # Determine if we should clear caches
                    should_clear = (clear_on_success and not exception_occurred) or (
                        clear_on_error and exception_occurred
                    )

                    if should_clear:
                        try:
                            jit_Guard.clear_all_caches()
                            if force_gc:
                                gc.collect()
                        except Exception as cleanup_error:
                            cleanup_msg = f"Cache cleanup error in {func.__name__}: {cleanup_error}"
                            if suppress_cleanup_errors:
                                print(f"Warning: {cleanup_msg}")
                            else:
                                # If original exception exists, chain it
                                if original_exception:
                                    raise cleanup_error from original_exception
                                else:
                                    raise cleanup_error

            return wrapper

        return decorator

    @staticmethod
    def test_isolation() -> Callable[[F], F]:
        """
        Decorator specifically designed for test functions that ensures complete isolation.
        Clears caches before and after test execution.

        Example:
            @jit_Guard.test_isolation()
            def test_simulation_forward():
                # Test runs with clean JAX state
                pass
        """

        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Clear before test
                jit_Guard.clear_all_caches()

                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    # Always clear after test, regardless of outcome
                    jit_Guard.clear_all_caches()
                    gc.collect()

            return wrapper

        return decorator

    @staticmethod
    def on_jit_error(retry_count: int = 1) -> Callable[[F], F]:
        """
        Decorator that clears caches and retries function on JAX/JIT errors.

        Args:
            retry_count: Number of times to retry after clearing caches

        Example:
            @jit_Guard.on_jit_error(retry_count=2)
            def compile_simulation(self):
                # Will retry up to 2 times if JIT errors occur
                pass
        """

        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None

                for attempt in range(retry_count + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        error_msg = str(e).lower()

                        # Check if this looks like a JIT/compilation error
                        is_jit_error = any(
                            keyword in error_msg
                            for keyword in [
                                "truth value of an array",
                                "jit",
                                "compilation",
                                "abstract",
                                "tracer",
                                "concrete value",
                            ]
                        )

                        if is_jit_error and attempt < retry_count:
                            print(f"JIT error on attempt {attempt + 1}: {e}")
                            print(
                                f"Clearing caches and retrying... ({retry_count - attempt} attempts remaining)"
                            )
                            jit_Guard.clear_all_caches()
                        else:
                            # Either not a JIT error or out of retries
                            raise

                # This should never be reached, but just in case
                raise last_exception

            return wrapper

        return decorator

    @staticmethod
    def safe_jit_operation(func, *args, **kwargs):
        """
        Safely execute a JIT operation with automatic cache clearing on failure.

        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func(*args, **kwargs)

        Raises:
            RuntimeError: If operation fails after cache clearing
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "truth value of an array" in str(e) or "JIT" in str(e).upper():
                print(f"JIT operation failed: {e}")
                print("Clearing caches and retrying...")
                jit_Guard.clear_all_caches()

                try:
                    return func(*args, **kwargs)
                except Exception as retry_error:
                    raise RuntimeError(
                        f"Operation failed even after cache clearing: {retry_error}"
                    ) from e
            else:
                raise


# Convenience context manager function
@contextmanager
def jit_guard(simulation_obj: T, cleanup_on_exit: bool = False) -> Generator[T, None, None]:
    """
    Convenience function to create jit_Guard context manager.

    Args:
        simulation_obj: Simulation object to manage
        cleanup_on_exit: If True, delete simulation object on context exit

    Yields:
        The managed simulation object
    """
    guard = jit_Guard(simulation_obj, cleanup_on_exit)
    try:
        yield guard.__enter__()
    finally:
        guard.__exit__(None, None, None)
