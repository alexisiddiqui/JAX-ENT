"""Pytest configuration for JAX-ENT tests.

This module configures the runtime environment for all tests,
setting up the testing mode (CPU, strict types, JIT enabled).
"""

import os
import pytest

# Set default test mode before any imports
os.environ.setdefault("JAXENT_MODE", "testing")

from jaxent.src.config.runtime import configure_runtime, RuntimeMode


@pytest.fixture(scope="session", autouse=True)
def configure_test_runtime():
    """Configure runtime for testing.

    Automatically applied to all test sessions.
    Tests run in TESTING mode: CPU, strict types, JIT enabled.
    """
    configure_runtime(mode=RuntimeMode.TESTING)


@pytest.fixture
def debug_mode():
    """Fixture for tests that need debug mode (JIT disabled).

    Example:
        def test_something_with_debug(debug_mode):
            # This test runs in debug mode
            pass
    """
    original_config = configure_runtime(mode=RuntimeMode.DEBUG)
    yield
    # Note: In a more sophisticated implementation, we could restore the original config here
    # but for now we just leave it in debug mode for the remainder of the test


@pytest.fixture
def development_mode():
    """Fixture for tests that need development mode.

    Example:
        def test_something_with_dev(development_mode):
            # This test runs in development mode
            pass
    """
    original_config = configure_runtime(mode=RuntimeMode.DEVELOPMENT)
    yield


@pytest.fixture
def production_mode():
    """Fixture for tests that need production mode.

    Example:
        def test_something_with_prod(production_mode):
            # This test runs in production mode (on GPU if available)
            pass
    """
    original_config = configure_runtime(mode=RuntimeMode.PRODUCTION)
    yield
