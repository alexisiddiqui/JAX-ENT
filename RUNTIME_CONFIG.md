# JAX-ENT Runtime Configuration System

## Overview

The JAX-ENT runtime configuration system provides centralized, mode-based configuration for JAX, beartype, chex, and other runtime settings. This eliminates scattered configuration across test files and enables switching between development, testing, and production modes without code changes.

## Quick Start

### Environment Variable Configuration

Set `JAXENT_MODE` before running your script:

```bash
# Development mode (CPU, JIT enabled, lenient type checking)
export JAXENT_MODE=development
python my_script.py

# Debug mode (CPU, JIT disabled, lenient type checking, NaN checking)
export JAXENT_MODE=debug
python my_script.py

# Testing mode (CPU, JIT enabled, strict type checking)
export JAXENT_MODE=testing
pytest

# Production mode (GPU, JIT enabled, strict type checking)
export JAXENT_MODE=production
python my_script.py

# Performance mode (GPU, JIT enabled, chex assertions disabled)
export JAXENT_MODE=performance
python benchmark.py
```

### Programmatic Configuration

```python
# Must be called BEFORE importing jaxent
from jaxent.src.config.runtime import configure_runtime
configure_runtime(mode="debug")

import jaxent
```

### In Tests

Tests automatically run in testing mode via `conftest.py`:

```python
def test_something():
    # Runs in testing mode (CPU, strict types)
    pass

def test_debug_feature(debug_mode):
    # Uses debug_mode fixture for this test only
    pass
```

## Configuration Modes

| Mode | Platform | JIT | Type Checking | Chex | NaN Debug | Use Case |
|------|----------|-----|---------------|------|-----------|----------|
| **PRODUCTION** | GPU | Enabled | Strict (raises) | Enabled | No | Production runs |
| **DEVELOPMENT** | CPU | Enabled | Lenient (warns) | Enabled | No | Interactive development |
| **DEBUG** | CPU | Disabled | Lenient (warns) | Enabled | Yes | Step-through debugging |
| **TESTING** | CPU | Enabled | Strict (raises) | Enabled | No | Unit/integration tests |
| **PERFORMANCE** | GPU | Enabled | Strict (raises) | Disabled | No | Benchmarking |

## Environment Variables

### Primary Configuration

- **`JAXENT_MODE`**: Set the runtime mode (production/development/debug/testing/performance)
  - Default: `development`
  - Example: `JAXENT_MODE=debug python script.py`

### Overrides

- **`JAXENT_PLATFORM`**: Override platform (cpu/gpu/tpu)
  - Example: `JAXENT_PLATFORM=cpu` to force CPU for production mode

- **`JAXENT_DISABLE_JIT`**: Override JIT setting (true/false)
  - Example: `JAXENT_DISABLE_JIT=true` to disable JIT in production

- **`JAXENT_ENABLE_CHEX`**: Override chex assertions (true/false)
  - Example: `JAXENT_ENABLE_CHEX=false` to disable chex in testing mode

## API Reference

### RuntimeMode Enum

```python
from jaxent.src.config.runtime import RuntimeMode

RuntimeMode.PRODUCTION
RuntimeMode.DEVELOPMENT
RuntimeMode.DEBUG
RuntimeMode.TESTING
RuntimeMode.PERFORMANCE
```

### RuntimeConfig Class

Immutable configuration object:

```python
from jaxent.src.config.runtime import RuntimeConfig

# Create from preset mode
config = RuntimeConfig.from_mode("debug")

# Create with overrides
config = RuntimeConfig.from_mode(
    "production",
    platform_override="cpu",
    disable_jit_override=True,
    enable_chex_override=False
)

# Access properties
print(config.mode)              # RuntimeMode.DEBUG
print(config.platform)          # "cpu"
print(config.disable_jit)       # True
print(config.beartype_conf)     # BeartypeConf object
print(config.enable_chex_asserts)  # True
```

### Configuration Functions

#### `configure_runtime()`

Configure the runtime environment:

```python
from jaxent.src.config.runtime import configure_runtime

# By mode name
configure_runtime(mode="debug")

# By RuntimeMode enum
from jaxent.src.config.runtime import RuntimeMode
configure_runtime(mode=RuntimeMode.TESTING)

# With overrides
configure_runtime(mode="production", platform_override="cpu")

# By custom config
config = RuntimeConfig.from_mode("testing")
configure_runtime(config=config)
```

#### `get_runtime_config()`

Get the current configuration:

```python
from jaxent.src.config.runtime import get_runtime_config

config = get_runtime_config()
print(config.mode.value)        # "development"
print(config.platform)          # "cpu"
```

#### `finalize_runtime_config()`

Apply JAX configuration after import (called automatically by `jaxent/__init__.py`):

```python
from jaxent.src.config.runtime import finalize_runtime_config

# Usually called automatically, but can be called manually if needed
finalize_runtime_config()
```

## Usage Examples

### Example 1: Debug a Failing Test

```bash
# Make a specific test run in debug mode
JAXENT_MODE=debug pytest jaxent/tests/some_test.py -xvs
```

### Example 2: Development on CPU

```bash
# Default development mode (already CPU)
python my_development_script.py
```

### Example 3: Production Run with Custom GPU

```bash
# Production mode on specific GPU
export JAXENT_MODE=production
python production_script.py
```

### Example 4: Programmatic Configuration

```python
#!/usr/bin/env python
# Configure before importing jaxent
from jaxent.src.config.runtime import configure_runtime, RuntimeMode

if is_debug_mode():
    configure_runtime(mode=RuntimeMode.DEBUG)
else:
    configure_runtime(mode=RuntimeMode.PRODUCTION)

import jaxent
# Now jaxent uses the configured runtime
```

### Example 5: Test Fixtures

```python
import pytest

def test_normal_operation():
    # Runs in testing mode (default for pytest)
    pass

def test_with_debug_mode(debug_mode):
    # This test specifically uses debug mode
    pass

def test_with_development_mode(development_mode):
    # This test specifically uses development mode
    pass
```

## Type Checking Modes

### Strict Mode (Production, Testing, Performance)

```python
from some_module import some_function

# This will RAISE an error
some_function(42)  # Expected string, got int
```

### Lenient Mode (Development, Debug)

```python
from some_module import some_function

# This will WARN instead of raising
some_function(42)  # Expected string, got int
# UserWarning: type violation...
```

## JIT Configuration

### Enabled (Production, Development, Testing, Performance)

```python
import jax

@jax.jit
def f(x):
    return x + 1

# JIT compilation occurs automatically
```

### Disabled (Debug)

```python
import jax

@jax.jit
def f(x):
    return x + 1

# Function runs eagerly without JIT compilation
# Useful for debugging with print statements, breakpoints, etc.
```

## Backward Compatibility

The runtime configuration system maintains full backward compatibility:

1. **Existing environment variable setup** still works:
   ```python
   os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
   import jaxent
   ```

2. **Direct JAX config updates** still work:
   ```python
   import jaxent
   import jax
   jax.config.update("jax_platform_name", "gpu")
   ```

3. **Manual beartype decorators** still work:
   ```python
   from beartype import beartype
   @beartype
   def func(x: int) -> str: ...
   ```

## Initialization Order

The initialization happens in this critical order:

1. **Import `jaxent.src.config.runtime`**: Reads `JAXENT_MODE` env var, sets XLA env vars
2. **Import `jaxent.src.__init__.py`**: Applies beartype configuration
3. **Import JAX and other modules**: JAX uses the pre-set environment variables
4. **Call `finalize_runtime_config()`**: Applies JAX and chex configuration

This order ensures:
- XLA settings take effect before JAX initialization
- Beartype is applied to the entire package
- JAX configuration is set after JAX is imported

## Testing

### Unit Tests

```bash
pytest jaxent/tests/unit/config/test_runtime.py -v
```

### Integration Tests

```bash
# All tests automatically use testing mode
pytest jaxent/tests/ -x
```

### Manual Verification

```bash
# Test each mode
JAXENT_MODE=debug python -c "import jaxent; from jaxent.src.config.runtime import get_runtime_config; print(get_runtime_config().mode.value)"
JAXENT_MODE=production python -c "import jaxent; from jaxent.src.config.runtime import get_runtime_config; print(get_runtime_config().mode.value)"
```

## Troubleshooting

### Issue: JIT is still enabled in debug mode

**Solution**: Make sure `configure_runtime()` is called BEFORE importing other jaxent modules:

```python
# ✓ Correct
from jaxent.src.config.runtime import configure_runtime
configure_runtime(mode="debug")
import jaxent

# ✗ Wrong
import jaxent
from jaxent.src.config.runtime import configure_runtime
configure_runtime(mode="debug")
```

### Issue: Type checking not enforcing rules

**Solution**: Ensure you're in strict mode (production/testing/performance):

```bash
# ✓ Use strict mode for type checking
JAXENT_MODE=testing pytest

# ✗ Lenient mode allows violations
JAXENT_MODE=development pytest  # Will warn, not fail
```

### Issue: Different behavior on GPU vs CPU

**Solution**: Check the platform setting:

```python
from jaxent.src.config.runtime import get_runtime_config
config = get_runtime_config()
print(f"Platform: {config.platform}")
```

## Future Enhancements

Potential future improvements (not yet implemented):

- Config file support (`jaxent.toml`, `~/.jaxent/config.toml`)
- Runtime mode switching with context managers
- Config validation and conflict detection
- CLI integration (`--jaxent-mode debug`)
- Detailed logging of configuration decisions

## See Also

- `jaxent/src/config/runtime.py` - Runtime configuration module
- `jaxent/src/__init__.py` - Package initialization with runtime config
- `jaxent/tests/conftest.py` - Pytest configuration
- `jaxent/tests/unit/config/test_runtime.py` - Unit tests
