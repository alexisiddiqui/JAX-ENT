#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
PROJECT_DIR_NAME="JAX-ENT"
VENV_DIR=".venv"
TEST_DATA_ZIP="jaxent/tests/inst.zip"
TEST_DATA_DIR="jaxent/tests/inst"

# --- 1. Check if in the correct directory ---
if [ "$(basename "$(pwd)")" != "$PROJECT_DIR_NAME" ]; then
    echo "Error: This script must be run from the '$PROJECT_DIR_NAME' project root directory."
    exit 1
fi

echo "--- Running in $(pwd) ---"

# --- User confirmation ---
echo ""
echo "WARNING: This setup process is intended to be handled by the test scripts."
echo "Running this script directly may not work as expected."
read -p "Do you want to continue? [y/N]: " CONTINUE
if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
    echo "Aborting setup."
    exit 0
fi

# --- 2. Set up virtual environment and install dependencies ---
if [ ! -d "$VENV_DIR" ]; then
    echo "--- Creating virtual environment in '$VENV_DIR' using uv... ---"
    uv venv
else
    echo "--- Virtual environment '$VENV_DIR' already exists. ---"
fi

echo "--- Activating virtual environment ---"
source "$VENV_DIR/bin/activate"

echo "--- Installing project in editable mode with 'test' dependencies... ---"
uv pip install -e .[test]

echo "--- Dependencies installed successfully. ---"


# --- 3. Unpack test data ---
if [ -f "$TEST_DATA_ZIP" ]; then
    echo "--- Unpacking test data from '$TEST_DATA_ZIP' to '$TEST_DATA_DIR'... ---"
    unzip -o "$TEST_DATA_ZIP" -d "$(dirname "$TEST_DATA_DIR")"
    echo "--- Test data unpacked successfully. ---"
else
    echo "--- Warning: Test data zip file not found at '$TEST_DATA_ZIP'. ---"
fi

# --- 4. Placeholder for search and replace ---
# The user specified that a search and replace is needed to "correct the suffix before 'JAX-ENT'".
# Add the specific command here when it is known.
# For example:
# find jaxent/tests/inst -type f -name "*.config" -exec sed -i 's/some_suffix-JAX-ENT/corrected_suffix-JAX-ENT/g' {} +
echo "--- Placeholder for search and replace. Add the required command here. ---"


# --- Done ---
echo ""
echo "--- Test setup complete. ---"
echo "To run the tests, make sure you are in the virtual environment:"
echo "WARNING: this test script has changed your virtual environment to use the dev branch."
echo "This may have changed some of the dependencies."
echo "You can activate the virtual environment with:"
echo "source .venv/bin/activate"
echo "Then run pytest:"
echo "pytest -vv --tb=short --ignore=jaxent/tests/manual --ignore=jaxent/tests/slow"
