#!/usr/bin/env bash
# --------------------------------------------------
# Run the full SoloRAG test suite.
#
# This script ensures that tests are always run from the project root
# and activates a local virtual environment if it exists.
#
# Usage:
#   ./scripts/run_tests.sh           # Run all tests (skip GPU if not available)
#   ./scripts/run_tests.sh --gpu     # Run only GPU tests
#   ./scripts/run_tests.sh --no-gpu  # Skip GPU tests
#
# You can also pass any standard pytest arguments:
#   ./scripts/run_tests.sh -v -k "test_api"
# --------------------------------------------------

set -euo pipefail

# Navigate to the project root directory
cd "$(dirname "$0")/.."

echo "🐍 Activating virtual environment..."
# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Parse special arguments
GPU_ONLY=false
NO_GPU=false
PYTEST_ARGS=()

for arg in "$@"; do
    case $arg in
        --gpu)
            GPU_ONLY=true
            ;;
        --no-gpu)
            NO_GPU=true
            ;;
        *)
            PYTEST_ARGS+=("$arg")
            ;;
    esac
done

if [ "$GPU_ONLY" = true ] && [ "$NO_GPU" = true ]; then
    echo "❌ Error: Cannot use both --gpu and --no-gpu flags"
    exit 1
fi

echo "🧪 Running test suite..."

if [ "$GPU_ONLY" = true ]; then
    echo "🎮 Running GPU tests only..."
    pytest app/tests -m gpu "${PYTEST_ARGS[@]}"
elif [ "$NO_GPU" = true ]; then
    echo "🚫 Skipping GPU tests..."
    pytest app/tests -m "not gpu" "${PYTEST_ARGS[@]}"
else
    echo "🔄 Running all tests (GPU tests will be skipped if hardware not available)..."
    pytest app/tests "${PYTEST_ARGS[@]}"
fi

echo "✅ All tests passed." 