#!/usr/bin/env bash
# --------------------------------------------------
# Run the full SoloRAG test suite.
#
# This script ensures that tests are always run from the project root
# and activates a local virtual environment if it exists.
#
# Usage:
#   ./scripts/run_tests.sh
#
# You can also pass any standard pytest arguments to it:
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

echo "🧪 Running test suite..."
pytest app/tests "$@"

echo "✅ All tests passed." 