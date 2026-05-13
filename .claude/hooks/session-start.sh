#!/bin/bash
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(pwd)}"

# Install Python dependencies from lockfile
uv sync --frozen

# Export PYTHONPATH so src/ imports work without install
echo 'export PYTHONPATH="${CLAUDE_PROJECT_DIR:-.}"' >> "$CLAUDE_ENV_FILE"
