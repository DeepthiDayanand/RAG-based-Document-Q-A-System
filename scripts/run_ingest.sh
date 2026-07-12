#!/usr/bin/env bash
# Convenience wrapper to (re)build the vector index from data/sample_docs
# or a directory passed as the first argument.
set -euo pipefail

DOCS_DIR="${1:-data/sample_docs}"
cd "$(dirname "$0")/.."
python -m app.ingest --docs_dir "$DOCS_DIR"
