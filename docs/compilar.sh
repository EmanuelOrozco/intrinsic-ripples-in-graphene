#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v tectonic >/dev/null 2>&1; then
  echo "No se encontró tectonic"
  exit 1
fi
cd "$ROOT"
tectonic --keep-logs reimplementacion.tex
tectonic --keep-logs resultados.tex
echo "PDF: $ROOT/reimplementacion.pdf"
echo "PDF: $ROOT/resultados.pdf"
