#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:$ROOT/../.tools:$PATH"
cd "$ROOT"
if [ -x "$ROOT/../.tools/tectonic" ]; then
  TECTONIC="$ROOT/../.tools/tectonic"
elif command -v tectonic >/dev/null 2>&1; then
  TECTONIC="$(command -v tectonic)"
else
  echo "No se encontró tectonic"
  exit 1
fi
"$TECTONIC" --keep-logs reimplementacion.tex
"$TECTONIC" --keep-logs resultados.tex
if [ -f resultados_n8640_macros.tex ]; then
  "$TECTONIC" --keep-logs resultados_n8640.tex
  echo "PDF: $ROOT/resultados_n8640.pdf"
fi
echo "PDF: $ROOT/reimplementacion.pdf"
echo "PDF: $ROOT/resultados.pdf"
