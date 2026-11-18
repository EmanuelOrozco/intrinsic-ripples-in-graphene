#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.local/bin:$ROOT/../.tools:$PATH"
cd "$ROOT"
if [ -x "$ROOT/../.tools/tectonic" ]; then
  TECTONIC="$ROOT/../.tools/tectonic"
elif [ -x "$ROOT/../../.tools/tectonic" ]; then
  TECTONIC="$ROOT/../../.tools/tectonic"
elif command -v tectonic >/dev/null 2>&1; then
  TECTONIC="$(command -v tectonic)"
else
  echo "No se encontró tectonic"
  exit 1
fi
"$TECTONIC" --keep-logs informe_tersoff.tex
"$TECTONIC" --keep-logs informe_lcbopii.tex
"$TECTONIC" --keep-logs informe_comparacion.tex
"$TECTONIC" --keep-logs informe_codigo.tex
echo "PDF: $ROOT/informe_tersoff.pdf"
echo "PDF: $ROOT/informe_lcbopii.pdf"
echo "PDF: $ROOT/informe_comparacion.pdf"
echo "PDF: $ROOT/informe_codigo.pdf"
