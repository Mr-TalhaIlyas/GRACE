#!/usr/bin/env bash
set -e  # exit on first error

SCRIPTS=(
  test1.py
  test2.py
  test3.py
)

for SCRIPT in "${SCRIPTS[@]}"; do
  echo "→ Running $SCRIPT"
  python "$SCRIPT"
  echo "✓ $SCRIPT finished, resources freed."
done
