#!/usr/bin/env bash
# Verify that all service idle.py copies are identical.
# Run from the repo root: ./scripts/check-idle-sync.sh
set -euo pipefail

files=(services/*/src/*/idle.py)

if [[ ${#files[@]} -lt 2 ]]; then
    echo "SKIP: fewer than 2 idle.py files found"
    exit 0
fi

reference="${files[0]}"
rc=0
for f in "${files[@]:1}"; do
    if ! diff -q "$reference" "$f" > /dev/null 2>&1; then
        echo "FAIL: $reference and $f have diverged"
        diff --unified "$reference" "$f" || true
        rc=1
    fi
done

if [[ $rc -eq 0 ]]; then
    echo "OK: all idle.py copies are identical (${#files[@]} files)"
fi
exit $rc
