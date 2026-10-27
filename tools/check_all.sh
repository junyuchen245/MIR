#!/bin/bash
# Run every behaviour-preservation gate: public API, golden numerical outputs (incl. pretrained loading), unit tests.
# Usage: tools/check_all.sh <api_snapshot.json> <golden.json>
cd "$(dirname "$0")/.."
API=${1:?api snapshot json}; GOLD=${2:?golden json}
rc=0
echo "== public API";      python3.8 tools/api_snapshot.py --check "$API" 2>/dev/null | grep -E "^  !|breaking" || rc=1
echo "== golden outputs";  python3.8 tools/golden_outputs.py --check "$GOLD" 2>/dev/null | grep -E "^  !|problem" || rc=1
echo "== unit tests";      python3.8 -m pytest tests -o addopts="" -q -p no:warnings 2>&1 | tail -1
exit $rc
