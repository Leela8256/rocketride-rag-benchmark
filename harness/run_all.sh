#!/usr/bin/env bash
# One campaign from start to end: screening, finalize, timed runs, analysis, report.
#   bash harness/run_all.sh <campaign folder>
# For a long campaign run it detached:
#   nohup bash harness/run_all.sh /data/rag-run > /data/rag-run/chain.out 2>&1 < /dev/null &
# A second start continues where the first stopped: finished runs are kept.
# HOURS (default 24) is the time limit of each stage: no run starts when it could not finish before it.
set -uo pipefail
HARNESS="$(cd "$(dirname "$0")" && pwd)"
C="${1:?usage: run_all.sh <campaign folder>}"
note() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$C/chain.log"; }

note "chain start"
if ! MANIFEST="$C/source/corpus-manifest.json" python3 -c "import json,os,sys; sys.exit(0 if json.load(open(os.environ['MANIFEST']))['stage'] == 'final' else 1)"; then
  note "screening start"
  python3 -u "$HARNESS/campaign.py" "$C" --stage screening --hours "${HOURS:-24}" >> "$C/screening.log" 2>&1
  note "screening end rc=$?"
  python3 "$HARNESS/finalize.py" "$C" >> "$C/finalize.log" 2>&1 || { note "finalize FAILED, see finalize.log"; exit 1; }
  note "PDFs and questions fixed: $(tail -n 2 "$C/finalize.log" | head -n 1 | cut -c1-300)"
fi
note "timed runs start"
python3 -u "$HARNESS/campaign.py" "$C" --stage campaign --hours "${HOURS:-24}" >> "$C/campaign.log" 2>&1
RC=$?
note "timed runs end rc=$RC"
python3 "$HARNESS/analyze.py" "$C" >> "$C/analyze.log" 2>&1 && python3 "$HARNESS/report.py" "$C" >> "$C/analyze.log" 2>&1 \
  && note "report written: $C/REPORT.md" || note "analysis FAILED, see analyze.log"
note "chain complete"
exit $RC
