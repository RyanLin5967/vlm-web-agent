#!/bin/bash
# Push Kaggle jobs in order. Retries while the account's GPU-session cap (2) is full, while a job's dependencies
# are still running, through network errors, and - slowly - while the weekly GPU quota is exhausted.
# usage: QUEUE_HOURS=96 kaggle/queue.sh JOB.json [JOB.json ...]   (bounded: gives up on a job after QUEUE_HOURS)
cd "$(dirname "$0")/.."
hours=${QUEUE_HOURS:-10}
for job in "$@"; do
  deadline=$(( $(date +%s) + hours * 3600 ))
  while :; do
    out=$(timeout 200 uv run python kaggle/launch.py "$job" 2>&1 | tail -1)
    echo "$(date -u +%m-%d %H:%M:%S) $job: $out"
    case "$out" in *successfully*) break;; esac
    [ "$(date +%s)" -gt "$deadline" ] && { echo "GAVE UP on $job"; exit 1; }
    case "$out" in *"weekly GPU quota"*) sleep 1800;; *) sleep 180;; esac
  done
done
echo QUEUE_DONE
