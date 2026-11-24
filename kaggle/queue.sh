#!/bin/bash
# Push Kaggle jobs in order, retrying while the account's GPU-session cap (2) is full.
# usage: kaggle/queue.sh JOB.json [JOB.json ...]   (bounded: gives up on a job after 10 h of retries)
cd "$(dirname "$0")/.."
for job in "$@"; do
  deadline=$(( $(date +%s) + 36000 ))
  while :; do
    out=$(timeout 200 uv run python kaggle/launch.py "$job" 2>&1 | tail -1)
    echo "$(date -u +%H:%M:%S) $job: $out"
    case "$out" in *successfully*) break;; esac
    # anything else (network drop, Kaggle 5xx, git fetch failure) is retried too; the deadline bounds it
    [ "$(date +%s)" -gt "$deadline" ] && { echo "GAVE UP on $job"; exit 1; }
    sleep 120
  done
done
echo QUEUE_DONE
