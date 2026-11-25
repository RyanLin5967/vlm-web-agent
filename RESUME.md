# Resume state (read this first)

Branch: `dev`. Decisions and measurements: `decision-log.md`. Queue log: written by `kaggle/queue.sh` (stdout).

## Blocker as of 2026-11-25 11:40 UTC
Kaggle refused GPU pushes: "Maximum weekly GPU quota of 30.00 hours reached" (account-wide, shared with the
diff-attention project). Reset time is not documented (Saturday 00:00 UTC per an old source; rolling per another).
CPU kernels still work. No paid compute is allowed (Ryan: $0, no Modal charges).

## Done
- Sites on Kaggle CPU via crane rootfs + proot: shopping, shopping_admin, reddit (480/812 tasks in scope).
  GitLab parked (flaky 500s), map blocked (S3 403).
- Teacher Qwen3-VL-30B-A3B AWQ on 2xT4. Results: reddit 7.5% (8/106), shopping_admin 0/107 (partial).
- Collections: wa-shopping-collect (86 verified episodes / 291 steps), wa-relabel-shopping (16 / 69 steps),
  wa-reddit-collect (COMPLETE, not yet summarised), wa-shopping-admin-collect (running at the block; includes relabel).

## Next actions, in order (the queue does all of them; each waits on its deps)
1. `kaggle/jobs/relabel_reddit.json` (after wa-reddit-collect)
2. `kaggle/jobs/train_r0.json` (after wa-relabel-reddit) -> merged 2B checkpoint + data_summary.json
3. `kaggle/jobs/{reddit,shopping,shopping_admin}_2b_r0_eval.json` (after wa-train-r0) -> dev success + WebArena results
Restart: `QUEUE_HOURS=96 nohup kaggle/queue.sh kaggle/jobs/relabel_reddit.json kaggle/jobs/train_r0.json kaggle/jobs/reddit_2b_r0_eval.json kaggle/jobs/shopping_2b_r0_eval.json kaggle/jobs/shopping_admin_2b_r0_eval.json >> queue.log 2>&1 &`
After those: add wa-shopping-admin-collect to training inputs (round 1), rejection-sampling rounds
(collect --tasks-file --judge defer with the 2B, then `wa.run judge` with the teacher), 32B baseline jobs.
