# Decision log — VLM web agent (spec: ~/projects/resume/specs/vlm-web-agent-spec.md)

Append-only. Newest at the bottom. (Not named DECISIONS.md: the global spend-guard hook treats every
`DECISIONS.*` as the Cat's Eye ledger.)

## 2026-11-24
- **Budget: Ryan's hard cap is $5 total for Modal** ("like 5$ wtf"), and "make sure you waste no money".
  Numbers only need to be "somewhat close" to the spec's targets.
- **Modal is unusable right now**: probe of shopping_admin in a Modal Sandbox was refused with
  "Container terminated due to reaching billing cycle spend limit" (`modal billing summary`, profile ryan-lin:
  $55.00 metered this cycle, -$30 credits, $13.76 billed — spent by other projects before this one started).
  Modal rates (`modal billing rates`): Sandbox CPU $0.1419/core-h is 3x Function CPU $0.0473/core-h, so if
  Modal is ever used for sites, use a Function + `modal.forward`, never a Sandbox.
- **Compute plan: $0 via Kaggle free kernels** (credentials verified with `kaggle kernels list --mine`).
  GPU work (2B training, 2B/teacher inference) on free T4x2/P100. Sites: probing whether a
  WebArena-Verified image can boot inside a Kaggle kernel without Docker (chroot/proot of the pulled
  rootfs) — `kaggle/site_probe/`.
- **Teacher: open VLM self-hosted** (Ryan picked "open teacher, frontier row only"). Frontier baseline row
  needs an API key Ryan does not have configured; it is not run until a key exists AND a 20-task cost pilot
  is approved. No API spend without that.
- **Benchmark: WebArena-Verified** (ServiceNow; 812 tasks, Docker Hub images `am1n3e/webarena-verified-*`,
  deterministic evaluators — no LLM judge, so no OpenAI key needed). Deviation from spec's BrowserGym
  evaluators, which call GPT-4 for fuzzy_match.
- Task counts by site set (assets/dataset/webarena-verified.json): shopping 187, shopping_admin 182,
  gitlab 180, map 109, reddit 106, multi-site 48.
- **No Modal at all** (Ryan, after raising the spend limit: "just whatever you do make sure i don't get charged on
  modal"). Verified no sandboxes running; stopped the empty `wa-sites` app (0 tasks); deleted `infra/` and the
  `modal` dependency so no code path in this repo can launch Modal work. Kaggle free tier only. The earlier
  sandbox probe never started (refused at image build by the old spend limit).
- Kaggle facts measured 2026-11-24: batch kernels run as root with CAP_SYS_CHROOT and CAP_MKNOD but not
  CAP_SYS_ADMIN, so chroot works and mount fails; WebArena services die without /proc,/dev -> boot under proot.
  `/` has ~2.5 TB free. Max 5 concurrent batch sessions. Requesting machine_shape "NvidiaTeslaT4" queued 30+ min;
  the default GPU shape gave T4x2 immediately.
- Site boot root cause (probe v2): services died because my streaming layer extractor LOST FILES (missing nginx
  conf sample, php-fpm.conf, InnoDB .ibd files; rootfs 1.7 GB), not only for lack of /proc. Switched to
  `crane export` (go-containerregistry), which flattens layers with correct whiteout semantics.
- **Student = Qwen3-VL-2B-Instruct** (kaggle/gpu_probe2, T4 fp16): Qwen3.5-2B cannot start under vLLM 0.34 on
  Turing (engine init fails, eager too); Qwen3-VL-2B runs under vLLM eager: 1.3 s single step, 16-batch 3.67 s,
  correct grounding on the synthetic SoM test. Fast batched serving is what makes 812-task evals and rejection
  sampling fit free GPU hours. Spec allows "newest ~2B Qwen-VL". fp16 forward of Qwen3.5-2B gave a finite loss,
  so fp16 LoRA training on T4 is viable.
- **Large open baseline = Qwen3-VL-32B-Instruct** (named in the spec; 16x the student), as the 4-bit AWQ build
  QuantTrio/Qwen3-VL-32B-Instruct-AWQ to fit 2xT4. Being probed with the 30B-A3B AWQ (teacher candidate) in gpu_probe3.
- 2026-11-24 ~16:00 UTC: reddit smoke is a real pass (logged in as MarvelsGrantMan136, elements extracted,
  2 scripted steps/episode, official eval ran). shopping_admin smoke v1 was a HOLLOW pass: every episode hit
  ERR_CONNECTION_REFUSED before any policy call, env-ctrl init had failed (image env not passed), and the run
  still wrote a 0% results file. Fixed + guarded: browser preflight REFUSEs, all-env-failure REFUSEs
  (tests/test_run_guards.py forces it to fire).
- GPU contention: the diff-attention project (kernel diffattn-stage-a) runs on the same Kaggle account; Kaggle
  allows 2 GPU sessions per account and the weekly GPU quota is shared. This project effectively has 1 slot.
- SFT path verified (kaggle wa-sft-smoke, tiny random Qwen3-VL, CPU): 2 LoRA updates, finite loss, adapter
  saved, supervised tokens decode to exactly target + <|im_end|> (assistant-only loss mask correct).
- Sites verified on Kaggle free CPU under proot (crane rootfs): shopping_admin (boot 174 s, admin dashboard,
  logged in), reddit (boot 271 s, logged in), shopping (boot 285 s, customer logged in). 475/812 single-site tasks.
- 32B AWQ on 2xT4 loads (Marlin AWQ kernel works on sm75, 9.85 GiB weights/GPU) but leaves 0.87 GiB KV cache:
  vLLM refused max_model_len 8192 (needs 1.0 GiB). KV/token/GPU at TP2: 32B ~131 KB vs 30B-A3B ~49 KB.
  -> **Teacher = Qwen3-VL-30B-A3B-Instruct AWQ** (more concurrency, 3B active params, faster decode).
  -> 32B baseline runs separately: max_model_len 6144, util 0.95, max_num_batched_tokens 4096, concurrency 3.
  Collection runs BEFORE eval in teacher kernels (data is the critical path).
- 2026-11-24 ~20:05 UTC: **:80 bug** confirmed and fixed. Base URL "http://localhost:80" made every
  NetworkEventEvaluator regex miss (browser records http://localhost/...) and made the crawler reject every link
  (3 synthetic tasks total). Same 106 reddit episodes of Qwen3-VL-30B-A3B re-scored with the official evaluator
  under the corrected URL (wa/rescore.py, kernel wa-rescore-reddit-teacher): 0/106 -> 8/106 = 7.5%
  (95% bootstrap CI 2.8-13.2%), mean 20.4 steps, 44.8k input tokens/task. results/reddit__Qwen3-VL-30B-A3B-AWQ__zs__s0.json
- Teacher throughput (vLLM log, 2xT4, conc 8): ~300 prompt tok/s, ~23 gen tok/s, KV ~25% -> ~0.3 steps/s.
  Collection raised to concurrency 16. Realistic data scale on free tier: low thousands of verified steps per site.
- GitLab root cause (smoke v4 logs): puma "Peer authentication failed for user gitlab" - under proot setuid is
  simulated, so Postgres peer auth sees uid root. Pre-boot hook switches local pg_hba lines to trust.
- Quoting bug in both sh() helpers: `bash -lc {c!r}` let the OUTER bash expand $vars/$(...) whenever repr chose
  double quotes (old wrapper printed '' for a test that must print "inner single $x sub"). Now shlex.quote;
  tests/test_sh_quoting.py. Earlier diagnostics loops printing only headers were this bug.
- 30B-A3B shopping_admin (re-scored, PARTIAL: 107/182 episodes finished before the 3 h step cap): 0/107.
  Verified genuine, not a scoring bug: evaluator compares real wrong values (1 vs 6, 9 vs 3), 6 wrong task_type,
  85/107 hit the 30-step limit (mean 24.6 steps, 55k input tokens/task).
- Guard gap: launcher steps ran `cmd | tee | tail` without pipefail, so a REFUSE exit read as rc=0 (gitlab smoke v6
  preflight refused HTTP 500 yet stage said ok). Fixed (bash -o pipefail) + test. Earlier "ok" stages with
  results files were cross-checked by reading their outputs, not by rc; the smoke runs that matter were read by hand.
- GitLab PARKED: boots under proot after the pg_hba trust fix, logs in (Rails log: POST /users/sign_in 302,
  GET / 200 as byteblaze), but the same URLs flap between 200 and 500 and gitaly-ruby sidecar sockets fail to
  connect. Not reliable enough to score 180 tasks. Revisit after the 3 working sites have results.
- Map BLOCKED (external): all three map data tarballs on webarena-map-server-data.s3.amazonaws.com return
  403 AccessDenied (checked 2026-11-24 21:10 UTC with curl -I from the Mac and from a Kaggle kernel); no mirror
  found. Rebuilding tile/Nominatim/OSRM databases from raw OSM is out of scope on free CPU.
- **Evaluation scope = tasks whose sites are all in {shopping, shopping_admin, reddit}: 187+182+106+5 = 480/812
  (59%).** Every model in the table is scored on exactly this set; the table says so. GitLab (180+24) parked
  (flaky 500s under proot), map/wikipedia (109+17+2+6) blocked.
- Shopping collection (30B-A3B, wa-shopping-collect): 402 episodes (ran out of tasks), 86 program-pass, 86
  judge-pass, 291 verified steps (3.4 steps/episode, 82 RETRIEVE). Too little and too short.
  -> **Trajectory-first relabeling** (wa/relabel.py): every coherent synthetic teacher trajectory becomes a
  demonstration of the task it actually accomplished (teacher writes intent + per-step reasoning; verified by
  answer-on-page check + the same LLM judge). Refuses eval-run inputs (rule 1). Collect jobs now append a relabel
  step; standalone relabel jobs for finished collections. Leakage gate still runs over all intents in build_data.
- relabel_shopping (wa-relabel-shopping): 402 episodes in -> 16 relabeled (69 steps), all NAVIGATE, 3-9 steps,
  intents look natural. Yield low; drop-reason counters added for the next runs.
- Packed kernel outputs (out_*.tar + summary/) fixed the Kaggle API rate limit on downloads.
- Pipeline is now dependency-gated in the queue (launch.py "after"): relabel_reddit <- reddit collect;
  train_r0 (shopping+reddit data) <- relabel_reddit; 2B r0 evals <- train_r0. Gate fire-checked: exit 3,
  "DEPS PENDING: wa-reddit-collect is RUNNING", no kernel created.
- 2026-11-25 11:40 UTC **BLOCKED: Kaggle weekly GPU quota (30 h) exhausted**, account-wide (this project + the
  diff-attention project). Push error: "Maximum weekly GPU quota of 30.00 hours reached". Reset time undocumented.
  Queue restarted detached (QUEUE_HOURS=96, 30-min retry on quota). Already-running wa-shopping-admin-collect
  continues. wa-reddit-collect COMPLETE (old unpacked format; consumed by relabel via mount). See RESUME.md.
