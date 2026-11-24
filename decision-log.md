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
