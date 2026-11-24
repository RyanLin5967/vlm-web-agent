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
