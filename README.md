# vlm-web-agent

A ~2B vision-language web agent (Qwen3.5-2B) trained only on self-collected, verified screenshot→action
trajectories from synthetic tasks on self-hosted WebArena sites, evaluated on all 812
[WebArena-Verified](https://github.com/ServiceNow/webarena-verified) tasks in one shared harness against a
10x+ larger open VLM (Qwen3.5-27B).

**Status: in progress.** No results yet. Every number will come from a results JSON written by a script.

## Layout
- `wa/obs.py`: 1280x720 screenshot with set-of-marks overlay, plus element list and visible text
- `wa/actions.py`: action space (click/type/select/scroll/go_back/goto/stop), parser, executor
- `wa/policy.py`: the only per-model part. Identical messages go to any OpenAI-compatible endpoint (vLLM)
- `wa/episode.py`: 30-step episode loop. Writes `agent_response.json` + `network.har` for the official evaluator
- `wa/taskgen.py`: crawl sites, teacher proposes grounded tasks with programmatic checks
- `wa/verify.py`: programmatic check plus LLM judge for synthetic episodes
- `wa/leakage.py`: fails the build if any training intent, template or answer matches the 812 test tasks
  (exact, template regex, MiniLM cosine > 0.9, answer match)
- `wa/dataset.py`, `wa/sft.py`: SFT rows and LoRA trainer
- `kaggle/`: free-tier Kaggle kernels (all compute runs there)
- `decision-log.md`: design decisions and why

## Rules
- Test isolation: synthetic tasks only, with the leakage check above.
- Same harness for every model: same observation, actions, 30-step limit, viewport and evaluator.

## Attribution
`data/webarena-verified.json` and `wa/prompts/*.md` come from ServiceNow/webarena-verified (Apache-2.0).
The source commit and hash are in `data/webarena-verified.SOURCE`.
