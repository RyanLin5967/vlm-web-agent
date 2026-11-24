"""Generate + push a Kaggle kernel for one job. Code reaches the kernel by cloning this public repo at a
pinned commit, so a run is reproducible from its commit + job JSON (both saved in the output).

usage: python kaggle/launch.py JOB.json        (JOB: see kaggle/jobs/*.json)
Job keys: name, site, mode ("eval"|"collect"), model, vllm_args, run_args, gpu (bool), commit (default: HEAD of origin/dev)"""
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
REPO = "https://github.com/RyanLin5967/vlm-web-agent.git"

KERNEL = r'''
import json, os, subprocess, sys, time, traceback
JOB = json.loads(%(job)r)
OUT = "/kaggle/working"
def sh(c, t=3600):
    print("+", c[:200], flush=True)
    r = subprocess.run(["bash", "-lc", f"timeout {t} bash -lc {c!r}"], capture_output=True, text=True)
    print((r.stdout + r.stderr)[-2000:], flush=True); return r.returncode
status = {"job": JOB, "stages": {}}
def stage(n, ok, **kw):
    status["stages"][n] = {"ok": ok, "t": round(time.time() - T0), **kw}; json.dump(status, open(f"{OUT}/status.json", "w"), indent=1)
T0 = time.time()
try:
    assert sh(f"git clone -q {JOB['repo']} /kaggle/repo && cd /kaggle/repo && git checkout -q {JOB['commit']}") == 0
    sys.path.insert(0, "/kaggle/repo"); os.chdir("/kaggle/repo")
    sh("pip install -q pillow playwright webarena-verified 2>&1 | tail -2 && python -m playwright install --with-deps chromium 2>&1 | tail -1", 1800)
    if JOB.get("gpu"):
        sh("pip install -q -U vllm 2>&1 | tail -2", 2400)
    stage("setup", True)
    from wa.rootfs import Site
    site = Site(JOB["site"]); boot_s = site.boot(log=lambda m: print(m, flush=True))
    stage("site", True, boot_s=round(boot_s), mode=site.mode)
    k, v = site.config_entry()
    json.dump({"environments": {k: v}}, open("/kaggle/wa_config.json", "w"))
    if JOB.get("gpu"):
        vl = open(f"{OUT}/vllm.log", "w")
        subprocess.Popen(["bash", "-lc", f"vllm serve {JOB['model']} --port 8000 {JOB.get('vllm_args', '')}"], stdout=vl, stderr=subprocess.STDOUT)
        for _ in range(240):
            if sh("curl -sf localhost:8000/v1/models > /dev/null", 20) == 0: break
            time.sleep(10)
        else:
            raise RuntimeError("vllm did not come up")
        stage("vllm", True)
    args = JOB.get("run_args", "")
    rc = sh(f"python -m wa.run {JOB['mode']} --config /kaggle/wa_config.json --sites {JOB['site']} --model {JOB['model']} "
            f"--endpoint {JOB.get('endpoint', 'http://localhost:8000/v1')} --out {OUT}/out {args} 2>&1 | tee {OUT}/run.log | tail -40", 11 * 3600)
    stage("run", rc == 0, rc=rc)
except Exception as e:
    status["error"] = traceback.format_exc()[-3000:]; json.dump(status, open(f"{OUT}/status.json", "w"), indent=1); print(status["error"])
finally:
    sh(f"cd {OUT} && tar czf runs_trimmed.tgz --exclude='*.har' out 2>/dev/null; du -sh {OUT}/* | tail -20")
'''


def main():
    job = json.loads(pathlib.Path(sys.argv[1]).read_text())
    job.setdefault("repo", REPO)
    if "commit" not in job:
        subprocess.run(["git", "fetch", "-q", "origin"], check=True)
        job["commit"] = subprocess.check_output(["git", "rev-parse", "origin/dev"], text=True).strip()
    slug = "wa-" + job["name"].replace("_", "-")
    d = HERE / "kernels" / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(KERNEL % {"job": json.dumps(job)})
    meta = {"id": f"ryanlin5967/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
            "is_private": True, "enable_gpu": bool(job.get("gpu")), "enable_internet": True,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
    if job.get("gpu"):
        meta["machine_shape"] = "NvidiaTeslaT4"
    (d / "kernel-metadata.json").write_text(json.dumps(meta))
    env = dict(os.environ, KAGGLE_API_TOKEN=pathlib.Path("~/.config/kaggle/token").expanduser().read_text().strip())
    r = subprocess.run(["uvx", "--quiet", "--with", "six", "kaggle", "kernels", "push", "-p", str(d)], env=env,
                       capture_output=True, text=True, timeout=180)
    print((r.stdout + r.stderr).strip().splitlines()[-1], "commit", job["commit"][:10])


if __name__ == "__main__":
    main()
