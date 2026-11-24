"""Boot a WebArena-Verified site without Docker (Kaggle kernels have no Docker daemon).

Pull the image's layers from Docker Hub, apply them (with whiteouts) into a rootfs directory, then run the
image's ENTRYPOINT+CMD under chroot (root, with /proc,/dev,/sys bind-mounted) or, if chroot is not allowed,
proot. One site per kernel, on its native port."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

SITES = {  # image suffix, port the site serves on inside the container, health path
    "shopping_admin": ("shopping_admin", 80, "/admin"),
    "shopping": ("shopping", 80, "/"),
    "reddit": ("reddit", 80, "/"),
    "gitlab": ("gitlab", 8023, "/explore"),
}


def sh(c: str, t: int = 600) -> tuple[int, str]:
    p = subprocess.run(["bash", "-lc", f"timeout {t} bash -lc {c!r}"], capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-3000:]


def _rm(p):
    if os.path.islink(p) or os.path.isfile(p):
        os.remove(p)
    elif os.path.isdir(p):
        shutil.rmtree(p)


CRANE_URL = "https://github.com/google/go-containerregistry/releases/latest/download/go-containerregistry_Linux_x86_64.tar.gz"


def pull_rootfs(image: str, root: str, log=print) -> dict:
    """Flatten the image with `crane export` (correct whiteout semantics) and untar it into root.
    A hand-rolled streaming extractor lost files here: an opaque-whiteout marker met mid-layer also deleted that
    same layer's files (shopping_admin rootfs came out 1.7 GB, missing nginx/php-fpm configs and InnoDB files)."""
    if sh("command -v crane")[0] != 0:
        rc, out = sh(f"curl -sL {CRANE_URL} | tar -xz -C /usr/local/bin crane && chmod +x /usr/local/bin/crane", 300)
        if rc != 0:
            raise RuntimeError(f"crane download failed: {out}")
    os.makedirs(root, exist_ok=True)
    rc, out = sh(f"crane export --platform linux/amd64 {image}:latest - | tar -x -p --numeric-owner -C {root}", 3600)
    if rc != 0:
        raise RuntimeError(f"crane export failed: {out}")
    cfg = json.loads(subprocess.check_output(["crane", "config", "--platform", "linux/amd64", f"{image}:latest"], text=True))["config"]
    for f in ("etc/resolv.conf", "etc/hosts"):
        try:
            os.remove(os.path.join(root, f))
        except FileNotFoundError:
            pass
        shutil.copy(f"/{f}", os.path.join(root, f))
    log(f"rootfs {sh(f'du -sh {root}')[1].split()[0]}")
    return cfg


class Site:
    def __init__(self, name: str, root: str = "/tmp/rootfs"):
        self.name, self.root = name, root
        img, self.port, self.health = SITES[name]
        self.image = f"am1n3e/webarena-verified-{img}"
        self.proc = None
        self.mode = None

    @property
    def base_url(self) -> str:
        # browsers drop a default :80, so the HAR records http://localhost/...; the evaluator's URL regexes are built
        # from this string, so it must match exactly (an explicit :80 scored every network event as missing)
        return "http://localhost" if self.port == 80 else f"http://localhost:{self.port}"

    def runner(self) -> str:
        if self.mode == "chroot":
            return f"chroot {self.root}"
        return f"proot -0 -r {self.root} -b /proc -b /dev -b /sys"

    def boot(self, log=print, deadline_s: int = 1500) -> float:
        t0 = time.time()
        cfg = pull_rootfs(self.image, self.root, log)
        self.pull_s = time.time() - t0
        t_boot = time.time()  # health deadline counts boot only, not the download (gitlab: 31.6 GB)
        # chroot alone is not enough: services need /proc and /dev inside the root. Kaggle allows chroot but not
        # mount (no CAP_SYS_ADMIN; measured 2026-11-24, kaggle/site_probe), so the probe is "can /proc be mounted".
        ok = sh(f"chroot {self.root} /bin/true && mount -t proc proc {self.root}/proc && echo PROC_OK")[1]
        if "PROC_OK" in ok:
            self.mode = "chroot"
            sh(f"mount --rbind /dev {self.root}/dev; mount --rbind /sys {self.root}/sys")
        else:
            self.mode = "proot"
            sh("command -v proot || (curl -sL -o /usr/local/bin/proot https://proot.gitlab.io/proot/bin/proot && chmod +x /usr/local/bin/proot)")
        entry = " ".join((cfg.get("Entrypoint") or []) + (cfg.get("Cmd") or []))
        env = " ".join(f"{e.split('=', 1)[0]}='{e.split('=', 1)[1]}'" for e in cfg.get("Env", []))
        self.env_prefix = f"env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root {env}"
        wd = cfg.get("WorkingDir") or "/"
        self.log = open(f"/kaggle/working/site_{self.name}.log", "w") if os.path.isdir("/kaggle/working") else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            ["bash", "-lc", f"{self.env_prefix} {self.runner()} /bin/sh -c 'cd {wd}; exec {entry}'"], stdout=self.log, stderr=subprocess.STDOUT)
        while time.time() - t_boot < deadline_s:
            _, code = sh(f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{self.port}{self.health}", 30)
            if code.strip() in {"200", "302"}:
                break
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} entrypoint exited {self.proc.returncode}\n{self.diagnose()}")
            time.sleep(10)
        else:
            raise RuntimeError(f"{self.name} not healthy after {deadline_s}s\n{self.diagnose()}")
        # env-ctrl reads WA_ENV_CTRL_TYPE etc. from the image env: run it with the same env as the entrypoint
        rc, out = sh(f"{self.env_prefix} {self.runner()} env-ctrl init --base-url {self.base_url}/ 2>&1 | tail -5", 900)
        log(f"env-ctrl init rc={rc}: {out[-500:]}")
        if rc != 0 or "Error" in out:
            raise RuntimeError(f"env-ctrl init failed rc={rc}: {out[-1500:]}")
        return time.time() - t0

    def selfcheck(self) -> dict:
        """Facts a reader needs when the browser cannot reach the site: listeners, per-address curl, redirects."""
        chk = {"listen": sh("ss -ltnp 2>/dev/null | head -30 || netstat -ltnp | head -30")[1]}
        for host in ("localhost", "127.0.0.1", "[::1]"):
            chk[host] = sh(f"curl -s -o /dev/null -w '%{{http_code}} -> %{{redirect_url}}' http://{host}:{self.port}{self.health}", 60)[1]
        chk["redirect_chain"] = sh(f"curl -sIL -o /dev/null -w '%{{url_effective}} %{{http_code}}' --max-redirs 10 {self.base_url}{self.health}", 60)[1]
        return chk

    def diagnose(self) -> str:
        out = [sh(f"{self.runner()} sh -c 'supervisorctl status || gitlab-ctl status'", 60)[1],
               sh(f"curl -s -m 30 -o /dev/null -w '%{{http_code}}' http://localhost:{self.port}{self.health}", 60)[1]]
        out.append(sh(f"for f in {self.root}/var/log/supervisor/*.log {self.root}/var/log/*.log {self.root}/var/log/*/*.log; do "
                      f"echo == $f; tail -n 15 $f; done 2>/dev/null | tail -n 150", 60)[1])
        out.append(sh(f"for f in {self.root}/var/log/gitlab/puma/current {self.root}/var/log/gitlab/puma/puma_stderr.log "
                      f"{self.root}/var/log/gitlab/gitlab-rails/application_json.log {self.root}/var/log/gitlab/sidekiq/current; do "
                      f"echo == $f; tail -n 25 $f; done 2>/dev/null; free -g; dmesg 2>/dev/null | tail -5", 60)[1])
        full = "\n".join(out)
        if os.path.isdir("/kaggle/working"):  # untruncated copy; the exception message keeps only the tail
            with open(f"/kaggle/working/diagnose_{self.name}.txt", "w") as f:
                f.write(full)
            sh(f"cp -r {self.root}/var/log/gitlab /kaggle/working/gitlab_logs 2>/dev/null; "
               f"find /kaggle/working/gitlab_logs -size +5M -delete 2>/dev/null")
        return full[-12000:]

    def config_entry(self) -> tuple[str, dict]:
        key = f"__{self.name.upper()}__"
        url = self.base_url + ("/admin" if self.name == "shopping_admin" else "")
        return key, {"urls": [url]}
