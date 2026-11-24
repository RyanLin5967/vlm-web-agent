"""Boot a WebArena-Verified site without Docker (Kaggle kernels have no Docker daemon).

Pull the image's layers from Docker Hub, apply them (with whiteouts) into a rootfs directory, then run the
image's ENTRYPOINT+CMD under chroot (root, with /proc,/dev,/sys bind-mounted) or, if chroot is not allowed,
proot. One site per kernel, on its native port."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import time
import urllib.request

SITES = {  # image suffix, port the site serves on inside the container, health path
    "shopping_admin": ("shopping_admin", 80, "/admin"),
    "shopping": ("shopping", 80, "/"),
    "reddit": ("reddit", 80, "/"),
    "gitlab": ("gitlab", 8023, "/explore"),
}
ACCEPT = ("application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, "
          "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json")


def sh(c: str, t: int = 600) -> tuple[int, str]:
    p = subprocess.run(["bash", "-lc", f"timeout {t} bash -lc {c!r}"], capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-3000:]


def _rm(p):
    if os.path.islink(p) or os.path.isfile(p):
        os.remove(p)
    elif os.path.isdir(p):
        shutil.rmtree(p)


def pull_rootfs(image: str, root: str, log=print) -> dict:
    tok = json.load(urllib.request.urlopen(
        f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{image}:pull"))["token"]

    def get(path, accept=None):
        req = urllib.request.Request(f"https://registry-1.docker.io/v2/{image}/{path}", headers={"Authorization": f"Bearer {tok}"})
        if accept:
            req.add_header("Accept", accept)
        return urllib.request.urlopen(req, timeout=900)

    m = json.load(get("manifests/latest", ACCEPT))
    if "manifests" in m:
        d = [x for x in m["manifests"] if x.get("platform", {}).get("architecture") == "amd64"][0]["digest"]
        m = json.load(get(f"manifests/{d}", ACCEPT))
    cfg = json.load(get(f"blobs/{m['config']['digest']}"))["config"]
    os.makedirs(root, exist_ok=True)
    for i, layer in enumerate(m["layers"]):
        with get(f"blobs/{layer['digest']}") as resp, tarfile.open(fileobj=resp, mode="r|*") as tf:
            for mem in tf:
                name = mem.name.lstrip("./")
                base, leaf = os.path.split(name)
                if leaf == ".wh..wh..opq":
                    d = os.path.join(root, base)
                    if os.path.isdir(d):
                        for x in os.listdir(d):
                            _rm(os.path.join(d, x))
                    continue
                if leaf.startswith(".wh."):
                    _rm(os.path.join(root, base, leaf[4:]))
                    continue
                if mem.ischr() or mem.isblk() or mem.isfifo():
                    continue
                dst = os.path.join(root, name)
                if os.path.lexists(dst) and not (os.path.isdir(dst) and not os.path.islink(dst) and mem.isdir()):
                    _rm(dst)
                try:
                    tf.extract(mem, root, set_attrs=True, numeric_owner=True)
                except Exception:  # noqa: BLE001
                    pass
        log(f"layer {i + 1}/{len(m['layers'])}")
    for f in ("etc/resolv.conf", "etc/hosts"):
        shutil.copy(f"/{f}", os.path.join(root, f))
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
        return f"http://localhost:{self.port}"

    def runner(self) -> str:
        if self.mode == "chroot":
            return f"chroot {self.root}"
        return f"proot -0 -r {self.root} -b /proc -b /dev -b /sys"

    def boot(self, log=print, deadline_s: int = 1800) -> float:
        t0 = time.time()
        cfg = pull_rootfs(self.image, self.root, log)
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
        wd = cfg.get("WorkingDir") or "/"
        self.log = open(f"/kaggle/working/site_{self.name}.log", "w") if os.path.isdir("/kaggle/working") else subprocess.DEVNULL
        self.proc = subprocess.Popen(
            ["bash", "-lc", f"env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root {env} "
                            f"{self.runner()} /bin/sh -c 'cd {wd}; exec {entry}'"], stdout=self.log, stderr=subprocess.STDOUT)
        while time.time() - t0 < deadline_s:
            _, code = sh(f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{self.port}{self.health}", 30)
            if code.strip() in {"200", "302"}:
                break
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} entrypoint exited {self.proc.returncode}")
            time.sleep(10)
        else:
            raise RuntimeError(f"{self.name} not healthy after {deadline_s}s")
        rc, out = sh(f"{self.runner()} env-ctrl init --base-url {self.base_url}/ 2>&1 | tail -3", 900)
        log(f"env-ctrl init rc={rc}: {out[-300:]}")
        return time.time() - t0

    def config_entry(self) -> tuple[str, dict]:
        key = f"__{self.name.upper()}__"
        url = self.base_url + ("/admin" if self.name == "shopping_admin" else "")
        return key, {"urls": [url]}
