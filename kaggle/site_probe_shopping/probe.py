"""Free-tier feasibility probe (runs on a Kaggle CPU kernel, $0).
Pull a WebArena-Verified image from Docker Hub without docker, unpack to a rootfs, and try to boot
the site under chroot, falling back to proot. Writes /kaggle/working/probe.json."""
import io, json, os, subprocess, tarfile, time, urllib.request, shutil

SITE = "shopping"
PORT, HEALTH = {"shopping_admin": (80, "/admin"), "shopping": (80, "/"), "reddit": (80, "/"), "gitlab": (8023, "/explore")}[SITE]
IMG = f"am1n3e/webarena-verified-{SITE}"
ROOT = "/tmp/rootfs"
R = {"img": IMG, "t0": time.time()}
def sh(c, t=600):
    p = subprocess.run(["bash", "-lc", f"timeout {t} bash -lc {c!r}"], capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-3000:]
def save():
    json.dump(R, open("/kaggle/working/probe.json", "w"), indent=1)

R["whoami"] = sh("whoami; nproc; free -g | head -2; df -h /tmp /kaggle/working | tail -2; grep Cap /proc/self/status")[1]
save()
tok = json.load(urllib.request.urlopen(f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{IMG}:pull"))["token"]
def get(path, accept=None):
    req = urllib.request.Request(f"https://registry-1.docker.io/v2/{IMG}/{path}", headers={"Authorization": f"Bearer {tok}"})
    if accept: req.add_header("Accept", accept)
    return urllib.request.urlopen(req, timeout=600)
acc = ("application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, "
       "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json")
m = json.load(get("manifests/latest", acc))
if "manifests" in m:
    d = [x for x in m["manifests"] if x.get("platform", {}).get("architecture") == "amd64"][0]["digest"]
    m = json.load(get(f"manifests/{d}", acc))
cfg = json.load(get(f"blobs/{m['config']['digest']}"))["config"]
R["layers"] = len(m["layers"]); R["compressed_gb"] = round(sum(l["size"] for l in m["layers"]) / 1e9, 2)
os.makedirs(ROOT, exist_ok=True)
for i, l in enumerate(m["layers"]):
    with get(f"blobs/{l['digest']}") as resp:
        with tarfile.open(fileobj=resp, mode="r|*") as tf:
            for mem in tf:
                name = mem.name.lstrip("./")
                base, leaf = os.path.split(name)
                if leaf == ".wh..wh..opq":
                    d = os.path.join(ROOT, base)
                    if os.path.isdir(d):
                        for x in os.listdir(d):
                            p = os.path.join(d, x); shutil.rmtree(p) if os.path.isdir(p) and not os.path.islink(p) else os.remove(p)
                    continue
                if leaf.startswith(".wh."):
                    p = os.path.join(ROOT, base, leaf[4:])
                    if os.path.islink(p) or os.path.isfile(p): os.remove(p)
                    elif os.path.isdir(p): shutil.rmtree(p)
                    continue
                if mem.ischr() or mem.isblk() or mem.isfifo(): continue
                dst = os.path.join(ROOT, name)
                if (os.path.lexists(dst) and not os.path.isdir(dst)) or (os.path.isdir(dst) and not mem.isdir() and not os.path.islink(dst)):
                    shutil.rmtree(dst) if os.path.isdir(dst) and not os.path.islink(dst) else os.remove(dst)
                try: tf.extract(mem, ROOT, set_attrs=True, numeric_owner=True)
                except Exception as e: R.setdefault("extract_errs", []).append(f"{name}: {e!r}"[:200])
    R["layers_done"] = i + 1; R["extract_s"] = round(time.time() - R["t0"]); save()
R["rootfs_du"] = sh(f"du -sh {ROOT}")[1]
for f in ("etc/resolv.conf", "etc/hosts"):
    shutil.copy(f"/{f}", os.path.join(ROOT, f))
entry = " ".join((cfg.get("Entrypoint") or []) + (cfg.get("Cmd") or []))
env = " ".join(f"{e.split('=',1)[0]}='{e.split('=',1)[1]}'" for e in cfg.get("Env", []))
R["entry"] = entry
R["chroot_true"] = sh(f"chroot {ROOT} /bin/true && echo CHROOT_OK")
R["mount_proc"] = sh(f"mount -t proc proc {ROOT}/proc && echo MOUNT_OK; mount --rbind /dev {ROOT}/dev && echo DEV_OK; mount --rbind /sys {ROOT}/sys && echo SYS_OK")
save()
mode = "chroot" if "CHROOT_OK" in R["chroot_true"][1] else "proot"
if mode == "proot":
    R["proot_get"] = sh("curl -sL -o /usr/local/bin/proot https://proot.gitlab.io/proot/bin/proot && chmod +x /usr/local/bin/proot && proot --version | head -2")
    runner = f"proot -0 -r {ROOT} -b /proc -b /dev -b /sys -w {cfg.get('WorkingDir') or '/'}"
else:
    runner = f"chroot {ROOT}"
R["mode"] = mode
log = open("/kaggle/working/site.log", "w")
proc = subprocess.Popen(["bash", "-lc", f"cd {ROOT}; env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root {env} {runner} /bin/sh -c 'cd {cfg.get('WorkingDir') or '/'}; exec {entry}'"],
                        stdout=log, stderr=subprocess.STDOUT)
t1 = time.time(); R["polls"] = []
while time.time() - t1 < 1500:
    rc, out = sh(f"curl -s -o /dev/null -w '%{{http_code}}' http://localhost:{PORT}{HEALTH}", 30)
    R["polls"].append([round(time.time() - t1), out.strip()]); R["polls"] = R["polls"][-20:]; save()
    if out.strip() in ("200", "302"):
        R["boot_s"] = round(time.time() - t1)
        rc, body = sh(f"curl -sL http://localhost:{PORT}{HEALTH} | grep -o '<title>[^<]*' | head -2", 60)
        R["admin_title"] = body; break
    if proc.poll() is not None:
        R["proc_exit"] = proc.returncode; break
    time.sleep(15)
R["services"] = sh(f"{runner} supervisorctl status 2>&1 | head -20", 60)
R["ps"] = sh("ps aux --sort=-%mem | head -15")
R["total_s"] = round(time.time() - R["t0"]); save()
proc.terminate()
