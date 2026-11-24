"""Probe v2 (free CPU kernel): why do mysqld/nginx/php-fpm/elasticsearch die under chroot without /proc,/dev?
A: chroot + mknod'd /dev nodes, run each service's own config test / foreground start, capture stderr.
B: proot with -b /proc -b /dev -b /sys, full boot, health + timed page loads."""
import json, os, subprocess, sys, time
R = {}; OUT = "/kaggle/working/probe2.json"
def save(): json.dump(R, open(OUT, "w"), indent=1)
def sh(c, t=600):
    p = subprocess.run(["bash", "-lc", f"timeout {t} bash -lc {c!r}"], capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-2500:]
subprocess.run("git clone -q -b dev https://github.com/RyanLin5967/vlm-web-agent.git /kaggle/repo", shell=True, check=True)
sys.path.insert(0, "/kaggle/repo")
from wa.rootfs import pull_rootfs
ROOT = "/tmp/rootfs"; t = time.time()
cfg = pull_rootfs("am1n3e/webarena-verified-shopping_admin", ROOT, log=lambda m: None)
R["pull_s"] = round(time.time() - t); R["cfg"] = {k: cfg.get(k) for k in ("Entrypoint", "Cmd", "WorkingDir")}
for f in ("etc/resolv.conf", "etc/hosts"): pass
# A: device nodes
R["mknod"] = sh(f"""cd {ROOT}/dev && for d in 'null 1 3' 'zero 1 5' 'full 1 7' 'random 1 8' 'urandom 1 9' 'tty 5 0'; do set -- $d; rm -f $1; mknod -m 666 $1 c $2 $3; done; mkdir -p shm pts; chmod 1777 shm; ls -l {ROOT}/dev; echo x > {ROOT}/dev/null && echo DEVNULL_OK""")
R["sup_conf"] = sh(f"cat {ROOT}/etc/supervisor/conf.d/*.conf {ROOT}/etc/supervisord.conf 2>/dev/null | grep -E '^\\[program|command|user|stdout_logfile|stderr' | head -60")[1]
R["entry_sh"] = sh(f"cat {ROOT}/custom-entrypoint.sh | head -80")[1]
C = f"chroot {ROOT}"
R["A_nginx_t"] = sh(f"{C} nginx -t")
R["A_phpfpm_t"] = sh(f"{C} sh -c 'php-fpm -t || /usr/local/sbin/php-fpm -t'")
R["A_mysqld"] = sh(f"{C} sh -c 'ls -ld /var/lib/mysql /var/run/mysqld; mysqld --user=mysql --version; timeout 20 mysqld --user=mysql 2>&1 | tail -20'", 60)
R["A_es"] = sh(f"{C} sh -c 'grep -A3 elasticsearch /etc/supervisor/conf.d/*.conf | head; ls /usr/share/elasticsearch/bin 2>&1 | head; which java'")
save()
# B: proot
R["proot_get"] = sh("curl -sL -o /usr/local/bin/proot https://proot.gitlab.io/proot/bin/proot && chmod +x /usr/local/bin/proot && proot --version | head -3")
env = " ".join(f"{e.split('=',1)[0]}='{e.split('=',1)[1]}'" for e in cfg.get("Env", []))
entry = " ".join((cfg.get("Entrypoint") or []) + (cfg.get("Cmd") or []))
P = f"proot -0 -r {ROOT} -b /proc -b /dev -b /sys -w {cfg.get('WorkingDir') or '/'}"
log = open("/kaggle/working/site2.log", "w")
proc = subprocess.Popen(["bash", "-lc", f"env -i PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin HOME=/root {env} {P} /bin/sh -c 'exec {entry}'"], stdout=log, stderr=subprocess.STDOUT)
t1 = time.time(); R["B_polls"] = []
while time.time() - t1 < 1500:
    rc, out = sh("curl -s -o /dev/null -w '%{http_code}' http://localhost:80/admin", 60)
    R["B_polls"].append([round(time.time() - t1), out.strip()]); R["B_polls"] = R["B_polls"][-10:]; save()
    if out.strip() in ("200", "302"):
        R["B_boot_s"] = round(time.time() - t1); break
    if proc.poll() is not None: R["B_exit"] = proc.returncode; break
    time.sleep(15)
R["B_services"] = sh(f"{P} supervisorctl status", 60)
if "B_boot_s" in R:
    R["B_init"] = sh(f"{P} env-ctrl init --base-url http://localhost:80/ 2>&1 | tail -5", 900)
    times = []
    for path in ["/admin", "/admin/sales/order/", "/admin/catalog/product/", "/admin/customer/index/", "/admin/reports/report_sales/bestsellers/"]:
        rc, out = sh(f"curl -sL -o /tmp/pg.html -w '%{{http_code}} %{{time_total}}' -c /tmp/cj -b /tmp/cj -H 'X-M2-Admin-Auto-Login: admin:admin1234' http://localhost:80{path}; grep -o '<title>[^<]*' /tmp/pg.html | head -1", 180)
        times.append([path, out.strip()])
    R["B_page_times"] = times
R["B_ps"] = sh("ps aux --sort=-%cpu | head -12")[1]
R["total_s"] = round(time.time() - t); save()
