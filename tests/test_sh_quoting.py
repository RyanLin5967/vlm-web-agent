import json, subprocess, sys
from wa.rootfs import sh
import kaggle.launch as L

def test_rootfs_sh_keeps_vars_and_quotes():
    rc, out = sh("x=inner && echo \"$x\" 'single $x' $(echo sub)")
    assert rc == 0 and out.strip() == "inner single $x sub", out

def test_kernel_sh_keeps_vars(tmp_path):
    src = L.KERNEL % {"job": json.dumps({"name": "x"})}
    head = src.split("T0 = time.time()")[0].replace('OUT = "/kaggle/working"', f'OUT = "{tmp_path}"')
    ns = {}; exec(head, ns)
    assert ns["sh"]("x=inner && test \"$x\" = inner && echo 'a $x' | grep -q 'a \\$x'") == 0
