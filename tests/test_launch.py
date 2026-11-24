import json
import kaggle.launch as L

def test_kernel_template_renders_and_compiles():
    src = L.KERNEL % {"job": json.dumps({"name": "x", "steps": [{"cmd": "echo '%s'"}]})}
    compile(src, "kernel", "exec")
