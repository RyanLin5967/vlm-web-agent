"""Free-tier GPU probe (Kaggle T4x2, $0): does vLLM serve the student (Qwen3.5-2B) and the open teacher
(Qwen3.5-35B-A3B GPTQ-Int4) on Turing, and at what latency for a screenshot+text step? Writes gpu_probe.json."""
import json, os, subprocess, sys, time, traceback
R = {}
OUT = "/kaggle/working/gpu_probe.json"
def save(): json.dump(R, open(OUT, "w"), indent=1)
def sh(c, t=1800):
    p = subprocess.run(["bash", "-lc", f"timeout {t} bash -lc {c!r}"], capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-2500:]
R["smi"] = sh("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv; nproc; free -g|head -2; df -h /kaggle/working /tmp|tail -2")[1]; save()
t = time.time(); R["pip"] = sh("pip install -q -U vllm 2>&1 | tail -5; python -c 'import vllm,torch;print(vllm.__version__,torch.__version__)'", 2400); R["pip_s"] = round(time.time()-t); save()

CHILD = r'''
import json, sys, time
from PIL import Image, ImageDraw
from vllm import LLM, SamplingParams
model, tp, extra = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3])
img = Image.new("RGB", (1280, 720), "white"); d = ImageDraw.Draw(img)
for i in range(40):
    x, y = 30 + (i % 8) * 155, 30 + (i // 8) * 130
    d.rectangle([x, y, x + 140, y + 40], outline="red", width=2); d.text((x + 4, y + 4), f"[{i}] button {i}", fill="black")
els = "\n".join(f"[{i}] button 'button {i}'" for i in range(40))
msg = [{"role": "user", "content": [{"type": "image_pil", "image_pil": img},
        {"type": "text", "text": "Task: click button 17.\nElements:\n" + els +
         '\nReply with one reasoning line then JSON like {"action":"click","id":3}.'}]}]
t0 = time.time()
llm = LLM(model=model, tensor_parallel_size=tp, dtype="float16", max_model_len=8192, gpu_memory_utilization=0.88,
          limit_mm_per_prompt={"image": 1}, enforce_eager=False, **extra)
load_s = time.time() - t0
sp = SamplingParams(max_tokens=96, temperature=0)
llm.chat(msg, sp)  # warmup
t1 = time.time(); o = llm.chat(msg, sp); single_s = time.time() - t1
t2 = time.time(); outs = llm.chat([msg] * 16, sp); batch16_s = time.time() - t2
print("RESULT " + json.dumps({"load_s": round(load_s), "single_s": round(single_s, 2), "batch16_s": round(batch16_s, 2),
      "prompt_tokens": len(o[0].prompt_token_ids), "text": o[0].outputs[0].text[:300]}))
'''
open("/tmp/child.py", "w").write(CHILD)
for name, model, tp, extra in [("student", "Qwen/Qwen3.5-2B", 1, {}),
                               ("teacher", "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", 2, {}),
                               ("teacher_alt", "Qwen/Qwen3-VL-30B-A3B-Instruct", 2, {"quantization": "fp8"})]:
    if name == "teacher_alt" and "RESULT" in str(R.get("teacher", "")): continue
    t = time.time(); rc, out = sh(f"python /tmp/child.py {model} {tp} '{json.dumps(extra)}' 2>&1", 2400)
    line = [l for l in out.splitlines() if l.startswith("RESULT ")]
    R[name] = {"model": model, "rc": rc, "wall_s": round(time.time()-t), **(json.loads(line[-1][7:]) if line else {"tail": out[-1800:]})}
    save()
