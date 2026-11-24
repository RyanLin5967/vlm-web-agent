"""GPU probe 2 (free T4x2): full logs + root causes for vLLM start failures, and fallbacks.
Each variant writes its whole log to /kaggle/working/<name>.log; summary in gpu_probe2.json."""
import json, os, re, subprocess, time
R = {}; W = "/kaggle/working"
def save(): json.dump(R, open(f"{W}/gpu_probe2.json", "w"), indent=1)
def run(name, cmd, t=1800):
    t0 = time.time()
    with open(f"{W}/{name}.log", "w") as f:
        rc = subprocess.run(["bash", "-lc", f"timeout {t} {cmd}"], stdout=f, stderr=subprocess.STDOUT).returncode
    txt = open(f"{W}/{name}.log").read()
    errs = [l for l in txt.splitlines() if re.search(r"Error|error:|not supported|requires|capability|CUDA|Triton|OOM|out of memory", l)][:25]
    res = [l for l in txt.splitlines() if l.startswith("RESULT ")]
    R[name] = {"rc": rc, "wall_s": round(time.time() - t0), "result": json.loads(res[-1][7:]) if res else None, "error_lines": errs}
    save()
run("pip", "pip install -q -U vllm 2>&1 | tail -3; python -c 'import vllm,torch,transformers;print(vllm.__version__,torch.__version__,transformers.__version__)'", 2400)
open("/tmp/child.py", "w").write(r'''
import json, sys, time
from PIL import Image, ImageDraw
from vllm import LLM, SamplingParams
model, tp, extra = sys.argv[1], int(sys.argv[2]), json.load(open(sys.argv[3]))
img = Image.new("RGB", (1280, 720), "white"); d = ImageDraw.Draw(img)
for i in range(40):
    x, y = 30 + (i % 8) * 155, 30 + (i // 8) * 130
    d.rectangle([x, y, x + 140, y + 40], outline="red", width=2); d.text((x + 4, y + 4), f"[{i}] button {i}", fill="black")
els = "\n".join(f"[{i}] button 'button {i}'" for i in range(40))
msg = [{"role": "user", "content": [{"type": "image_pil", "image_pil": img}, {"type": "text", "text": "Task: click button 17.\nElements:\n" + els + '\nReply with one reasoning line then JSON like {"action":"click","id":3}.'}]}]
t0 = time.time()
llm = LLM(model=model, tensor_parallel_size=tp, dtype="float16", max_model_len=8192, gpu_memory_utilization=0.88, limit_mm_per_prompt={"image": 1}, **extra)
load_s = time.time() - t0
sp = SamplingParams(max_tokens=96, temperature=0)
kw = {"chat_template_kwargs": {"enable_thinking": False}} if "3.5" in model else {}
llm.chat(msg, sp, **kw)
t1 = time.time(); o = llm.chat(msg, sp, **kw); single = time.time() - t1
t2 = time.time(); llm.chat([msg] * 16, sp, **kw); b16 = time.time() - t2
print("RESULT " + json.dumps({"load_s": round(load_s), "single_s": round(single, 2), "batch16_s": round(b16, 2), "prompt_tokens": len(o[0].prompt_token_ids), "text": o[0].outputs[0].text[:300]}))
''')
def vllm_variant(name, model, tp, extra, env=""):
    json.dump(extra, open(f"/tmp/{name}.json", "w"))
    run(name, f"env {env} python /tmp/child.py {model} {tp} /tmp/{name}.json", 1500)
vllm_variant("s35_eager", "Qwen/Qwen3.5-2B", 1, {"enforce_eager": True})
vllm_variant("s3vl_2b", "Qwen/Qwen3-VL-2B-Instruct", 1, {"enforce_eager": True})
open("/tmp/hf.py", "w").write(r'''
import json, sys, time, torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor
m = sys.argv[1]
proc = AutoProcessor.from_pretrained(m); model = AutoModelForImageTextToText.from_pretrained(m, torch_dtype=torch.float16, device_map={"": 0})
img = Image.new("RGB", (1280, 720), "white"); d = ImageDraw.Draw(img)
for i in range(40):
    x, y = 30 + (i % 8) * 155, 30 + (i // 8) * 130; d.rectangle([x, y, x + 140, y + 40], outline="red", width=2); d.text((x + 4, y + 4), f"[{i}]", fill="black")
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": 'Task: click button 17. Reply with one reasoning line then JSON like {"action":"click","id":3}.'}]}]
kw = {"enable_thinking": False} if "3.5" in m else {}
text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **kw)
inp = proc(text=[text], images=[img], return_tensors="pt").to(0)
with torch.no_grad():
    model.generate(**inp, max_new_tokens=8)
    t = time.time(); out = model.generate(**inp, max_new_tokens=64, do_sample=False); s = time.time() - t
    loss = model(**inp, labels=inp["input_ids"]).loss.item()
print("RESULT " + json.dumps({"gen64_s": round(s, 2), "prompt_tokens": inp["input_ids"].shape[1], "fp16_loss_finite": loss == loss and abs(loss) < 1e4, "loss": loss,
      "text": proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0][:200]}))
''')
run("hf_s35", "python /tmp/hf.py Qwen/Qwen3.5-2B", 1200)
run("smi_after", "nvidia-smi --query-gpu=memory.used --format=csv")
