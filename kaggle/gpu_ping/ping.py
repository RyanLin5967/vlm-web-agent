import subprocess
print(subprocess.run("nvidia-smi; nproc; free -g", shell=True, capture_output=True, text=True).stdout)
