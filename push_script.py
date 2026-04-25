import os
if "HF_TOKEN" in os.environ:
    del os.environ["HF_TOKEN"]

import huggingface_hub
try:
    huggingface_hub.login(token="hf_nVKzvZEhRPAkciiKsbLiYSzppLQoEqtZwqw", add_to_git_credential=False)
    print("HF login successful")
except Exception as e:
    print("HF login failed:", e)

import subprocess
env = os.environ.copy()
env['PYTHONIOENCODING'] = 'utf-8'
print("Running openenv push...")
res = subprocess.run(
    ["uv", "run", "openenv", "push", "--repo-id", "Ponmurugaiya72/broadbandcare-env"], 
    capture_output=True, 
    text=True, 
    env=env,
    encoding="utf-8",
    errors="replace"
)
print("CODE:", res.returncode)
print("OUT:", res.stdout[:2000])
print("ERR:", res.stderr[:2000])
