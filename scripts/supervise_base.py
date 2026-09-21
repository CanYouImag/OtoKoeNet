import os
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__)) + os.sep + ".."
GATE_LOG = os.path.join(ROOT, "log", "a1_gate.log")


def gate_done():
    if not os.path.exists(GATE_LOG):
        return False
    with open(GATE_LOG, encoding="utf-8", errors="replace") as f:
        text = f.read()
    return "done" in text


while True:
    if gate_done():
        break
    time.sleep(60)

time.sleep(30)
os.chdir(ROOT)
subprocess.Popen(["cmd", "/c", r"scripts\run_train_base.bat"])
print("base launched after gate done")