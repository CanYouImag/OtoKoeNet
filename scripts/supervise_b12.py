import os
import subprocess
import time

ROOT = os.path.dirname(os.path.abspath(__file__)) + os.sep + ".."
BASE_LOG = os.path.join(ROOT, "log", "a1_base.log")


def is_done():
    if not os.path.exists(BASE_LOG):
        return False
    with open(BASE_LOG, encoding="utf-8", errors="replace") as f:
        return "done" in f.read()


while not is_done():
    time.sleep(120)

time.sleep(60)
wb = r"C:\Users\11831\PycharmProjects\WebLogin_Anomaly_Detection"
subprocess.Popen(["cmd", "/c", os.path.join(wb, "scripts", "run_b12_full.bat")],
                 cwd=wb)
print("b12 launched after base done")