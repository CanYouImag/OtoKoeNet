import os
import subprocess
import time

B12_LOG = r"C:\Users\11831\PycharmProjects\WebLogin_Anomaly_Detection\results\cicids_b12_full.log"
ROOT = r"C:\Users\11831\PycharmProjects\OtoKoeNet"


def b12_done():
    if not os.path.exists(B12_LOG):
        return False
    with open(B12_LOG, encoding="utf-8", errors="replace") as f:
        return "Full results:" in f.read()


while not b12_done():
    time.sleep(120)

time.sleep(60)
os.chdir(ROOT)
subprocess.Popen(["cmd", "/c", os.path.join(ROOT, "scripts", "run_train_gate03.bat")])
print("a1b gate03 launched after b12")