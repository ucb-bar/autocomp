"""
Master comparison script: runs all 9 kernel comparison scripts in sequence.
Run from the nki-migration-testing directory:
    python compare_all.py
"""
import subprocess
import sys
import os

SCRIPTS = [
    "compare_cumsum.py",
    "compare_transpose.py",
    "compare_maxpool.py",
    "compare_rope.py",
    "compare_conv1d.py",
    "compare_conv2d.py",
    "compare_csa2048.py",
    "compare_csa16384.py",
    "compare_mha.py",
]

results = {}
for script in SCRIPTS:
    kernel = script.replace("compare_", "").replace(".py", "")
    print(f"\n{'='*60}")
    print(f"Running {script} ...")
    print(f"{'='*60}")
    ret = subprocess.run([sys.executable, script])
    results[kernel] = ret.returncode == 0

print(f"\n{'='*60}")
print("Summary:")
for kernel, passed in results.items():
    status = "PASS" if passed else "FAIL"
    print(f"  {kernel:20s}: {status}")
print(f"{'='*60}")

if not all(results.values()):
    sys.exit(1)
