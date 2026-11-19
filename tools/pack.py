"""Build a self-contained submission: agent.py + shared modules + data files, zipped.

    python -m tools.pack agents/v5 --name v5

Writes submissions/<name>/ and submissions/<name>.zip, then plays 4 games with the official
catan_kit.evaluate on the packed folder to prove it loads alone.
"""
import argparse
import os
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMON = os.path.join(ROOT, "agents", "common")
BANNED = (".so", ".pyd", ".dll", ".dylib", ".exe", ".pyc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--name", required=True)
    ap.add_argument("--games", type=int, default=4)
    a = ap.parse_args()
    out = os.path.join(ROOT, "submissions", a.name)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    for d in (COMMON, a.src):
        for f in os.listdir(d):
            p = os.path.join(d, f)
            if os.path.isfile(p) and not f.endswith(BANNED):
                shutil.copy(p, out)
    shutil.copy(os.path.join(ROOT, "engine", "obs_np.py"), out)   # pure numpy encoder
    z = out + ".zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(os.listdir(out)):
            zf.write(os.path.join(out, f), f)
    print(f"{z}: {os.path.getsize(z)/1e6:.2f} MB, files {sorted(os.listdir(out))}")
    subprocess.run([sys.executable, "-m", "catan_kit.evaluate", "--a", os.path.join(out, "agent.py"),
                    "--b", "examples/f_agent.py", "--games", str(a.games)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
