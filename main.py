"""
Convenience entry point — runs both training + test pipelines sequentially.

    python main.py
"""
import subprocess
import sys

COMMANDS = [
    [sys.executable, "train.py", "--task", "detection", "--epochs", "5"],
    [sys.executable, "test.py",  "--task", "detection",
     "--ckpt", "checkpoints/best_detection.pt"],
    [sys.executable, "train.py", "--task", "subtype", "--epochs", "5"],
    [sys.executable, "test.py",  "--task", "subtype",
     "--ckpt", "checkpoints/best_subtype.pt"],
]

if __name__ == "__main__":
    for cmd in COMMANDS:
        print("\n$ " + " ".join(cmd))
        ret = subprocess.call(cmd)
        if ret != 0:
            sys.exit(ret)