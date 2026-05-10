#!/usr/bin/env python3
"""Batch runner to repeatedly call new_testing.py with offsets.

Usage example:
  python batch_run.py 0 100 5

This will run `new_testing.py` for offsets [0-100), then wait 5 minutes,
then run [100-200), etc., until the end reaches 3720 (by default).
"""
import argparse
import subprocess
import time
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Batch-run new_testing.py over offset ranges")
    p.add_argument("x", type=int, help="start offset (integer)")
    p.add_argument("y", type=int, help="limit per run (integer)")
    p.add_argument("z", type=float, help="wait time between runs (minutes)")
    p.add_argument("--script", default="new_testing.py", help="path to the target script")
    p.add_argument("--max-end", type=int, default=3720, help="stop when end reaches this value")
    p.add_argument("--python", default=sys.executable, help="python executable to use")
    p.add_argument("--extra", nargs=argparse.REMAINDER, help="extra args appended to the called script")
    return p.parse_args()


def run_batch(start: int, limit: int, csv_name: str, python_exe: str, script_path: Path, extra_args):
    cmd = [python_exe, str(script_path), "--offset", str(start), "--limit", str(limit), f"--csv={csv_name}"]
    if extra_args:
        cmd += extra_args
    print("Running:", " ".join(cmd))
    res = subprocess.run(cmd)
    return res.returncode


def main():
    args = parse_args()
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"Error: script not found: {script_path}")
        sys.exit(2)

    start = args.x
    chunk = args.y
    wait_seconds = args.z * 60.0
    max_end = args.max_end

    if start < 0 or chunk <= 0 or wait_seconds < 0:
        print("Invalid numeric arguments: ensure x>=0, y>0, z>=0")
        sys.exit(2)

    while start < max_end:
        end = min(start + chunk, max_end)
        limit = end - start
        csv_name = f"CSV_results/{start}-{end}.csv"

        rc = run_batch(start, limit, csv_name, args.python, script_path, args.extra)
        if rc != 0:
            print(f"Run failed with exit code {rc}. Stopping.")
            sys.exit(rc)

        if end >= max_end:
            print("Reached target end. All done.")
            break

        print(f"Sleeping {args.z} minutes before next run...")
        time.sleep(wait_seconds)
        start = end


if __name__ == "__main__":
    main()
