#!/usr/bin/env python3
"""Profile compilation performance between main and joint_derivative branches."""

import subprocess
import sys
import os
import time
from pathlib import Path


def run_with_compile_count(script_path: str, branch: str) -> dict:
    """Run a script and count JAX compilations."""
    print(f"\n{'='*60}")
    print(f"Running on branch: {branch}")
    print(f"{'='*60}\n")

    # Set environment variable to log compilations
    env = os.environ.copy()
    env['JAX_LOG_COMPILES'] = '1'

    start_time = time.time()

    # Run the script and capture output
    result = subprocess.run(
        [sys.executable, script_path],
        env=env,
        capture_output=True,
        text=True,
        cwd=script_path.parent,
    )

    elapsed_time = time.time() - start_time

    # Count compilation messages in stderr
    compile_count = result.stderr.count('Compiling')

    print(f"\n{'='*60}")
    print(f"Branch: {branch}")
    print(f"Elapsed time: {elapsed_time:.2f}s")
    print(f"Compilations: {compile_count}")
    print(f"{'='*60}\n")

    # Show any errors
    if result.returncode != 0:
        print(f"Error on {branch}:")
        print(result.stderr)

    return {
        'branch': branch,
        'time': elapsed_time,
        'compilations': compile_count,
        'returncode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr,
    }


def main():
    """Profile both branches."""
    script_path = Path("/home/dan/research/frm/danare/main.py")

    if not script_path.exists():
        print(f"Error: Script not found at {script_path}")
        sys.exit(1)

    # Get current branch
    current_branch = subprocess.check_output(
        ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
        text=True,
    ).strip()

    results = []

    # Run on current branch (should be joint_derivative)
    print(f"Current branch: {current_branch}")
    results.append(run_with_compile_count(script_path, current_branch))

    # Switch to main and run
    print("\nSwitching to main branch...")
    subprocess.run(['git', 'checkout', 'main'], check=True)
    results.append(run_with_compile_count(script_path, 'main'))

    # Switch back to original branch
    print(f"\nSwitching back to {current_branch}...")
    subprocess.run(['git', 'checkout', current_branch], check=True)

    # Print comparison
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)

    for r in results:
        print(f"\n{r['branch']}:")
        print(f"  Time: {r['time']:.2f}s")
        print(f"  Compilations: {r['compilations']}")

    if len(results) == 2:
        time_diff = results[0]['time'] - results[1]['time']
        compile_diff = results[0]['compilations'] - results[1]['compilations']

        print(f"\nDifference ({results[0]['branch']} - {results[1]['branch']}):")
        print(f"  Time: {time_diff:+.2f}s ({time_diff/results[1]['time']*100:+.1f}%)")
        print(f"  Compilations: {compile_diff:+d} ({compile_diff/results[1]['compilations']*100:+.1f}%)")

    print("="*60)


if __name__ == "__main__":
    main()
