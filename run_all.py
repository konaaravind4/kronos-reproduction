"""
run_all.py — Master script to reproduce all Kronos paper experiments
Usage:
    python run_all.py                    # Full run (requires model download)
    python run_all.py --no-model         # Simulation mode (no GPU/download needed)
    python run_all.py --skip-data        # Skip data download (use cached)
"""
import argparse
import subprocess
import sys
import os
import json
from pathlib import Path

ROOT = Path(__file__).parent

STEPS = [
    ("📥 Step 1: Download Market Data",
     ["python", "scripts/fetch_data.py"]),
    ("📈 Step 2: Price Series Forecasting",
     ["python", "scripts/experiment_forecasting.py"]),
    ("📉 Step 3: Volatility Forecasting",
     ["python", "scripts/experiment_volatility.py"]),
    ("🎲 Step 4: Synthetic K-line Generation",
     ["python", "scripts/experiment_generation.py"]),
    ("⏱  Step 5: Test-Time Scaling Analysis",
     ["python", "scripts/experiment_test_time_scaling.py"]),
]


def run_step(label: str, cmd: list, no_model: bool = False, extra_args: list = None):
    print("\n" + "=" * 65)
    print(f"  {label}")
    print("=" * 65)
    full_cmd = cmd.copy()
    if no_model:
        full_cmd.append("--no-model")
    if extra_args:
        full_cmd.extend(extra_args)
    result = subprocess.run(full_cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"  [WARN] Step returned code {result.returncode} — continuing...")
    return result.returncode == 0


def main(args):
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   Kronos: A Foundation Model for Financial Markets           ║")
    print("║   Paper Reproduction — All Experiments                       ║")
    print("║   Paper: arXiv:2508.02739  (AAAI 2026)                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    if args.no_model:
        print("  ⚠️  Running in simulation mode (--no-model).")
        print("     Results are calibrated to match paper's qualitative trends.")
        print("     To use the real model: remove --no-model flag.")
    else:
        print("  ℹ️  Will download Kronos-small (24.7M params) from HuggingFace.")
        print("     Ensure internet connection is available.")
    print()

    results = {}

    if not args.skip_data:
        ok = run_step(*STEPS[0], no_model=False)
        results["data"] = ok
    else:
        print("\n  [SKIP] Data download (--skip-data)")

    for label, cmd in STEPS[1:]:
        ok = run_step(label, cmd, no_model=args.no_model)
        results[label] = ok

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "╔" + "═" * 62 + "╗")
    print("║   Experiment Summary                                         ║")
    print("╠" + "═" * 62 + "╣")
    for step, ok in results.items():
        status = "✅ PASS" if ok else "⚠️  WARN"
        short = step[:50]
        print(f"║  {status}  {short:<50} ║")
    print("╚" + "═" * 62 + "╝")

    print("\n📂 Results saved to: results/")
    print("🌐 Open interface/index.html in a browser to view the dashboard.")
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Kronos paper results")
    parser.add_argument("--no-model",  action="store_true",
                        help="Skip model download; use simulation mode")
    parser.add_argument("--skip-data", action="store_true",
                        help="Skip data download step")
    args = parser.parse_args()
    main(args)
