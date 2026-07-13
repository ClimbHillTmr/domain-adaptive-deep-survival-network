"""
Phase 4 – Full Orchestrator
==============================
Legacy Phase 4 entry point. The locked submission pipeline is the only
supported route for result generation.

Usage:
  cd /Users/cht/GitHub/domain-adaptive-deep-survival-network
  python scripts/phase4_all.py

To skip ablation training (if already done):
  python scripts/phase4_all.py --skip-ablation
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd, critical=True):
    print(f"\n>>> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        msg = f"Command failed (exit {result.returncode}): {cmd}"
        if critical:
            print(f"ERROR: {msg}")
            sys.exit(result.returncode)
        else:
            print(f"WARNING: {msg} (continuing)")
    return result.returncode


def print_summary():
    print("\n" + "=" * 60)
    print("PHASE 4 SUMMARY")
    print("=" * 60)

    # Main results
    eval_path = Path("experiments/results/evaluation_results.json")
    cox_path = Path("experiments/results/local_cox_update_results.json")
    if eval_path.exists() and cox_path.exists():
        import json
        with open(eval_path) as f:
            ev = json.load(f)
        with open(cox_path) as f:
            cx = json.load(f)
        print("\n[Main Results — Target Centre (Fuding)]")
        zs = ev["sweeps"]["Zero-shot"]["metrics"]
        da = ev["sweeps"]["CDAN-GSN (Ours)"]["metrics"]
        cm = cx["metrics"]
        print(f"  Zero-shot:    C-index {zs['C-index']:.4f} [{zs['C-index_CI_lower']:.4f}, {zs['C-index_CI_upper']:.4f}]")
        print(f"  CoxPH:        C-index {cm['C-index']:.4f} [{cm['C-index_CI_lower']:.4f}, {cm['C-index_CI_upper']:.4f}]")
        print(f"  CDAN-GSN:     C-index {da['C-index']:.4f} [{da['C-index_CI_lower']:.4f}, {da['C-index_CI_upper']:.4f}]")
        print(f"  DA gain:      +{da['C-index'] - zs['C-index']:.4f} vs zero-shot")
        print(f"  vs CoxPH:     {'+' if da['C-index'] > cm['C-index'] else ''}{da['C-index'] - cm['C-index']:.4f}")

    # Ablation summary
    abl_path = Path("experiments/results/ablation_results_v2.json")
    if abl_path.exists():
        with open(abl_path) as f:
            abls = json.load(f)
        full_c = da["C-index"] if eval_path.exists() else None
        print("\n[Ablation Study]")
        for abl in abls:
            if "metrics" not in abl:
                continue
            c = abl["metrics"]["C-index"]
            delta = f"{c - full_c:+.4f}" if full_c else ""
            print(f"  {abl['name']:20s}: C-index {c:.4f}  {delta}")

    # Statistical tests
    stat_path = Path("experiments/results/statistical_tests.json")
    if stat_path.exists():
        with open(stat_path) as f:
            st = json.load(f)
        print("\n[Statistical Tests]")
        for horizon, res in st.get("paired_cluster_auc", {}).items():
            print(f"  Paired cluster AUC {horizon}: Δ={res['observed_diff']:+.4f} [{res['ci_lower']:.4f}, {res['ci_upper']:.4f}]")
        cbi = st.get("paired_cluster_cindex", {}).get("CDAN-GSN_vs_CoxPH")
        if cbi:
            print(f"  Paired cluster C-index: Δ={cbi['observed_diff']:+.4f} [{cbi['ci_lower']:.4f}, {cbi['ci_upper']:.4f}]")

    # Figure inventory
    print("\n[Generated Files]")
    for p in sorted(Path("figures/Main_Figures").glob("*.pdf")):
        print(f"  {p}")
    for p in sorted(Path("tables").glob("*.csv")):
        print(f"  {p}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ablation", action="store_true", help="Skip ablation training (use cached results)")
    args = parser.parse_args()

    print("=" * 60)
    print("PHASE 4: delegating to the locked submission pipeline")
    print("=" * 60)

    if args.skip_ablation:
        raise SystemExit("Cached or partial runs are no longer supported.")
    run("PYTHONPATH=. python src/pipeline/run_locked_submission.py", critical=True)
    print_summary()


if __name__ == "__main__":
    main()
