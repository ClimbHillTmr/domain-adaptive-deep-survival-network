"""
Phase 4 – Full Orchestrator
==============================
Runs all Phase 4 steps in order:
  1. Re-run ablation study (4 variants, with bootstrap CI)
  2. Generate publication figures (Fig1–Fig6)
  3. Build tables 2 & 3 + ablation figure
  4. Run statistical tests (DeLong AUC + bootstrap C-index permutation)
  5. Print final summary

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
        for horizon, res in st.get("delong_auc", {}).items():
            sig = "*" if res["significant_0.05"] else "ns"
            print(f"  DeLong AUC {horizon}: CDAN={res['AUC_CDAN-GSN']:.4f} vs Cox={res['AUC_CoxPH']:.4f}  p={res['p_value']:.4f} ({sig})")
        cbi = st.get("cindex_bootstrap", {}).get("CDAN-GSN_vs_CoxPH")
        if cbi:
            sig = "*" if cbi["significant_0.05"] else "ns"
            print(f"  C-index permutation: diff={cbi['observed_diff']:+.4f}  p={cbi['p_value_one_sided']:.4f} ({sig})")

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
    print("PHASE 4: Ablation + Figures + Statistical Tests")
    print("=" * 60)

    # Step 1: Ablation study
    if not args.skip_ablation:
        run("PYTHONPATH=. python scripts/run_ablations.py", critical=True)
    else:
        print("\n[Step 1] Skipped ablation training (--skip-ablation)")

    # Step 2: Publication figures (Fig1–Fig6)
    run("PYTHONPATH=. python src/visualization/build_publication_figures.py", critical=False)

    # Step 3: Export real predictions (Fig7 calibration, Fig8 DCA)
    run("PYTHONPATH=. python src/evaluate/export_real_predictions.py", critical=False)

    # Step 4: Tables + ablation figure
    run("PYTHONPATH=. python scripts/build_tables_and_figs.py", critical=False)

    # Step 5: Statistical tests
    run("PYTHONPATH=. python scripts/statistical_tests.py", critical=False)

    # Final summary
    print_summary()


if __name__ == "__main__":
    main()
