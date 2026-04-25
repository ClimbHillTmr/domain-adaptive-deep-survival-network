#!/usr/bin/env python3
"""
Real-time experiment monitor.
Watches the latest log file and reports progress.
"""
import os
import sys
import time
import glob
import re


def find_latest_log():
    """Find the most recent ablation log file."""
    log_dir = os.path.join(os.path.dirname(__file__), "runs")
    if not os.path.exists(log_dir):
        return None

    # Find all date directories
    date_dirs = sorted(glob.glob(os.path.join(log_dir, "20*")))
    if not date_dirs:
        return None

    # Find latest time directory
    time_dirs = sorted(glob.glob(os.path.join(date_dirs[-1], "*")))
    if not time_dirs:
        return None

    log_file = os.path.join(time_dirs[-1], "run_dadsn_ablation.log")
    return log_file if os.path.exists(log_file) else None


def parse_log(log_file):
    """Parse log file and extract experiment progress."""
    if not os.path.exists(log_file):
        return None

    with open(log_file, "r") as f:
        content = f.read()

    # Extract epoch lines
    epoch_pattern = r"Epoch (\d+)/(\d+) \| SurvLoss: ([\d.]+) \| CoralLoss: ([\d.]+) \| Target C-index: ([\d.]+) \| Target IBS: ([\d.]+) \| LR: ([\d.]+)"
    epochs = re.findall(epoch_pattern, content)

    # Extract experiment headers
    exp_pattern = r"### Experiment (\d+): (.+) ###"
    experiments = re.findall(exp_pattern, content)

    # Extract final results
    result_pattern = r"Target C-index: ([\d.]+), IBS: ([\d.]+)"
    results = re.findall(result_pattern, content)

    # Extract early stopping
    early_stop_pattern = r"Early stopping at epoch (\d+)"
    early_stops = re.findall(early_stop_pattern, content)

    return {
        "epochs": epochs,
        "experiments": experiments,
        "results": results,
        "early_stops": early_stops,
        "file_size": os.path.getsize(log_file),
        "last_modified": os.path.getmtime(log_file),
    }


def monitor(interval=30):
    """Monitor experiment progress."""
    print("=" * 80)
    print("DA-DSN Experiment Monitor")
    print("=" * 80)

    last_epoch_count = 0
    last_exp_count = 0

    while True:
        log_file = find_latest_log()
        if not log_file:
            print("\r[WAITING] No log file found yet...", end="", flush=True)
            time.sleep(interval)
            continue

        data = parse_log(log_file)
        if not data:
            print("\r[WAITING] Log file empty...", end="", flush=True)
            time.sleep(interval)
            continue

        # Clear line
        print("\r" + " " * 100 + "\r", end="", flush=True)

        # Print experiment status
        print(f"\n{'='*80}")
        print(f"Log: {log_file}")
        print(
            f"Last updated: {time.strftime('%H:%M:%S', time.localtime(data['last_modified']))}"
        )
        print(f"File size: {data['file_size'] / 1024:.1f} KB")

        # Print experiments
        if data["experiments"]:
            print(f"\nExperiments started: {len(data['experiments'])}")
            for exp_num, exp_name in data["experiments"]:
                print(f"  [{exp_num}] {exp_name}")

        # Print latest epoch progress
        if data["epochs"]:
            latest = data["epochs"][-1]
            epoch, total, surv, coral, cidx, ibs, lr = latest
            progress = int(epoch) / int(total) * 100

            print(f"\nCurrent Epoch: {epoch}/{total} ({progress:.1f}%)")
            print(f"  SurvLoss: {surv} | CoralLoss: {coral}")
            print(f"  C-index: {cidx} | IBS: {ibs}")
            print(f"  Learning Rate: {lr}")

        # Print completed results
        if data["results"]:
            print(f"\nCompleted Results: {len(data['results'])}")
            for i, (cidx, ibs) in enumerate(data["results"][-3:]):
                print(f"  [{i+1}] C-index: {cidx}, IBS: {ibs}")

        # Print early stopping info
        if data["early_stops"]:
            print(
                f"\nEarly Stopping triggered at epochs: {', '.join(data['early_stops'])}"
            )

        # Check if process is still running
        import subprocess

        result = subprocess.run(
            ["pgrep", "-f", "run_dadsn_ablation"], capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"\n[COMPLETED] Experiment process has finished!")
            break

        # Wait for next update
        print(f"\nNext update in {interval}s... (Ctrl+C to stop)")
        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n\nMonitor stopped.")
            break


if __name__ == "__main__":
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    monitor(interval)
