import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    if "--train" not in sys.argv:
        subprocess.run(
            [sys.executable, "src/evaluate/audit_binary_data.py"],
            cwd=root,
            env={**__import__("os").environ, "PYTHONPATH": str(root)},
            check=True,
        )
        print("Static binary-data audit completed. No data were regenerated and no model was trained.")
        return
    subprocess.run(
        [sys.executable, "src/main_binary.py", "--train"],
        cwd=root,
        env={**__import__("os").environ, "PYTHONPATH": str(root)},
        check=True,
    )


if __name__ == "__main__":
    main()
