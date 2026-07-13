import subprocess
import sys
from pathlib import Path


def main():
    if "--train" not in sys.argv:
        print("Refusing to run training by default. Use: python run_all.py --train")
        return

    root = Path(__file__).resolve().parent
    subprocess.run(
        [sys.executable, "src/pipeline/run_locked_submission.py"],
        cwd=root,
        env={**__import__("os").environ, "PYTHONPATH": str(root)},
        check=True,
    )


if __name__ == "__main__":
    main()
