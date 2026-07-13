import subprocess
import sys


def main():
    if "--train" not in sys.argv:
        print("Refusing to run training by default. Use: python run_all.py --train")
        return

    subprocess.run([sys.executable, "src/pipeline/run_locked_submission.py"], check=True)


if __name__ == "__main__":
    main()
