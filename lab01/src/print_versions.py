"""Print Python and every pinned package version -> save to results/versions.txt."""
import platform
import sys
from importlib import metadata
from pathlib import Path

REQ = Path(__file__).resolve().parents[2] / "requirements.txt"


def main():
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    names = []
    if REQ.exists():
        for line in REQ.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line.split("==")[0].strip())
    else:
        names = ["numpy", "scikit-learn", "joblib", "psutil", "memory-profiler"]
    for n in sorted(names, key=str.lower):
        try:
            print(f"{n}=={metadata.version(n)}")
        except metadata.PackageNotFoundError:
            print(f"{n}: NOT INSTALLED")


if __name__ == "__main__":
    main()
