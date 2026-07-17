"""
Print the cross-provider leaderboard from judged runs in results/runs/.

Each runner script scores itself completely (per-bucket metrics + equal-weight overall live in
every judged file's `summary`); this just lines the latest run per config up side by side.

Usage: python scripts/leaderboard.py
"""

import glob
import json
from pathlib import Path

RUNS = Path(__file__).parent.parent / "results" / "runs"
BUCKETS = ["general_enrichment", "recruiting", "sales", "due_diligence"]


def main():
    latest = {}
    for f in sorted(glob.glob(str(RUNS / "*_judged.json"))):
        data = json.loads(Path(f).read_text())
        if data.get("summary", {}).get("scores"):
            latest[data.get("name") or Path(f).stem] = data["summary"]["scores"]
    if not latest:
        print(f"No scored runs in {RUNS}. Run a provider script first (e.g. python scripts/sixtyfour.py --tier high).")
        return

    print("overall = equal-weight average of the four bucket scores (25% per use case)\n")
    for metric in ("weighted", "accuracy", "precision"):
        print(f"=== {metric} ===")
        print(f"{'system':22s} " + " ".join(f"{b[:12]:>12s}" for b in BUCKETS) + f" {'OVERALL':>9s}")
        for name in sorted(latest, key=lambda n: -latest[n]["overall"][metric]):
            sc = latest[name]
            cells = [f"{sc['buckets'][b][metric]:12.1f}" if b in sc["buckets"] else f"{'—':>12s}" for b in BUCKETS]
            print(f"{name:22s} " + " ".join(cells) + f" {sc['overall'][metric]:9.1f}")
        print()


if __name__ == "__main__":
    main()
