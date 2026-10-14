"""Task 4: compare measurements to the four budgets -> results/budget_table.csv.

Memory is judged on peak process RSS (conservative: includes the Python +
sklearn runtime, not just the model). Latency/size use the model measurements.
"""
import csv
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
# (mem_max_MB or None for no cap, latency_max_ms, size_max_KB)
BUDGETS = {
    "Cloud":  (None,      100, 500 * 1024),
    "Edge":   (1024,       50,  50 * 1024),
    "Mobile": (256,        20,  10 * 1024),
    "TinyML": (256 / 1024, 10,         100),  # 256 KB = 0.25 MB
}


def main():
    rows = list(csv.DictReader(open(RES / "system_cost.csv")))
    out = []
    for r in rows:
        mem = max(float(r["peak_rss_train_MB"]), float(r["peak_rss_infer_MB"]))
        lat, size = float(r["infer_latency_ms_median"]), float(r["model_size_KB"])
        for tier, (mem_max, lat_max, size_max) in BUDGETS.items():
            ok_mem = mem_max is None or mem <= mem_max
            ok_lat, ok_size = lat <= lat_max, size <= size_max
            out.append({
                "model": r["model"], "tier": tier,
                "memory_ok": ok_mem, "latency_ok": ok_lat, "size_ok": ok_size,
                "fits": ok_mem and ok_lat and ok_size,
            })
    with open(RES / "budget_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for o in out:
        print(o)


if __name__ == "__main__":
    main()
