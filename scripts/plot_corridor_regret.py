#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from statistics import mean


def load_rows(path: str):
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    # coerce to floats where appropriate
    for r in rows:
        for k in [
            "uniform_mean_last50",
            "hybrid_mean_last50",
            "delta",
            "coin_reward",
            "uniform_mixture",
            "hybrid_mixture",
        ]:
            if k in r:
                try:
                    r[k] = float(r[k])
                except Exception:
                    pass
    return rows


def main(args: argparse.Namespace) -> None:
    rows = load_rows(args.csv)
    summary = {
        "n": len(rows),
        "uniform_mean": mean(r["uniform_mean_last50"] for r in rows),
        "hybrid_mean": mean(r["hybrid_mean_last50"] for r in rows),
        "delta_mean": mean(r["delta"] for r in rows),
    }

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rows": rows, "summary": summary}, f, indent=2)

    # Optional plotting if matplotlib is available
    if args.out:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            deltas = [r["delta"] for r in rows]
            plt.figure(figsize=(6, 3))
            plt.axhline(0, color="#888", linewidth=1)
            plt.plot(
                deltas, marker="o", linestyle="-", label="delta = hybrid - uniform"
            )
            plt.title("Corridor Retrieval Regret (delta)")
            plt.xlabel("seed index")
            plt.ylabel("delta (last-50 mean reward)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(args.out, dpi=150)
            print(f"saved plot -> {args.out}")
        except Exception as e:  # pragma: no cover
            print(f"plotting skipped: {e}")

    print("summary:", summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--out", type=str, default="", help="optional PNG path")
    p.add_argument("--json", type=str, default="", help="optional JSON dump path")
    main(p.parse_args())
