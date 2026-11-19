"""Summarize an 8-GPU RSI Arena round into a sorted Markdown leaderboard."""
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: summarize_rsi.py ROUND_DIR")
    d = Path(sys.argv[1])
    rows = []
    for p in sorted(d.glob("gpu*.json")):
        rows.append(json.loads(p.read_text()))
    if not rows:
        sys.exit(f"No gpu*.json found in {d}")

    base = next((r for r in rows if r["idea"] in ("base", "opt_base")), rows[0])
    b8 = base["final_val_bpb"]["8"]
    b16 = base["final_val_bpb"].get("16", float("nan"))

    rows.sort(key=lambda r: r["final_val_bpb"]["8"])
    print(f"| Rank | Idea | val_bpb@8 | Δ@8 vs base | val_bpb@16 | Δ@16 vs base | val_bpb@4 | collapse@8 | tok/s | mem GB |")
    print(f"|---|---|---|---|---|---|---|---|---|---|")
    for idx, r in enumerate(rows, 1):
        v8 = r["final_val_bpb"]["8"]
        v16 = r["final_val_bpb"].get("16", float("nan"))
        v4 = r["final_val_bpb"].get("4", float("nan"))
        d8 = v8 - b8
        d16 = v16 - b16
        col = r.get("final_collapse_r8") or float("nan")
        print(f"| {idx} | `{r['idea']}` | **{v8:.5f}** | {d8:+.5f} | {v16:.5f} | {d16:+.5f} | {v4:.5f} | {col:.4f} | {r['tok_per_sec']/1e3:.1f}k | {r['peak_mem_gb']:.1f} |")
        if r.get("extra"):
            print(f"  extra({r['idea']}): {r['extra']}")


if __name__ == "__main__":
    main()
