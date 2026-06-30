#!/usr/bin/env python3
"""
diagnostics.py - sanity checks for the Redrob ranker.

Verifies the things that get a submission disqualified or down-scored:
  * Honeypot rate in the top 100 (must be <= 10%; we target 0%).
  * That detected honeypots are actually pushed out of the top 100.
  * Title / location / experience distribution of the top 100.
  * A peek at the lowest-ranked survivors (rank 90-100).

Usage:
    python diagnostics.py --candidates ./candidates.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import date

import rank as R


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--top-k", type=int, default=100)
    args = ap.parse_args()

    candidates = R.load_candidates(args.candidates)
    print(f"Loaded {len(candidates):,} candidates\n")

    # Reference "today" exactly as rank.py computes it.
    today = date(2026, 6, 28)
    for c in candidates:
        d = R._parse_date((c.get("redrob_signals", {}) or {}).get("last_active_date"))
        if d and d > today:
            today = d

    # Pool-wide honeypot detection.
    honeypot_ids = set()
    for c in candidates:
        hard, _ = R.honeypot_report(c, today)
        if hard:
            honeypot_ids.add(c.get("candidate_id"))
    print(f"Honeypots detected in full pool: {len(honeypot_ids)}")

    # Run the real ranker and inspect the top-K.
    top = R.rank(candidates, top_k=args.top_k)
    top_ids = {r["candidate_id"] for r in top}
    by_id = {c.get("candidate_id"): c for c in candidates}

    hp_in_top = top_ids & honeypot_ids
    rate = 100.0 * len(hp_in_top) / max(1, len(top))
    print(f"Honeypots in top {len(top)}: {len(hp_in_top)} ({rate:.1f}%) "
          f"-> {'PASS' if rate <= 10 else 'FAIL'} (limit 10%)\n")

    # Distributions over the top-K.
    titles, locs, bands = Counter(), Counter(), Counter()
    for r in top:
        c = by_id[r["candidate_id"]]
        _, tkind = R.score_title(c)
        _, lkind = R.score_location(c)
        titles[tkind] += 1
        locs[lkind] += 1
        yoe = (c.get("profile", {}) or {}).get("years_of_experience")
        if isinstance(yoe, (int, float)):
            bands["5-9 (in band)" if 5 <= yoe <= 9 else "out of band"] += 1

    print("Title tier distribution (top):", dict(titles))
    print("Location distribution (top): ", dict(locs))
    print("Experience band (top):       ", dict(bands), "\n")

    print("Lowest-ranked survivors (rank 90-100):")
    for pos, r in enumerate(top[89:], start=90):
        print(f"  {pos:>3}  {r['candidate_id']}  {r['score']:.4f}  "
              f"{r['reasoning'][:90]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
