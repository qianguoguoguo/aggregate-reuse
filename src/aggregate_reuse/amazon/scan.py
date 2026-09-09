"""Eligibility scan for the frozen Amazon Reviews 2023 domain."""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path

from .preprocessing import first_present


def stable_review_fields(obj):
    asin = first_present(obj, ["parent_asin", "asin", "item_id"])
    user = first_present(obj, ["user_id", "reviewer_id"])
    rating = first_present(obj, ["rating", "overall"])
    timestamp = first_present(
        obj,
        ["timestamp", "time", "unixReviewTime", "review_time"],
    )
    return asin, user, rating, timestamp


def finite_rating(x):
    try:
        y = float(x)
    except Exception:
        return None
    if not math.isfinite(y):
        return None
    return y


def quantile_from_counts(counts, q):
    if not counts:
        return None
    xs = sorted(counts)
    idx = int(round(q * (len(xs) - 1)))
    return xs[idx]


def scan_category(
    path,
    out_json,
    out_csv,
    *,
    category="home_and_kitchen",
    progress_every=1_000_000,
):
    path = Path(path)
    out_json = Path(out_json)
    out_csv = Path(out_csv)

    total = 0
    blank = 0
    malformed = 0
    invalid_missing = Counter()
    invalid_rating = 0
    valid_records = 0
    duplicate_user_item = 0

    item_counts_before = Counter()
    item_counts_after = Counter()
    reviewers = set()
    seen_user_item = set()

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            total += 1
            if progress_every and total % progress_every == 0:
                print(
                    f"{total:,} records | valid={valid_records:,} | "
                    f"items={len(item_counts_before):,} | "
                    f"duplicates={duplicate_user_item:,}",
                    flush=True,
                )

            if not line.strip():
                blank += 1
                continue

            try:
                obj = json.loads(line)
            except Exception:
                malformed += 1
                continue

            asin, user, rating_raw, timestamp = stable_review_fields(obj)

            missing = False
            if asin is None or str(asin).strip() == "":
                invalid_missing["asin"] += 1
                missing = True
            if user is None or str(user).strip() == "":
                invalid_missing["user_id"] += 1
                missing = True
            if rating_raw is None:
                invalid_missing["rating"] += 1
                missing = True
            if timestamp is None or str(timestamp).strip() == "":
                invalid_missing["timestamp"] += 1
                missing = True
            if missing:
                continue

            rating = finite_rating(rating_raw)
            if rating is None or rating < 1.0 or rating > 5.0:
                invalid_rating += 1
                continue

            asin = str(asin)
            user = str(user)

            valid_records += 1
            item_counts_before[asin] += 1
            reviewers.add(user)

            pair = (user, asin)
            if pair in seen_user_item:
                duplicate_user_item += 1
                continue
            seen_user_item.add(pair)
            item_counts_after[asin] += 1

    thresholds = [50, 100, 200, 300, 500, 1000]
    before_vals = list(item_counts_before.values())
    after_vals = list(item_counts_after.values())

    summary = {
        "category": category,
        "input_path": f"external/{path.name}",
        "total_lines": total,
        "blank_lines": blank,
        "malformed_json": malformed,
        "missing_required_fields": dict(invalid_missing),
        "invalid_rating": invalid_rating,
        "valid_records_before_user_item_dedup": valid_records,
        "duplicate_user_item_records": duplicate_user_item,
        "usable_records_after_user_item_dedup":
            valid_records - duplicate_user_item,
        "unique_items_before_dedup": len(item_counts_before),
        "unique_items_after_dedup": len(item_counts_after),
        "unique_reviewers_valid_records": len(reviewers),
        "threshold_counts_before_dedup": {
            str(t): sum(c >= t for c in before_vals) for t in thresholds
        },
        "threshold_counts_after_dedup": {
            str(t): sum(c >= t for c in after_vals) for t in thresholds
        },
        "usable_reviews_per_item_after_dedup_quantiles": {
            "p50": quantile_from_counts(after_vals, 0.50),
            "p75": quantile_from_counts(after_vals, 0.75),
            "p90": quantile_from_counts(after_vals, 0.90),
            "p95": quantile_from_counts(after_vals, 0.95),
            "p99": quantile_from_counts(after_vals, 0.99),
            "p999": quantile_from_counts(after_vals, 0.999),
            "max": max(after_vals) if after_vals else None,
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "asin",
            "reviews_before_user_item_dedup",
            "usable_reviews_after_dedup",
        ])
        all_items = sorted(set(item_counts_before) | set(item_counts_after))
        for asin in all_items:
            writer.writerow([
                asin,
                item_counts_before.get(asin, 0),
                item_counts_after.get(asin, 0),
            ])

    return summary
