from __future__ import annotations

import csv
import datetime as dt
import gzip
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple


STAR_RATINGS = {1, 2, 3, 4, 5}


def first_present(obj: Dict, names: Iterable[str]):
    for name in names:
        if name in obj and obj[name] is not None:
            return obj[name]
    return None


def extract_review_fields(obj: Dict):
    asin = first_present(obj, ["parent_asin", "asin", "item_id"])
    user = first_present(obj, ["user_id", "reviewer_id"])
    rating = first_present(obj, ["rating", "overall"])
    timestamp = first_present(
        obj, ["timestamp", "time", "unixReviewTime", "review_time"]
    )
    return asin, user, rating, timestamp


def normalize_star_rating(value) -> Tuple[Optional[int], str]:
    """
    Returns (rating, status).

    Accepted ratings are exactly 1,2,3,4,5 (allowing representations such
    as 5.0). Rating 0 is explicitly distinguished from other unsupported
    values for auditability.
    """
    try:
        x = float(value)
    except Exception:
        return None, "non_numeric"

    if not math.isfinite(x):
        return None, "non_finite"

    if x == 0.0:
        return None, "zero"

    rounded = int(round(x))
    if abs(x - rounded) > 1e-12 or rounded not in STAR_RATINGS:
        return None, "unsupported"

    return rounded, "ok"


def normalize_timestamp(value) -> Tuple[Optional[int], str]:
    """
    Normalize the dataset timestamp to an integer key used only for ordering.

    Amazon Reviews 2023 normally provides integer millisecond timestamps.
    Numeric strings/floats are accepted if integral. A small ISO-8601 fallback
    is included for portability.
    """
    if isinstance(value, bool):
        return None, "invalid"

    if isinstance(value, int):
        return int(value), "ok"

    if isinstance(value, float):
        if math.isfinite(value) and float(value).is_integer():
            return int(value), "ok"
        return None, "invalid"

    s = str(value).strip()
    if not s:
        return None, "missing"

    try:
        x = float(s)
        if math.isfinite(x) and x.is_integer():
            return int(x), "ok"
    except Exception:
        pass

    try:
        z = s.replace("Z", "+00:00")
        parsed = dt.datetime.fromisoformat(z)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        # microseconds since epoch, exact integer ordering key
        return int(round(parsed.timestamp() * 1_000_000)), "ok_iso"
    except Exception:
        return None, "invalid"


def load_preliminary_eligible_asins(
    counts_csv: str | Path,
    threshold: int = 300,
) -> set[str]:
    eligible = set()
    with open(counts_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"asin", "usable_reviews_after_dedup"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"{counts_csv} does not contain required columns {sorted(required)}"
            )
        for row in reader:
            if int(row["usable_reviews_after_dedup"]) >= threshold:
                eligible.add(row["asin"])
    return eligible


def connect_work_db(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute("PRAGMA cache_size=-262144")  # ~256 MiB page cache
    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            asin TEXT NOT NULL,
            user_id TEXT NOT NULL,
            rating INTEGER NOT NULL,
            timestamp INTEGER NOT NULL,
            source_line INTEGER NOT NULL,
            PRIMARY KEY (asin, user_id)
        ) WITHOUT ROWID
        """
    )
    return conn


UPSERT_SQL = """
INSERT INTO reviews(asin,user_id,rating,timestamp,source_line)
VALUES(?,?,?,?,?)
ON CONFLICT(asin,user_id) DO UPDATE SET
    rating=excluded.rating,
    timestamp=excluded.timestamp,
    source_line=excluded.source_line
WHERE
    excluded.timestamp < reviews.timestamp
    OR (
        excluded.timestamp = reviews.timestamp
        AND excluded.source_line < reviews.source_line
    )
"""


def ingest_preliminary_eligible_reviews(
    *,
    reviews_gz: str | Path,
    preliminary_eligible: set[str],
    conn: sqlite3.Connection,
    progress_every: int = 1_000_000,
    batch_size: int = 50_000,
) -> Dict:
    stats = Counter()
    batch = []

    with gzip.open(reviews_gz, "rt", encoding="utf-8", errors="replace") as f:
        for source_line, line in enumerate(f, start=1):
            stats["total_lines"] += 1

            if progress_every and stats["total_lines"] % progress_every == 0:
                print(
                    f"{stats['total_lines']:,} lines | "
                    f"candidate-valid={stats['candidate_valid_records']:,} | "
                    f"rating0-skipped={stats['rating_zero_skipped']:,}",
                    flush=True,
                )

            if not line.strip():
                stats["blank_lines"] += 1
                continue

            try:
                obj = json.loads(line)
            except Exception:
                stats["malformed_json"] += 1
                continue

            asin, user, rating_raw, timestamp_raw = extract_review_fields(obj)

            if asin is None or str(asin).strip() == "":
                stats["missing_asin"] += 1
                continue
            asin = str(asin)
            if asin not in preliminary_eligible:
                stats["noncandidate_records"] += 1
                continue

            if user is None or str(user).strip() == "":
                stats["candidate_missing_user"] += 1
                continue
            user = str(user)

            if rating_raw is None:
                stats["candidate_missing_rating"] += 1
                continue

            rating, rating_status = normalize_star_rating(rating_raw)
            if rating_status != "ok":
                if rating_status == "zero":
                    stats["rating_zero_skipped"] += 1
                else:
                    stats[f"rating_{rating_status}_skipped"] += 1
                continue

            if timestamp_raw is None:
                stats["candidate_missing_timestamp"] += 1
                continue

            timestamp, ts_status = normalize_timestamp(timestamp_raw)
            if timestamp is None:
                stats["candidate_invalid_timestamp"] += 1
                continue
            if ts_status == "ok_iso":
                stats["candidate_iso_timestamp"] += 1

            stats["candidate_valid_records"] += 1
            batch.append((asin, user, rating, timestamp, source_line))

            if len(batch) >= batch_size:
                before = conn.total_changes
                conn.executemany(UPSERT_SQL, batch)
                conn.commit()
                stats["sqlite_changes"] += conn.total_changes - before
                batch.clear()

    if batch:
        before = conn.total_changes
        conn.executemany(UPSERT_SQL, batch)
        conn.commit()
        stats["sqlite_changes"] += conn.total_changes - before

    return dict(stats)


def finalize_stage1(
    *,
    conn: sqlite3.Connection,
    preliminary_eligible: set[str],
    out_jsonl_gz: str | Path,
    out_manifest_csv: str | Path,
    out_excluded_csv: str | Path,
    threshold: int = 300,
) -> Dict:
    # Authoritative unique-user counts after the exact earliest-review rule.
    counts = dict(
        conn.execute(
            "SELECT asin, COUNT(*) FROM reviews GROUP BY asin"
        ).fetchall()
    )

    final_eligible = {
        asin for asin, n in counts.items() if int(n) >= threshold
    }
    excluded = sorted(preliminary_eligible - final_eligible)

    # This index makes chronological export deterministic and efficient.
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_reviews_asin_time
        ON reviews(asin, timestamp, source_line)
        """
    )
    conn.commit()

    out_jsonl_gz = Path(out_jsonl_gz)
    out_manifest_csv = Path(out_manifest_csv)
    out_excluded_csv = Path(out_excluded_csv)
    out_jsonl_gz.parent.mkdir(parents=True, exist_ok=True)
    out_manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    rating_totals = Counter()
    n_items_written = 0
    n_reviews_written = 0

    manifest_fields = [
        "asin",
        "usable_unique_user_reviews",
        "exported_reviews",
        "first_timestamp",
        "last_timestamp",
        "first_source_line",
        "last_source_line",
        "rating_1",
        "rating_2",
        "rating_3",
        "rating_4",
        "rating_5",
    ]

    with gzip.open(
        out_jsonl_gz, "wt", encoding="utf-8", compresslevel=6
    ) as jout, open(
        out_manifest_csv, "w", encoding="utf-8", newline=""
    ) as mf:
        mw = csv.DictWriter(mf, fieldnames=manifest_fields)
        mw.writeheader()

        query = """
        SELECT asin,user_id,rating,timestamp,source_line
        FROM reviews
        ORDER BY asin,timestamp,source_line
        """

        current_asin = None
        selected = []

        def flush_item(asin, reviews):
            nonlocal n_items_written, n_reviews_written
            if asin is None or asin not in final_eligible:
                return
            if len(reviews) != threshold:
                raise RuntimeError(
                    f"Invariant failure for {asin}: expected {threshold}, "
                    f"got {len(reviews)} exported reviews."
                )

            users = [r["user_id"] for r in reviews]
            if len(users) != len(set(users)):
                raise RuntimeError(f"Duplicate user remained within item {asin}")

            rc = Counter(r["rating"] for r in reviews)
            if set(rc) - STAR_RATINGS:
                raise RuntimeError(f"Invalid rating survived for {asin}: {rc}")

            obj = {"asin": asin, "reviews": reviews}
            jout.write(json.dumps(obj, separators=(",", ":")) + "\n")

            mw.writerow({
                "asin": asin,
                "usable_unique_user_reviews": counts[asin],
                "exported_reviews": len(reviews),
                "first_timestamp": reviews[0]["timestamp"],
                "last_timestamp": reviews[-1]["timestamp"],
                "first_source_line": reviews[0]["source_line"],
                "last_source_line": reviews[-1]["source_line"],
                "rating_1": rc.get(1, 0),
                "rating_2": rc.get(2, 0),
                "rating_3": rc.get(3, 0),
                "rating_4": rc.get(4, 0),
                "rating_5": rc.get(5, 0),
            })
            rating_totals.update(rc)
            n_items_written += 1
            n_reviews_written += len(reviews)

        for asin, user, rating, timestamp, source_line in conn.execute(query):
            if asin != current_asin:
                flush_item(current_asin, selected)
                current_asin = asin
                selected = []

            if asin in final_eligible and len(selected) < threshold:
                selected.append({
                    "position": len(selected) + 1,
                    "user_id": user,
                    "rating": int(rating),
                    "timestamp": int(timestamp),
                    "source_line": int(source_line),
                })

        flush_item(current_asin, selected)

    with open(out_excluded_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["asin", "usable_unique_user_reviews_stage1"])
        for asin in excluded:
            writer.writerow([asin, counts.get(asin, 0)])

    if n_items_written != len(final_eligible):
        raise RuntimeError(
            f"Final item-count invariant failed: expected "
            f"{len(final_eligible)}, wrote {n_items_written}"
        )
    if n_reviews_written != n_items_written * threshold:
        raise RuntimeError("Final exported review-count invariant failed.")

    return {
        "preliminary_eligible_items": len(preliminary_eligible),
        "final_eligible_items": len(final_eligible),
        "preliminary_items_excluded_at_stage1": len(excluded),
        "exported_reviews": n_reviews_written,
        "reviews_per_item": threshold,
        "rating_totals_first300": {
            str(k): int(rating_totals.get(k, 0)) for k in range(1, 6)
        },
    }
