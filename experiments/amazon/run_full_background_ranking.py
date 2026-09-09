#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import pearsonr, spearmanr, t

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aggregate_reuse.amazon.references import (
    exact_predictive_w1_baseline,
    shrunk_reference,
    load_reference_table,
)
from aggregate_reuse.amazon.paths import AmazonPathLayout
from aggregate_reuse.amazon.provenance import (
    build_provenance,
    require_artifact_category,
    resolve_raw_dataset_sha256,
)


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------

def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_attack_world(path: str | Path):
    out = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            asin = str(row["asin"])
            out[asin] = {
                "asin": asin,
                "treatment_block": row["treatment_block"],
                "treated_positions": [int(x) for x in json.loads(row["treated_positions"])],
                "treated_source_lines": [int(x) for x in json.loads(row["treated_source_lines"])],
                "original_ratings": [int(x) for x in json.loads(row["original_ratings"])],
                "replacement_ratings": [int(x) for x in json.loads(row["replacement_ratings"])],
                "clean_counts": np.asarray(json.loads(row["clean_counts"]), dtype=int),
                "attack_counts": np.asarray(json.loads(row["attack_counts"]), dtype=int),
                "clean_w1": float(row["clean_w1"]),
                "attack_w1": float(row["attack_w1"]),
                "d_cf": float(row["d_cf"]),
            }
    return out


def load_r8_assignment(path: str | Path):
    """
    Return slot -> synthetic identity and validate one-to-one slot assignment.
    Slot key is (asin, treatment_block, treated_position, treated_source_line).
    """
    slot_to_syn = {}
    by_asin = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            key = (
                str(row["asin"]),
                row["treatment_block"],
                int(row["treated_position"]),
                int(row["treated_source_line"]),
            )
            if key in slot_to_syn:
                raise RuntimeError(f"Duplicate Stage-4 r=8 slot assignment: {key}")
            syn = str(row["synthetic_account_id"])
            slot_to_syn[key] = syn
            by_asin[key[0]].append((key, syn))
    return slot_to_syn, by_asin


def load_selected_blocks(roles_path: str | Path, attack):
    """
    Stream the Stage-2 roles file and retain only the selected experimental
    block for the 2,000 Stage-4 items.
    """
    wanted = set(attack)
    blocks = {}
    with gzip.open(roles_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            asin = str(obj["asin"])
            if asin not in wanted:
                continue
            role = attack[asin]["treatment_block"]
            if role not in obj["blocks"]:
                raise RuntimeError(f"{asin}: missing selected role {role}")
            block = obj["blocks"][role]
            if len(block) != 30:
                raise RuntimeError(f"{asin}/{role}: expected 30 reviews, got {len(block)}")
            blocks[asin] = block
            if len(blocks) == len(wanted):
                break

    missing = wanted - set(blocks)
    if missing:
        ex = sorted(missing)[:5]
        raise RuntimeError(
            f"Stage-2 roles file is missing {len(missing)} selected items; examples={ex}"
        )
    return blocks


def write_accounts(path: Path, rows):
    fields = [
        "account_type",
        "account_id",
        "frequency",
        "raw_w1_score",
        "predictive_centered_w1_score",
        "coactivity_score",
        "combined_score",
    ]
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def hist(block):
    """Return 5-bin rating counts for one 30-review Amazon block."""
    counts = np.zeros(5, dtype=int)
    for rev in block:
        rating = int(rev["rating"])
        if rating < 1 or rating > 5:
            raise RuntimeError(f"Unsupported Amazon rating: {rating}")
        counts[rating - 1] += 1
    return counts


def w1_counts(counts, q):
    """Wasserstein-1 for 5-star histograms on unit-spaced support 1,...,5."""
    p = np.asarray(counts, dtype=float)
    if p.sum() <= 0:
        raise RuntimeError("Empty rating histogram.")
    p = p / p.sum()
    q = np.asarray(q, dtype=float)
    if len(q) != 5:
        raise RuntimeError(f"Expected 5-bin reference, got {len(q)} bins.")
    return float(np.abs(np.cumsum(p - q)[:-1]).sum())


# ---------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------

def _finite(x):
    return np.asarray(x, dtype=float)[np.isfinite(np.asarray(x, dtype=float))]


def background_percentiles(plant_scores, background_scores):
    """
    Midrank percentile of each planted account relative to background only:
      (# background strictly below + 0.5 # tied) / # background.
    High values are better.
    """
    p = np.asarray(plant_scores, dtype=float)
    b = np.sort(np.asarray(background_scores, dtype=float))
    if len(b) == 0:
        return np.full(len(p), np.nan, dtype=float)

    out = np.empty(len(p), dtype=float)
    for j, s in enumerate(p):
        left = np.searchsorted(b, s, side="left")
        right = np.searchsorted(b, s, side="right")
        out[j] = (left + 0.5 * (right - left)) / len(b)
    return out


def expected_top_capture(scores, planted, alpha):
    """
    Expected planted capture in the top ceil(alpha*N) accounts under uniform
    random ordering within exact score ties. This avoids arbitrary lexical
    tie-breaking.
    """
    scores = np.asarray(scores, dtype=float)
    planted = np.asarray(planted, dtype=bool)
    n = len(scores)
    nplant = int(planted.sum())
    if n == 0 or nplant == 0:
        return {
            "cutoff_accounts": 0,
            "cutoff_fraction": float("nan"),
            "planted_capture": float("nan"),
            "enrichment_over_random": float("nan"),
        }

    K = max(1, int(math.ceil(float(alpha) * n)))
    order = np.argsort(-scores, kind="mergesort")
    ss = scores[order]
    pp = planted[order]

    captured = 0.0
    used = 0
    i = 0
    while i < n and used < K:
        j = i + 1
        while j < n and ss[j] == ss[i]:
            j += 1
        g = j - i
        gp = int(pp[i:j].sum())
        take = min(g, K - used)
        captured += gp * (take / g)
        used += take
        i = j

    capture = captured / nplant
    random_rate = K / n
    enrich = capture / random_rate if random_rate > 0 else float("nan")
    return {
        "cutoff_accounts": int(K),
        "cutoff_fraction": float(random_rate),
        "planted_capture": float(capture),
        "enrichment_over_random": float(enrich),
    }


def expected_inspection_burden(scores, planted, target_fraction):
    """
    Minimum number of accounts that must be inspected to reach target_fraction
    of planted identities in expectation under uniform random ordering within
    exact score ties.
    """
    scores = np.asarray(scores, dtype=float)
    planted = np.asarray(planted, dtype=bool)
    n = len(scores)
    nplant = int(planted.sum())
    if n == 0 or nplant == 0:
        return {"accounts": None, "fraction_of_universe": float("nan")}

    target = float(target_fraction) * nplant
    order = np.argsort(-scores, kind="mergesort")
    ss = scores[order]
    pp = planted[order]

    exp_plants = 0.0
    inspected = 0
    i = 0
    while i < n:
        j = i + 1
        while j < n and ss[j] == ss[i]:
            j += 1
        g = j - i
        gp = int(pp[i:j].sum())

        if gp == 0:
            inspected += g
            i = j
            continue

        if exp_plants + gp >= target:
            need = max(0.0, target - exp_plants)
            # Expected planted count from s positions in this tie group is
            # s * gp / g.
            slots = int(math.ceil(need * g / gp - 1e-12))
            slots = max(0, min(g, slots))
            inspected += slots
            return {
                "accounts": int(inspected),
                "fraction_of_universe": float(inspected / n),
            }

        exp_plants += gp
        inspected += g
        i = j

    return {"accounts": int(n), "fraction_of_universe": 1.0}


def ranking_metrics(scores, planted, frequency, *, top_fractions, burden_targets):
    scores = np.asarray(scores, dtype=float)
    planted = np.asarray(planted, dtype=bool)
    frequency = np.asarray(frequency, dtype=int)

    ps = scores[planted]
    bs = scores[~planted]

    if len(ps) == 0:
        raise RuntimeError("Ranking universe contains no planted identities.")

    med_score = float(np.median(ps))
    n_background = int((~planted).sum())

    if n_background > 0:
        percentiles = background_percentiles(ps, bs)
        q25 = float(np.quantile(percentiles, 0.25))
        q50 = float(np.quantile(percentiles, 0.50))
        q75 = float(np.quantile(percentiles, 0.75))
        above95 = float(np.mean(percentiles >= 0.95))
        above99 = float(np.mean(percentiles >= 0.99))
        bg_above_median = int(np.sum(bs > med_score))
        background_relative_available = True
    else:
        q25 = q50 = q75 = float("nan")
        above95 = above99 = float("nan")
        bg_above_median = 0
        background_relative_available = False

    out = {
        "n_accounts": int(len(scores)),
        "n_planted": int(planted.sum()),
        "n_background": n_background,
        "background_relative_metrics_available": background_relative_available,
        "planted_background_percentile_q25": q25,
        "planted_background_percentile_median": q50,
        "planted_background_percentile_q75": q75,
        "fraction_planted_above_background_p95": above95,
        "fraction_planted_above_background_p99": above99,
        "background_accounts_strictly_above_planted_median_score": bg_above_median,
        "planted_median_score": med_score,
    }

    for a in top_fractions:
        r = expected_top_capture(scores, planted, float(a))
        tag = str(a).replace(".", "p")
        out[f"top_{tag}_cutoff_accounts"] = r["cutoff_accounts"]
        out[f"top_{tag}_cutoff_fraction"] = r["cutoff_fraction"]
        out[f"top_{tag}_planted_capture"] = r["planted_capture"]
        out[f"top_{tag}_enrichment_over_random"] = r["enrichment_over_random"]

    for q in burden_targets:
        r = expected_inspection_burden(scores, planted, float(q))
        tag = str(q).replace(".", "p")
        out[f"burden_{tag}_accounts"] = r["accounts"]
        out[f"burden_{tag}_fraction_of_universe"] = r["fraction_of_universe"]

    # Correlation is descriptive only; historical accounts are not negatives.
    if len(np.unique(frequency)) > 1 and np.std(scores) > 0:
        out["spearman_score_frequency_all"] = float(
            spearmanr(scores, frequency).statistic
        )
        out["pearson_score_frequency_all"] = float(
            pearsonr(scores, frequency).statistic
        )
    else:
        out["spearman_score_frequency_all"] = float("nan")
        out["pearson_score_frequency_all"] = float("nan")

    bfreq = frequency[~planted]
    if len(bs) > 1 and len(np.unique(bfreq)) > 1 and np.std(bs) > 0:
        out["spearman_score_frequency_background"] = float(
            spearmanr(bs, bfreq).statistic
        )
        out["pearson_score_frequency_background"] = float(
            pearsonr(bs, bfreq).statistic
        )
    else:
        out["spearman_score_frequency_background"] = float("nan")
        out["pearson_score_frequency_background"] = float("nan")

    return out


def stratum_mask(frequency, planted, kind):
    frequency = np.asarray(frequency, dtype=int)
    planted = np.asarray(planted, dtype=bool)
    if kind == "all":
        return np.ones(len(frequency), dtype=bool)
    if kind == "freq_eq_8":
        # All planted accounts have r=8; keep background accounts with exact
        # observed frequency 8.
        return planted | ((~planted) & (frequency == 8))
    if kind == "freq_7_9":
        return planted | ((~planted) & (frequency >= 7) & (frequency <= 9))
    raise ValueError(kind)


# ---------------------------------------------------------------------
# Co-activity
# ---------------------------------------------------------------------

def coactivity_from_item_participants(item_participants, all_accounts):
    """
    C_u = sum_{v != u} max(N_uv - 1, 0), where N_uv is the number of
    scored items shared by accounts u and v.

    Internal account keys are prefixed with P: or H: so planted and historical
    namespaces cannot collide.
    """
    pair_counts = defaultdict(int)

    for asin, participants in item_participants.items():
        if len(participants) != len(set(participants)):
            raise RuntimeError(f"{asin}: duplicate account inside scored item.")
        for u, v in combinations(sorted(participants), 2):
            pair_counts[(u, v)] += 1

    score = {u: 0.0 for u in all_accounts}
    repeated_pairs = 0
    max_overlap = 0
    for (u, v), n in pair_counts.items():
        max_overlap = max(max_overlap, n)
        if n > 1:
            extra = float(n - 1)
            score[u] += extra
            score[v] += extra
            repeated_pairs += 1

    return score, {
        "n_distinct_account_pairs_sharing_at_least_one_item": int(len(pair_counts)),
        "n_repeated_pairs": int(repeated_pairs),
        "max_pair_overlap": int(max_overlap),
    }


# ---------------------------------------------------------------------
# One seed
# ---------------------------------------------------------------------

def run_seed(
    seed,
    cfg,
    *,
    attack_root,
    reuse_root,
    roles_path,
    refs,
    freeze,
    out_root,
    overwrite,
):
    attack_seed_dir = Path(attack_root) / f"seed_{seed:03d}"
    reuse_seed_dir = Path(reuse_root) / f"seed_{seed:03d}"

    attack_summary_path = attack_seed_dir / "attack_summary.json"
    reuse_summary_path = reuse_seed_dir / "reuse_summary.json"
    attack_path = attack_seed_dir / "attack_world.csv.gz"
    assignment_path = reuse_seed_dir / "identity_assignment_r8.csv.gz"

    for p in [
        attack_summary_path,
        reuse_summary_path,
        attack_path,
        assignment_path,
    ]:
        if not p.exists():
            raise FileNotFoundError(p)

    out_seed = Path(out_root) / f"seed_{seed:03d}"
    out_seed.mkdir(parents=True, exist_ok=True)
    summary_path = out_seed / "summary.json"
    if summary_path.exists() and not overwrite:
        print(f"Seed {seed}: already complete, skipping.")
        existing = read_json(summary_path)
        require_artifact_category(
            existing,
            expected=cfg["category"],
            label=f"existing Stage-10 seed {seed}",
        )
        return existing

    s4 = read_json(attack_summary_path)
    reuse_summary = read_json(reuse_summary_path)
    require_artifact_category(
        s4, expected=cfg["category"], label="Stage-4 attack summary"
    )
    require_artifact_category(
        reuse_summary, expected=cfg["category"], label="Stage-4 reuse summary"
    )

    if s4.get("experiment") != "amazon_primary_attack":
        raise RuntimeError(f"Seed {seed}: unexpected Phase-2I attack summary.")
    if int(s4["m"]) != int(cfg["primary_modified_ratings_k"]):
        raise RuntimeError("Phase-2I intervention strength differs from Stage-10 config.")
    if not all(bool(v) for v in s4["invariants"].values()):
        raise RuntimeError("Phase-2I attack invariants did not all pass.")
    if not all(bool(v) for v in reuse_summary["invariants"].values()):
        raise RuntimeError("Phase-2J reuse invariants did not all pass.")
    if reuse_summary["attack_world_sha256"] != s4["attack_world_sha256"]:
        raise RuntimeError("Phase-2J attack hash differs from Phase-2I.")
    if "8" not in reuse_summary["identity_assignment_sha256_by_reuse"]:
        raise RuntimeError("Phase-2J summary does not contain r=8 identity assignment.")

    attack = load_attack_world(attack_path)
    slot_to_syn, assignment_by_asin = load_r8_assignment(assignment_path)

    # Verify exact r=8 identity-assignment hash before ranking.
    hh = hashlib.sha256()
    with gzip.open(assignment_path, "rt", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for e in rd:
            payload = {
                "asin": e["asin"],
                "treatment_block": e["treatment_block"],
                "treated_position": int(e["treated_position"]),
                "treated_source_line": int(e["treated_source_line"]),
                "synthetic_account_id": e["synthetic_account_id"],
                "d_cf": float(e["d_cf"]),
            }
            hh.update(
                json.dumps(
                    payload, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            )
            hh.update(b"\n")
    if hh.hexdigest() != reuse_summary["identity_assignment_sha256_by_reuse"]["8"]:
        raise RuntimeError("Phase-2J r=8 identity-assignment hash mismatch.")

    if len(attack) != int(cfg["n_attacked_items_per_seed"]):
        raise RuntimeError(
            f"Seed {seed}: expected {cfg['n_attacked_items_per_seed']} attack items, "
            f"got {len(attack)}."
        )

    lam = float(freeze["selected_lambda"])

    # -----------------------------------------------------------------
    # Full experimental-period stream over ALL eligible items.
    # We score both 30-review experimental blocks for every eligible item:
    #   A = 181--210, B = 211--240.
    # For the 2,000 attacked items, exactly one of these two blocks is the
    # frozen Stage-4 treated block and is modified; the other remains clean.
    # -----------------------------------------------------------------
    frequency = defaultdict(int)
    raw_score = defaultdict(float)
    pred_score = defaultdict(float)
    planted_ids = set()
    historical_ids = set()
    item_block_participants = {}

    observed_attack_slots = set()
    n_eligible_items = 0
    n_scored_blocks = 0
    n_scored_reviews = 0

    with gzip.open(roles_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            asin = str(obj["asin"])
            if asin not in refs:
                continue

            n_eligible_items += 1
            ref_counts, loo_probs = refs[asin]
            alpha, q = shrunk_reference(ref_counts, loo_probs, lam)
            bpred = exact_predictive_w1_baseline(alpha, 30)

            for block_name in ["experimental_A", "experimental_B"]:
                if block_name not in obj["blocks"]:
                    raise RuntimeError(f"{asin}: missing {block_name}")
                block = obj["blocks"][block_name]
                if len(block) != 30:
                    raise RuntimeError(f"{asin}/{block_name}: expected 30 reviews.")

                clean_counts = hist(block)
                obs_counts = clean_counts.copy()

                # Frozen Stage-4 treatment only on the selected block.
                donor_map = {}
                if asin in attack and block_name == attack[asin]["treatment_block"]:
                    for key, syn in assignment_by_asin[asin]:
                        if key[1] != block_name:
                            continue
                        donor_map[(key[2], key[3])] = syn
                    if len(donor_map) != int(cfg["primary_modified_ratings_k"]):
                        raise RuntimeError(
                            f"{asin}/{block_name}: expected six Stage-4 manipulated slots."
                        )
                    # Use the frozen Stage-4 attacked histogram exactly.
                    obs_counts = attack[asin]["attack_counts"].copy()

                raw_inc = w1_counts(obs_counts, q)
                pred_inc = raw_inc - bpred

                participants = []
                nplant_block = 0

                for rev in block:
                    pos = int(rev["position"])
                    src_line = int(rev["source_line"])

                    if (pos, src_line) in donor_map:
                        syn = donor_map[(pos, src_line)]
                        internal = "P:" + syn
                        planted_ids.add(internal)
                        nplant_block += 1
                        observed_attack_slots.add((asin, block_name, pos, src_line))
                    else:
                        internal = "H:" + str(rev["user_id"])
                        historical_ids.add(internal)

                    participants.append(internal)
                    frequency[internal] += 1
                    raw_score[internal] += raw_inc
                    pred_score[internal] += pred_inc
                    n_scored_reviews += 1

                if len(participants) != 30:
                    raise RuntimeError(f"{asin}/{block_name}: participant count != 30.")
                if len(set(participants)) != 30:
                    raise RuntimeError(f"{asin}/{block_name}: duplicate account in block.")

                if asin in attack and block_name == attack[asin]["treatment_block"]:
                    if nplant_block != int(cfg["primary_modified_ratings_k"]):
                        raise RuntimeError(
                            f"{asin}/{block_name}: expected six planted identities."
                        )
                elif nplant_block != 0:
                    raise RuntimeError(
                        f"{asin}/{block_name}: planted identity leaked into an untreated block."
                    )

                item_block_participants[(asin, block_name)] = participants
                n_scored_blocks += 1

    if observed_attack_slots != set(slot_to_syn):
        missing = set(slot_to_syn) - observed_attack_slots
        extra = observed_attack_slots - set(slot_to_syn)
        raise RuntimeError(
            f"Stage-10 full-background slot mismatch: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    expected_plants = (
        int(cfg["n_attacked_items_per_seed"])
        * int(cfg["primary_modified_ratings_k"])
        // int(cfg["reuse_r"])
    )
    if len(planted_ids) != expected_plants:
        raise RuntimeError(
            f"Expected {expected_plants} planted identities, got {len(planted_ids)}."
        )
    if any(frequency[u] != int(cfg["reuse_r"]) for u in planted_ids):
        raise RuntimeError("Not every planted identity has exactly r=8 exposures.")

    if planted_ids & historical_ids:
        raise RuntimeError("Internal account namespaces collided.")

    all_accounts = sorted(planted_ids | historical_ids)
    planted_vec = np.asarray([u in planted_ids for u in all_accounts], dtype=bool)
    freq_vec = np.asarray([frequency[u] for u in all_accounts], dtype=int)

    # IMPORTANT: for this operational experiment, start with the three channels
    # that are not confounded by Stage-4 incidence topology.
    score_channels = {
        "frequency": {u: float(frequency[u]) for u in all_accounts},
        "raw_w1": {u: float(raw_score[u]) for u in all_accounts},
        "predictive_centered_w1": {u: float(pred_score[u]) for u in all_accounts},
    }

    metrics = {}
    for channel, mapping in score_channels.items():
        full_scores = np.asarray([mapping[u] for u in all_accounts], dtype=float)
        metrics[channel] = {}

        for stratum in cfg["activity_strata"]:
            mask = stratum_mask(freq_vec, planted_vec, stratum)
            if int(planted_vec[mask].sum()) != expected_plants:
                raise RuntimeError(f"{stratum}: planted population changed.")

            metrics[channel][stratum] = ranking_metrics(
                full_scores[mask],
                planted_vec[mask],
                freq_vec[mask],
                top_fractions=cfg["top_fractions"],
                burden_targets=cfg["inspection_burden_targets"],
            )

    # Save account-level scores for audit. Historical reviewers remain unlabeled
    # background accounts, not verified negatives.
    account_rows = []
    for j, u in enumerate(all_accounts):
        is_plant = u.startswith("P:")
        account_rows.append({
            "account_type": "planted" if is_plant else "historical_background",
            "account_id": u[2:],
            "frequency": int(freq_vec[j]),
            "raw_w1_score": float(raw_score[u]),
            "predictive_centered_w1_score": float(pred_score[u]),
            "coactivity_score": "",
            "combined_score": "",
        })
    write_accounts(out_seed / "account_scores.csv.gz", account_rows)

    hist_freq_counts = defaultdict(int)
    for u in historical_ids:
        hist_freq_counts[int(frequency[u])] += 1

    summary = {
        "stage": "amazon_stage10_one_world_investigative_ranking_burden_full_background",
        "category": cfg["category"],
        "seed": int(seed),
        "source_stage4": {
            "attack_world_sha256": s4["attack_world_sha256"],
            "modified_ratings_k": int(cfg["primary_modified_ratings_k"]),
            "reuse_r": int(cfg["reuse_r"]),
            "n_attacked_items": int(len(attack)),
            "selected_lambda": lam,
        },
        "stream": {
            "n_eligible_items": int(n_eligible_items),
            "blocks_per_item": 2,
            "n_scored_blocks": int(n_scored_blocks),
            "n_scored_reviews": int(n_scored_reviews),
            "block_roles": ["experimental_A", "experimental_B"],
        },
        "population": {
            "n_planted_accounts": int(len(planted_ids)),
            "n_historical_background_accounts": int(len(historical_ids)),
            "n_total_accounts": int(len(all_accounts)),
            "n_historical_frequency_eq_8": int(
                sum(frequency[u] == 8 for u in historical_ids)
            ),
            "n_historical_frequency_7_9": int(
                sum(7 <= frequency[u] <= 9 for u in historical_ids)
            ),
            "historical_frequency_max": int(
                max([frequency[u] for u in historical_ids], default=0)
            ),
            "historical_frequency_histogram_1_12": {
                str(k): int(hist_freq_counts.get(k, 0)) for k in range(1, 13)
            },
        },
        "metrics": metrics,
        "invariants": {
            "all_eligible_items_scored_in_both_experimental_blocks": True,
            "historical_donor_identity_displaced_at_each_manipulated_slot": True,
            "exactly_30_accounts_per_scored_block": True,
            "exactly_6_planted_accounts_in_each_treated_stage4_block": True,
            "all_planted_accounts_have_frequency_8": True,
            "stage4_attack_world_reused_without_modification": True,
            "stage4_r8_identity_assignment_reused_without_modification": True,
            "historical_accounts_not_treated_as_verified_negatives": True,
            "no_auc_precision_fpr_or_recall_against_historical_background_reported": True,
            "coactivity_and_combined_scores_not_reported_in_full_background_stage10": True,
        },
        "interpretation_guard": (
            "This is a one-world investigative ranking-burden analysis over the "
            "full eligible experimental-period stream. Historical Amazon reviewers "
            "are an unlabeled background population, not verified benign accounts. "
            "Reported quantities describe where planted identities appear in an "
            "investigative ranking; they are not precision, false-positive rate, "
            "recall, or fraud-detection AUC."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


# ---------------------------------------------------------------------
# Across-seed aggregation
# ---------------------------------------------------------------------

def mean_t_ci(values):
    x = _finite(values)
    if len(x) == 0:
        return {
            "mean": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n": 0,
        }
    mean = float(x.mean())
    if len(x) == 1:
        return {"mean": mean, "ci_lower": mean, "ci_upper": mean, "n": 1}
    se = float(x.std(ddof=1) / math.sqrt(len(x)))
    crit = float(t.ppf(0.975, df=len(x) - 1))
    return {
        "mean": mean,
        "ci_lower": mean - crit * se,
        "ci_upper": mean + crit * se,
        "n": int(len(x)),
    }


def aggregate_all(
    seed_summaries,
    cfg,
    out_root,
    *,
    layout,
    config_path,
    raw_dataset_sha256,
    freeze,
    upstream_artifacts,
):
    channels = [
        "frequency",
        "raw_w1",
        "predictive_centered_w1",
    ]
    strata = list(cfg["activity_strata"])

    # Only aggregate scalar ranking outputs, preserving the nested schema.
    final_metrics = {}
    for channel in channels:
        final_metrics[channel] = {}
        for stratum in strata:
            example = seed_summaries[0]["metrics"][channel][stratum]
            final_metrics[channel][stratum] = {}
            for key, val in example.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    vals = [
                        s["metrics"][channel][stratum][key]
                        for s in seed_summaries
                    ]
                    final_metrics[channel][stratum][key] = mean_t_ci(vals)

    pop_keys = [
        "n_planted_accounts",
        "n_historical_background_accounts",
        "n_total_accounts",
        "n_historical_frequency_eq_8",
        "n_historical_frequency_7_9",
        "historical_frequency_max",
    ]
    population = {
        k: mean_t_ci([s["population"][k] for s in seed_summaries])
        for k in pop_keys
    }

    final = {
        "stage": "amazon_stage10_one_world_investigative_ranking_burden_full_background",
        "category": layout.category,
        "raw_dataset_sha256": raw_dataset_sha256,
        "selected_lambda": float(freeze["selected_lambda"]),
        "n_seeds": len(seed_summaries),
        "seed_ids": [int(s["seed"]) for s in seed_summaries],
        "population": population,
        "metrics": final_metrics,
        "top_fractions": cfg["top_fractions"],
        "inspection_burden_targets": cfg["inspection_burden_targets"],
        "activity_strata": cfg["activity_strata"],
        "interpretation_guard": seed_summaries[0]["interpretation_guard"],
    }

    final["provenance"] = build_provenance(
        category=layout.category,
        config_path=config_path,
        raw_dataset_sha256=raw_dataset_sha256,
        selected_lambda=freeze["selected_lambda"],
        seed_ids=final["seed_ids"],
        upstream_artifacts=upstream_artifacts,
        repo_root=ROOT,
        shared_root=layout.shared_category_root,
    )
    out_path = Path(out_root) / layout.artifact_name(
        "stage10_full_background_summary"
    )
    out_path.write_text(json.dumps(final, indent=2, sort_keys=True), encoding="utf-8")
    return final


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(ROOT / "configs" / "amazon_full_background.yaml"),
    )
    ap.add_argument(
        "--shared-root",
        type=Path,
        help="Operational shared-root override; does not alter the config file.",
    )
    ap.add_argument(
        "--attack-dir",
        default=None,
    )
    ap.add_argument(
        "--reuse-dir",
        default=None,
    )
    ap.add_argument(
        "--roles",
        default=None,
    )
    ap.add_argument(
        "--references",
        default=None,
    )
    ap.add_argument(
        "--freeze",
        default=None,
    )
    ap.add_argument(
        "--out-dir",
        default=None,
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed", type=int)
    g.add_argument("--all-seeds", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    layout = AmazonPathLayout.from_config(
        cfg,
        repo_root=ROOT,
        shared_root=args.shared_root,
    )
    shared = cfg["shared_data"]
    roles_path = (
        Path(args.roles).resolve() if args.roles
        else layout.resolve_shared_path(shared["roles_file"])
    )
    references_path = (
        Path(args.references).resolve() if args.references
        else layout.resolve_shared_path(shared["reference_histograms"])
    )
    freeze_path = (
        Path(args.freeze).resolve() if args.freeze
        else layout.resolve_shared_path(shared["reference_freeze"])
    )
    attack_root = (
        Path(args.attack_dir).resolve() if args.attack_dir
        else layout.resolve_shared_path(shared["attack_dir"])
    )
    reuse_root = (
        Path(args.reuse_dir).resolve() if args.reuse_dir
        else layout.resolve_shared_path(shared["reuse_dir"])
    )
    out_root = (
        Path(args.out_dir).resolve() if args.out_dir
        else layout.resolve_shared_path(shared["output_subdir"])
    )
    refs = load_reference_table(references_path)
    freeze = read_json(freeze_path)
    require_artifact_category(
        freeze,
        expected=cfg["category"],
        label="Stage-3 freeze",
        allow_legacy_home_missing=False,
    )
    out_root.mkdir(parents=True, exist_ok=True)

    if not roles_path.exists():
        raise FileNotFoundError(
            f"Stage-10 requires the Stage-2 roles file with historical user IDs: "
            f"{roles_path}"
        )

    seed_ids = [int(x) for x in cfg["seed_ids"]]

    if args.seed is not None:
        if args.seed not in seed_ids:
            raise ValueError(f"Seed {args.seed} is not in the frozen Stage-10 seed list.")
        s = run_seed(
            args.seed,
            cfg,
            attack_root=attack_root,
            reuse_root=reuse_root,
            roles_path=roles_path,
            refs=refs,
            freeze=freeze,
            out_root=out_root,
            overwrite=args.overwrite,
        )
        print(json.dumps({
            "stream": s["stream"],
            "population": s["population"],
            "frequency": s["metrics"]["frequency"],
            "predictive_centered_w1": s["metrics"]["predictive_centered_w1"],
        }, indent=2))
        print(f"Summary: {out_root / f'seed_{args.seed:03d}' / 'summary.json'}")
        return

    summaries = []
    for seed in seed_ids:
        print(f"\n=== Stage 10 seed {seed} ===")
        summaries.append(
            run_seed(
                seed,
                cfg,
                attack_root=attack_root,
                reuse_root=reuse_root,
                roles_path=roles_path,
                refs=refs,
                freeze=freeze,
                out_root=out_root,
                overwrite=args.overwrite,
            )
        )

    upstream_artifacts = {
        "roles_file": roles_path,
        "reference_histograms": references_path,
        "reference_freeze": freeze_path,
    }
    for label, path in {
        "stage4_attack_summary": attack_root / layout.artifact_name(
            "stage4_attack_summary"
        ),
        "stage4_reuse_summary": reuse_root / layout.artifact_name(
            "stage4_reuse_summary"
        ),
    }.items():
        if path.is_file():
            artifact = read_json(path)
            require_artifact_category(
                artifact, expected=cfg["category"], label=label
            )
            upstream_artifacts[label] = path
    raw_dataset_sha256 = resolve_raw_dataset_sha256(layout, freeze)
    final = aggregate_all(
        summaries,
        cfg,
        out_root,
        layout=layout,
        config_path=config_path,
        raw_dataset_sha256=raw_dataset_sha256,
        freeze=freeze,
        upstream_artifacts=upstream_artifacts,
    )
    print("\nStage 10 complete.")
    print(json.dumps({
        "population": final["population"],
        "frequency_all": final["metrics"]["frequency"]["all"],
        "predictive_centered_w1_all": final["metrics"]["predictive_centered_w1"]["all"],
        "predictive_centered_w1_freq_eq_8": final["metrics"]["predictive_centered_w1"]["freq_eq_8"],
        "predictive_centered_w1_freq_7_9": final["metrics"]["predictive_centered_w1"]["freq_7_9"],
    }, indent=2))
    print(
        "Summary:",
        out_root / layout.artifact_name("stage10_full_background_summary"),
    )


if __name__ == "__main__":
    main()
