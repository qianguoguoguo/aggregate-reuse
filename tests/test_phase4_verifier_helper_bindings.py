from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HELPERS = {
    "compare_legacy_json_with_report",
    "aggregate_provenance_matches",
    "per_seed_category_matches",
}

VERIFIERS = [
    "verify_primary_attack.py",
    "verify_primary_reuse.py",
    "verify_matched_twins.py",
    "verify_amazon_shape.py",
    "verify_complementarity.py",
    "verify_reference_history.py",
    "verify_strength_fixed.py",
    "verify_k6_population_audit.py",
    "verify_full_background_ranking.py",
    "verify_self_influence.py",
]


def module_bound_names(tree: ast.Module) -> set[str]:
    bound: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    bound.add(target.id)
    return bound


def test_phase4_verifiers_bind_every_new_helper_name() -> None:
    failures: list[str] = []
    for name in VERIFIERS:
        path = ROOT / "tools" / name
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        loaded = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        needed = loaded & HELPERS
        missing = needed - module_bound_names(tree)
        if missing:
            failures.append(f"{name}: {sorted(missing)}")
    assert not failures, "unbound Phase-4 verifier helpers: " + "; ".join(failures)
