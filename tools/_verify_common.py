"""Shared, verifier-only comparison helpers.

The comparison policy deliberately separates derived floating-point values
from structural data:

* floating-point leaves use ``math.isclose`` with both relative and absolute
  tolerance;
* dictionary keys, list lengths, strings, booleans, and integer-valued leaves
  remain exact;
* callers that historically treat matching NaNs as equal retain that behavior
  through :func:`recursive_compare_nan_equal`.

The compatibility wrappers continue to return ``[(path, reason), ...]`` so
existing verifiers can adopt the shared policy without changing their report
format. New or migrated verifiers can use :func:`compare_with_report` to also
record the measured numerical discrepancy envelope.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any


# These are the artifact-wide candidate tolerances already used by the generic
# verifier in tools/verify_results.py. The measured Linux/Windows envelope is
# recorded separately by verifier reports rather than encoded here.
DEFAULT_RTOL = 1e-10
DEFAULT_ATOL = 1e-12


@dataclass
class ComparisonReport:
    """Result and numerical envelope for one recursive comparison."""

    rtol: float
    atol: float
    equal_nan: bool
    differences: list[tuple[str, str]] = field(default_factory=list)
    float_values_checked: int = 0
    nonzero_float_differences: int = 0
    max_abs_diff: float = 0.0
    max_abs_path: str | None = None
    max_abs_expected: float | None = None
    max_abs_actual: float | None = None
    max_rel_diff: float = 0.0
    max_rel_path: str | None = None
    max_rel_expected: float | None = None
    max_rel_actual: float | None = None

    @property
    def status(self) -> str:
        return "PASS" if not self.differences else "FAIL"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, auditable comparison summary."""

        return {
            "status": self.status,
            "differences": [
                {"path": path, "reason": reason}
                for path, reason in self.differences
            ],
            "tolerances": {
                "rtol": self.rtol,
                "atol": self.atol,
                "equal_nan": self.equal_nan,
            },
            "numeric_envelope": {
                "float_values_checked": self.float_values_checked,
                "nonzero_float_differences": self.nonzero_float_differences,
                "max_abs_diff": self.max_abs_diff,
                "max_abs_path": self.max_abs_path,
                "max_abs_expected": self.max_abs_expected,
                "max_abs_actual": self.max_abs_actual,
                "max_rel_diff": self.max_rel_diff,
                "max_rel_path": self.max_rel_path,
                "max_rel_expected": self.max_rel_expected,
                "max_rel_actual": self.max_rel_actual,
            },
        }


def float_close(
    actual: Real,
    expected: Real,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    equal_nan: bool = False,
) -> bool:
    """Compare two floating-point values under the shared policy."""

    actual_float = float(actual)
    expected_float = float(expected)
    if math.isnan(actual_float) or math.isnan(expected_float):
        return (
            equal_nan
            and math.isnan(actual_float)
            and math.isnan(expected_float)
        )
    return math.isclose(
        actual_float,
        expected_float,
        rel_tol=rtol,
        abs_tol=atol,
    )


def _is_integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_float(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, (Integral, bool))
    )


def _record_float(
    report: ComparisonReport,
    *,
    path: str,
    expected: Real,
    actual: Real,
) -> None:
    expected_float = float(expected)
    actual_float = float(actual)
    report.float_values_checked += 1

    # NaNs and infinities do not have a meaningful finite error envelope. The
    # equality decision is still made explicitly below.
    if math.isfinite(expected_float) and math.isfinite(actual_float):
        abs_diff = abs(actual_float - expected_float)
        rel_diff = abs_diff / max(abs(expected_float), 1e-300)
        if abs_diff != 0.0:
            report.nonzero_float_differences += 1
        if abs_diff > report.max_abs_diff:
            report.max_abs_diff = abs_diff
            report.max_abs_path = path
            report.max_abs_expected = expected_float
            report.max_abs_actual = actual_float
        if rel_diff > report.max_rel_diff:
            report.max_rel_diff = rel_diff
            report.max_rel_path = path
            report.max_rel_expected = expected_float
            report.max_rel_actual = actual_float

    if not float_close(
        actual_float,
        expected_float,
        rtol=report.rtol,
        atol=report.atol,
        equal_nan=report.equal_nan,
    ):
        report.differences.append((path, "float"))


def _compare_value(
    expected: Any,
    actual: Any,
    *,
    path: str,
    report: ComparisonReport,
) -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            report.differences.append((path, "type"))
            return
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            report.differences.append((path, "keys"))
        for key in sorted(expected_keys & actual_keys, key=repr):
            _compare_value(
                expected[key],
                actual[key],
                path=f"{path}.{key}",
                report=report,
            )
        return

    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            report.differences.append((path, "list"))
            return
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual)
        ):
            _compare_value(
                expected_item,
                actual_item,
                path=f"{path}[{index}]",
                report=report,
            )
        return

    if isinstance(expected, bool):
        if not isinstance(actual, bool) or expected is not actual:
            report.differences.append((path, "bool"))
        return

    if _is_integer(expected):
        if not _is_integer(actual) or expected != actual:
            report.differences.append((path, "int"))
        return

    if _is_float(expected):
        if not _is_float(actual):
            report.differences.append((path, "float_type"))
            return
        _record_float(
            report,
            path=path,
            expected=expected,
            actual=actual,
        )
        return

    if type(expected) is not type(actual) or expected != actual:
        report.differences.append((path, "value"))


def compare_with_report(
    expected: Any,
    actual: Any,
    path: str = "$",
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    equal_nan: bool = False,
) -> ComparisonReport:
    """Recursively compare values and return differences plus error metrics."""

    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    report = ComparisonReport(rtol=rtol, atol=atol, equal_nan=equal_nan)
    _compare_value(expected, actual, path=path, report=report)
    return report


def recursive_compare(
    expected: Any,
    actual: Any,
    path: str = "$",
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> list[tuple[str, str]]:
    """Compatibility wrapper with strict NaN semantics.

    ``atol`` intentionally remains the fourth positional parameter so existing
    verifier calls retain their historical API.
    """

    return compare_with_report(
        expected,
        actual,
        path,
        rtol=rtol,
        atol=atol,
        equal_nan=False,
    ).differences


def recursive_compare_nan_equal(
    expected: Any,
    actual: Any,
    path: str = "$",
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> list[tuple[str, str]]:
    """Compatibility wrapper that accepts NaNs at matching locations."""

    return compare_with_report(
        expected,
        actual,
        path,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    ).differences

# Phase-4 category-aware Amazon summaries add portable provenance metadata to
# legacy Home-and-Kitchen scientific JSON.  Legacy gold intentionally remains
# immutable, so verifiers compare the legacy scientific schema while allowing
# only these explicitly enumerated top-level additions.
LEGACY_ADDITIVE_TOP_LEVEL_KEYS = frozenset({
    "category",
    "raw_dataset_sha256",
    "selected_lambda",
    "selected_lambda_scope",
    "provenance",
})


def compare_legacy_json_with_report(
    expected: Any,
    actual: Any,
    path: str = "$",
    *,
    allowed_extra_keys: frozenset[str] = LEGACY_ADDITIVE_TOP_LEVEL_KEYS,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    equal_nan: bool = False,
) -> ComparisonReport:
    """Compare legacy gold to a category-aware JSON artifact safely.

    Only explicitly whitelisted *top-level* additive metadata fields may be
    absent from the immutable legacy gold.  Missing legacy fields or any other
    unexpected fields still fail.  The scientific subtree is compared using
    the normal strict structural / tolerant-floating policy.
    """

    if not isinstance(expected, dict) or not isinstance(actual, dict):
        return compare_with_report(
            expected,
            actual,
            path,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
        )

    report = ComparisonReport(rtol=rtol, atol=atol, equal_nan=equal_nan)
    expected_keys = set(expected)
    actual_keys = set(actual)
    missing = expected_keys - actual_keys
    unexpected = actual_keys - expected_keys - set(allowed_extra_keys)
    if missing:
        report.differences.append((path, f"missing_keys:{sorted(missing)}"))
    if unexpected:
        report.differences.append((path, f"unexpected_keys:{sorted(unexpected)}"))

    for key in sorted(expected_keys & actual_keys, key=repr):
        _compare_value(
            expected[key],
            actual[key],
            path=f"{path}.{key}",
            report=report,
        )
    return report
