#!/usr/bin/env python3
"""Check the environment needed for exact controlled-result reproduction."""

from __future__ import annotations

import platform
import sys

import numpy
import scipy
import sklearn
import yaml


EXPECTED = {
    "python": "3.12.10",
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "sklearn": "1.9.0",
    "pyyaml": "6.0.3",
}


def current():
    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "pyyaml": yaml.__version__,
    }


def main() -> int:
    got = current()
    ok = True
    print("Exact paper-reproduction environment check")
    print("=" * 56)
    for key, expected in EXPECTED.items():
        actual = got[key]
        match = actual == expected
        ok &= match
        print(f"{key:10s} expected={expected:10s} actual={actual:10s} "
              f"{'PASS' if match else 'FAIL'}")
    print("=" * 56)
    print("STATUS:", "PASS" if ok else "FAIL")
    if not ok:
        print(
            "The code may still run, but exact paper reproduction requires the "
            "frozen scientific environment above. Version differences can change "
            "RNG streams or numerical/statistical behavior and therefore may not "
            "reproduce the frozen per-seed results exactly."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
