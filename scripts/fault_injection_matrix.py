"""Regenerate the detection matrix of the validation checks (#697): every check against the fault library on
Darcy 2-D and Burgers 1-D.

    python scripts/fault_injection_matrix.py --out docs/guides        # full size, writes check_power_matrix.md
    python scripts/fault_injection_matrix.py --small                  # CI size, prints, exit 1 on a failure

All the work is in ``pinneapple_analysis.validation.fault_injection``; this file only calls it.
"""
import sys

from pinneapple_analysis.validation.fault_injection.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
