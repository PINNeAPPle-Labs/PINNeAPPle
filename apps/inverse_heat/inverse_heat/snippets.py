"""Complete scripts shown in the app (each case's tab and the Code tab): they reproduce the case with
``pip install pinneapple`` and nothing else. The files live in ``scripts/`` and are run by tests/test_inverse_heat.py.

The page fills each script's parameter block (lines ``NAME = value``) with the inputs and readings of the run."""
import os

_DIR = os.path.join(os.path.dirname(__file__), "scripts")
_CASES = {
    "fin": ("1D fin: h from thermocouples along a pin fin", "fin_h_from_thermocouples.py", "about 30 s on a CPU"),
    "plate": ("2D plate: h and the hot spot of a heat spreader", "plate_h_from_thermocouples.py", "about 3 min on a CPU"),
    "block": ("3D block: the chip temperature under a fan-cooled block", "block_chip_temperature.py",
              "about 30 min on a CPU, minutes on a GPU"),
}


def _load():
    out = {}
    for case, (title, name, runtime) in _CASES.items():
        with open(os.path.join(_DIR, name), encoding="utf-8") as fh:
            out[case] = {"title": title, "file": name, "runtime": runtime, "code": fh.read()}
    return out


SNIPPETS = _load()
