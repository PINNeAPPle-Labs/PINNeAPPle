"""MuJoCo model loading and data initialization for PINNeAPPle.

Wraps ``mujoco.MjModel`` / ``mujoco.MjData`` construction with a structured config so the rest of the pipeline stays
framework-agnostic, and hides the differences between the MuJoCo 3.x and 4.x wheels:

* 4.0 split the distribution: ``mujoco`` (bindings) now depends on ``mujoco-core`` (engine) and ``mujoco-studio``,
  and MJX moved out into its own ``mujoco-mjx`` package (``from mujoco import mjx`` still works once it is installed).
* 4.0 requires Python ≥ 3.11; 3.17 is the last line that installs on 3.10.
* ``mujoco.batch`` (thread-pool batch stepping) exists since 3.17, so both lines can use
  :class:`~.batch.BatchRunner`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "MUJOCO_MIN_VERSION",
    "MuJoCoConfig",
    "mujoco_version",
    "mujoco_available",
    "require_mujoco",
    "has_batch",
    "has_mjx",
    "load_model",
    "make_data",
    "reset_data",
    "load_spec",
    "spec_to_xml",
    "model_summary",
]

#: Oldest MuJoCo line this layer supports (``mujoco.batch`` and the 3.16 compiler land there).
MUJOCO_MIN_VERSION = (3, 17)


def require_mujoco():
    """Import and return the ``mujoco`` module, with an actionable message when it is missing."""
    try:
        import mujoco
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "mujoco is required for this integration. Install it with:\n"
            '  pip install "pinneapple[mujoco]"\n'
            "(MuJoCo 4.0 needs Python >= 3.11; on 3.10 the extra installs the 3.17 line.)"
        ) from e
    return mujoco


def mujoco_available() -> bool:
    """Whether the MuJoCo bindings can be imported here."""
    import importlib.util

    return importlib.util.find_spec("mujoco") is not None


def mujoco_version() -> tuple | None:
    """Installed MuJoCo version as a tuple of ints (``(4, 0, 0)``), or ``None`` when it is not installed."""
    import importlib.util

    if importlib.util.find_spec("mujoco") is None:
        return None
    mujoco = require_mujoco()
    parts = []
    for piece in str(mujoco.__version__).split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or None


def _version_at_least(version: tuple) -> bool:
    return mujoco_version() is not None and tuple(mujoco_version()) >= tuple(version)


def has_batch() -> bool:
    """Whether ``mujoco.batch`` (thread-pool batch stepping, 3.17+) is available."""
    if not mujoco_available() or not _version_at_least((3, 17)):
        return False
    import importlib.util

    return importlib.util.find_spec("mujoco.batch") is not None


def has_mjx() -> bool:
    """Whether MJX (JAX rollouts) can be imported.

    In MuJoCo 4.0 MJX ships in the separate ``mujoco-mjx`` package; on 3.x it came inside ``mujoco``.
    The import path is ``mujoco.mjx`` in both cases.
    """
    if not mujoco_available() or not _version_at_least((3, 5)):
        return False
    import importlib.util

    return importlib.util.find_spec("mujoco.mjx") is not None


@dataclass
class MuJoCoConfig:
    """Configuration for a MuJoCo simulation.

    Parameters
    ----------
    model_path : path to a ``.xml`` (MJCF/URDF) or ``.mjb`` binary model file.
        Mutually exclusive with ``model_xml``.
    model_xml : raw XML string (MJCF). Mutually exclusive with ``model_path``.
    assets : dict mapping virtual filenames to bytes — forwarded to ``mujoco.MjModel.from_xml_string`` when
        using ``model_xml`` (meshes and textures referenced by the XML).
    dt : override for ``model.opt.timestep`` (seconds). ``None`` keeps the model default.
    n_steps : number of ``mj_step`` calls per :meth:`~.runner.MuJoCoRunner.simulate` run.
    n_substeps : how many ``mj_step`` calls to batch per loop iteration (``mj_step(..., nstep=n_substeps)``).
    gravity : override gravity vector (3-tuple). ``None`` keeps the model default.
    keyframe : name or index of the ``<keyframe>`` the runner resets to. ``None`` resets to ``qpos0``.
    record_energy : turn on ``mjENBL_ENERGY`` so the runner records potential and kinetic energy per step.
    record_contacts : record the number of active contacts per step.
    extra_opts : additional key-value pairs written into ``model.opt`` before the simulation runs.
    """

    model_path: str | Path | None = None
    model_xml: str | None = None
    assets: dict[str, bytes] = field(default_factory=dict)
    dt: float | None = None
    n_steps: int = 1000
    n_substeps: int = 1
    gravity: tuple | None = None
    keyframe: str | int | None = None
    record_energy: bool = True
    record_contacts: bool = True
    extra_opts: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.model_path is None and self.model_xml is None:
            raise ValueError("Provide either model_path or model_xml.")
        if self.model_path is not None and self.model_xml is not None:
            raise ValueError("Provide only one of model_path or model_xml.")
        if self.n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {self.n_steps}")
        if self.n_substeps < 1:
            raise ValueError(f"n_substeps must be >= 1, got {self.n_substeps}")
        if self.dt is not None and not self.dt > 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.gravity is not None and len(tuple(self.gravity)) != 3:
            raise ValueError(f"gravity must have three components, got {self.gravity}")


def _is_binary(path: str | Path) -> bool:
    return str(path).lower().endswith(".mjb")


def load_model(config: MuJoCoConfig):
    """Load a ``mujoco.MjModel`` from the config and apply its overrides.

    Raises
    ------
    ImportError
        When ``mujoco`` is not installed.
    ValueError, mujoco.FatalError
        When the model cannot be compiled (invalid XML, missing mesh, ...).
    """
    mujoco = require_mujoco()

    if config.model_path is not None:
        path = str(config.model_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"MuJoCo model not found: {path}")
        if _is_binary(path):
            model = mujoco.MjModel.from_binary_path(path)
        else:
            model = mujoco.MjModel.from_xml_path(path)
    else:
        model = mujoco.MjModel.from_xml_string(config.model_xml, config.assets or {})

    if config.dt is not None:
        model.opt.timestep = float(config.dt)
    if config.gravity is not None:
        model.opt.gravity[:] = config.gravity
    for key, value in config.extra_opts.items():
        setattr(model.opt, key, value)
    if config.record_energy:
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY
    return model


def make_data(model):
    """Allocate a fresh ``mujoco.MjData`` for ``model``."""
    mujoco = require_mujoco()
    return mujoco.MjData(model)


def reset_data(model, data, keyframe: str | int | None = None) -> None:
    """Reset ``data`` to ``keyframe`` (name or index) or to ``qpos0``, and sync the derived fields."""
    mujoco = require_mujoco()
    if keyframe is None:
        mujoco.mj_resetData(model, data)
    else:
        key = keyframe
        if isinstance(keyframe, str):
            key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            if key < 0:
                available = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i) for i in range(model.nkey)]
                raise KeyError(f"unknown keyframe {keyframe!r}; the model has {available}")
        if not 0 <= int(key) < model.nkey:
            raise IndexError(f"keyframe index {key} out of range (model has {model.nkey})")
        mujoco.mj_resetDataKeyframe(model, data, int(key))
    mujoco.mj_forward(model, data)


def load_spec(source: str | Path, assets: dict[str, bytes] | None = None):
    """Load an editable ``mujoco.MjSpec`` from an XML file, an XML string or a binary ``.mjb``.

    ``MjSpec`` is the model-editing API: change a body, joint or option, then ``spec.compile()`` to get the model.
    """
    mujoco = require_mujoco()
    text = str(source)
    if os.path.exists(text):
        if _is_binary(text):
            return mujoco.MjSpec.from_binary_path(text)
        return mujoco.MjSpec.from_file(text)
    if text.lstrip().startswith("<"):
        if assets:
            return mujoco.MjSpec.from_string(text, assets)
        return mujoco.MjSpec.from_string(text)
    raise FileNotFoundError(f"Neither a file nor an MJCF string: {text[:80]!r}")


def spec_to_xml(spec) -> str:
    """Serialize an ``MjSpec`` back to MJCF (usable with ``MjModel.from_xml_string``)."""
    if not hasattr(spec, "to_xml"):  # pragma: no cover - only very old 3.x builds
        raise NotImplementedError("MjSpec.to_xml() needs MuJoCo >= 3.2")
    return spec.to_xml()


def model_summary(model) -> dict:
    """Sizes, options and version of a compiled model — the provenance PINNeAPPle records for every run."""
    mujoco = require_mujoco()

    def _name(enum, value):
        try:
            return enum(value).name.replace("mj", "").replace("_", "").lower()
        except Exception:  # pragma: no cover - defensive, enum names differ across versions
            return int(value)

    return {
        "mujoco": mujoco.__version__,
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "nsensor": int(model.nsensor),
        "nkey": int(model.nkey),
        "dt": float(model.opt.timestep),
        "integrator": _name(mujoco.mjtIntegrator, model.opt.integrator),
        "solver": _name(mujoco.mjtSolver, model.opt.solver),
        "gravity": [float(v) for v in model.opt.gravity],
    }
