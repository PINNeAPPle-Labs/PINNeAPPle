"""Run a PICT case and bring its fields back as arrays PINNeAPPle can check and store."""
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .case import CavityCase, build_cavity_domain, pict_availability

__all__ = ["PICTRunConfig", "PICTResult", "run_cavity"]


@dataclass
class PICTRunConfig:
    """The PISO settings PICT takes, in the names its ``Simulation`` uses.

    ``substeps`` is either a positive number of substeps per output step or ``"ADAPTIVE"`` (PICT's CFL-based split).
    ``differentiable`` records the steps on the torch graph, which is what a network trained through the solver
    needs and what makes the run several times slower.
    """

    iterations: int = 100
    time_step: float = 0.25
    substeps: Any = "ADAPTIVE"
    corrector_steps: int = 2
    adaptive_cfl: float = 0.8
    non_orthogonal: bool = False
    pressure_tol: float = 1e-5
    advect_non_ortho_steps: int = 1
    pressure_non_ortho_steps: int = 1
    velocity_corrector: str = "FD"
    differentiable: bool = False
    record_every: int = 10
    log_dir: str | None = None

    def to_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``PISOtorch_simulation.Simulation(...)``."""
        return {
            "time_step": self.time_step,
            "substeps": self.substeps,
            "corrector_steps": self.corrector_steps,
            "adaptive_CFL": self.adaptive_cfl,
            "non_orthogonal": self.non_orthogonal,
            "advect_non_ortho_steps": self.advect_non_ortho_steps,
            "pressure_non_ortho_steps": self.pressure_non_ortho_steps,
            "pressure_tol": self.pressure_tol,
            "velocity_corrector": self.velocity_corrector,
            "differentiable": self.differentiable,
            "log_interval": 0,
            "log_dir": self.log_dir,
        }


@dataclass
class PICTResult:
    """What a PICT run left behind: the recorded velocity and pressure, and the case they came from.

    ``velocity`` is ``(snapshots, 2, ny, nx)`` in the order PICT stores a field (channels first), ``velocity[0]``
    being the x component; ``pressure`` is ``(snapshots, ny, nx)`` and is the pressure PICT reports, whose datum is
    arbitrary (only differences drive the flow).
    """

    velocity: np.ndarray
    pressure: np.ndarray
    times: np.ndarray
    case_summary: dict[str, Any]
    config: dict[str, Any]
    device: str
    wall_time_s: float
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.velocity.ndim != 4 or self.velocity.shape[1] != 2:
            raise ValueError("velocity must be (snapshots, 2, ny, nx)")

    # -- what the fields mean -------------------------------------------------

    @property
    def nx(self) -> int:
        return int(self.velocity.shape[3])

    @property
    def ny(self) -> int:
        return int(self.velocity.shape[2])

    def component(self, name: str) -> np.ndarray:
        """``"u"`` or ``"v"``: ``(snapshots, ny, nx)``."""
        try:
            return self.velocity[:, {"u": 0, "v": 1}[name]]
        except KeyError:
            raise KeyError(f"component must be 'u' or 'v', not {name!r}") from None

    def cell_centers(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre coordinates from the case bounds (the recorded fields are cell-centred)."""
        x0, y0, x1, y1 = self.case_summary["bounds"]
        return (np.linspace(x0, x1, self.nx, endpoint=False) + 0.5 * (x1 - x0) / self.nx,
                np.linspace(y0, y1, self.ny, endpoint=False) + 0.5 * (y1 - y0) / self.ny)

    def checks(self) -> dict[str, Any]:
        """The physics checks of :func:`~pinneapple_simulation.external_solvers.pict.validation.cavity_checks`."""
        from .validation import cavity_checks

        return cavity_checks(self)

    def to_upd(self):
        """The recorded fields as a PINNeAPPle ``PhysicalSample`` (time, x, y and the three fields)."""
        import torch

        from pinneapple_data.physical_sample import PhysicalSample

        x, y = self.cell_centers()
        state = {
            "u": torch.as_tensor(self.component("u").reshape(len(self.times), -1), dtype=torch.float32),
            "v": torch.as_tensor(self.component("v").reshape(len(self.times), -1), dtype=torch.float32),
            "p": torch.as_tensor(self.pressure.reshape(len(self.times), -1), dtype=torch.float32),
        }
        return PhysicalSample(
            state=state,
            domain={"type": "grid", "coords": {"time": self.times.astype(np.float32),
                                               "x": x.astype(np.float32), "y": y.astype(np.float32)}},
            provenance={"version": "0.1", "source": "pict", "case": "lid_driven_cavity",
                        **self.case_summary, "config": self.config, "wall_time_s": self.wall_time_s},
            schema={"units": {"u": "m/s", "v": "m/s", "p": "Pa", "time": "s", "x": "m", "y": "m"}},
        )


def run_cavity(case: CavityCase, config: PICTRunConfig | None = None) -> PICTResult:
    """Run ``case`` on PICT and record ``config.record_every`` snapshots of velocity and pressure.

    Raises :class:`ImportError` naming the missing part when PICT cannot run here: no numbers are invented for a
    machine without the solver.
    """
    availability = pict_availability()
    if not availability.available:
        raise ImportError(f"PICT cannot run here: {'; '.join(availability.missing)}. {availability.hint}")

    import PISOtorch_simulation

    config = config or PICTRunConfig()
    device = availability.device
    domain, block = build_cavity_domain(case, device=device)
    domain.CreatePressureOnBlocks()

    sim = PISOtorch_simulation.Simulation(domain=domain, **config.to_kwargs())

    velocity, pressure, times = [], [], []
    started = time.perf_counter()
    done = 0
    while done < config.iterations:
        chunk = min(config.record_every, config.iterations - done)
        sim.run(iterations=chunk, log_domain=False)
        done += chunk
        velocity.append(block.getVelocity(False).detach().cpu().numpy()[0])       # (2, ny, nx)
        pressure.append(block.pressure.detach().cpu().numpy()[0][0])             # (ny, nx)
        times.append(done * config.time_step)

    return PICTResult(velocity=np.stack(velocity), pressure=np.stack(pressure), times=np.asarray(times),
                      case_summary=case.summary(), config=vars(config), device=device,
                      wall_time_s=time.perf_counter() - started,
                      meta={"pict": "PICT (tum-pbs), multi-block PISO on CUDA", "iterations": config.iterations})
