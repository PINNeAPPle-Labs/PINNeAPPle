"""Run OpenRadioss (starter + engine + ANIM->VTK conversion) through Docker.

OpenRadioss ships Linux binaries (x86-64 and aarch64); on macOS the simplest
reproducible route is a plain Ubuntu container with the unpacked release
mounted read-only. To keep disk usage bounded, each ANIM frame is converted to
legacy-VTK and immediately reduced inside the container to its nodal
displacement block (``disp_XXX.txt``); frame 0 is kept as a full VTK so the
mesh (points, cells, part ids) can be read back.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

__all__ = ["OpenRadiossDockerConfig", "run_openradioss_case", "run_openradioss_cases"]

_SCRIPT = r"""
set -e
OR=/opt/OpenRadioss
ARCH={arch}
export OPENRADIOSS_PATH=$OR RAD_CFG_PATH=$OR/hm_cfg_files
export RAD_H3D_PATH=$OR/extlib/h3d/lib/$ARCH
export LD_LIBRARY_PATH=$OR/extlib/ArmFlang_runtime/$ARCH:$OR/extlib/h3d/lib/$ARCH:$OR/extlib/hm_reader/$ARCH:$LD_LIBRARY_PATH
export OMP_STACKSIZE=400m OMP_NUM_THREADS={threads}
cd /case
$OR/exec/starter_$ARCH -i {starter} -np 1 > starter.log 2>&1
$OR/exec/engine_$ARCH -i {engine} > engine.log 2>&1
mkdir -p frames
for f in {run}A[0-9][0-9][0-9]; do
  n=${{f: -3}}
  $OR/exec/anim_to_vtk_$ARCH "$f" > tmp.vtk
  if [ "$n" = "000" ] || [ "$n" = "001" ]; then cp tmp.vtk frames/frame_$n.vtk; fi
  awk '/^VECTORS Displacement/{{on=1; next}} /^(CELL_DATA|SCALARS|VECTORS|POINT_DATA)/{{on=0}} on' tmp.vtk > frames/disp_$n.txt
  grep -m1 -A1 '^TIME' tmp.vtk | tail -1 > frames/time_$n.txt
  rm -f "$f" tmp.vtk
done
echo OK > done.flag
"""


@dataclass(frozen=True)
class OpenRadiossDockerConfig:
    openradioss_dir: str                 # unpacked release (contains exec/, extlib/, hm_cfg_files/)
    image: str = "ubuntu:22.04"
    arch: Optional[str] = None           # "linuxa64" (arm64) or "linux64_gf" (x86-64); auto if None
    threads: int = 1
    docker_bin: str = "docker"
    extra_path: Sequence[str] = ("/Applications/Docker.app/Contents/Resources/bin",)

    def resolved_arch(self) -> str:
        if self.arch:
            return self.arch
        return "linuxa64" if platform.machine().lower() in ("arm64", "aarch64") else "linux64_gf"


def run_openradioss_case(case_dir: str | Path, run_name: str, cfg: OpenRadiossDockerConfig,
                         *, starter: Optional[str] = None, engine: Optional[str] = None) -> Path:
    """Run one staged case. Returns the case directory; raises on solver failure."""
    case = Path(case_dir).resolve()
    if (case / "done.flag").exists():
        return case
    starter = starter or f"{run_name}_0000.rad"
    engine = engine or f"{run_name}_0001.rad"
    script = _SCRIPT.format(arch=cfg.resolved_arch(), threads=cfg.threads,
                            starter=starter, engine=engine, run=run_name)
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([env.get("PATH", ""), *cfg.extra_path])
    cmd = [cfg.docker_bin, "run", "--rm",
           "-v", f"{Path(cfg.openradioss_dir).resolve()}:/opt/OpenRadioss:ro",
           "-v", f"{case}:/case", cfg.image, "bash", "-c", script]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0 or not (case / "done.flag").exists():
        tail = ""
        for log in ("engine.log", "starter.log"):
            if (case / log).exists():
                tail = (case / log).read_text()[-1500:]
                break
        raise RuntimeError(f"OpenRadioss failed in {case} (rc={proc.returncode}): {proc.stderr[-500:]}\n{tail}")
    return case


def run_openradioss_cases(case_dirs: Sequence[str | Path], run_name: str, cfg: OpenRadiossDockerConfig,
                          *, max_parallel: int = 4,
                          on_done: Optional[Callable[[Path, Optional[Exception]], None]] = None) -> List[Path]:
    """Run many cases with bounded parallelism (one single-threaded solver per case)."""
    def job(c):
        try:
            p = run_openradioss_case(c, run_name, cfg)
            if on_done:
                on_done(p, None)
            return p
        except Exception as e:  # keep the batch going; the failure is reported, not hidden
            if on_done:
                on_done(Path(c), e)
            return None

    with ThreadPoolExecutor(max_parallel) as ex:
        return list(ex.map(job, case_dirs))


def docker_available(cfg: OpenRadiossDockerConfig) -> bool:
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([env.get("PATH", ""), *cfg.extra_path])
    return shutil.which(cfg.docker_bin, path=env["PATH"]) is not None
