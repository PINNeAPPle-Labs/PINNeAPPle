"""Safe, logged external solver invocation without shell interpolation."""
import os
from pathlib import Path
import shutil
import subprocess

def checked(argv,cwd,timeout=3600,env=None):
    cwd=Path(cwd).resolve()
    cwd.mkdir(parents=True,exist_ok=True)
    if not argv or shutil.which(str(argv[0])) is None:
        raise RuntimeError(f"Missing solver executable: {argv[0] if argv else '<empty>'}")
    label=Path(str(argv[0])).name
    with (cwd/(label+".stdout.log")).open("w") as stdout, (cwd/(label+".stderr.log")).open("w") as stderr:
        proc=subprocess.run([str(a) for a in argv],cwd=str(cwd),env={**os.environ,**(env or {})},stdout=stdout,stderr=stderr,timeout=timeout,check=False)
    if proc.returncode:
        raise RuntimeError(f"{label} exited with code {proc.returncode}; see {cwd}")
    return proc.returncode
