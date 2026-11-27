"""Optional interactive HexaMSH bridge; no imports of Gmsh or HexaMSH."""
from pathlib import Path
import shutil
import subprocess
import tempfile


def import_hex_mesh(source, destination):
    """Preserve native connectivity and tags; reject non-Hex8 volume cells."""
    import meshio
    from .mesh_gen import quality

    source, destination = Path(source), Path(destination)
    mesh = meshio.read(source)
    volume_types = ("tetra", "hexahedron", "wedge", "pyramid", "voxel", "polyhedron")
    if any(c.type.startswith(volume_types) and c.type != "hexahedron" for c in mesh.cells):
        raise ValueError("Only linear hexahedral volume meshes are supported")
    if not any(c.type == "hexahedron" and len(c.data) for c in mesh.cells):
        raise ValueError("No Hex8 elements found")
    report = quality(str(source), "fea")
    if report["verdict"] != "pass":
        raise ValueError(f"Mesh quality gate: {report['verdict']}")
    if destination.suffix.lower() != ".msh" or source.suffix.lower() != ".msh":
        raise ValueError("Use native MSH files to preserve physical groups")
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copyfile(source, destination)
    return mesh, report


def generate_hex_mesh(cad, destination, *, executable="hexamsh", timeout=3600):
    """Launch upstream GUI in isolated directory; user must save one MSH file."""
    cad = Path(cad).resolve(strict=True)
    destination = Path(destination).resolve()
    if cad.suffix.lower() not in {".step", ".stp", ".brep"}:
        raise ValueError("Expected STEP/STP/BREP")
    command = shutil.which(executable)
    if command is None:
        raise RuntimeError("Install HexaMSH in a separate Python 3.12 environment")
    with tempfile.TemporaryDirectory(prefix="pinneapple-hexamsh-") as tmp:
        work = Path(tmp)
        shutil.copyfile(cad, work / cad.name)
        subprocess.run([command, cad.name], cwd=work, check=True, timeout=timeout)
        outputs = list(work.rglob("*.msh"))
        if len(outputs) != 1:
            raise RuntimeError("Save exactly one MSH file in the HexaMSH working directory")
        return import_hex_mesh(outputs[0], destination)
