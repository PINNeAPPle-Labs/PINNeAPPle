"""Fail-closed numerical validation gates for structural FEM cases.

These are necessary checks, not a sufficient engineering certification.
"""
from __future__ import annotations
import numpy as np


def verify_fem(nodes_m, tetrahedra, displacement_m, stress_pa,
               von_mises_pa, reaction_n, applied_forces_n,
               fixed_nodes, force_tolerance_n=1e-4):
    nodes = np.asarray(nodes_m)
    tets = np.asarray(tetrahedra)
    u = np.asarray(displacement_m)
    s = np.asarray(stress_pa)
    vm = np.asarray(von_mises_pa)
    reactions = np.asarray(reaction_n)
    forces = np.asarray(applied_forces_n)
    fixed = np.asarray(fixed_nodes, dtype=int)
    checks = {}
    checks["shapes"] = (nodes.ndim == 2 and nodes.shape[1] == 3 and
                        tets.ndim == 2 and tets.shape[1] == 4 and
                        u.shape == nodes.shape and s.shape == (len(tets), 6) and
                        vm.shape == (len(tets),) and reactions.shape == nodes.shape and
                        forces.shape == nodes.shape)
    checks["finite"] = all(np.all(np.isfinite(a)) for a in
                           (nodes, u, s, vm, reactions, forces))
    checks["fixed_boundary"] = (len(fixed) > 0 and
                                np.all((fixed >= 0) & (fixed < len(nodes))) and
                                np.max(np.abs(u[fixed])) <= 1e-10)
    residual = float(np.linalg.norm(reactions.sum(axis=0) + forces.sum(axis=0)))
    checks["force_balance"] = residual <= force_tolerance_n
    checks["nonnegative_vm"] = bool(np.all(vm >= 0))
    # Mesh convergence and independent FEM comparison cannot be inferred from
    # one case; mark explicitly unverified, never silently pass.
    checks["mesh_convergence"] = False
    checks["independent_reference"] = False
    return {"passed_basic_checks": bool(all(v for k, v in checks.items()
                                           if k not in ("mesh_convergence", "independent_reference"))),
            "certified": False, "checks": checks,
            "force_balance_residual_n": residual,
            "required_next": ["mesh convergence", "independent FEM comparison",
                              "BC and material validation", "domain-of-validity test"]}
