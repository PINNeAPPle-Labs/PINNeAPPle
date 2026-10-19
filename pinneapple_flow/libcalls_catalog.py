"""The curated allowlist of library calls exposed as nodes (``lib.<key>``). Parameter forms come from the
signatures; this file only says which callable, where it sits in the palette, which inputs come from ports and
which references back it. Add a capability by adding an entry."""
from __future__ import annotations

from .libcalls import LibCall, add

W = "pinneapple_flow.wrappers"
DS = {"data": ("dataset", "A,Y", "xy")}
DSX = {"data": ("dataset", "X,Y", "xy", False)}
SER = {"series": ("series", "x", "series")}
SUR = {"model": ("model", "fn", "surrogate")}

add(
    # ---- classical solvers and external tools
    LibCall("solve.preset", f"{W}:solve_preset", "Solve a library preset", "Solvers",
            "pp.solve on a preset problem: analytic, reference solver, FEM or a PINN variant",
            refs=("raissi2019pinn",)),
    LibCall("compare.preset", f"{W}:compare_preset", "Compare methods on a preset", "Validation",
            "pp.compare: same points, same reference, relative L2 / RMSE / max error, cost", refs=("oberkampf2010vv",)),
    LibCall("solver.lbm", f"{W}:lbm_channel", "Lattice Boltzmann channel", "Solvers", "LBM (D2Q9) channel flow"),
    LibCall("solver.beam", f"{W}:beam_deflection", "Beam deflection (FD)", "Solvers",
            "Euler-Bernoulli beam boundary-value problem by finite differences", refs=("leveque2007fd",)),
    LibCall("solver.xtfc", f"{W}:xtfc_pde", "X-TFC PDE solver", "Solvers",
            "extreme theory of functional connections (physics-informed random features)"),
    LibCall("solver.meshless", f"{W}:meshless_compare", "Meshless vs mesh-based", "Solvers",
            "Kansa, RBF-FD, FD and FEM on one problem and domain (meshless_study)"),
    LibCall("solver.terramechanics", f"{W}:wheel_on_soil", "Wheel on soil (Wong-Reece)", "Solvers",
            "terramechanics: sinkage, drawbar pull, torque of a wheel at steady state"),
    LibCall("solver.calculix", f"{W}:calculix_cantilever", "CalculiX cantilever", "Solvers",
            "a cantilever deck written by the library and solved with CalculiX when installed",
            node_type="solver.calculix"),
    LibCall("solver.fmu", f"{W}:fmu_simulate", "Simulate an FMU (FMI)", "Solvers",
            "Modelica / FMI co-simulation through fmpy", requires=("fmpy",), node_type="solver.fmu"),
    LibCall("solver.newton", f"{W}:newton_validate", "Rigid bodies (NVIDIA Newton)", "Solvers",
            "rigid-body solvers checked against closed forms", requires=("newton",), node_type="solver.newton"),
    LibCall("tools.availability", f"{W}:openfoam_available", "External tools available?", "Solvers",
            "is OpenFOAM / CalculiX on PATH"),
    # ---- models, ROMs, surrogates
    LibCall("model.architecture", f"{W}:model_architecture", "Any registry model: build + size", "Learning",
            "operators (DeepONet, FNO 1/2/3-D, UNO, GNO, Transolver, PINO), GNNs (MeshGraphNet, GraphCast), PINN "
            "variants, ROMs, recurrent and transformer models from the model registry",
            refs=("kovachki2023operator",)),
    LibCall("rom.compare", f"{W}:rom_compare", "ROM study (POD, DMD, OpInf, ...)", "Learning",
            "reduced-order models compared on snapshots at equal budget (rom_study)", inputs=DS,
            refs=("berkooz1993pod", "schmid2010dmd", "peherstorfer2016opinf")),
    LibCall("rom.parametric_pod", f"{W}:parametric_pod", "Parametric POD", "Learning",
            "POD basis + coefficient regression (RBF / GP / nearest / linear)", inputs=DS,
            refs=("berkooz1993pod", "benner2015pmor")),
    LibCall("surrogate.study", f"{W}:surrogate_study", "Classical ML / GP surrogates", "Learning",
            "ridge, Gaussian process, gradient boosting, MLP, FNO, DeepONet compared (run_study)", inputs=DSX,
            refs=("rasmussen2006gp",), node_type="surrogate.study"),
    # ---- uncertainty and sensitivity
    LibCall("uq.sobol", "pinneapple_analysis.sensitivity:sobol_indices", "Sobol indices (any surrogate)", "Analysis",
            "variance-based sensitivity of a surrogate's mean prediction over its training ranges",
            refs=("sobol2001indices", "saltelli2002sensitivity"), inputs=SUR, exclude=("space",)),
    LibCall("uq.morris", "pinneapple_analysis.sensitivity:morris", "Morris screening", "Analysis",
            "elementary effects of each input of a surrogate", refs=("morris1991screening",), inputs=SUR,
            exclude=("space",)),
    # ---- time series
    LibCall("ts.predictability", "pinneapple_systems.time_series.predictability:predictability_report",
            "Predictability report", "Time series", "surrogate tests, entropies, Lyapunov estimate of a series",
            refs=("theiler1992surrogate",), inputs=SER),
    # ---- data, design, exports
    LibCall("doe.design", f"{W}:doe_design", "Design of experiments", "Data",
            "pinneapple_data.doe.design: LHS / Sobol / Halton / maximin / factorial cases over a parameter space, "
            "with coverage metrics", refs=("mckay1979lhs",)),
    LibCall("design.intent", f"{W}:design_intent", "Design from a goal (text)", "Design",
            "pinneapple_design.intent.design: a plain-language goal to a verified design package, or the questions "
            "it still needs"),
    LibCall("export.geometry", f"{W}:geometry_export", "Export a library geometry", "Export",
            "pinneapple_design.geometry_library: build a parametric geometry and write STL / OBJ / glTF / USD / STEP"),
)
