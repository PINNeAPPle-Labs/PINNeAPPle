"""Espaço de projeto sobre uma malha real: FFD, RBF e modos de forma, com o portão de validade medindo tudo (#292).

O que este exemplo mostra, na ordem em que um estudo precisa das três coisas: uma malha que já existe no repositório
(o carro de `data/geometries`) vira um espaço de projeto por FFD com simetria, os handles de RBF movem pontos
escolhidos segurando a face de apoio, e os modos de forma resumem a família gerada. Em cada etapa o custo é medido, e
cada amostra passa pelo portão — estanqueidade, volume, área frontal, faces viradas, autointersecção — que aceita com
os números ou recusa com o motivo.

O que este exemplo não faz: rodar o CFD das amostras. A malha deformada entra no `pp.cfd.ExternalFlow` e o caso é
escrito (é a última seção), mas `snappyHexMesh` + `simpleFoam` exigem o OpenFOAM instalado, e as 200 amostras com
arrasto de ponta a ponta ficam como o próximo passo da issue.

    python examples/geometry/09_surface_morphing_design_space.py            # ~40 s numa máquina de portátil
    python examples/geometry/09_surface_morphing_design_space.py --samples 60 --out /tmp/morph
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pinneapple_design.geometry import morph as M

ROOT = Path(__file__).resolve().parents[2]
CAR = ROOT / "data" / "geometries" / "road_car" / "road_car.light.stl"


def median_seconds(fn, n: int = 30) -> float:
    """Custo mediano de uma chamada: a média esconde a primeira chamada, que paga a memória."""
    fn()                                             # aquece
    times = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return float(np.median(times))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=24, help="amostras válidas a coletar do espaço de projeto")
    ap.add_argument("--out", default=str(ROOT / "examples" / "geometry" / "_out" / "morphing"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report: dict = {"surface": str(CAR.relative_to(ROOT)), "samples": args.samples}

    # ------------------------------------------------------------------ 1. a malha de referência
    t = time.perf_counter()
    car = M.Surface.load(CAR)
    load_s = time.perf_counter() - t
    V, F = car.reference, car.faces
    print(f"1. {CAR.name}: {car.n_vertices} vértices, {car.n_faces} triângulos, carregada em {load_s:.3f} s")
    report["mesh"] = {"vertices": car.n_vertices, "faces": car.n_faces, "load_seconds": round(load_s, 4)}

    # ------------------------------------------------------------------ 2. FFD com simetria
    t = time.perf_counter()
    ffd = M.FFD.fit(V, control=(6, 4, 6), symmetry="y", padding=0.05)
    fit_s = time.perf_counter() - t
    n_plain = M.FFD.fit(V, control=(6, 4, 6), padding=0.05).n_parameters
    cell = (ffd.hi - ffd.lo) / (np.array(ffd.control.shape[:3]) - 1)
    identity = float(np.abs(ffd.deform(np.zeros(ffd.n_parameters)) - V).max())
    deform_s = median_seconds(lambda: ffd.deform(np.full(ffd.n_parameters, 1e-3 * cell.max())))
    print(f"2. FFD {ffd}: {n_plain} variáveis, {ffd.n_parameters} com simetria em y "
          f"({100 * ffd.n_parameters / n_plain:.0f} %), rede em {fit_s:.3f} s")
    print(f"   identidade em zero parâmetros: erro máximo {identity:.1e} m | uma deformação custa {1e3 * deform_s:.2f} ms")
    report["ffd"] = {"control": list(ffd.control.shape[:3]), "parameters": ffd.n_parameters,
                     "parameters_without_symmetry": n_plain, "cell_m": cell.round(4).tolist(),
                     "fit_seconds": round(fit_s, 4), "identity_max_error_m": identity,
                     "deform_seconds": round(deform_s, 5)}

    # ------------------------------------------------------------------ 3. o FFD contra uma família paramétrica
    # o corpo de Ahmed da própria biblioteca tem a mesma contagem de vértices em qualquer ângulo de rampa, então a
    # correspondência vértice a vértice é exata e o FFD pode ser medido contra o gerador: quanto do 35 graus um FFD
    # ajustado sobre o de 25 graus consegue expressar
    import pinneapple as pp

    a25, a35 = pp.bodies.ahmed_body(slant_deg=25.0), pp.bodies.ahmed_body(slant_deg=35.0)
    A0, Fa = M.as_surface(a25)
    A1, _ = M.as_surface(a35)
    ffd_a = M.FFD.fit(A0, control=(4, 2, 3), padding=0.0)
    W = ffd_a.weights(A0)
    basis = np.stack([(W @ ffd_a.full(np.eye(ffd_a.n_parameters)[j])).ravel()
                      for j in range(ffd_a.n_parameters)], axis=1)
    target = (A1 - A0).ravel()
    xa, *_ = np.linalg.lstsq(basis, target, rcond=None)
    Va = ffd_a.deform(xa)
    ahmed_rel = float(np.linalg.norm(Va - A1) / np.linalg.norm(target))
    ahmed_mm = float(1000 * np.abs(Va - A1).max())
    ahmed_ok = M.check_surface(Va, Fa, reference=(A0, Fa)).ok
    print(f"3. corpo de Ahmed: um FFD de {ffd_a.n_parameters} variáveis ajustado no corpo de 25° expressa o de 35° "
          f"com {100 * ahmed_rel:.1f} % de erro relativo L2 (pior vértice {ahmed_mm:.1f} mm numa mudança de "
          f"{1000 * np.abs(target).max():.1f} mm); o corpo deformado passa no portão: {ahmed_ok}")
    print(f"   o vetor de projeto desse ajuste tem deslocamentos de até {np.abs(xa).max():.2f} m: a base de Bernstein "
          f"é global, então o inverso é degenerado — é por isso que o espaço de projeto é uma caixa limitada, e não a "
          f"base inteira.")
    report["ahmed"] = {"parameters": ffd_a.n_parameters, "relative_l2": ahmed_rel, "max_mm": ahmed_mm,
                       "change_mm": float(1000 * np.abs(target).max()), "valid": ahmed_ok,
                       "max_control_displacement_m": float(np.abs(xa).max())}

    # ------------------------------------------------------------------ 4. o portão de validade
    t = time.perf_counter()
    reference = M.Reference.build(V, F)
    ref_s = time.perf_counter() - t
    gate = {"max_volume_change": 0.25, "contact_tolerance": 1.0}
    lo, hi = ffd.bounds(0.35)                       # ±35 % de uma célula: a banda que este estudo declara
    space = M.DesignSpace(ffd, F, V, lo=lo, hi=hi, gate=gate, gate_reference=reference)
    t = time.perf_counter()
    base = space.check(np.zeros(space.dim))
    gate_s = time.perf_counter() - t
    print(f"4. portão: referência em {ref_s:.2f} s ({reference.intersections} pares de faces já se cruzam num STL de "
          f"{reference.faces.shape[0]} triângulos, {reference.area:.1f} m² de área, volume {reference.volume:.2f}); "
          f"uma amostra custa {1e3 * gate_s:.0f} ms")
    print(f"   a referência passa: {base.ok} | volume {base.metrics['enclosed_volume']:.3f} m³, "
          f"área frontal {base.metrics['frontal_area']:.3f} m²")
    report["gate"] = {"reference_seconds": round(ref_s, 3), "sample_seconds": round(gate_s, 3),
                      "reference_self_intersections": reference.intersections,
                      "reference_volume_m3": reference.volume, "reference_area_m2": reference.area,
                      "reference_frontal_area_m2": reference.frontal_area, "identity_ok": base.ok}

    # ------------------------------------------------------------------ 5. o espaço de projeto
    report_space = space.sample_valid(args.samples, seed=0)
    summary = report_space.summary()
    print(f"5. espaço de projeto: {summary['accepted']}/{args.samples} amostras válidas em {summary['draws']} sorteios "
          f"({100 * summary['acceptance_rate']:.0f} % de aceitação), {summary['seconds']:.1f} s")
    for reason, count in sorted(summary["reasons"].items()):
        print(f"   recusa: {count:3d} x {reason}")
    if summary["accepted"]:
        vols = np.array([r.metrics["volume_ratio"] for r in report_space.reports])
        areas = np.array([r.metrics["frontal_area_change"] for r in report_space.reports])
        print(f"   volume das aceitas: {vols.min():.3f} a {vols.max():.3f} do original | "
              f"área frontal {100 * areas.min():+.1f} % a {100 * areas.max():+.1f} %")
        report["design_space"] = summary | {
            "volume_ratio": [float(vols.min()), float(vols.max())],
            "frontal_area_change": [float(areas.min()), float(areas.max())]}
    else:
        report["design_space"] = summary

    # a mesma malha com uma banda larga de propósito: é assim que o portão mostra o que recusa
    stress = M.DesignSpace(ffd, F, V, lo=ffd.bounds(2.0)[0], hi=ffd.bounds(2.0)[1], gate=gate,
                           gate_reference=reference)
    stressed = stress.sample_valid(8, seed=1, max_draws=8)
    stress_summary = stressed.summary()
    print(f"   banda larga (2 células): {stress_summary['accepted']}/8 aceitas; "
          + "; ".join(f"{c} x {r}" for r, c in sorted(stress_summary["reasons"].items())))
    report["stress_band"] = stress_summary

    # ------------------------------------------------------------------ 6. RBF: handles na superfície
    lo, hi = V.min(0), V.max(0)
    ground = np.nonzero(V[:, 2] < lo[2] + 0.02 * (hi[2] - lo[2]))[0]      # a face de apoio: fica parada
    want = [(0.5 * (lo[0] + hi[0]), 0.0, hi[2]), (lo[0] + 0.35 * (hi[0] - lo[0]), 0.0, hi[2]),
            (0.55 * (lo[0] + hi[0]), hi[1], lo[2] + 0.6 * (hi[2] - lo[2])),
            (0.55 * (lo[0] + hi[0]), lo[1], lo[2] + 0.6 * (hi[2] - lo[2]))]
    handles = [int(np.argmin(np.linalg.norm(V - np.asarray(p), axis=1))) for p in want]
    t = time.perf_counter()
    rbf = M.RBFMorph(handles, fixed=ground).fit(V)
    rbf_fit = time.perf_counter() - t
    rbf_s = median_seconds(lambda: rbf.deform(np.full(rbf.n_parameters, 2e-3)), 20)
    xr = np.zeros(rbf.n_parameters)
    xr[2] = 0.08                                                          # sobe o handle do teto
    Vr = rbf.deform(xr)
    handle_error = float(np.abs(Vr[handles[0]] - (V[handles[0]] + [0, 0, 0.08])).max())
    fixed_error = float(np.abs(Vr[ground] - V[ground]).max())
    # movimento rígido: a parte polinomial garante que uma translação dos handles translada a malha — mas só quando
    # nada está preso, porque um ponto fixo ancora o campo (é por isso que o apoio fica num RBF próprio)
    free = M.RBFMorph(handles).fit(V)
    tvec = np.array([0.05, -0.02, 0.03])
    rigid = float(np.abs(free.deform(np.tile(tvec, len(handles))) - (V + tvec)).max())
    pinned = float(np.abs(rbf.deform(np.tile(tvec, len(handles))) - (V + tvec)).max())
    print(f"6. RBF: {rbf}, {rbf.n_parameters} variáveis, {len(ground)} pontos fixos; ajuste em {rbf_fit:.3f} s, "
          f"deformação em {1e3 * rbf_s:.2f} ms")
    print(f"   handle exato a {handle_error:.1e} m, apoio parado a {fixed_error:.1e} m; uma translação dos handles "
          f"translada a malha a {rigid:.1e} m sem pontos fixos e a {pinned:.1e} m com eles")
    report["rbf"] = {"handles": len(handles), "fixed_points": int(len(ground)), "parameters": rbf.n_parameters,
                     "fit_seconds": round(rbf_fit, 4), "deform_seconds": round(rbf_s, 5),
                     "handle_error_m": handle_error, "fixed_error_m": fixed_error,
                     "rigid_error_m_free": rigid, "rigid_error_m_pinned": pinned}

    # ------------------------------------------------------------------ 7. modos de forma
    family = [ffd.deform(x) for x in report_space.parameters]
    if len(family) >= 3:
        t = time.perf_counter()
        modes = M.ShapeModes(energy=0.99, max_modes=len(family) - 1).fit(family)
        modes_s = time.perf_counter() - t
        energy = sum(modes.modal_energy())
        six = sum(modes.modal_energy()[:6])
        residual = max(modes.residual(f) for f in family)
        print(f"7. modos: {modes.n_parameters} modos levam a família de {len(family)} formas a {100 * energy:.1f} % "
              f"(o primeiro 6 carregam {100 * six:.1f} %, resíduo relativo máximo {residual:.1e}, "
              f"{modes_s:.3f} s)")
        print("   uma família sorteada de um FFD é de dimensão alta por construção; o que os modos resumem é uma "
              "família de variantes reais, com poucas direções.")
        report["modes"] = {"count_for_99pct": modes.n_parameters, "energy": energy, "first_six_energy": six,
                           "residual": residual, "fit_seconds": round(modes_s, 4), "family": len(family)}

    # ------------------------------------------------------------------ 8. a malha deformada entra no CFD
    flow_stats = None
    try:
        if summary["accepted"]:
            import tempfile

            Vd = space.build(report_space.parameters[-1]).vertices
            flow = pp.cfd.ExternalFlow({"road_car": (Vd, F)}, speed=30.0, resolution="coarse", ground=True)
            case = Path(tempfile.mkdtemp()) / "case_morph"                # o caso é escrito, não versionado
            flow.write(str(case), procs=1)
            body = case / "constant" / "triSurface" / "road_car.stl"
            flow_stats = {"ref_area_m2": float(flow.ref_area), "ref_length_m": float(flow.L),
                          "tri_surface_bytes": int(body.stat().st_size) if body.exists() else 0,
                          "written": bool(body.exists())}
            print(f"8. CFD: a malha deformada virou caso do ExternalFlow (área de referência "
                  f"{flow.ref_area:.3f} m², comprimento {flow.L:.3f} m, {flow_stats['tri_surface_bytes']} bytes de "
                  f"triSurface escritos).")
            print("   o solver não roda aqui: snappyHexMesh e simpleFoam precisam do OpenFOAM instalado.")
    except Exception as exc:                        # o caso escrito é bônus, não pré-requisito
        print(f"8. CFD: o caso não foi escrito ({type(exc).__name__}: {exc})")
    report["cfd_case"] = flow_stats

    (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"\nresumo em {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
