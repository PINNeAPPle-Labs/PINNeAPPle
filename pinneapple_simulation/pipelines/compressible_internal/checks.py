"""The checks that decide whether a passage result may be used — generic, and each one falsifiable.

A number from a solver is not a result until something independent agrees with it. Each function
here returns a ``{"code", "status", "value", "detail"}`` row with ``status`` one of ``pass``,
``fail`` or ``inconclusive``, and **nothing is ever passed by omission**: a check that could not run
says so, and says why.

==========================  ===============================================================  ==============
check                       what it compares                                                 reference
==========================  ===============================================================  ==============
``mesh``                    ``checkMesh`` verdict, non-orthogonality and aspect ratio          OpenFOAM
``converged``               the residual floor the solver reports                             OpenFOAM
``conservacao.de.massa``    mass flow in against mass flow out                                the run itself
``entalpia.de.estagnacao``  ``h0_out - h0_in`` across an adiabatic passage with no work       the EOS
``regime.de.escoamento``    ``p_out/p_in`` against the critical ratio, and ``Ma_max``           isentropic gas
``joule.thomson``           the outlet temperature against the library's isenthalpic path      Pinneapple
``densidade.gas.real``      the inlet density against Peng-Robinson and the ideal gas          the EOS
``coeficiente.de.fluxo``    the run's ``Kv`` against IEC 60534-2-1 for the same throat         IEC 60534
==========================  ===============================================================  ==============

The two that catch the reference document's run are ``regime.de.escoamento`` (a Mach of 0.9 at a
pressure ratio of 0.857 is impossible: methane is not choked there and the port cannot reach Mach 1)
and ``joule.thomson`` (a 90 K drop where the isenthalpic path gives 3.8 K). Those two are the
argument, and they need no CFD to make -- which is exactly why they are worth running against any
CFD.
"""
from __future__ import annotations

import math
from typing import Any

from .physics import Gas, peng_robinson

__all__ = ["run_checks", "critical_pressure_ratio", "stagnation_enthalpy", "iec_flow_coefficient",
           "reported_result_check", "ideal_mach_from_pressure_ratio"]


def _row(code: str, status: str, value: Any, detail: str) -> dict[str, Any]:
    return {"code": code, "status": status, "value": value, "detail": detail}


def critical_pressure_ratio(gas: Gas) -> float:
    """``p*/p0`` for a calorically perfect gas with ``gamma = Cp/(Cp - R)``."""
    g = gas.gamma_ig
    return (2.0 / (g + 1.0)) ** (g / (g - 1.0))


def stagnation_enthalpy(gas: Gas, p_pa: float, T_K: float, u_m_s: float) -> float:
    """``h + u^2/2`` with ``h`` carrying the Peng-Robinson departure, J/kg."""
    h = gas.Cp_ig * (T_K - 298.15) + peng_robinson(gas, p_pa, T_K)["h_dep"]
    return h + 0.5 * u_m_s ** 2


def iec_flow_coefficient(mdot_kg_s: float, p_in_pa: float, p_out_pa: float, rho_in: float,
                         gas: Gas, xT: float = 0.72) -> dict[str, Any]:
    """The flow coefficient a *valve* would need to pass ``mdot``, by IEC 60534-2-1 (compressible).

    ``Kv = Q [m3/h] sqrt((rho/rho_water)/dP [bar])`` is the same definition the run uses, so the two
    numbers are directly comparable; the IEC route differs only in that it takes the expansion
    factor ``Y`` and the choked-flow branch into account. When the passage is a valve, agreement
    means the CFD reproduces the standard's flow equation; when it is a nozzle, the same number is
    the discharge coefficient in disguise, and it is reported that way.
    """
    gamma = gas.gamma_ig
    x = (p_in_pa - p_out_pa) / max(p_in_pa, 1.0)          # IEC's pressure-drop ratio
    F_gamma = gamma / 1.4
    x_choked = F_gamma * xT
    choked = x >= x_choked
    Y = 1.0 if choked else 1.0 - x / (3.0 * F_gamma * xT)
    dp_bar = max(p_in_pa - p_out_pa, 0.0) / 1e5
    q_m3h = mdot_kg_s / rho_in * 3600.0
    # IEC 60534-2-1's compressible form, rearranged for the coefficient the flow implies
    # Q = Kv Y sqrt(dP / rho_rel)  =>  Kv = Q sqrt(rho_rel / dP) / Y
    kv = (q_m3h * math.sqrt((rho_in / 998.2) / dp_bar) / Y) if dp_bar > 0 else float("nan")
    return {"Kv_iec": kv, "Y": Y, "choked": bool(choked), "x": x, "x_choked": x_choked}


def ideal_mach_from_pressure_ratio(ratio: float, gas: Gas) -> float:
    """The Mach a passage could reach with ``p_out/p_in = ratio``, isentropically.

    ``Ma = sqrt(2/(gamma-1) ((p0/p)^((gamma-1)/gamma) - 1))``. It is the ceiling a result can be
    held to without a solver: a reported Mach above it, on a passage that is not choked, is not a
    flow feature, it is a field that never formed.
    """
    g = gas.gamma_ig
    if ratio >= 1.0:
        return 0.0
    return math.sqrt(max(2.0 / (g - 1.0) * ((1.0 / ratio) ** ((g - 1.0) / g) - 1.0), 0.0))


def reported_result_check(gas: Gas, conditions, reported: dict[str, Any]) -> dict[str, Any]:
    """A result somebody brought, checked against the physics of its own boundary conditions.

    A CFD report can be wrong in ways that need no run to see: a Mach number the pressure ratio
    cannot produce, a temperature drop larger than the isenthalpic path allows, or a physical time
    shorter than the field needs to cross the part. Each of those is computed here from the reported
    numbers themselves and the stated conditions, and any one of them failing is enough to say the
    result cannot be used.
    """
    p_in = float(reported.get("p_in_bar") or conditions.p_in_target_bar or 0.0) * 1e5
    p_out = float(reported.get("p_out_bar") or conditions.p_out_bar) * 1e5
    ratio = p_out / max(p_in, 1.0)
    ma_ceiling = ideal_mach_from_pressure_ratio(ratio, gas)
    problems: list[str] = []
    values: dict[str, Any] = {"p_out/p_in": round(ratio, 4), "Ma_isentropico_maximo": round(ma_ceiling, 3)}

    ma = reported.get("Ma_max")
    if ma is not None:
        values["Ma_max_reportado"] = float(ma)
        if ratio > critical_pressure_ratio(gas) and float(ma) > ma_ceiling * 1.25:
            problems.append(f"Mach {float(ma):.2f} reportado contra {ma_ceiling:.2f} que a razão de pressões "
                            f"permite com escoamento não afogado")

    t_lo, t_hi = reported.get("T_min_K"), reported.get("T_max_K")
    if t_lo is not None:
        t_lo, t_hi = float(t_lo), float(t_hi if t_hi is not None else conditions.T_in_K)
        dT_cold = t_lo - conditions.T_in_K
        ma_from_T = math.sqrt(max(2.0 / (gas.gamma_ig - 1.0) * ((conditions.T_in_K / max(t_lo, 1.0)) - 1.0), 0.0))
        values.update({"T_min_reportado_K": t_lo, "T_max_reportado_K": t_hi,
                       "dT_frio_reportado_K": round(dT_cold, 2),
                       "Ma_implicado_pela_T_min": round(ma_from_T, 3)})
        if ma_from_T > ma_ceiling * 1.25:
            problems.append(f"a temperatura mínima de {t_lo:.0f} K exigiria Mach {ma_from_T:.2f}, acima do "
                            f"que a razão de pressões permite ({ma_ceiling:.2f})")
        if t_hi > conditions.T_in_K + 10.0:
            problems.append(f"o campo mostra {t_hi:.0f} K, acima da entrada de {conditions.T_in_K:.0f} K: uma "
                            f"laminação adiabática não aquece o gás acima da estagnação")

    dt_us = reported.get("physical_time_us")
    transit_us = reported.get("acoustic_transit_us")
    if dt_us is not None and transit_us is not None:
        values.update({"tempo_fisico_us": float(dt_us), "travessia_acustica_us": float(transit_us)})
        if float(dt_us) < 0.5 * float(transit_us):
            problems.append(f"o tempo físico simulado ({float(dt_us):.0f} µs) é menos de metade da travessia "
                            f"acústica do componente ({float(transit_us):.0f} µs): o campo não se formou")

    return _row("resultado.do.cliente", "fail" if problems else "pass", values,
                "; ".join(problems) if problems else
                "os números trazidos são compatíveis com as condições de contorno que os acompanham")


def run_checks(result, fields, passage, gas: Gas, conditions, run, *, cfd_reported: dict[str, Any] | None = None
               ) -> list[dict[str, Any]]:
    """Every check, in the order the report reads them."""
    rows: list[dict[str, Any]] = []
    p_in, p_out = result.p_in_pa, result.p_out_pa
    throat_area = float(passage.meta.get("throttling_area_mm2", 0.0)) * 1e-6

    # ------------------------------------------------------------------ mesh & convergence
    rows.append(_row("mesh", "pass" if run.mesh_ok else "fail",
                     {"cellulas": passage.meta.get("n_cells")},
                     "checkMesh: " + ("Mesh OK" if run.mesh_ok else run.checkmesh_log.strip()[:300])))
    # A steady RANS on a valve with a 4 mm throttling curtain does not always reach a machine-level
    # residual: the shear layer over the plug keeps a limit cycle. That is reported as
    # ``inconclusive`` -- not "failed" (the field is bounded and the mean is stable) and not "passed"
    # (the solver did not declare convergence) -- and the KPIs are the mean over the second half of
    # the pseudo-time (OpenFOAM's fieldAverage), with the oscillation quoted where it matters.
    rows.append(_row("converged", "pass" if run.converged else "inconclusive",
                     {"iteracoes": run.n_iter, "residuos": {k: round(v, 8) for k, v in run.residuals.items()},
                      "media_em_pseudotempo": True},
                     "rhoSimpleFoam " + ("convergiu" if run.converged else
                                         f"parou em {run.n_iter} iterações no patamar "
                                         f"(p = {run.residuals.get('p', float('nan')):.1e}); os indicadores são a "
                                         f"média sobre a segunda metade do pseudo-tempo")))

    # ------------------------------------------------------------------ mass conservation
    imb = result.mass_imbalance
    rows.append(_row("conservacao.de.massa", "pass" if imb < 1e-3 else ("fail" if imb > 1e-2 else "inconclusive"),
                     {"entrada_kg_s": round(result.mdot_kg_s, 6), "saida_kg_s": round(result.mdot_out_kg_s, 6),
                      "desbalanco_relativo": round(imb, 9)},
                     "vazão mássica que entra contra a que sai; erro relativo "
                     f"{imb * 100:.4f} % (critério 0,1 %)"))

    # ------------------------------------------------------------------ stagnation enthalpy
    # h0 is dominated by the Peng-Robinson departure term, so a *relative* tolerance on h0 itself
    # would be meaningless (h0 passes through zero near 300 K at 70 bar). It is normalised by Cp T,
    # the scale of the energy the flow carries: 1 % of that is 6.6 kJ/kg here, far above the
    # discretisation error and far below the ~8 kJ/kg a solver that lost the Joule-Thomson term shows.
    h_ref = gas.Cp_ig * result.T_in_K
    h0_in = stagnation_enthalpy(gas, p_in, result.T_in_K, 0.0)
    u_out = result.mdot_kg_s / (peng_robinson(gas, p_out, result.T_out_K)["rho"]
                                * math.pi * (passage.spans[-1].radii[-1] * 1e-3) ** 2)
    h0_out = stagnation_enthalpy(gas, p_out, result.T_out_K, u_out)
    dh0 = (h0_out - h0_in) / h_ref
    rows.append(_row("entalpia.de.estagnacao",
                     "pass" if abs(dh0) < 0.01 else ("fail" if abs(dh0) > 0.05 else "inconclusive"),
                     {"h0_entrada_J_kg": round(h0_in, 1), "h0_saida_J_kg": round(h0_out, 1),
                      "variacao_relativa": round(dh0, 6)},
                     "passagem adiabática sem trabalho: h0 se conserva; variação "
                     f"{dh0 * 100:+.3f} % (critério 1 %)"))

    # ------------------------------------------------------------------ regime / choking
    ratio = p_out / p_in
    crit = critical_pressure_ratio(gas)
    choked_expected = ratio < crit
    ma = result.ma_max
    if choked_expected:
        ok = ma > 0.9
        detail = (f"p_saída/p_entrada = {ratio:.4f} < crítico {crit:.4f}: a passagem está afogada, "
                  f"então Ma no gargalo deve chegar a 1; medido Ma_max = {ma:.3f}")
    else:
        ok = ma < 1.0
        detail = (f"p_saída/p_entrada = {ratio:.4f} > crítico {crit:.4f}: a passagem NÃO está afogada, "
                  f"logo o escoamento é subsônico em todo o domínio; medido Ma_max = {ma:.3f}")
    rows.append(_row("regime.de.escoamento", "pass" if ok else "fail",
                     {"p_ratio": round(ratio, 5), "p_ratio_critico": round(crit, 5),
                      "afogado_esperado": choked_expected, "Ma_max": round(ma, 4)}, detail))

    # ------------------------------------------------------------------ Joule-Thomson
    rows.append(_joule_thomson_row(gas, conditions, result,
                                   passage.spans[-1].radii[-1] * 1e-3))

    # ------------------------------------------------------------------ real gas vs ideal
    rho_pr = peng_robinson(gas, p_in, result.T_in_K)["rho"]
    rho_ig = p_in / (gas.Rs * result.T_in_K)
    err = abs(result.rho_in - rho_pr) / rho_pr
    rows.append(_row("densidade.gas.real", "pass" if err < 0.02 else "fail",
                     {"rho_cfd": round(result.rho_in, 3), "rho_peng_robinson": round(rho_pr, 3),
                      "rho_gas_ideal": round(rho_ig, 3), "erro_relativo": round(err, 6)},
                     f"a densidade da entrada coincide com Peng-Robinson ({rho_pr:.2f} kg/m³) a "
                     f"{err * 100:.2f} %; o gás ideal daria {rho_ig:.2f} kg/m³ "
                     f"({(rho_ig / rho_pr - 1) * 100:+.1f} %)"))

    # ------------------------------------------------------------------ Kv against IEC 60534
    if throat_area > 0.0:
        iec = iec_flow_coefficient(result.mdot_kg_s, p_in, p_out, result.rho_in, gas,
                                   xT=float(passage.meta.get("xT", 0.72)))
        cd = result.mdot_kg_s / (throat_area * math.sqrt(max(2.0 * result.rho_in * (p_in - p_out), 1e-30)))
        rows.append(_row("coeficiente.de.fluxo",
                         "pass" if 0.4 < cd < 1.0 else "inconclusive",
                         {"Kv_cfd": round(result.kv, 5), "Kv_iec_60534": round(iec["Kv_iec"], 5),
                          "Y": round(iec["Y"], 4), "Cd_no_gargalo": round(cd, 4),
                          "area_gargalo_mm2": passage.meta.get("throttling_area_mm2")},
                         f"Kv do CFD {result.kv:.4f} contra {iec['Kv_iec']:.4f} pela IEC 60534-2-1 "
                         f"(Y = {iec['Y']:.3f}, {'afogado' if iec['choked'] else 'não afogado'}); "
                         f"Cd no gargalo {cd:.3f}"))

    if cfd_reported:
        rows.append(reported_result_check(gas, conditions, cfd_reported))
    return rows


def _joule_thomson_row(gas: Gas, conditions, result, passage_radius: float = 0.01) -> dict[str, Any]:
    """The outlet temperature against the library's isenthalpic path (GERG-2008 when it can)."""
    reference, how = None, ""
    try:                                             # the library's own real-gas reference
        from pinneapple_systems.process_components.joule_thomson import throttle
        r = throttle({gas.name.capitalize(): 1.0}, result.p_in_pa, result.T_in_K, result.p_out_pa,
                     eos="GERG-2008")
        reference, how = float(r["T2_K"]), "GERG-2008 (ISO 20765-2), via Pinneapple"
    except Exception as exc:                          # CoolProp missing, or the gas has no GERG entry
        try:
            from pinneapple_systems.process_components.joule_thomson import throttle
            r = throttle({gas.name.capitalize(): 1.0}, result.p_in_pa, result.T_in_K, result.p_out_pa,
                         eos="Peng-Robinson")
            reference, how = float(r["T2_K"]), f"Peng-Robinson (GERG indisponível: {type(exc).__name__})"
        except Exception as exc2:
            reference = None
            how = f"indisponível: {type(exc2).__name__}: {exc2}"
    if reference is None:
        return _row("joule.thomson", "inconclusive", {"dT_cfd": round(result.dT_K, 3)},
                    f"a referência isentálpica não pôde ser calculada ({how}); dT do CFD = "
                    f"{result.dT_K:+.3f} K")
    dT_ref = reference - result.T_in_K
    # static against static, with the kinetic energy: the reference is a gas brought to rest, the run
    # leaves at 100+ m/s, and T + u^2/(2 Cp) is what a throttle conserves. Without this the check
    # blames the CFD for a drop that is the flow accelerating out of the port.
    rho_out = peng_robinson(gas, result.p_out_pa, result.T_out_K)["rho"]
    u_out = result.mdot_kg_s / max(rho_out * math.pi * passage_radius ** 2, 1e-12)
    t_corrected = result.T_out_K + 0.5 * u_out ** 2 / gas.Cp_ig
    err = abs(t_corrected - reference)
    ideal = 0.0
    status = "pass" if err < 1.0 else ("fail" if err > 3.0 else "inconclusive")
    return _row("joule.thomson", status,
                {"T_saida_cfd_K": round(result.T_out_K, 3), "T_saida_referencia_K": round(reference, 3),
                 "dT_cfd_K": round(result.dT_K, 3), "dT_referencia_K": round(dT_ref, 3),
                 "u_saida_m_s": round(u_out, 2),
                 "correcao_cinetica_K": round(t_corrected - result.T_out_K, 3),
                 "dT_gas_ideal_K": ideal, "diferenca_com_correcao_K": round(t_corrected - reference, 3)},
                f"a laminação é isentálpica: referência {how:35s} dá {dT_ref:+.3f} K; o CFD mede "
                f"{result.dT_K:+.3f} K, que com a energia cinética de saída ({u_out:.0f} m/s, "
                f"{t_corrected - result.T_out_K:+.3f} K) vira {t_corrected - result.T_in_K:+.3f} K "
                f"(diferença {t_corrected - reference:+.3f} K, critério 1 K); um gás ideal daria 0,000 K")
