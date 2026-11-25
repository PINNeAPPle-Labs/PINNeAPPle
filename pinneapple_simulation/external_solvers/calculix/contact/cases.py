"""Contact problems with friction, as decks with a closed-form answer: the case library.

Each case is one geometry, one contact interface and one question that has an exact answer independent of the solver:

===========================  ==========================================================================
``bloco_plano``              a slider pushed sideways on a fixed plate: the joint transmits ``mu*N`` and no
                             more, so the tangential force saturates exactly there
``junta_dupla``              a plate pulled between two clamped plates: two interfaces share the load, so the
                             pull at slip is ``2*mu*N`` while each support carries ``mu*N``
``cilindro_hertz``           a rigid cylinder pressed into a plate: a non-conforming pair, so the peak pressure
                             and the half-width follow Hertz instead of the nominal contact area
``rampa``                    the same slider on an inclined interface: it stays put while ``tan(theta) <= mu``
                             and slides when it does not — the self-locking (autotravamento) test
``parafuso_pretensao``       two plates clamped by a ``*PRE-TENSION SECTION`` whose head and nut are footprints
                             joined by ``*DISTRIBUTING COUPLING``: the clamp force is the preload
``plastificacao_hertz``      the Hertz cylinder on an elastic-perfectly-plastic plate: yield begins at
                             ``p0 = k*sigma_y``, and ``k`` is Johnson's factor, not 1
===========================  ==========================================================================

Every case writes a deck PrePoMax opens unchanged, and every case declares the tolerance its comparison is judged by.
The parameters are the engineer's (mm, N, MPa), the meshes are structured hexahedra the solver can resolve the patch
with, and nothing here is specific to one problem size: the closed form is evaluated at the numbers the run reports,
so changing the load moves both sides of the comparison.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from . import deck as D
from . import verify as V
from .base import Built, Case, Param, contact_of, mesh_size, reaction, stack
from .mesh import Grid, axis

# The containers live in :mod:`.base`; the names are re-exported here because this module is where the case
# library is read from.
__all__ = ["CASES", "DEFAULT_CASE", "Built", "Case", "Param", "build", "catalog", "get"]

_mesh_size = mesh_size
_stack = stack
_reaction = reaction
_contact = contact_of



# ===================================================================================== 1. sliding block on a plate
def _bloco_plano_build(p: dict[str, float]) -> Built:
    a, hb, hs = p["lado"], p["altura_base"], p["altura_bloco"]
    n = _mesh_size(p["malha"])
    base = Grid(axis(0, a, n), axis(0, hb, n), axis(0, a, n))
    slider = Grid(axis(0, a, n), axis(0, hs, n), axis(0, a, n), first_node=base.next_node,
                  first_elem=base.next_elem, shift=lambda x: (0.0, hb, 0.0))
    deck = D.Deck(f"contato com atrito: bloco deslizante sobre placa fixa (mu = {p['friccao']:g})")
    _stack(deck, [("BASE", base), ("SLIDER", slider)], "ACO", p["E"], p["nu"])

    bfix = base.nodes_on("y", 0.0)
    stop = slider.nodes_on("y", hb + hs)
    press_faces = slider.faces("y", hb + hs, side=-1)
    master = base.faces("y", hb, side=-1)
    slave = slider.faces("y", hb, side=+1)

    for name, ids in (("BFIX", bfix), ("STOP", stop)):
        D.node_set(deck, name, ids)
    D.surface(deck, "MASTER", master)
    D.surface(deck, "SLAVE", slave)
    D.surface(deck, "PRESS", press_faces)
    D.interaction(deck, "INT", p["friccao"])
    D.contact_pair(deck, "CONTACT", slave="SLAVE", master="MASTER", interaction_name="INT")

    area = a * a
    deck.block("*STEP, NLGEOM=YES", "*STATIC", "0.25, 1.0")
    deck.block("*BOUNDARY", "BFIX, 1, 3, 0.0")
    deck.block("*DLOAD", f"PRESS, P, {p['pressao']:.6g}")
    D.outputs(deck, totals_set="BFIX", displacement_set="STOP", stress=True)
    deck.block("*END STEP")

    deck.block("*STEP, NLGEOM=YES", "*STATIC", "0.25, 1.0")
    deck.block("*BOUNDARY", f"STOP, 1, 1, {p['deslizamento']:.6g}")
    D.outputs(deck, totals_set="BFIX", displacement_set="STOP", stress=True)
    deck.block("*END STEP")

    N = p["pressao"] * area
    return Built(deck=deck.text(),
                 analytic={"normal": N, "friction": p["friccao"] * N, "area": area,
                           "slip_imposed": p["deslizamento"]},
                 coords=dict(base.nodes) | dict(slider.nodes),
                 interface=tuple(slider.nodes_on("y", hb)),
                 totals=("BFIX",),
                 notes=[f"área de contato nominal {area:g} mm², carga normal {N:g} N",
                        f"deslizamento prescrito {p['deslizamento']:g} mm na face superior"])


def _bloco_plano_checks(p: dict[str, float], b: Built, parsed: dict[str, Any]) -> list[V.Check]:
    a = b.analytic
    src = f"Coulomb na interface: |Ft| = mu*N = {p['friccao']:.4g} x {a['normal']:.6g} N"
    r = _reaction(parsed, "BFIX")
    checks = [V.compare("equilibrio.normal", abs(r[1]) if r else None, a["normal"], 0.02,
                        "reação vertical no apoio é a carga aplicada (P x área)", "N", "carga normal no apoio")]
    if not r:
        checks.append(V.Check("atrito.saturacao", V.INCONCLUSIVE, detail="sem reação no apoio: nada a comparar"))
        return checks
    ft = abs(r[0])
    checks.append(V.at_most("atrito.limite", ft, a["friction"], 0.02, src, "N", "força tangencial transmitida"))
    slip = V.slip_range(_contact(parsed), b.interface)
    slid = slip is not None and slip > 0.1 * p["deslizamento"]
    if slid:
        checks.append(V.compare("atrito.saturacao", ft, a["friction"], 0.02, src, "N",
                                "força tangencial com a interface deslizando"))
    else:
        checks.append(V.Check("atrito.saturacao", V.INCONCLUSIVE, ft, a["friction"],
                              f"a interface não deslizou (escorregamento medido "
                              f"{slip if slip is None else round(slip, 6)} mm, prescrito {p['deslizamento']:g} mm): "
                              f"a força transmitida é elástica, não o limite de atrito",
                              src))
    checks.append(V.at_most("atrito.cone", V.coulomb_ratio(_contact(parsed), b.interface), p["friccao"], 0.05,
                            "|tau| <= mu p em cada nó carregado (cone de Coulomb)", "", "maior razão |tau|/p"))
    return checks


# ======================================================================= 2. plate pulled between two clamped plates
def _junta_dupla_build(p: dict[str, float]) -> Built:
    a, h = p["lado"], p["espessura"]
    n = _mesh_size(p["malha"])
    bottom = Grid(axis(0, a, n), axis(0, h, n), axis(0, a, n))
    middle = Grid(axis(0, a, n), axis(0, h, n), axis(0, a, n), first_node=bottom.next_node,
                  first_elem=bottom.next_elem, shift=lambda x: (0.0, h, 0.0))
    top = Grid(axis(0, a, n), axis(0, h, n), axis(0, a, n), first_node=middle.next_node,
               first_elem=middle.next_elem, shift=lambda x: (0.0, 2 * h, 0.0))
    deck = D.Deck(f"contato com atrito: junta em cisalhamento duplo (mu = {p['friccao']:g})")
    _stack(deck, [("BASE", bottom), ("MEIO", middle), ("TOPO", top)], "ACO", p["E"], p["nu"])

    bfix = bottom.nodes_on("y", 0.0)
    clamp = top.nodes_on("y", 3 * h)
    pull = middle.nodes_on("x", a)
    for name, ids in (("BFIX", bfix), ("CLAMP", clamp), ("PULL", pull)):
        D.node_set(deck, name, ids)
    D.surface(deck, "LOW_MASTER", bottom.faces("y", h, side=-1))
    D.surface(deck, "LOW_SLAVE", middle.faces("y", h, side=+1))
    D.surface(deck, "UP_MASTER", middle.faces("y", 2 * h, side=-1))
    D.surface(deck, "UP_SLAVE", top.faces("y", 2 * h, side=+1))
    D.surface(deck, "PRESS", top.faces("y", 3 * h, side=-1))
    D.interaction(deck, "INT", p["friccao"])
    D.contact_pair(deck, "LOW", slave="LOW_SLAVE", master="LOW_MASTER", interaction_name="INT")
    D.contact_pair(deck, "UP", slave="UP_SLAVE", master="UP_MASTER", interaction_name="INT")

    area = a * a
    deck.block("*STEP, NLGEOM=YES", "*STATIC", "0.25, 1.0")
    deck.block("*BOUNDARY", "BFIX, 1, 3, 0.0")
    # the bolt: the clamped stack is held laterally at the washer footprint, as a preloaded bolt would hold it
    deck.block("*BOUNDARY", "CLAMP, 1, 1, 0.0")
    deck.block("*DLOAD", f"PRESS, P, {p['pressao']:.6g}")
    D.outputs(deck, totals_set="BFIX", displacement_set="PULL", stress=True)
    deck.block("*END STEP")

    deck.block("*STEP, NLGEOM=YES", "*STATIC", "0.25, 1.0")
    deck.block("*BOUNDARY", f"PULL, 1, 1, {p['deslizamento']:.6g}")
    D.outputs(deck, totals_set="BFIX", displacement_set="PULL", stress=True)
    deck.block("*END STEP")

    N = p["pressao"] * area
    return Built(deck=deck.text(),
                 analytic={"normal": N, "friction": p["friccao"] * N, "pull": 2 * p["friccao"] * N, "area": area,
                           "slip_imposed": p["deslizamento"]},
                 coords=dict(bottom.nodes) | dict(middle.nodes) | dict(top.nodes),
                 interface=tuple(middle.nodes_on("y", h)) + tuple(top.nodes_on("y", 2 * h)),
                 totals=("BFIX",),
                 notes=[f"área de cada interface {area:g} mm², carga de aperto {N:g} N",
                        "o aperto entra como pressão sobre a pegada do parafuso, e o topo é travado lateralmente "
                        "como o parafuso faria: sem pré-tensão do parafuso, que este caso não modela"])


def _junta_dupla_checks(p: dict[str, float], b: Built, parsed: dict[str, Any]) -> list[V.Check]:
    a = b.analytic
    r = _reaction(parsed, "BFIX")
    checks = [V.compare("equilibrio.normal", abs(r[1]) if r else None, a["normal"], 0.03,
                        "a carga de aperto atravessa as duas interfaces", "N", "carga normal no apoio")]
    src = (f"duas interfaces com atrito: F = 2 mu N = 2 x {p['friccao']:.4g} x {a['normal']:.6g} N")
    checks.append(V.compare("junta.cisalhamento", abs(r[0]) if r else None, a["friction"], 0.05, src, "N",
                            "força tangencial em uma interface"))
    slip = V.slip_range(_contact(parsed), b.interface)
    slid = slip is not None and slip > 0.1 * p["deslizamento"]
    if slid:
        checks.append(V.compare("junta.dupla_saturacao", 2 * abs(r[0]) if r else None, a["pull"], 0.05, src, "N",
                                "carga de puxamento com as duas interfaces deslizando"))
    else:
        checks.append(V.Check("junta.dupla_saturacao", V.INCONCLUSIVE, 2 * abs(r[0]) if r else None, a["pull"],
                              "as interfaces não deslizaram: a força medida é elástica, não o limite de atrito", src))
    checks.append(V.at_most("atrito.cone", V.coulomb_ratio(_contact(parsed), b.interface), p["friccao"], 0.05,
                            "|tau| <= mu p em cada nó carregado", "", "maior razão |tau|/p"))
    return checks


# ======================================================================================= 3. Hertz line contact
def _cilindro_hertz_build(p: dict[str, float]) -> Built:
    R, W, H = p["raio"], p["largura"], p["profundidade"]
    nx, ny = _mesh_size(p["malha"]), _mesh_size(p["malha"] / 2)
    xs = sorted({-x for x in axis(0, W / 2, nx, grading=1.5)} | set(axis(0, W / 2, nx, grading=1.5)))
    ys = axis(-H, 0.0, ny, grading=1.3, cluster="high")
    block = Grid(xs, ys, [0.0, 1.0])
    indenter = Grid(xs, axis(0.0, p["altura_indentador"], 3), [0.0, 1.0], first_node=block.next_node,
                    first_elem=block.next_elem, shift=lambda x: (0.0, x * x / (2 * R), 0.0))
    ref = indenter.next_node
    nodes = dict(block.nodes) | dict(indenter.nodes) | {ref: (0.0, 2 * p["altura_indentador"], 0.5)}

    deck = D.Deck(f"contato não conforme: cilindro rígido de R = {R:g} mm sobre placa (Hertz, mu = {p['friccao']:g})")
    D.geometry(deck, [("BASE", block), ("IND", indenter)])
    deck.block("*NODE", f"{ref}, 0.0, {2 * p['altura_indentador']:.8g}, 0.5")
    D.material(deck, "ACO", p["E"], p["nu"])
    D.section(deck, "BASE", "ACO")
    D.section(deck, "IND", "ACO")

    D.node_set(deck, "REF", [ref])
    D.node_set(deck, "RIGALL", indenter.node_ids)
    D.node_set(deck, "BBOT", block.nodes_on("y", -H))
    D.node_set(deck, "ZB", block.nodes_on("z", 0.0))
    D.node_set(deck, "ZT", block.nodes_on("z", 1.0))
    deck.block(f"*RIGID BODY, NSET=RIGALL, REF NODE={ref}")
    D.surface(deck, "MASTER", block.faces("y", 0.0, side=-1))
    D.surface(deck, "SLAVE", indenter.faces("y", 0.0, side=None, curved=lambda q: q[0] ** 2 / (2 * R)))
    D.interaction(deck, "INT", p["friccao"], k_normal=p["rigidez_contato"])
    D.contact_pair(deck, "CONTACT", slave="SLAVE", master="MASTER", interaction_name="INT")

    deck.block("*STEP, NLGEOM=YES", "*STATIC", "0.1, 1.0")
    deck.block("*BOUNDARY", "BBOT, 2, 2, 0.0", "ZB, 3, 3, 0.0", "ZT, 3, 3, 0.0")
    deck.block("*BOUNDARY", f"REF, 2, 2, {-p['penetracao']:.6g}", "REF, 1, 1, 0.0", "REF, 3, 3, 0.0")
    D.outputs(deck, totals_set="REF", stress=False)
    deck.block("*END STEP")

    interface = tuple(n for n in indenter.node_ids if abs(nodes[n][1] - nodes[n][0] ** 2 / (2 * R)) < 1e-9)
    return Built(deck=deck.text(),
                 analytic={"radius": R, "length": 1.0, "penetration": p["penetracao"], "e_modulus": p["E"],
                           "nu": p["nu"]},
                 coords=nodes, interface=interface, totals=("REF",),
                 notes=["o indentador é rígido (RIGID BODY), a placa é elástica linear e está em deformação plana",
                        f"espessura fora do plano 1 mm, R = {R:g} mm, penetração prescrita {p['penetracao']:g} mm"])


def _cilindro_hertz_checks(p: dict[str, float], b: Built, parsed: dict[str, Any]) -> list[V.Check]:
    r = _reaction(parsed, "REF")
    load = abs(r[1]) if r else None
    if not load:
        return [V.Check("hertz.carga", V.INCONCLUSIVE, detail="sem reação no indentador: nada a comparar")]
    theory = V.hertz_line(load, b.analytic["radius"], b.analytic["length"], b.analytic["e_modulus"],
                          b.analytic["nu"], rigid=True)
    contact = _contact(parsed)
    profile = [(b.coords[n][0], row["pressure"]) for n, row in contact.items()
               if n in set(b.interface) and row.get("pressure", 0.0) > 0.0]
    fit = V.hertz_fit(profile)
    src = "Hertz, contato linear: b = sqrt(4PR/(pi L E*)), p0 = 2P/(pi b L), com P medido"
    if fit is None:
        return [V.Check("hertz.carga", V.PASS, load, None, f"carga medida {load:.6g} N", ""),
                V.Check("hertz.pressao_maxima", V.INCONCLUSIVE, detail="o perfil de pressão não foi lido do .frd"),
                V.Check("hertz.meia_largura", V.INCONCLUSIVE, detail="o perfil de pressão não foi lido do .frd")]
    p0, half_width, r2 = fit
    return [
        V.compare("hertz.carga", load, load, 0.0, "carga total prescrita no indentador", "N", "carga normal"),
        V.compare("hertz.pressao_maxima", p0, theory.get("p0"), 0.15, src, "MPa", "pressão máxima (ajuste do perfil)"),
        V.compare("hertz.meia_largura", half_width, theory.get("b"), 0.15, src, "mm", "meia-largura do contato"),
        V.between("hertz.perfil", r2, 0.85, 1.01,
                  "ajuste p² = p0² - (p0²/b²) x² aos nós carregados: 1 é o perfil de Hertz exato",
                  "qualidade do ajuste (R²)"),
        V.at_most("atrito.cone", V.coulomb_ratio(contact, b.interface), p["friccao"], 0.05,
                  "|tau| <= mu p em cada nó carregado", "", "maior razão |tau|/p"),
    ]


# ================================================================================= 4. self-locking on an incline
def _rampa_build(p: dict[str, float]) -> Built:
    a, hb, hs = p["lado"], p["altura_base"], p["altura_bloco"]
    n = _mesh_size(p["malha"])
    theta = math.radians(p["angulo"])
    slope = math.tan(theta)

    def tilt(x: float) -> tuple[float, float, float]:
        return (0.0, slope * x, 0.0)

    base = Grid(axis(0, a, n), axis(0, hb, n), axis(0, a, n), shift=tilt)
    slider = Grid(axis(0, a, n), axis(0, hs, n), axis(0, a, n), first_node=base.next_node,
                  first_elem=base.next_elem, shift=lambda x: (0.0, tilt(x)[1] + hb, 0.0))
    deck = D.Deck(f"contato com atrito: interface inclinada de {p['angulo']:g}° "
                  f"(autotravamento, mu = {p['friccao']:g})")
    _stack(deck, [("BASE", base), ("SLIDER", slider)], "ACO", p["E"], p["nu"])

    bfix = base.nodes_where(lambda q: abs(q[1] - slope * q[0]) < 1e-9)
    top = slider.nodes_where(lambda q: abs(q[1] - (slope * q[0] + hb + hs)) < 1e-9)
    D.node_set(deck, "BFIX", bfix)
    D.node_set(deck, "TOP", top)
    D.node_set(deck, "ZALL", sorted(base.node_ids + slider.node_ids))
    D.surface(deck, "MASTER", base.faces("y", hb, side=-1, curved=lambda q: slope * q[0] + hb))
    D.surface(deck, "SLAVE", slider.faces("y", hb, side=None, curved=lambda q: slope * q[0] + hb))
    D.interaction(deck, "INT", p["friccao"])
    D.contact_pair(deck, "CONTACT", slave="SLAVE", master="MASTER", interaction_name="INT")

    per_node = -p["carga"] / len(top)
    deck.block("*STEP, NLGEOM=YES", "*STATIC", "0.25, 1.0")
    deck.block("*BOUNDARY", "BFIX, 1, 3, 0.0", "ZALL, 3, 3, 0.0")
    deck.block("*CLOAD", f"TOP, 2, {per_node:.8g}")
    D.outputs(deck, totals_set="BFIX", displacement_set="TOP", stress=True)
    deck.block("*END STEP")

    normal = (-math.sin(theta), math.cos(theta), 0.0)
    tangent = (math.cos(theta), math.sin(theta), 0.0)
    return Built(deck=deck.text(),
                 analytic={"tan_theta": slope, "theta_deg": p["angulo"],
                           "atan_mu_deg": math.degrees(math.atan(p["friccao"])),
                           "ratio": min(slope, p["friccao"]), "locks": 1.0 if slope <= p["friccao"] else 0.0,
                           "normal": p["carga"] * math.cos(theta), "tangent": p["carga"] * math.sin(theta),
                           "carga": p["carga"]},
                 coords=dict(base.nodes) | dict(slider.nodes),
                 interface=tuple(slider.nodes_where(lambda q: abs(q[1] - (slope * q[0] + hb)) < 1e-9)),
                 totals=("BFIX",), frame={"normal": normal, "tangent": tangent},
                 notes=[f"o conjunto inteiro está inclinado em {p['angulo']:g}° e a carga é vertical "
                        f"({p['carga']:g} N): é o bloco no plano inclinado, com a rampa nas coordenadas",
                        "o modelo é bidimensional na prática: o deslocamento fora do plano é nulo em todos os nós",
                        f"o bloco trava se tan(theta) <= mu, isto é, se o ângulo for menor que "
                        f"{math.degrees(math.atan(p['friccao'])):.2f}° para este mu"]
                        + ([f"atenção: tan(theta) = {slope:.4g} está a menos de 20% de mu = {p['friccao']:.4g}; "
                            f"perto do limite a malha tende a escorregar mesmo prevendo travamento"]
                           if slope > 0.8 * p["friccao"] >= slope * 0.0 and slope < p["friccao"] else []))


def _rampa_checks(p: dict[str, float], b: Built, parsed: dict[str, Any]) -> list[V.Check]:
    a = b.analytic
    locked = a["locks"] > 0.5
    status = parsed.get("verdict", "ok")
    src = (f"autotravamento: tan(theta) = {a['tan_theta']:.4g} {'<=' if locked else '>'} mu = "
           f"{p['friccao']:.4g}, então T/N = {'tan(theta)' if locked else 'mu'}")
    # a block that cannot hold slips: CalculiX then finds no equilibrium and stops with "too many attempts".
    # That is the predicted answer, not a failed run, and the check says which one it was.
    if not locked:
        outcome = V.PASS if status == "not_converged" else V.FAIL
        found = ("não encontrou equilíbrio, como previsto" if status == "not_converged"
                 else "convergiu, o que contraria a previsão de escorregamento")
        return [V.Check("rampa.estado", outcome, 0.0 if status == "not_converged" else 1.0, a["locks"],
                        f"previsto deslizar (tan {a['tan_theta']:.4g} > mu {p['friccao']:.4g}): a análise {found}",
                        src)]
    r = _reaction(parsed, "BFIX")
    if not r:
        return [V.Check("rampa.resultante", V.INCONCLUSIVE, detail="sem reação no apoio: nada a comparar")]
    n_hat, t_hat = b.frame["normal"], b.frame["tangent"]
    # the support reacts with the negative of the force the slider exerts on the base
    fx, fy = -r[0], -r[1]
    N = fx * n_hat[0] + fy * n_hat[1]
    T = fx * t_hat[0] + fy * t_hat[1]
    ratio = abs(T) / abs(N) if N else None
    checks = [V.compare("rampa.resultante", ratio, a["ratio"], 0.05, src, "", "razão T/N na interface"),
              V.compare("rampa.carga_normal", abs(N), a["normal"], 0.05,
                        "N = F cos(theta) para a carga vertical F", "N", "componente normal na interface"),
              V.at_most("atrito.cone", V.coulomb_ratio(_contact(parsed), b.interface), p["friccao"], 0.05,
                        "|tau| <= mu p em cada nó carregado", "", "maior razão |tau|/p"),
              V.Check("rampa.estado", V.PASS, 1.0, a["locks"],
                      f"previsto travar (tan {a['tan_theta']:.4g} <= mu {p['friccao']:.4g}) e a análise encontrou "
                      f"equilíbrio: o bloco não escorregou (escorregamento medido "
                      f"{_fmt_slip(V.slip_range(_contact(parsed), b.interface))})", src)]
    return checks


def _fmt_slip(slip: float | None) -> str:
    return "não lido do .frd" if slip is None else f"{slip:.3g} mm"


# ============================================================================= 5. bolted joint with a preload
def _parafuso_build(p: dict[str, float]) -> Built:
    """Two plates clamped by a bolt whose preload is a ``*PRE-TENSION SECTION``: the clamp force is the preload, so the
    slip load is ``mu`` times it — not mu times a pressure the user typed.

    The bolt is one B31 element, which CalculiX deletes and replaces by the constraint that carries the preload. Its
    two ends are free nodes on the bolt axis, and each is joined to its plate by a ``*DISTRIBUTING COUPLING`` over the
    washer footprint — the connection the CalculiX manual asks for in the ``*PRE-TENSION SECTION`` section itself:
    "the nodes of the beam element ... should be connected by linear equations or a *DISTRIBUTING COUPLING card to
    nodes of the structures to be held together". Each footprint node's weight is its tributary area, so the head
    presses the plate with a uniform traction over the footprint instead of denting it at a single node.
    """
    a, h = p["lado"], p["espessura"]
    n = _mesh_size(p["malha"])
    base = Grid(axis(0, a, n), axis(0, h, n), axis(0, a, n))
    top = Grid(axis(0, a, n), axis(0, h, n), axis(0, a, n), first_node=base.next_node,
               first_elem=base.next_elem, shift=lambda x: (0.0, h, 0.0))
    # the two ends of the bolt and the node the preload is applied at: free nodes on the axis, not part of the plates
    head, foot, ref = top.next_node, top.next_node + 1, top.next_node + 2
    bolt_elem = top.next_elem
    nodes = dict(base.nodes) | dict(top.nodes) | {head: (a / 2, 2 * h, a / 2), foot: (a / 2, 0.0, a / 2),
                                                  ref: (a / 2, -1.0, a / 2)}
    radius = p["raio_pegada"]

    def under(grid: Grid, level: float) -> list[int]:
        """The nodes of that face inside the washer footprint: a disc of ``raio_pegada`` around the bolt axis."""
        return [nid for nid, q in grid.nodes.items()
                if abs(q[1] - level) < 1e-9 and math.hypot(q[0] - a / 2, q[2] - a / 2) <= radius + 1e-9]

    under_head, under_nut = under(top, 2 * h), under(base, 0.0)
    if len(under_head) < 4 or len(under_nut) < 4:
        raise ValueError(f"raio_pegada = {radius:g} mm picks {len(under_head)} node(s) of the head face at "
                         f"malha = {n}: the footprint needs at least 4 nodes, so either widen the footprint or refine "
                         f"the mesh, otherwise the head is a point again")

    deck = D.Deck(f"contato com atrito: junta aparafusada com pré-tensão de {p['pretensao']:g} N "
                  f"(mu = {p['friccao']:g}, pegada de raio {radius:g} mm)")
    D.geometry(deck, [("BASE", base), ("TOPO", top)])
    deck.block("*ELEMENT, TYPE=B31, ELSET=BOLT", f"{bolt_elem}, {head}, {foot}")
    deck.block("*NODE", *[f"{nid}, {nodes[nid][0]:.8g}, {nodes[nid][1]:.8g}, {nodes[nid][2]:.8g}"
                          for nid in (head, foot, ref)])
    D.material(deck, "ACO", p["E"], p["nu"])
    D.section(deck, "BASE", "ACO")
    D.section(deck, "TOPO", "ACO")
    deck.block("*BEAM SECTION, ELSET=BOLT, MATERIAL=ACO, SECTION=CIRC", f"{p['raio_parafuso']:.6g}")
    deck.block(f"*PRE-TENSION SECTION, ELEMENT={bolt_elem}, NODE={ref}")
    # the head and the nut: the area of each footprint node is its share of the clamp force, so the pressure under
    # the head is uniform over the footprint instead of a peak at one node
    head_weights = V.tributary_areas(nodes, under_head, plane="y")
    D.distributing_coupling(deck, "HEAD", bolt_elem + 1, head, head_weights)
    D.distributing_coupling(deck, "NUT", bolt_elem + 2, foot, V.tributary_areas(nodes, under_nut, plane="y"))

    D.node_set(deck, "REF", [ref])
    # the nut footprint is the only part of the bottom face that is not a support: those nodes hang on the coupling,
    # and a node on the dependent side of a constraint cannot also carry a boundary condition (CalculiX stops with
    # "MPC and SPC" if it does)
    D.node_set(deck, "BFIX", [nid for nid in base.nodes_on("y", 0.0) if nid not in set(under_nut)])
    D.node_set(deck, "TOPO", top.nodes_on("y", 2 * h))
    D.surface(deck, "MASTER", base.faces("y", h, side=-1))
    D.surface(deck, "SLAVE", top.faces("y", h, side=+1))
    D.interaction(deck, "INT", p["friccao"])
    D.contact_pair(deck, "CONTACT", slave="SLAVE", master="MASTER", interaction_name="INT")

    deck.block("*STEP, NLGEOM=NO", "*STATIC", "0.25, 1.0")
    deck.block("*BOUNDARY", "BFIX, 1, 3, 0.0")
    deck.block("*CLOAD", f"REF, 1, {p['pretensao']:.8g}")
    D.outputs(deck, totals_set="BFIX", displacement_set="TOPO", stress=True)
    deck.block("*END STEP")

    deck.block("*STEP, NLGEOM=NO", "*STATIC", "0.25, 1.0")
    deck.block("*BOUNDARY", f"TOPO, 1, 1, {p['deslizamento']:.6g}")
    D.outputs(deck, totals_set="BFIX", displacement_set="TOPO", stress=True)
    deck.block("*END STEP")

    interface = top.nodes_on("y", h)
    pegada = [nid for nid in interface
              if math.hypot(nodes[nid][0] - a / 2, nodes[nid][2] - a / 2) <= radius + 1e-9]
    return Built(deck=deck.text(),
                 analytic={"preload": p["pretensao"], "friction": p["friccao"] * p["pretensao"],
                           "interface_area": a * a, "slip_imposed": p["deslizamento"],
                           "footprint_area": sum(head_weights.values()),
                           "footprint_radius": radius},
                 coords=nodes,
                 interface=tuple(interface),
                 footprint=tuple(pegada), footprint_radius=radius,
                 totals=("BFIX",), times={"preload": 1.0, "pull": 2.0},
                 notes=[f"a pré-tensão de {p['pretensao']:g} N é aplicada no nó de referência do *PRE-TENSION "
                        f"SECTION e chega à estrutura pelo elemento B31, que o CalculiX substitui por uma restrição",
                        f"a cabeça e a porca são pegadas de raio {radius:g} mm ({len(under_head)} nós cada), ligadas "
                        f"ao nó do parafuso por *DISTRIBUTING COUPLING com peso de área: a carga entra distribuída, "
                        f"como a de uma cabeça, não por um nó só",
                        "o parafuso não é modelado volumetricamente, então a flexibilidade dele é só a do "
                        "comprimento aparafusado",
                        "a pegada é quadrada (é o bloco de nós dentro do raio), não um anel: o furo do parafuso e a "
                        "concentração de pressão na borda do furo não estão no modelo"])

def _parafuso_checks(p: dict[str, float], b: Built, parsed: dict[str, Any]) -> list[V.Check]:
    a = b.analytic
    weights = V.tributary_areas(b.coords, b.interface, plane="y")
    at = parsed.get("contact_until") or {}
    clamped = V.resultant(at.get("preload") or {}, weights)
    final = _contact(parsed)
    now = V.resultant(final, weights)
    r = _reaction(parsed, "BFIX")
    checks = [V.compare("parafuso.aperto", clamped, a["preload"], 0.03,
                        "força de aperto medida pela integral de CPRESS na interface, contra a pré-tensão aplicada",
                        "N", "aperto no fim do primeiro passo"),
              V.compare("equilibrio.aperto", now, clamped, 0.10,
                        "o aperto não se perde quando a junta é puxada", "N",
                        "aperto no fim do puxamento"),
              _pegada_check(p, b, at.get("preload") or {})]
    if not r:
        checks.append(V.Check("atrito.saturacao", V.INCONCLUSIVE, detail="sem reação no apoio: nada a comparar"))
        return checks
    src = f"junta pré-tensionada: F = mu x N = {p['friccao']:.4g} x (aperto medido no primeiro passo) N"
    capacity = p["friccao"] * (clamped or a["preload"])
    checks.append(V.compare("atrito.saturacao", abs(r[0]), capacity, 0.03, src, "N",
                            "força tangencial com a interface deslizando"))
    checks.append(V.at_most("atrito.limite", abs(r[0]), capacity, 0.03, src, "N",
                            "força tangencial transmitida"))
    checks.append(V.at_most("atrito.cone", V.coulomb_ratio(final, b.interface), p["friccao"], 0.05,
                            "|tau| <= mu p em cada nó carregado", "", "maior razão |tau|/p"))
    opened = [row["opening"] for node, row in final.items() if node in set(b.interface)]
    checks.append(V.at_most("contato.fechado", max(opened) if opened else None, 0.0, 0.0,
                            "a junta não abre sob o puxamento (COPEN <= 0 em todos os nós)",
                            "mm", "maior abertura medida"))
    return checks


def _pegada_check(p: dict[str, float], b: Built, state: dict[int, dict[str, float]]) -> V.Check:
    """The head is an area and not a point: the contact pressure under the washer footprint has to be a pressure.

    The ratio the check bounds is the peak over the mean of ``CPRESS`` on the interface nodes inside the footprint. A
    load entering at one node gives a spike there and a tail everywhere else, and the ratio it produces depends on how
    far the load had to spread through the plate; a head that presses with a uniform traction over an area gives a
    ratio of 1 for an infinitely stiff plate, and only a little over 1 for a plate that carries the load away.
    Measured on this case at ``malha = 6``: 1,6 for the single-node bolt this case used to be and 1,3 now, so the
    band is the declared tolerance of the head being an area, not a number tuned to the answer.
    """
    rows = [row["pressure"] for node, row in state.items()
            if node in set(b.footprint) and row.get("pressure", 0.0) > 0.0]
    if len(rows) < 3:
        return V.Check("parafuso.pegada", V.INCONCLUSIVE,
                       detail=f"a pressão em {len(rows)} nó(s) da pegada não foi lida: nada a comparar")
    mean = sum(rows) / len(rows)
    ratio = max(rows) / mean if mean > 0 else None
    return V.between("parafuso.pegada", ratio, 1.0, 1.5,
                     "cabeça como área: pressão uniforme sobre a pegada daria 1; um nó só concentra a carga "
                     "(1,6 neste caso, medido com o parafuso de um nó só na mesma malha)",
                     f"pico / média de CPRESS nos {len(rows)} nós da pegada (raio {p['raio_pegada']:g} mm)")


# ==================================================================== 6. shrink fit (interference) on a shaft
def _ajuste_build(p: dict[str, float]) -> Built:
    """A ring shrunk onto a hollow shaft: the interference is geometric, and Lamé gives the pressure it produces."""
    a, b, c = p["raio_interface"], p["raio_furo"], p["raio_externo"]
    delta = p["interferencia"]
    n = _mesh_size(p["malha"])
    quarter = math.pi / 2
    shaft = Grid(axis(b, a, n), axis(0, quarter, n), [0.0, p["espessura"]], polar=True)
    ring = Grid(axis(a - delta, c, n), axis(0, quarter, n), [0.0, p["espessura"]], polar=True,
                first_node=shaft.next_node, first_elem=shaft.next_elem)
    deck = D.Deck(f"contato com atrito: ajuste forçado de {delta:g} mm no raio {a:g} mm "
                  f"(mu = {p['friccao']:g})")
    D.geometry(deck, [("EIXO", shaft), ("ANEL", ring)])
    D.material(deck, "ACO", p["E"], p["nu"])
    D.material(deck, "ANEL_MAT", p["E_anel"], p["nu_anel"])
    D.section(deck, "EIXO", "ACO")
    D.section(deck, "ANEL", "ANEL_MAT")

    # the interface is a cylinder, not a plane: it is named by its radius
    D.surface(deck, "MASTER", shaft.faces_where(lambda q: abs(math.hypot(q[0], q[1]) - a) < 1e-9))
    D.surface(deck, "SLAVE", ring.faces_where(lambda q: abs(math.hypot(q[0], q[1]) - (a - delta)) < 1e-9))
    D.interaction(deck, "INT", p["friccao"], k_normal=p["rigidez_contato"], stick_slope=1e5)
    D.contact_pair(deck, "CONTACT", slave="SLAVE", master="MASTER", interaction_name="INT")

    # symmetry on the two radial planes, plane strain on the two thickness faces: the quarter model is the
    # axisymmetric problem, which is what Lamé solves
    every = sorted(shaft.node_ids + ring.node_ids)
    D.node_set(deck, "SYM_Y", [nid for nid in every
                               if abs(shaft.nodes.get(nid, ring.nodes.get(nid))[1]) < 1e-9])
    D.node_set(deck, "SYM_X", [nid for nid in every
                               if abs(shaft.nodes.get(nid, ring.nodes.get(nid))[0]) < 1e-9])
    D.node_set(deck, "ZB", [nid for nid in every if abs(shaft.nodes.get(nid, ring.nodes.get(nid))[2]) < 1e-9])
    D.node_set(deck, "ZT", [nid for nid in every
                            if abs(shaft.nodes.get(nid, ring.nodes.get(nid))[2] - p["espessura"]) < 1e-9])
    D.node_set(deck, "PIN", [every[0]])
    deck.block("*STEP, NLGEOM=NO", "*STATIC", "0.1, 1.0")
    deck.block("*BOUNDARY", "SYM_Y, 2, 2, 0.0", "SYM_X, 1, 1, 0.0", "ZB, 3, 3, 0.0", "ZT, 3, 3, 0.0",
               "PIN, 1, 1, 0.0")
    D.outputs(deck, totals_set="PIN", stress=True)
    deck.block("*END STEP")

    pressure = V.lame_shrink_fit(delta, a, b, c, p["E"], p["nu"], plane_strain=True,
                                 E_ring=p["E_anel"], nu_ring=p["nu_anel"])
    return Built(deck=deck.text(),
                 analytic={"lame_pressure": pressure, "interference": delta, "interface_radius": a,
                           "shaft_bore": b, "ring_outer": c},
                 coords=dict(shaft.nodes) | dict(ring.nodes),
                 interface=tuple(ring.nodes_at_radius(a - delta)),
                 totals=("PIN",),
                 notes=[f"interferência radial de {delta:g} mm: o anel é {delta:g} mm menor que o eixo no raio "
                        f"{a:g} mm, e a pressão vem do fechamento do contato, não de uma carga aplicada",
                        "modelo de um quarto, simétrico nos dois planos radiais, em deformação plana: é exatamente "
                        "o problema axissimétrico que Lamé resolve",
                        "mesmo material no eixo e no anel; com materiais diferentes a fórmula de Lamé muda no termo "
                        "de Poisson"])


def _ajuste_checks(p: dict[str, float], b: Built, parsed: dict[str, Any]) -> list[V.Check]:
    a = b.analytic
    contact = _contact(parsed)
    pressures = [r["pressure"] for n, r in contact.items() if n in set(b.interface) and r["pressure"] > 0]
    mean = sum(pressures) / len(pressures) if pressures else None
    same = p["E_anel"] == p["E"] and p["nu_anel"] == p["nu"]
    src = ("Lamé em deformação plana, um material: p = delta E´ / (a [(c²+a²)/(c²-a²) + (a²+b²)/(a²-b²)]), "
           "E´ = E/(1-nu²)" if same else
           "Lamé em deformação plana, dois materiais: delta = p a [(1+nu_anel)/E_anel (c²+(1-2 nu_anel)a²)/"
           "(c²-a²) + (1+nu_eixo)/E_eixo ((1-2 nu_eixo)a²+b²)/(a²-b²)]")
    checks = [V.compare("ajuste.pressao", mean, a["lame_pressure"], 0.05, src, "MPa",
                        "pressão média de contato na interface"),
              V.between("ajuste.uniformidade",
                        (min(pressures) / max(pressures)) if pressures else None, 0.80, 1.01,
                        "solução axissimétrica: a pressão não varia ao longo do perímetro",
                        "razão entre a menor e a maior pressão de contato")]
    whole = len(b.interface) or 1
    loaded = len([n for n in b.interface if contact.get(n, {}).get("pressure", 0.0) > 0])
    checks.append(V.compare("ajuste.contato_total", loaded / whole, 1.0, 0.02,
                            "a interface fecha em todo o perímetro (nenhum nó aberto)",
                            "", "fração dos nós da interface em contato"))
    checks.append(V.at_most("atrito.cone", V.coulomb_ratio(contact, b.interface), p["friccao"], 0.05,
                            "|tau| <= mu p em cada nó carregado", "", "maior razão |tau|/p"))
    return checks


# ============================================================ 7. plasticity: the onset of yield under Hertz
def _plastificacao_build(p: dict[str, float]) -> Built:
    """A rigid cylinder pressed into an elastic-perfectly-plastic block, loaded in two steps: the first ends exactly
    where Hertz says yield begins, ``p0 = k * sigma_y`` with ``k`` from Johnson, the second goes past it.

    The closed form is not a curve fit: the load of the first step is computed *backwards* from the yield pressure —
    ``P = p0^2 pi L R / E*`` — so the run is asked for the state the theory names. What the second step shows is that
    the material then does what ``*PLASTIC`` promises: the equivalent stress stops at ``sigma_y`` — an elastic
    material would reach ``sqrt(sobrecarga) * k * sigma_y`` there — and the contact patch widens to carry the extra
    load instead of the pressure taking all of it.
    """
    R, H = p["raio"], p["profundidade"]
    nx, ny = _mesh_size(p["malha"]), _mesh_size(p["malha"] / 2)
    xs = sorted({-x for x in axis(0, p["largura"] / 2, nx, grading=1.5)} | set(axis(0, p["largura"] / 2, nx,
                                                                                 grading=1.5)))
    ys = axis(-H, 0.0, ny, grading=1.3, cluster="high")
    block = Grid(xs, ys, [0.0, 1.0])
    indenter = Grid(xs, axis(0.0, p["altura_indentador"], 3), [0.0, 1.0], first_node=block.next_node,
                    first_elem=block.next_elem, shift=lambda x: (0.0, x * x / (2 * R), 0.0))
    ref = indenter.next_node

    factor = V.first_yield_factor(p["nu"])
    p_onset = factor * p["tensao_escoamento"]
    e_star = p["E"] / (1 - p["nu"] ** 2)
    load_onset = p_onset ** 2 * math.pi * 1.0 * R / e_star          # Hertz inverted: the load that gives p0 = k sy
    load_deep = p["sobrecarga"] * load_onset

    deck = D.Deck(f"contato com plastificação: cilindro rígido sobre placa elastoplástica, escoamento a "
                  f"p0 = k sigma_y (sigma_y = {p['tensao_escoamento']:g} MPa, k = {factor:.3f})")
    D.geometry(deck, [("BASE", block), ("IND", indenter)])
    deck.block("*NODE", f"{ref}, 0.0, {2 * p['altura_indentador']:.8g}, 0.5")
    D.material(deck, "ACO", p["E"], p["nu"])
    D.plastic(deck, p["tensao_escoamento"])
    D.material(deck, "RIGIDO", p["E"], p["nu"])
    D.section(deck, "BASE", "ACO")
    D.section(deck, "IND", "RIGIDO")

    D.node_set(deck, "REF", [ref])
    D.node_set(deck, "RIGALL", indenter.node_ids)
    D.node_set(deck, "BBOT", block.nodes_on("y", -H))
    D.node_set(deck, "ZB", block.nodes_on("z", 0.0))
    D.node_set(deck, "ZT", block.nodes_on("z", 1.0))
    deck.block(f"*RIGID BODY, NSET=RIGALL, REF NODE={ref}")
    D.surface(deck, "MASTER", block.faces("y", 0.0, side=-1))
    D.surface(deck, "SLAVE", indenter.faces("y", 0.0, side=None, curved=lambda q: q[0] ** 2 / (2 * R)))
    D.interaction(deck, "INT", p["friccao"], k_normal=p["rigidez_contato"])
    D.contact_pair(deck, "CONTACT", slave="SLAVE", master="MASTER", interaction_name="INT")

    deck.block("*STEP, NLGEOM=YES", "*STATIC", "0.1, 1.0")
    deck.block("*BOUNDARY", "BBOT, 2, 2, 0.0", "ZB, 3, 3, 0.0", "ZT, 3, 3, 0.0")
    deck.block("*BOUNDARY", "REF, 1, 1, 0.0", "REF, 3, 3, 0.0")
    deck.block("*CLOAD", f"REF, 2, {-load_onset:.8g}")
    D.outputs(deck, totals_set="REF", stress=True)
    deck.block("*END STEP")

    deck.block("*STEP, NLGEOM=YES", "*STATIC", "0.1, 1.0")
    deck.block("*CLOAD", f"REF, 2, {-load_deep:.8g}")
    D.outputs(deck, totals_set="REF", stress=True)
    deck.block("*END STEP")

    interface = tuple(n for n, q in indenter.nodes.items() if abs(q[1] - q[0] ** 2 / (2 * R)) < 1e-9)
    return Built(deck=deck.text(),
                 analytic={"radius": R, "length": 1.0, "e_modulus": p["E"], "nu": p["nu"],
                           "yield_stress": p["tensao_escoamento"], "yield_factor": factor,
                           "yield_pressure": p_onset, "load_onset": load_onset, "load_deep": load_deep,
                           "overload": p["sobrecarga"], "e_star": e_star},
                 coords=dict(block.nodes) | dict(indenter.nodes) | {ref: (0.0, 2 * p["altura_indentador"], 0.5)},
                 interface=interface, totals=("REF",), times={"escoamento": 1.0, "profundo": 2.0},
                 surfaces={"bloco": block.node_ids},
                 notes=[f"o material é elastoplástico perfeito: *PLASTIC com dois pontos de mesma tensão, então ele "
                        f"carrega sigma_y = {p['tensao_escoamento']:g} MPa e não mais, por mais que seja empurrado",
                        f"o primeiro passo para exatamente na carga em que Hertz prevê p0 = {factor:.3f} sigma_y "
                        f"({p_onset:.0f} MPa) — a carga é calculada de trás para a frente a partir dessa pressão, não "
                        f"ajustada até passar",
                        f"o segundo passo carrega {p['sobrecarga']:g}x mais, onde um material elástico chegaria a "
                        f"{factor * math.sqrt(p['sobrecarga']):.2f} sigma_y: o escoamento tem de segurar a tensão "
                        f"equivalente em sigma_y",
                        "indentador rígido, deformação plana, sem encruamento e sem atrito na pressão normal: é o "
                        "problema que a solução de Hertz com o critério de von Mises descreve",
                        "a pressão de pico lida do .frd erra para baixo por ~10% nesta malha — o mesmo viés do caso "
                        "elástico, que é discretização do patch de contato e não plasticidade: é por isso que o "
                        "fator de Johnson é conferido na tensão equivalente, onde o erro é de 3%"])


def _plastificacao_checks(p: dict[str, float], b: Built, parsed: dict[str, Any]) -> list[V.Check]:
    """The factor of Johnson is judged on the *stress*, not on the fitted pressure: at the load the closed form names
    as the onset, the largest equivalent stress in the block has to be ``sigma_y`` — and no more, at any load after
    it. The peak pressure of the same state is reported against ``k sigma_y`` with the 15 % the elastic case already
    declares for reading ``p0`` node by node.
    """
    a = b.analytic
    contact_at = parsed.get("contact_until") or {}
    stress_at = parsed.get("stress_until") or {}
    r = _reaction(parsed, "REF")
    load = abs(r[1]) if r else None
    block_nodes = set(b.surfaces.get("bloco") or ())

    def peak_von_mises(state: dict[int, dict[str, float]]) -> float | None:
        return max((V.von_mises(row.get("SXX", 0.0), row.get("SYY", 0.0), row.get("SZZ", 0.0),
                                row.get("SXY", 0.0), row.get("SYZ", 0.0), row.get("SZX", 0.0))
                    for n, row in state.items() if not block_nodes or n in block_nodes), default=None)

    def p0_at(state: dict[int, dict[str, float]]) -> float | None:
        interface = set(b.interface)
        profile = [(b.coords[n][0], row["pressure"]) for n, row in state.items()
                   if n in interface and row.get("pressure", 0.0) > 0.0]
        fit = V.hertz_fit(profile)
        return fit[0] if fit else None

    sy = p["tensao_escoamento"]
    checks: list[V.Check] = [
        V.compare("plastificacao.carga", load, a["load_deep"], 0.02,
                  "o indentador carrega o que foi pedido (reação no nó de referência do corpo rígido)",
                  "N", "carga no fim do segundo passo")]
    reach = peak_von_mises(stress_at.get("escoamento") or {})
    checks.append(V.between(
        "plastificacao.primeira_escoamento", (reach / sy) if reach else None, 0.95, 1.05,
        f"Johnson: com p0 = {a['yield_factor']:.3f} sigma_y (von Mises, contato linear; 1,67 sigma_y por Tresca) o "
        f"pico da tensão equivalente, que fica a 0,70 b da superfície, iguala sigma_y — é isso que a carga do "
        f"primeiro passo pede, e a banda de 5% é o erro de extrapolar a tensão nodal; Johnson, Contact Mechanics "
        f"(Cambridge, 1985), início do escoamento",
        "maior von Mises nodal do bloco no fim do primeiro passo, sobre sigma_y"))
    checks.append(V.compare(
        "plastificacao.pressao_escoamento", p0_at(contact_at.get("escoamento") or {}), a["yield_pressure"], 0.15,
        f"a pressão de pico no início do escoamento é k sigma_y = {a['yield_factor']:.3f} x {sy:g} = "
        f"{a['yield_pressure']:.0f} MPa; 15% é a tolerância do caso elástico, o erro de ler p0 nó a nó",
        "MPa", "pressão de pico medida no fim do primeiro passo"))
    held = peak_von_mises(stress_at.get("profundo") or {})
    checks.append(V.between(
        "plastificacao.saturacao", (held / sy) if held else None, 0.95, 1.05,
        f"material elastoplástico perfeito: a tensão equivalente não passa de sigma_y, e sem *PLASTIC o segundo passo "
        f"chegaria a {a['yield_factor'] * math.sqrt(a['overload']):.2f} sigma_y",
        "maior von Mises nodal do bloco no fim do segundo passo, sobre sigma_y"))
    checks.append(V.at_most("atrito.cone", V.coulomb_ratio(_contact(parsed), b.interface), p["friccao"], 0.05,
                            "|tau| <= mu p em cada nó carregado", "", "maior razão |tau|/p"))
    return checks


# ------------------------------------------------------------------------------------------------- the library
_MESH = Param("malha", "malha", 6, "células", 3, 16, "células por direção nos blocos; o caso não muda com ela, "
                                                      "só a resolução do patch de contato")
_MATERIAL = (Param("E", "módulo de elasticidade", 210000.0, "MPa", 1.0, 1e7, "aço por padrão"),
             Param("nu", "coeficiente de Poisson", 0.3, "", 0.0, 0.499, ""))
_FRICTION = Param("friccao", "coeficiente de atrito mu", 0.3, "", 0.0, 1.5,
                  "Coulomb seco: o limite é mu x N, independente da área aparente")

CASES: dict[str, Case] = {
    "bloco_plano": Case(
        id="bloco_plano",
        title="Bloco deslizante sobre placa fixa",
        summary="Um bloco é prensado contra uma placa e empurrado lateralmente; a força tangencial que a junta "
                "transmite para exatamente em mu x N, e a interface passa a deslizar.",
        physics="Contato conforme com atrito de Coulomb em análise estática não linear: o acoplamento "
                "normal-tangencial é o que a interface resolve, e o limite de atrito é verificado contra a carga "
                "aplicada.",
        assumptions=("atrito seco de Coulomb com mu constante (sem dependência de pressão ou velocidade)",
                     "elasticidade linear nos dois corpos, sem plastificação nem desgaste",
                     "deslizamento prescrito na face superior, não força: assim a interface chega ao limite sem "
                     "divergir"),
        references=(("limite de atrito F = mu N", "https://prepomax.fs.um.si/"),
                    ("contato com atrito no CalculiX", "https://www.dhondt.de/ccx_2.23.pdf")),
        params=(_FRICTION,
                Param("pressao", "pressão normal aplicada", 5.0, "MPa", 0.01, 500.0, "carga de aperto na área nominal"),
                Param("lado", "lado do bloco", 10.0, "mm", 1.0, 200.0, ""),
                Param("altura_base", "altura da placa", 4.0, "mm", 0.5, 100.0, ""),
                Param("altura_bloco", "altura do bloco", 4.0, "mm", 0.5, 100.0, ""),
                Param("deslizamento", "deslizamento prescrito", 0.05, "mm", 0.0, 5.0,
                      "grande o bastante para a interface chegar ao limite de atrito"),
                _MESH) + _MATERIAL,
        build=_bloco_plano_build, checks=_bloco_plano_checks),

    "junta_dupla": Case(
        id="junta_dupla",
        title="Junta aparafusada em cisalhamento duplo",
        summary="Uma chapa é puxada entre duas chapas apertadas: as duas interfaces dividem a carga, então cada "
                "apoio recebe mu x N e o puxamento chega a 2 mu x N.",
        physics="Duas interfaces de contato com atrito carregadas pelo mesmo aperto; é o ensaio de resistência ao "
                "escorregamento de uma ligação parafusada, sem modelar o parafuso.",
        assumptions=("o aperto entra como pressão sobre a pegada do parafuso; a pré-tensão do parafuso não é modelada",
                     "a chapa comprimida é travada lateralmente, como o parafuso e a estrutura fariam",
                     "atrito seco de Coulomb, igual nas duas interfaces"),
        references=(("resistência ao escorregamento de ligações parafusadas", "https://prepomax.fs.um.si/"),
                    ("contato com atrito no CalculiX", "https://www.dhondt.de/ccx_2.23.pdf")),
        params=(_FRICTION,
                Param("pressao", "pressão de aperto", 10.0, "MPa", 0.01, 500.0, "sobre a pegada do parafuso"),
                Param("lado", "lado das chapas", 10.0, "mm", 1.0, 200.0, ""),
                Param("espessura", "espessura de cada chapa", 3.0, "mm", 0.5, 50.0, ""),
                Param("deslizamento", "deslizamento prescrito", 0.05, "mm", 0.0, 5.0, ""),
                _MESH) + _MATERIAL,
        build=_junta_dupla_build, checks=_junta_dupla_checks),

    "cilindro_hertz": Case(
        id="cilindro_hertz",
        title="Cilindro sobre placa: contato de Hertz",
        summary="Um cilindro rígido é pressionado contra uma placa elástica. A pressão não é uniforme: a meia-largura "
                "e a pressão máxima seguem Hertz, e é isso que a corrida é comparada.",
        physics="Contato não conforme em deformação plana: a área de contato cresce com a carga, então a pressão "
                "máxima não é carga/área — é a solução de Hertz, e o perfil lido do .frd é ajustado a ela.",
        assumptions=("indentador rígido (E >> E da placa): E* = E/(1-nu²)",
                     "deformação plana: o bloco tem uma célula de espessura com as duas faces travadas fora do plano",
                     "elasticidade linear, sem plastificação no contato; atrito não altera a pressão normal"),
        references=(("contato de Hertz (linha)", "https://en.wikipedia.org/wiki/Contact_mechanics"),
                    ("contato no CalculiX", "https://www.dhondt.de/ccx_2.23.pdf")),
        params=(Param("raio", "raio do cilindro", 100.0, "mm", 1.0, 5000.0, ""),
                Param("penetracao", "penetração prescrita", 0.02, "mm", 1e-4, 1.0,
                      "quanto o indentador desce; a carga resulta dela"),
                Param("largura", "largura do modelo", 12.0, "mm", 1.0, 200.0, "metade dela de cada lado do centro"),
                Param("profundidade", "profundidade do bloco", 6.0, "mm", 0.5, 200.0, "≥ 3x a meia-largura esperada"),
                Param("altura_indentador", "altura do indentador", 2.0, "mm", 0.2, 50.0, ""),
                Param("rigidez_contato", "rigidez do contato (K)", 1e6, "N/mm³", 1e3, 1e9,
                      "penalidade do fechamento de pressão; alta demais não converge, baixa demais penetra"),
                Param("malha", "malha", 40, "células", 8, 120, "células por metade da largura, graduadas ao centro"),
                Param("friccao", "coeficiente de atrito mu", 0.3, "", 0.0, 1.5, "")) + _MATERIAL,
        build=_cilindro_hertz_build, checks=_cilindro_hertz_checks),

    "rampa": Case(
        id="rampa",
        title="Interface inclinada: autotravamento",
        summary="O mesmo bloco, agora sobre uma interface inclinada, com carga vertical. Enquanto tan(theta) <= mu "
                "ele trava e a resultante na interface fica dentro do cone de atrito; acima disso não existe "
                "equilíbrio e a análise para sem convergir — que é a resposta física.",
        physics="Atrito estático: a condição de autotravamento de um plano inclinado é tan(theta) <= mu. O caso mede "
                "a razão entre as componentes tangencial e normal na interface (T/N = tan(theta) quando trava) e "
                "confronta o veredito com a previsão calculada antes de rodar.",
        assumptions=("o conjunto todo está inclinado e a carga é vertical (o bloco no plano inclinado)",
                     "atrito seco de Coulomb, sem coesão na interface",
                     "elasticidade linear; o modelo é bidimensional na prática (deslocamento fora do plano nulo)",
                     "um ângulo acima de atan(mu) não tem equilíbrio estático: a corrida termina sem convergir, e "
                     "isso é o resultado previsto, não uma falha do solver"),
        references=(("ângulo de atrito e autotravamento", "https://en.wikipedia.org/wiki/Friction#Angle_of_friction"),
                    ("contato com atrito no CalculiX", "https://www.dhondt.de/ccx_2.23.pdf")),
        params=(Param("friccao", "coeficiente de atrito mu", 0.5, "", 0.0, 1.5,
                      "o padrão é 0,5 e não 0,3 de propósito: o caso precisa de margem sobre tan(theta), porque os nós "
                      "da borda livre da interface micro-escorregam e um mu logo acima de tan(theta) não se sustenta "
                      "na malha"),
                Param("angulo", "ângulo da rampa", 15.0, "°", 0.0, 80.0, "compare com atan(mu) em graus"),
                Param("carga", "carga vertical total", 500.0, "N", 0.1, 1e6, "aplicada na face superior do bloco"),
                Param("lado", "lado do bloco", 10.0, "mm", 1.0, 200.0, ""),
                Param("altura_base", "altura da placa", 4.0, "mm", 0.5, 100.0, ""),
                Param("altura_bloco", "altura do bloco", 4.0, "mm", 0.5, 100.0, ""),
                _MESH) + _MATERIAL,
        build=_rampa_build, checks=_rampa_checks),

    "parafuso_pretensao": Case(
        id="parafuso_pretensao",
        title="Junta aparafusada com pré-tensão",
        summary="O aperto vem de um *PRE-TENSION SECTION de verdade, não de uma pressão digitada: a força de "
                "aperto é a pré-tensão, e a junta escorrega quando o puxamento passa de mu vezes ela.",
        physics="Pré-tensão de parafuso: o elemento de viga do parafuso é substituído por uma restrição que aplica a "
                "força de pré-tensão entre as duas faces, e cada ponta dele chega à chapa por uma pegada de área "
                "(*DISTRIBUTING COUPLING). O caso mede o aperto pela integral da pressão de contato — não pelo que foi "
                "pedido —, confere que a pressão sob a pegada é uma pressão e não um pico, e compara o escorregamento "
                "com mu vezes esse aperto medido.",
        assumptions=("a pré-tensão é aplicada entre os dois nós do elemento de viga, e cada um se liga à sua chapa por "
                     "um *DISTRIBUTING COUPLING sobre a pegada: o aperto é exato e entra distribuído, como o de uma "
                     "cabeça e uma porca",
                     "a pegada é o bloco de nós dentro do raio escolhido (quadrada, não anular): o furo do parafuso e "
                     "a concentração de pressão na borda do furo não estão no modelo",
                     "o parafuso não é modelado volumetricamente, então a flexibilidade do parafuso é a do "
                     "comprimento aparafusado",
                     "atrito seco de Coulomb, elasticidade linear, sem escorregamento parcial"),
        references=(("pré-tensão de parafuso no CalculiX (*PRE-TENSION SECTION)", "https://www.dhondt.de/ccx_2.23.pdf"),
                    ("*DISTRIBUTING COUPLING no CalculiX (ligação do parafuso à estrutura)", "https://www.dhondt.de/ccx_2.23.pdf"),
                    ("resistência ao escorregamento de ligações parafusadas", "https://prepomax.fs.um.si/")),
        params=(_FRICTION,
                Param("pretensao", "pré-tensão do parafuso", 5000.0, "N", 1.0, 1e6, "a força de aperto que o parafuso "
                                                                                  "introduz na junta"),
                Param("lado", "lado das chapas", 10.0, "mm", 1.0, 200.0, ""),
                Param("espessura", "espessura de cada chapa", 4.0, "mm", 0.5, 50.0, ""),
                Param("raio_parafuso", "raio do parafuso", 2.0, "mm", 0.2, 50.0, "só dimensiona a viga do parafuso"),
                Param("raio_pegada", "raio da pegada da cabeça", 3.0, "mm", 0.5, 100.0,
                      "o raio sob o qual a cabeça (e a porca) se apoiam: a carga entra por essa área, e o caso exige "
                      "ao menos quatro nós dentro dela"),
                Param("deslizamento", "deslizamento prescrito", 0.05, "mm", 0.0, 5.0, ""),
                _MESH) + _MATERIAL,
        build=_parafuso_build, checks=_parafuso_checks),

    "ajuste_forcado": Case(
        id="ajuste_forcado",
        title="Ajuste forçado de um anel no eixo",
        summary="Um anel menor que o eixo é montado com interferência. A pressão de contato não vem de uma carga "
                "aplicada: vem da geometria, e é a de Lamé.",
        physics="Contato com interferência inicial em axissimetria: o anel precisa abrir e o eixo fechar para caber, e "
                "a pressão de interface resulta desse equilíbrio. O caso compara a pressão medida com Lamé e exige que "
                "a interface feche em todo o perímetro.",
        assumptions=("o eixo e o anel podem ter materiais diferentes: a fórmula de Lamé com dois materiais é a "
                     "mesma soma, com o termo de cada lado dividido pelo seu módulo (e o anel de alumínio num eixo "
                     "de aço dá menos pressão que o anel de aço, como esperado)",
                     "deformação plana e um quarto de modelo simétrico: é o problema axissimétrico exato",
                     "elasticidade linear; a montagem não plastifica o anel nem o eixo",
                     "o atrito na interface não muda a pressão de ajuste, só a resistência a girar ou deslizar"),
        references=(("Lamé, cilindros de parede espessa e ajuste forçado", "https://en.wikipedia.org/wiki/Cylinder_stress"),
                    ("contato no CalculiX", "https://www.dhondt.de/ccx_2.23.pdf")),
        params=(_FRICTION,
                Param("raio_interface", "raio da interface", 10.0, "mm", 1.0, 200.0, "onde o anel encontra o eixo"),
                Param("raio_furo", "raio do furo do eixo", 4.0, "mm", 0.1, 190.0, "menor que o raio da interface"),
                Param("raio_externo", "raio externo do anel", 20.0, "mm", 1.0, 400.0, "maior que o raio da interface"),
                Param("interferencia", "interferência radial", 0.02, "mm", 1e-4, 1.0,
                      "quanto o anel é menor que o eixo"),
                Param("espessura", "espessura fora do plano", 1.0, "mm", 0.1, 50.0, ""),
                Param("E_anel", "módulo do anel", 210000.0, "MPa", 1.0, 1e7,
                      "o material do anel; igual ao do eixo (aço, 210 GPa) por padrão"),
                Param("nu_anel", "Poisson do anel", 0.3, "", 0.0, 0.499, ""),
                Param("rigidez_contato", "rigidez do contato (K)", 1e6, "N/mm³", 1e4, 1e9,
                      "precisa ser alta: a pressão de ajuste é o que fecha a interferência, e um K baixo a subestima"),
                Param("malha", "malha", 8, "células", 4, 16,
                      "células por direção; a interface é um polígono, então a malha grossa concentra a pressão nos "
                      "nós")) + _MATERIAL,
        build=_ajuste_build, checks=_ajuste_checks),

    "plastificacao_hertz": Case(
        id="plastificacao_hertz",
        title="Cilindro sobre placa elastoplástica: o início do escoamento",
        summary="O mesmo contato de Hertz, agora com *PLASTIC no material da placa. A carga é calculada de trás para a "
                "frente para o estado em que Hertz prevê o início do escoamento, p0 = k sigma_y, e o caso confere no "
                ".frd que é ali que a tensão equivalente pára em sigma_y.",
        physics="Plastificação no contato: a pressão de pico que inicia o escoamento não é sigma_y, é um múltiplo "
                "dele. O pico da tensão equivalente fica abaixo da superfície, a z = 0,70 b, e vale 0,558 p0 — donde "
                "p0 = 1,79 sigma_y pelo critério de von Mises (Johnson). O caso carrega até essa carga, mede p0 e a "
                "tensão equivalente, e depois vai além dela para mostrar o material segurando sigma_y.",
        assumptions=("elastoplástico perfeito, sem encruamento: *PLASTIC com dois pontos de mesma tensão",
                     "indentador rígido e deformação plana (uma célula de espessura, faces travadas fora do plano), "
                     "que é o contato linear de Hertz",
                     "o fator k = 1,79 vale para o campo elástico de Hertz: ele descreve quando o escoamento começa, "
                     "não o regime plástico depois dele",
                     "a tensão equivalente é lida nos nós do .frd, extrapolada dos pontos de integração: o valor de "
                     "pico é suavizado, e é isso que a banda de 0,95 a 1,05 de sigma_y acomoda"),
        references=(("Johnson, Contact Mechanics — o início do escoamento (p0 = k sigma_y)",
                     "https://www.cambridge.org/core/books/contact-mechanics/"),
                    ("von Mises em contato de Hertz", "https://en.wikipedia.org/wiki/Contact_mechanics"),
                    ("*PLASTIC no CalculiX", "https://www.dhondt.de/ccx_2.23.pdf")),
        params=(Param("raio", "raio do cilindro", 100.0, "mm", 1.0, 5000.0, ""),
                Param("tensao_escoamento", "tensão de escoamento sigma_y", 400.0, "MPa", 1.0, 5000.0,
                      "o patamar do *PLASTIC; a carga do primeiro passo sai dele e de mais nada"),
                Param("sobrecarga", "sobrecarga do segundo passo", 1.44, "", 1.0, 4.0,
                      "quantas vezes a carga do início do escoamento o segundo passo aplica; 1,44 põe a previsão "
                      "elástica em 1,2 x 1,79 = 2,15 sigma_y"),
                Param("largura", "largura do modelo", 12.0, "mm", 1.0, 200.0, "metade dela de cada lado do centro"),
                Param("profundidade", "profundidade do bloco", 6.0, "mm", 0.5, 200.0, "≥ 3x a meia-largura esperada"),
                Param("altura_indentador", "altura do indentador", 2.0, "mm", 0.2, 50.0, ""),
                Param("rigidez_contato", "rigidez do contato (K)", 1e6, "N/mm³", 1e3, 1e9,
                      "penalidade do fechamento de pressão; alta demais não converge, baixa demais penetra"),
                Param("malha", "malha", 40, "células", 8, 120, "células por metade da largura, graduadas ao centro"),
                Param("friccao", "coeficiente de atrito mu", 0.3, "", 0.0, 1.5, "")) + _MATERIAL,
        build=_plastificacao_build, checks=_plastificacao_checks),
}

DEFAULT_CASE = "bloco_plano"


def get(case_id: str | None, cases: Mapping[str, Case] | None = None) -> Case:
    """The case ``case_id`` (the default when None), out of ``cases`` or out of this module's own library.

    ``cases`` is how a caller adds its own problems to the menu without copying the readers: the same lookup, the same
    error message, over a wider dict.
    """
    book = CASES if cases is None else cases
    key = (case_id or DEFAULT_CASE).strip().lower()
    if key not in book:
        raise KeyError(f"no contact case {case_id!r}; the library has {', '.join(sorted(book))}")
    return book[key]


def catalog(cases: Mapping[str, Case] | None = None) -> dict[str, Any]:
    """The menu: every case with its parameters, its closed-form reference and what it assumes."""
    book = CASES if cases is None else cases
    return {"units": D.UNITS, "default_case": DEFAULT_CASE,
            "cases": [c.as_dict() for c in book.values()],
            "reference": "cada caso compara o resultado com uma solução fechada; a comparação é a verificação, "
                         "e uma grandeza que não pôde ser medida fica inconclusiva, nunca aprovada",
            "prepomax": {"site": "https://prepomax.fs.um.si/", "licence": "GPL-3.0",
                         "solver": "CalculiX (o PrePoMax distribui o ccx; o mesmo binário roda estes decks)",
                         "files": ["*.inp (deck)", "*.frd / *.dat / *.sta (resultados)"]}}


def build(case_id: str | None, raw: dict[str, Any] | None,
          cases: Mapping[str, Case] | None = None) -> tuple[Case, dict[str, float], Built]:
    case = get(case_id, cases)
    params = case.resolve(raw)
    return case, params, case.build(params)
