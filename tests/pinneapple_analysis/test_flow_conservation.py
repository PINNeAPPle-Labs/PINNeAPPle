"""Conservação e condição de contorno medidas, com solução exata como referência (#159).

O aceite do #159 é que massa, energia e violação de contorno sejam **calculadas**, não fornecidas pelo chamador, "para
ao menos dois problemas de benchmark". Os dois aqui têm solução exata fechada, então cada número medido é comparado
com um número escrito à mão:

* **Poiseuille em canal plano** — ``u(y) = 4 U y (H - y) / H²``, ``v = 0``, paredes em não-deslizamento. A divergência
  é exatamente zero e o que entra é igual ao que sai;
* **condução 1-D em parede com geração uniforme** — ``T(x) = T_w + q''' (L² - x²) / (2k)`` (Incropera & DeWitt,
  Fundamentos de Transferência de Calor e Massa, cap. 3, parede plana com geração). O fluxo condutivo líquido que
  atravessa o domínio é exatamente ``q''' V``.

Em ambos, o campo exato tem de dar resíduo no erro de arredondamento, e uma perturbação conhecida tem de dar o valor
analítico — não apenas "falhar".
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_analysis.trust.flow_card import FlowSpec, energy_balance, flow_checks, mass_balance
from pinneapple_analysis.trust.trust_report import TrustReport

# ---------------------------------------------------------------------------------------------- benchmark 1: massa
U, H, L = 1.0, 1.0, 2.0
Y_AXIS = np.linspace(0.0, H, 41)
X_AXIS = np.linspace(0.0, L, 81)
XG, YG = np.meshgrid(X_AXIS, Y_AXIS, indexing="ij")
POISEUILLE = 4.0 * U * YG * (H - YG) / H ** 2                     # u(y), nada em x
ZERO = np.zeros_like(POISEUILLE)
WALLS = FlowSpec.mass(components={"u": "x", "v": "y"},
                      boundary={"y_min": {"u": 0.0, "v": 0.0}, "y_max": {"u": 0.0, "v": 0.0}})


def _by_name(checks):
    return {c.name: c for c in checks}


def test_poiseuille_conserves_mass_to_round_off():
    """O campo exato: o balanço por volume e o global são zero, e as paredes batem com a especificação."""
    card = _by_name(flow_checks({"x": X_AXIS, "y": Y_AXIS}, {"u": POISEUILLE, "v": ZERO}, spec=WALLS))
    assert card["conservation.mass"].passed and card["conservation.mass"].value < 1e-12
    assert card["conservation.mass.local"].passed and card["conservation.mass.local"].value < 1e-12
    assert card["boundary y_min (u)"].value == 0 and card["boundary y_max (u)"].value == 0
    assert card["conservation.mass"].detail.startswith("mass: o que sai pelas faces 0 contra a fonte integrada 0")


def test_a_field_that_creates_mass_fails_the_global_balance_by_the_analytic_amount():
    """``u = 4 (1 + x/L) y (H - y)``: a vazão cresce ao longo de x, então o domínio cria massa.

    O analítico é exato e não depende do código: entra ``4/6`` e sai ``8/6`` por unidade de largura, ou seja um
    desequilíbrio de ``2/3`` contra um fluxo de contorno de ``2`` — um terço do que atravessa o contorno.
    """
    growing = 4.0 * (1.0 + XG / L) * YG * (H - YG) / H ** 2
    card = _by_name(flow_checks({"x": X_AXIS, "y": Y_AXIS}, {"u": growing, "v": ZERO}, spec=WALLS))
    check = card["conservation.mass"]
    assert check.passed is False and check.critical
    assert check.value == pytest.approx((8 / 6 - 4 / 6) / (8 / 6 + 4 / 6), rel=1e-3)
    assert "desequilíbrio 0.666" in check.detail


def test_a_total_that_closes_does_not_hide_the_cell_that_does_not():
    """``u + 0.02 sin(pi x)``: entra e sai a mesma coisa (o total fecha em exatamente zero), e a divergência local é
    a da perturbação. O resíduo por volume vale ``A sin(pi h) cos(pi x) h_y`` no discreto e ``A pi h_x h_y`` no
    contínuo: o medido tem de ser esse número, não "algo diferente de zero"."""
    perturbed = POISEUILLE + 0.02 * np.sin(np.pi * XG)
    card = _by_name(flow_checks({"x": X_AXIS, "y": Y_AXIS}, {"u": perturbed, "v": ZERO}, spec=WALLS))
    assert card["conservation.mass"].value == pytest.approx(0.0, abs=1e-12)      # o total fecha
    hx, hy = X_AXIS[1] - X_AXIS[0], Y_AXIS[1] - Y_AXIS[0]
    analytic = 0.02 * np.pi * hx * hy                                            # divergência × volume do volume
    assert card["conservation.mass.local"].passed is True
    worst = float(card["conservation.mass.local"].detail.split("|balanço|")[1].split("contra")[0])
    assert worst == pytest.approx(analytic, rel=2e-3)
    assert 0 < worst < 1e-4                                                      # medido, e não zero


def test_flow_that_blows_through_a_wall_shows_up_as_mass_the_domain_created():
    """``v = 0,5 y``: o campo empurra massa pela parede, que a especificação diz impermeável. O balanço global é do
    próprio campo previsto, então o desequilíbrio é exatamente ``v(H) x largura`` = 1,0 — e o contorno acusa a
    parede em 0,5."""
    blowing = 0.5 * YG / H
    card = _by_name(flow_checks({"x": X_AXIS, "y": Y_AXIS}, {"u": POISEUILLE, "v": blowing}, spec=WALLS))
    assert card["conservation.mass"].passed is False
    assert "desequilíbrio 1" in card["conservation.mass"].detail
    assert card["boundary y_max (v)"].passed is False


def test_a_wall_that_does_not_hold_the_flow_still_is_reported_with_its_measured_slip():
    slip = POISEUILLE.copy()
    slip[:, 0] = 0.1                                                 # o primeiro nó em y é a parede y_min
    card = _by_name(flow_checks({"x": X_AXIS, "y": Y_AXIS}, {"u": slip, "v": ZERO}, spec=WALLS))
    wall = card["boundary y_min (u)"]
    assert wall.passed is False and wall.critical
    assert wall.value == pytest.approx(0.1 / np.ptp(slip), rel=1e-12)            # medido, não informado
    # um deslizamento tangencial não move massa por aquela face — quem o pega é a checagem de contorno, e é por isso
    # que as duas existem
    assert card["conservation.mass.local"].value < 1e-12


# ---------------------------------------------------------------------------------------------- benchmark 2: energia
Q, K, WALL_L = 2.0, 1.0, 1.0
TX = np.linspace(0.0, WALL_L, 21)
TY = np.linspace(0.0, 1.0, 11)
TXG, _ = np.meshgrid(TX, TY, indexing="ij")
SLAB = Q * (WALL_L ** 2 - TXG ** 2) / (2 * K)                       # T(x) = q''' (L² - x²) / 2k
SLAB_SPEC = FlowSpec.energy("T", "x", source=Q, thermal_conductivity=K,
                            boundary={"x_min": {"T": Q * WALL_L ** 2 / (2 * K)}, "x_max": {"T": 0.0}})


def test_a_wall_with_uniform_generation_balances_the_heat_that_leaves_it():
    card = _by_name(flow_checks({"x": TX, "y": TY}, {"T": SLAB}, spec=SLAB_SPEC))
    assert card["conservation.energy"].passed and abs(card["conservation.energy"].value) < 1e-12
    assert card["conservation.energy.local"].value < 1e-12
    assert card["boundary x_min (T)"].value == 0 and card["boundary x_max (T)"].value == 0
    # o que atravessa o contorno é exatamente q''' V = 2 x 1 x 1
    assert "o que sai pelas faces 2 contra a fonte integrada 2" in card["conservation.energy"].detail


def test_a_temperature_field_that_does_not_balance_its_source_is_off_by_the_analytic_amount():
    """``T + 0.05 (x - L/2)²``: a curvatura extra é 0,1, então o fluxo líquido fica 0,1 abaixo da fonte.

    Nada aqui vem do código: o termo ``+0.05 (x - 0.5)²`` tem segunda derivada exatamente ``0.1``, e a integral disso
    no domínio é ``0.1 x L x altura`` = 0,1. As faces do contorno também deixam de valer o prescrito, em
    ``0.05 x 0.25 = 0.0125``.
    """
    perturbed = SLAB + 0.05 * (TXG - WALL_L / 2) ** 2
    card = _by_name(flow_checks({"x": TX, "y": TY}, {"T": perturbed}, spec=SLAB_SPEC))
    imbalance = -0.1 * K * WALL_L * 1.0                                       # desequilíbrio analítico, em watts
    assert "desequilíbrio -0.1" in card["conservation.energy"].detail
    assert card["conservation.energy"].value == pytest.approx(abs(imbalance) / 2.0, rel=1e-12)
    assert card["boundary x_min (T)"].value == pytest.approx(0.0125 / 1.0, rel=1e-12)   # 0,05 x 0,25 na parede
    tighter = FlowSpec.energy("T", "x", source=Q, thermal_conductivity=K, boundary=SLAB_SPEC.boundary,
                              tolerances={"boundary": 0.01, "conservation": 0.01})
    tight = _by_name(flow_checks({"x": TX, "y": TY}, {"T": perturbed}, spec=tighter))
    assert tight["boundary x_min (T)"].passed is False and tight["conservation.energy"].passed is False


def test_a_linear_profile_does_not_balance_a_constant_source():
    """Sem curvatura não há fluxo líquido: para ``q''' = 2`` o balanço tem de acusar exatamente a fonte."""
    linear = 1.0 - TXG
    card = _by_name(flow_checks({"x": TX, "y": TY}, {"T": linear}, spec=SLAB_SPEC))
    assert card["conservation.energy"].passed is False
    assert "desequilíbrio -2" in card["conservation.energy"].detail


# ---------------------------------------------------------------------------------------------- comportamento
def test_the_shortcuts_agree_with_the_spec_they_build():
    by_spec = _by_name(flow_checks({"x": TX, "y": TY}, {"T": SLAB}, spec=SLAB_SPEC))
    by_shortcut = _by_name(energy_balance({"x": TX, "y": TY}, SLAB, name="T", axis="x", source=Q,
                                          thermal_conductivity=K, boundary=SLAB_SPEC.boundary))
    assert by_shortcut["conservation.energy"].value == by_spec["conservation.energy"].value
    mass = _by_name(mass_balance({"x": X_AXIS, "y": Y_AXIS}, {"u": POISEUILLE, "v": ZERO},
                                 boundary=WALLS.boundary))
    assert mass["conservation.mass"].passed is True


def test_a_prediction_without_a_component_the_spec_names_is_refused_not_guessed():
    with pytest.raises(ValueError, match="que a previsão não traz"):
        flow_checks({"x": X_AXIS, "y": Y_AXIS}, {"u": POISEUILLE}, spec=WALLS)
    with pytest.raises(ValueError, match="não está na grade"):
        flow_checks({"x": X_AXIS, "y": Y_AXIS}, {"u": POISEUILLE, "v": ZERO},
                    spec=FlowSpec.mass({"u": "z", "v": "y"}))


def test_a_non_finite_field_is_one_failing_sanity_check_and_nothing_else():
    bad = POISEUILLE.copy()
    bad[3, 3] = np.nan
    checks = flow_checks({"x": X_AXIS, "y": Y_AXIS}, {"u": bad, "v": ZERO}, spec=WALLS)
    assert len(checks) == 1 and checks[0].passed is False and checks[0].critical
    assert "NaN" in checks[0].detail


def test_a_grid_too_coarse_for_a_one_sided_derivative_is_refused():
    with pytest.raises(ValueError, match="at least 3 points"):
        flow_checks({"x": [0.0, 1.0]}, {"T": np.array([1.0, 0.0])}, spec=FlowSpec.energy())
    with pytest.raises(ValueError, match="strictly increasing"):
        flow_checks({"x": [0.0, 1.0, 0.5]}, {"T": np.zeros(3)}, spec=FlowSpec.energy())


def test_the_spec_refuses_a_quantity_it_does_not_know():
    with pytest.raises(ValueError, match="mass.*energy"):
        FlowSpec(quantity="momentum", components={"u": "x"})
    with pytest.raises(ValueError, match="components"):
        FlowSpec(quantity="mass", components={})
    with pytest.raises(ValueError, match="xmin"):
        FlowSpec.mass({"u": "x"}, boundary={"xmin": {"u": 0.0}})


def test_a_failing_conservation_check_rejects_the_deploy_report():
    """As checagens entram no relatório de confiança com o mesmo peso das outras: uma crítica que falha rejeita."""
    from pinneapple_analysis.trust.trust_report import CONTRADICTS, SUPPORTS, Check

    checks = flow_checks({"x": X_AXIS, "y": Y_AXIS}, {"u": POISEUILLE, "v": ZERO}, spec=WALLS)
    status = {True: SUPPORTS, False: CONTRADICTS, None: "not_run"}
    report = TrustReport([Check(c.name, status[c.passed], critical=c.critical, value=c.value,
                               limit=c.threshold) for c in checks], model="poiseuille")
    assert report.decisao == "APPROVED"
    slip = POISEUILLE.copy()
    slip[0, :] = 0.2
    checks = flow_checks({"x": X_AXIS, "y": Y_AXIS}, {"u": slip, "v": ZERO}, spec=WALLS)
    report = TrustReport([Check(c.name, status[c.passed], critical=c.critical, value=c.value,
                               limit=c.threshold) for c in checks], model="poiseuille")
    assert report.decisao == "REJECT"
