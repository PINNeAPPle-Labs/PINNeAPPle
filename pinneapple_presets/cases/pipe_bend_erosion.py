"""Erosão por areia num joelho de tubulação de 90°: impactos lagrangianos, desgaste pelo DNV-RP-O501 e cena 3-D.

O caso é o joelho de 90° de uma linha de 3" por onde passa gás a 15 m/s arrastando areia: as partículas entram com o
fluido, não conseguem acompanhar a curva (St ~ 40) e impactam o extradorso. Três passos, os mesmos do gêmeo digital do
PINNeAPPle-CFD (``_runs/erosion-twin/build_twin.py``, o caso já rodado e medido):

1. rastreamento lagrangiano de cada partícula num escoamento pistonado dentro do joelho (arrasto de Stokes, sem
   ricochete): mede a velocidade ``|v|`` e o ângulo ``α`` com a parede de cada impacto;
2. cada impacto entra no modelo de desgaste da norma — ``ER = K V^n F(α)`` (DNV-RP-O501 Rev. 4.2, eq. 7.1-7.3) com
   ``K``, ``n`` e ``ρ_t`` da Tabela 7-2 — e a taxa por célula da parede é a massa erodida dividida pela área e por
   ``ρ_t``, integrada no período;
3. a segunda rota do mesmo modelo, a triagem fechada da §8.4 para curvas (α = arctan(1/√(2R/D)), o fator G de
   partícula fina e o ``C₁`` empírico), é calculada junto: as duas precisam concordar dentro do fator declarado.

O escoamento contínuo é pistonado, não um RANS: o que o preset mede é a física de impactação inercial (onde e com que
velocidade e ângulo a areia bate) e a taxa da norma, não um campo de velocidade com camada limite.
"""
from __future__ import annotations

import math

import numpy as np

from ..core import Output, Param, Preset, Reference, check

DNV = 'DNV-RP-O501 Rev. 4.2 (2007), "Erosive wear in piping systems"'
ANNEX = ("PINNeAPPle-CFD, docs/anexos/erosao-tubulacoes.md §2.2 (F(α) e Tabela 7-2) e §5 (§8.4, curva): transcrição "
         "da norma; as constantes devem ser reconferidas na edição 2015/2025")
DNV_A = (9.370, 42.295, 110.864, 175.804, 170.137, 98.398, 31.211, 4.170)      # eq. 7.3, A1..A8
DNV_K, DNV_N, DNV_RHO_WALL = 2.0e-9, 2.6, 7800.0                                # Tabela 7-2, aço carbono
SECONDS_PER_YEAR = 3.156e7            # a constante do gêmeo (365 dias); a norma usa 3,15e7 dentro de C_unit
C_UNIT = 3.15e10                      # DNV §8.4: conversão para mm/ano dentro da fórmula de triagem
C1_BEND = 2.5                         # DNV §8.4 rev. 4.2 (era 5 nas revisões anteriores)
TOL_SCREENING = 0.35                  # o quanto a triagem fechada e o rastreamento podem divergir (ver references())
ST_MIN = 5.0                          # abaixo deste Stokes a areia acompanha parte da curva: as checagens de regime
                                      # inercial não são executadas (o relatório diz por quê)

# malha da parede e alisamento do mapa (os do gêmeo: 180 x 72 células e média móvel 5 x 5)
N_ARC, N_THETA, SMOOTH = 180, 72, 5
# integração do rastreamento (a do gêmeo): passo de tempo, teto de passos e semente
DT, MAX_STEPS, SEED = 2e-5, 6000, 11


def f_dnv(alpha_rad):
    """F(α) do DNV-RP-O501 eq. 7.3: ``Σ (-1)^(i+1) A_i α^i``, com α em rad (a norma escreve α em graus × π/180)."""
    a = np.asarray(alpha_rad, float)
    return sum(((-1) ** (i + 1)) * DNV_A[i - 1] * a ** i for i in range(1, 9))


def stokes_time(rho_particle: float, d_particle: float, mu_gas: float) -> float:
    """Tempo de resposta da partícula ao arrasto de Stokes, ``τ_p = ρ_p d_p² / (18 μ)`` [s]."""
    return rho_particle * d_particle ** 2 / (18.0 * mu_gas)


def dnv_bend_screening(*, d_inner: float, r_over_d: float, u_gas: float, sand_rate: float, d_particle: float,
                       rho_particle: float, rho_gas: float, mu_gas: float, k_dnv: float = DNV_K,
                       n_dnv: float = DNV_N, rho_wall: float = DNV_RHO_WALL) -> dict:
    """Triagem fechada do DNV-RP-O501 §8.4 para uma curva de 90° [mm/ano], com os passos intermediários.

    ``α = arctan(1/√(2R/D))`` é o ângulo efetivo da norma, ``A = Re_D tan α / β`` (β = ρ_p/ρ_m), ``γ/γ_c`` o fator G de
    partícula fina (G = 1 quando as partículas são grandes o bastante para bater na parede) e
    ``Ė = ṁ_p K F(α) sin α U^n / (ρ_t A_pipe) · G · C₁ · C_unit``.
    """
    alpha = math.atan(1.0 / math.sqrt(2.0 * r_over_d))
    re_d = rho_gas * u_gas * d_inner / mu_gas
    beta = rho_particle / rho_gas
    a_stokes = re_d * math.tan(alpha) / beta
    gamma = d_particle / d_inner
    den = 1.88 * math.log(a_stokes) - 6.04 if a_stokes > 0 else 0.0
    raw = 1.0 / (beta * den) if den > 0 else 0.0
    gamma_c = raw if 0.0 < raw < 0.1 else 0.1
    g = gamma / gamma_c if gamma < gamma_c else 1.0
    a_pipe = math.pi * d_inner ** 2 / 4.0
    rate = (sand_rate * k_dnv * float(f_dnv(alpha)) * math.sin(alpha) * u_gas ** n_dnv
            / (rho_wall * a_pipe) * g * C1_BEND * C_UNIT)
    return {"alpha_deg": math.degrees(alpha), "re_d": re_d, "beta": beta, "a_stokes": a_stokes, "gamma": gamma,
            "gamma_c": gamma_c, "g": g, "area_pipe": a_pipe, "rate_mm_year": rate}


def bend_mesh(d_inner: float, r_over_d: float, n_arc: int = N_ARC, n_theta: int = N_THETA):
    """Superfície do joelho de 90° (vértice ``i*n_theta + j``): (V, F) com o eixo do tubo sobre um quarto de círculo.

    ``R/D`` é o raio do eixo da curva sobre o diâmetro do tubo (``R = 1,5 D`` no joelho de raio longo da ASME B16.9).
    ``φ`` corre de 0 (entrada) a 90° (saída) e ``θ`` é a posição circunferencial: ``θ = 0`` é o extradorso, onde a
    areia bate.
    """
    r_pipe, r_bend = d_inner / 2.0, r_over_d * d_inner
    phi = np.linspace(0.0, np.pi / 2, n_arc)
    th = np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)
    P, T = np.meshgrid(phi, th, indexing="ij")
    rho = r_bend + r_pipe * np.cos(T)
    V = np.stack([rho * np.cos(P), rho * np.sin(P), r_pipe * np.sin(T)], -1).reshape(-1, 3)
    i, j = np.meshgrid(np.arange(n_arc - 1), np.arange(n_theta), indexing="ij")
    a = (i * n_theta + j).ravel()
    b = (i * n_theta + (j + 1) % n_theta).ravel()
    F = np.vstack([np.stack([a, b, a + n_theta], 1), np.stack([b, b + n_theta, a + n_theta], 1)])
    return V, F


def track_particles(*, d_inner: float, r_over_d: float, u_gas: float, rho_particle: float, d_particle: float,
                    mu_gas: float, n_particles: int, seed: int = SEED, dt: float = DT,
                    max_steps: int = MAX_STEPS) -> dict:
    """Rastreamento lagrangiano das partículas no joelho: arrasto de Stokes sobre o escoamento pistonado, sem
    ricochete (a partícula é absorvida no primeiro contato com a parede).

    Devolve, por partícula que tocou a parede dentro do joelho, a posição (``phi``, ``theta``), a velocidade e o
    ângulo de impacto; ``n_exit`` conta as que atravessaram o joelho sem tocar e ``n_stuck`` as que ficaram dentro do
    tubo no teto de passos (a conservação que a checagem confere).
    """
    r_pipe, r_bend = d_inner / 2.0, r_over_d * d_inner
    rng = np.random.default_rng(seed)
    r = r_pipe * np.sqrt(rng.random(n_particles))               # entrada uniforme na seção do tubo
    theta = rng.random(n_particles) * 2 * np.pi
    x = r_bend + r * np.cos(theta)
    y = np.zeros(n_particles)
    z = r * np.sin(theta)
    vx, vy, vz = np.zeros(n_particles), np.full(n_particles, u_gas), np.zeros(n_particles)
    phi_hit = np.full(n_particles, np.nan)
    theta_hit = np.full(n_particles, np.nan)
    speed_hit = np.full(n_particles, np.nan)
    alpha_hit = np.full(n_particles, np.nan)
    alive = np.ones(n_particles, bool)
    tau_p = stokes_time(rho_particle, d_particle, mu_gas)
    for _ in range(max_steps):
        if not alive.any():
            break
        k = np.flatnonzero(alive)
        xx, yy, zz = x[k], y[k], z[k]
        phi = np.arctan2(yy, xx)
        cp, sp = np.cos(phi), np.sin(phi)
        ux, uy = -u_gas * sp, u_gas * cp                      # tangente ao joelho (escoamento pistonado)
        vx[k] += (ux - vx[k]) / tau_p * dt
        vy[k] += (uy - vy[k]) / tau_p * dt
        xx, yy, zz = xx + vx[k] * dt, yy + vy[k] * dt, zz + vz[k] * dt
        x[k], y[k], z[k] = xx, yy, zz
        rho = np.sqrt(xx * xx + yy * yy)
        dr = rho - r_bend
        r[k] = np.sqrt(dr * dr + zz * zz)
        theta[k] = np.arctan2(zz, dr)
        gone = np.zeros(n_particles, bool)
        gone[k] = np.arctan2(yy, xx) > np.pi / 2             # saiu da região do joelho
        hit = alive & ((r >= r_pipe) | gone)
        if hit.any():
            h = np.flatnonzero(hit)
            ph_h = np.arctan2(y[h], x[h])
            th_h = theta[h]
            cph, sph, cth, sth = np.cos(ph_h), np.sin(ph_h), np.cos(th_h), np.sin(th_h)
            nx, ny, nz = cth * cph, cth * sph, sth           # normal da parede (θ = 0 no extradorso)
            vxh, vyh, vzh = vx[h], vy[h], vz[h]
            sp_h = np.sqrt(vxh * vxh + vyh * vyh + vzh * vzh)
            sin_a = np.abs(vxh * nx + vyh * ny + vzh * nz) / np.maximum(sp_h, 1e-9)
            phi_hit[h], theta_hit[h], speed_hit[h] = ph_h, th_h, sp_h
            alpha_hit[h] = np.degrees(np.arcsin(np.clip(sin_a, 0.0, 1.0)))
            alive[h] = False
    inside = ~np.isnan(phi_hit) & (phi_hit <= np.pi / 2)
    return {"phi": phi_hit, "theta": theta_hit, "speed": speed_hit, "alpha_deg": alpha_hit, "inside": inside,
            "n_impact": int(inside.sum()), "n_exit": int((~np.isnan(phi_hit) & ~inside).sum()),
            "n_stuck": int(np.isnan(phi_hit).sum()), "stokes": tau_p * u_gas / d_inner, "tau_p": tau_p}


def erosion_map(track: dict, *, d_inner: float, r_over_d: float, sand_rate: float, n_particles: int,
                rho_wall: float = DNV_RHO_WALL, k_dnv: float = DNV_K, n_dnv: float = DNV_N,
                n_arc: int = N_ARC, n_theta: int = N_THETA, smooth: int = SMOOTH) -> dict:
    """Taxa de erosão [mm/ano] por vértice da parede, a partir dos impactos do rastreamento.

    Cada impacto carrega a massa de areia que ele representa (``ṁ_sand / n_particles``) para a célula em que caiu; a
    célula recebe massa por unidade de área, o modelo da norma converte em massa de parede removida por massa de areia
    e ``ρ_t`` dá a profundidade. O mapa que se lê na parede é a média móvel 5 x 5 (o valor de uma sonda sobre a face,
    não o de um vértice isolado).
    """
    m = track["inside"]
    i = np.clip(np.rint(track["phi"][m] / (np.pi / 2) * (n_arc - 1)).astype(int), 0, n_arc - 1)
    j = np.rint(track["theta"][m] / (2 * np.pi) * n_theta).astype(int) % n_theta
    idx = i * n_theta + j
    n_hits = np.bincount(idx, minlength=n_arc * n_theta).astype(float)
    sum_v = np.bincount(idx, weights=track["speed"][m], minlength=n_arc * n_theta)
    sum_a = np.bincount(idx, weights=track["alpha_deg"][m], minlength=n_arc * n_theta)
    area = (r_over_d * d_inner) * (np.pi / 2) * (np.pi * d_inner) / (n_arc * n_theta)    # área de parede/célula
    v_cell = np.divide(sum_v, np.maximum(n_hits, 1))
    a_cell = np.divide(sum_a, np.maximum(n_hits, 1))
    flux = n_hits * (sand_rate / n_particles) / area                      # kg de areia / (m2 s)
    raw = k_dnv * v_cell ** n_dnv * f_dnv(np.radians(a_cell)) * flux / rho_wall * SECONDS_PER_YEAR * 1000.0
    rate = _smooth(raw, n_arc, n_theta, smooth)
    peak = int(np.argmax(rate))
    row = peak // n_theta
    hits = n_hits.reshape(n_arc, n_theta)
    window = np.take(np.take(hits, np.arange(row - smooth // 2, row + smooth // 2 + 1), axis=0, mode="clip"),
                     np.arange(peak % n_theta - smooth // 2, peak % n_theta + smooth // 2 + 1), axis=1, mode="wrap")
    return {"rate": rate, "raw": raw, "n_hits": n_hits, "v_cell": v_cell, "alpha_cell": a_cell, "flux_cell": flux,
            "area_cell": area, "peak": peak, "rate_peak": float(rate[peak]), "rate_raw_peak": float(raw[peak]),
            "peak_arc_deg": float(row) / (n_arc - 1) * 90.0,
            "peak_theta_deg": float(peak % n_theta) / n_theta * 360.0,
            "peak_window_impacts": int(window.sum())}


def _eroded_mass(track: dict, n_particles: int, sand_rate: float, k_dnv: float, n_dnv: float) -> float:
    """Massa de parede removida por segundo somando os impactos: ``(ṁ_areia / N) Σ K V^n F(α)`` [kg/s]."""
    m = track["inside"]
    if not m.any():
        return 0.0
    return float(np.sum(k_dnv * track["speed"][m] ** n_dnv * f_dnv(np.radians(track["alpha_deg"][m])))) \
        * sand_rate / n_particles


def _smooth(field, n_arc: int, n_theta: int, k: int):
    """Média móvel k x k na grade (arco x circunferência), com a circunferência periódica."""
    a = np.asarray(field, float).reshape(n_arc, n_theta)
    pad_x = np.pad(a, ((k // 2, k // 2), (0, 0)), mode="edge")
    c = np.apply_along_axis(lambda v: np.convolve(v, np.ones(k) / k, mode="valid"), 0, pad_x)
    pad_y = np.pad(c, ((0, 0), (k // 2, k // 2)), mode="wrap")
    return np.apply_along_axis(lambda v: np.convolve(v, np.ones(k) / k, mode="valid"), 1, pad_y).ravel()


class PipeBendErosion(Preset):
    name = "pipe_bend_erosion"
    title = "Joelho de tubulação com areia: taxa de erosão, perda de parede e vida pelo DNV-RP-O501"
    sector = "Fluidos / integridade de tubulação (óleo e gás)"
    problem = ("Joelho de 90° de uma linha de 3\" de aço carbono por onde passa gás com areia (poço de gás). As "
               "partículas são rastreadas no escoamento do joelho, cada impacto entra no modelo de desgaste do "
               "DNV-RP-O501 (ER = K V^n F(α), Tabela 7-2) e a taxa por célula da parede dá o mapa de erosão, o pico, "
               "a perda de parede no período e a vida até a sobrespessura admissível. A triagem fechada da §8.4 é "
               "calculada junto como segunda rota do mesmo modelo.")
    question = ("Qual a taxa de erosão de pico neste joelho, quanto da parede ele perde no período avaliado e quanto "
                "tempo resta até a sobrespessura admissível? Onde (arco e circunferência) fica o pico, e a triagem "
                "fechada da norma confirma a mesma ordem de grandeza?")
    params = (
        Param("d_inner", "m", 0.080, 'premissa: linha de 3" com diâmetro interno nominal de 80 mm (o NPS 3" Sch 40 da '
              "ASME B36.10M tem ID 77,9 mm e parede de 5,49 mm)", 0.025, 0.5),
        Param("r_over_d", "-", 1.5, "premissa: joelho de raio longo 1,5D, o padrão dos joelhos de 90° da ASME B16.9",
              1.0, 6.0),
        Param("u_gas", "m/s", 15.0, "premissa: velocidade média do gás na linha de produção do caso", 2.0, 40.0),
        Param("sand_rate", "kg/s", 20.0 / 86400.0, "premissa: 20 kg/dia de areia produzida (2,315e-4 kg/s); o piso da "
              "faixa é uma vazão desprezível, não o caso sem areia (sem areia não há erosão a medir)", 1e-9, 0.01),
        Param("d_particle", "m", 150e-6, "premissa: areia de 150 µm (a DNV cita 250-500 µm em poço sem gravel pack; a "
              "partícula fina é o caso que mais espalha o impacto pelo arco)", 20e-6, 1e-3),
        Param("rho_particle", "kg/m3", 2650.0, "quartzo (SiO2), a areia de produção", 1500.0, 8000.0),
        Param("rho_gas", "kg/m3", 50.0, "premissa: gás natural a ~65 bar e 40 °C: ρ = p M/(Z R T) = 65e5 × 0,0180 / "
              "(0,90 × 8,314 × 313) = 50 kg/m3 (entra na triagem pelo Re_D do fator G)", 0.5, 400.0),
        Param("mu_gas", "Pa s", 1.5e-5, "premissa: gás natural a 40 °C e 65 bar, μ ≈ 1,5e-5 Pa s", 8e-6, 1e-3),
        Param("k_dnv", "(m/s)^-n", DNV_K, DNV + " Tabela 7-2 (aço carbono): K = 2,0e-9 (m/s)^-n", 1e-10, 1e-7),
        Param("n_dnv", "-", DNV_N, DNV + " Tabela 7-2 (aço carbono): n = 2,6", 2.0, 4.0),
        Param("rho_wall", "kg/m3", DNV_RHO_WALL, DNV + " Tabela 7-2: ρ_t = 7800 kg/m3 (aço)", 1000.0, 9000.0),
        Param("t_wall", "m", 5.49e-3, 'ASME B36.10M, NPS 3" Sch 40: parede de 5,49 mm', 2e-3, 0.03),
        Param("t_allow", "m", 2e-3, "premissa: sobrespessura de erosão admissível no período (2 mm; parcela da "
              "sobre-espessura de 3 mm usual em projeto de aço carbono)", 0.0, 0.02),
        Param("years", "a", 5.0, "premissa: período avaliado / intervalo de inspeção de 5 anos", 0.5, 30.0),
        Param("n_particles", "-", 50000, "premissa: 50 mil partículas; o gêmeo rodado usou 200 mil e o pico varia "
              "menos de 2 % entre os dois. O pico é o máximo do mapa, então amostra pequena o sobrestima (5 mil dão "
              "~18 % acima de 50 mil): o piso de 5 mil é o mínimo em que a janela de 25 células ainda tem impactos",
              5000, 500000),
    )
    outputs = (
        Output("rate_peak_mm_year", "mm/a", "taxa de erosão de pico na parede do joelho"),
        Output("wall_loss_mm", "mm", "perda de parede no pico ao fim do período avaliado"),
        Output("alpha_median_deg", "grau", "ângulo de impacto mediano, entre a trajetória e a parede"),
        Output("impact_fraction", "-", "fração das partículas que impactam a parede dentro do joelho"),
        Output("peak_arc_deg", "grau", "posição do pico ao longo do arco (0° = entrada, 90° = saída)"),
        Output("peak_theta_deg", "grau", "posição circunferencial do pico (0° = extradorso)"),
        Output("rate_dnv_mm_year", "mm/a", "triagem fechada do DNV-RP-O501 §8.4 na mesma condição"),
        Output("life_years", "a", "vida no pico até consumir a sobrespessura admissível"),
        Output("remaining_wall_mm", "mm", "parede remanescente no pico ao fim do período"),
        Output("stokes", "-", "número de Stokes das partículas no joelho (τ_p U / D)"),
        Output("n_impacts", "-", "impactos medidos na parede (de n_particles lançadas)"),
    )
    validity = ("Escoamento pistonado monofásico (não é um RANS): a física medida é a impactação inercial da areia e a "
                "taxa do modelo da norma. Areia diluída e sem ricochete (uma partícula, um impacto, sem interação "
                "entre partículas), St >> 1, joelho de 90° com R/D >= 1, gás a Mach < 0,3. O pico é uma média móvel "
                "de 5 x 5 células na malha de 180 x 72 da parede (0,5° no arco e 5° na circunferência). A triagem "
                "§8.4 usa uma única velocidade, a média da linha. Abaixo de St ~ 5 (areia fina, gás muito viscoso ou "
                "tubo grande) a areia acompanha parte da curva: as checagens de regime inercial não são executadas e "
                "o relatório diz por quê.")
    pipeline = ("geometria: joelho de 90°, D x R/D (d_inner, r_over_d), malha da parede de 180 x 72 células",
                "modelo: arrasto de Stokes (τ_p = ρ_p d_p²/18μ) no escoamento pistonado; desgaste ER = K V^n F(α) do "
                "DNV-RP-O501 com K, n e ρ_t da Tabela 7-2",
                "solver: rastreamento lagrangiano até o impacto -> taxa por célula -> média móvel -> integração no "
                "período; triagem fechada da §8.4 na mesma condição",
                "pós: pico e sua posição, ângulo mediano, fração de impactos, perda de parede, vida e parede "
                "remanescente",
                "decisão: perda de parede no período <= sobrespessura admissível")
    library = ("pinneapple_presets.cases.pipe_bend_erosion (f_dnv, stokes_time, dnv_bend_screening, track_particles, "
               "erosion_map, bend_mesh); a cena usa pinneapple_tools.visualization.studio.scene",)
    has_3d = True
    has_variants = True
    variant_outputs = ("rate_peak_mm_year", "wall_loss_mm", "life_years")
    missing = {
        "surrogate": "o rastreamento de 50 mil partículas custa segundos e a triagem fechada da norma responde na "
                     "hora; uma varredura de engenharia usa a §8.4, não um substituto treinado",
        "inversion": "o caso não tem medição de campo; a inversão útil seria calibrar K e n contra espessuras medidas "
                     "por UT ao longo dos anos, e o dado de inspeção não faz parte do preset",
        "optimization": "não há objetivo de custo declarado; o ganho vem de R/D, D e velocidade, que a triagem "
                        "fechada da §8.4 responde sem rastreamento",
    }
    limitations = (
        "Escoamento pistonado: sem RANS, sem camada limite e sem turbulência, então a velocidade de impacto segue a "
        "velocidade média do gás (um RANS com camada limite muda a distribuição dentro da seção).",
        "Sem ricochete, sem dispersão turbulenta e sem interação entre partículas (areia diluída): a taxa é a de um "
        "único impacto por partícula.",
        "K, n e A1..A8 vêm de transcrição secundária da norma (anexo do PINNeAPPle-CFD): reconferir na edição "
        "2015/2025 antes de usar o número em contrato.",
        "O alisamento 5 x 5 do mapa e a malha de 180 x 72 fixam a resolução da posição do pico: um pico mais estreito "
        "que uma célula é lido como média da vizinhança.",
        "O pico é o máximo do mapa, e um máximo entre células com poucos impactos é sobrestimado pela amostragem: de "
        "50 mil para 200 mil partículas ele cai ~2 %, e 5 mil dão ~18 % acima.",
    )

    # --------------------------------------------------------------------------------------- solver
    def _screening(self, p):
        return dnv_bend_screening(d_inner=p["d_inner"], r_over_d=p["r_over_d"], u_gas=p["u_gas"],
                                  sand_rate=p["sand_rate"], d_particle=p["d_particle"],
                                  rho_particle=p["rho_particle"], rho_gas=p["rho_gas"], mu_gas=p["mu_gas"],
                                  k_dnv=p["k_dnv"], n_dnv=p["n_dnv"], rho_wall=p["rho_wall"])

    def solve(self, p):
        n = int(p["n_particles"])
        tr = track_particles(d_inner=p["d_inner"], r_over_d=p["r_over_d"], u_gas=p["u_gas"],
                             rho_particle=p["rho_particle"], d_particle=p["d_particle"], mu_gas=p["mu_gas"],
                             n_particles=n)
        mp = erosion_map(tr, d_inner=p["d_inner"], r_over_d=p["r_over_d"], sand_rate=p["sand_rate"], n_particles=n,
                         rho_wall=p["rho_wall"], k_dnv=p["k_dnv"], n_dnv=p["n_dnv"])
        scr = self._screening(p)
        inside = tr["inside"]
        alpha_med = float(np.nanmedian(tr["alpha_deg"][inside])) if inside.any() else float("nan")
        peak = mp["rate_peak"]
        wall_loss = peak * p["years"]
        allow = p["t_allow"] * 1000.0
        life = allow / peak if peak > 0 else float("inf")
        # a taxa de remoção de massa em 1,5x a velocidade: a integral sobre os impactos confere o expoente n da norma
        # sem a amostragem ruidosa de um pico de célula, e uma amostra menor basta (ambas as corridas usam a semente)
        n_fast = int(min(n, 5000))
        u_up = 1.5 * p["u_gas"]
        mass = [_eroded_mass(track_particles(d_inner=p["d_inner"], r_over_d=p["r_over_d"], u_gas=u,
                                             rho_particle=p["rho_particle"], d_particle=p["d_particle"],
                                             mu_gas=p["mu_gas"], n_particles=n_fast),
                             n_fast, p["sand_rate"], p["k_dnv"], p["n_dnv"])
                for u in (p["u_gas"], u_up)]
        V, F = bend_mesh(p["d_inner"], p["r_over_d"])
        return {
            "rate_peak_mm_year": peak, "wall_loss_mm": wall_loss, "alpha_median_deg": alpha_med,
            "impact_fraction": tr["n_impact"] / n, "peak_arc_deg": mp["peak_arc_deg"],
            "peak_theta_deg": mp["peak_theta_deg"], "rate_dnv_mm_year": scr["rate_mm_year"],
            "life_years": life, "remaining_wall_mm": p["t_wall"] * 1000.0 - wall_loss,
            "stokes": tr["stokes"], "n_impacts": tr["n_impact"],
            "diagnostics": {
                "tracking": {"n_impact": tr["n_impact"], "n_exit": tr["n_exit"], "n_stuck": tr["n_stuck"],
                             "n_particles": n,
                             "alpha_min": float(np.nanmin(tr["alpha_deg"][inside])) if inside.any() else 0.0,
                             "alpha_max": float(np.nanmax(tr["alpha_deg"][inside])) if inside.any() else 0.0},
                "peak_cell": {"raw": mp["rate_raw_peak"], "v": float(mp["v_cell"][mp["peak"]]),
                              "alpha_deg": float(mp["alpha_cell"][mp["peak"]]),
                              "flux": float(mp["flux_cell"][mp["peak"]]), "area": float(mp["area_cell"]),
                              "window_impacts": mp["peak_window_impacts"]},
                "screening": scr,
                "u_scaling": {"ratio": mass[1] / mass[0] if mass[0] > 0 else float("nan"), "speed": u_up / p["u_gas"],
                              "n_particles": n_fast, "mass_rate_kg_s": mass[0]}},
            "_fields": {"bend_vertices": V, "bend_faces": F, "rate": mp["rate"],
                        "wall_loss": p["years"] * mp["rate"], "years": p["years"]},
        }

    # --------------------------------------------------------------------------------------- checks
    def checks(self, p, out):
        dg = out["diagnostics"]
        track, cell, scr, sc = dg["tracking"], dg["peak_cell"], dg["screening"], dg["u_scaling"]
        # 1. a célula de pico reavaliada na forma escalar da norma: ER = K V^n F(α) por massa de areia, sobre ρ_t
        alpha_rad = math.radians(cell["alpha_deg"])
        f_scalar = sum(((-1) ** (i + 1)) * DNV_A[i - 1] * alpha_rad ** i for i in range(1, 9))
        closed = (p["k_dnv"] * cell["v"] ** p["n_dnv"] * f_scalar * cell["flux"] / p["rho_wall"]
                  * SECONDS_PER_YEAR * 1000.0)
        rel = abs(cell["raw"] - closed) / closed if closed > 0 else abs(cell["raw"] - closed)
        ratio = out["rate_peak_mm_year"] / out["rate_dnv_mm_year"] if out["rate_dnv_mm_year"] > 0 else float("nan")
        theta = (out["peak_theta_deg"] + 180.0) % 360.0 - 180.0
        # as checagens de regime só valem para areia inercial (St >= ST_MIN) e com o fator G da norma em 1: fora daí o
        # modelo pistonado e a triagem fechada descrevem regimes diferentes e a checagem seria uma comparação inválida
        inertial = out["stokes"] >= ST_MIN
        scaling = None if not inertial or not math.isfinite(sc["ratio"]) or sc["ratio"] <= 0 else \
            abs(sc["ratio"] / sc["speed"] ** p["n_dnv"] - 1.0) < 0.05
        return [
            check("penetração na célula de pico = formulação do DNV-RP-O501", rel < 1e-9, value=rel, limit=1e-9,
                  source=DNV + " eq. 7.1-7.3 e Tabela 7-2 (aço: K = 2,0e-9, n = 2,6, ρ_t = 7800 kg/m3); " + ANNEX,
                  reason=f"ER = K V^n F(α) com V = {cell['v']:.2f} m/s, α = {cell['alpha_deg']:.1f}°, "
                         f"F = {f_scalar:.4f}, fluxo de areia {cell['flux']:.4e} kg/(m2 s), "
                         f"ρ_t = {p['rho_wall']:.0f} kg/m3"),
            check("rastreamento x triagem fechada da DNV §8.4 (35 %)",
                  None if scr["g"] < 1.0 or not math.isfinite(ratio) else abs(ratio - 1.0) <= TOL_SCREENING,
                  value=ratio, limit=1.0 + TOL_SCREENING, source=DNV + " §8.4 (α = arctan(1/√(2R/D)), G, C1 = 2,5, "
                  "C_unit = 3,15e10); " + ANNEX,
                  reason=(f"a triagem aplica o fator de partícula fina G = {scr['g']:.2f} "
                          f"(γ/γ_c = {scr['gamma'] / scr['gamma_c']:.2f}): fora do regime inercial o modelo pistonado "
                          f"não tem essa correção e as duas rotas descrevem regimes diferentes" if scr["g"] < 1.0 else
                          f"pico do rastreamento {out['rate_peak_mm_year']:.3f} mm/ano contra "
                          f"{out['rate_dnv_mm_year']:.3f} mm/ano da triagem (α = {scr['alpha_deg']:.1f}°); a triagem "
                          f"troca a distribuição de impactos (mediana {out['alpha_median_deg']:.1f}°) por um ângulo "
                          f"efetivo único")),
            check("conservação do número de partículas",
                  track["n_impact"] + track["n_exit"] + track["n_stuck"] == track["n_particles"],
                  value=track["n_impact"] + track["n_exit"] + track["n_stuck"], limit=track["n_particles"],
                  source="rastreamento lagrangiano: cada partícula termina na parede do joelho, atravessando o joelho "
                         "sem tocar ou (no teto de passos) ainda dentro do tubo; nenhuma é criada ou perdida",
                  reason=f"{track['n_impact']} impactos, {track['n_exit']} atravessaram sem tocar, "
                         f"{track['n_stuck']} ficaram no tubo"),
            check("areia inercial: praticamente toda partícula impacta (St >> 1)",
                  None if not inertial else out["impact_fraction"] >= 0.99,
                  value=out["impact_fraction"], limit=0.99,
                  source="St = τ_p U / D com τ_p = ρ_p d_p²/(18 μ): com St >> 1 a partícula não acompanha a curva e "
                         "não sobra uma fração que fuja do extradorso",
                  reason=(f"St = {out['stokes']:.2f} abaixo de {ST_MIN:g}: a areia acompanha parte da curva "
                          f"({100 * (1 - out['impact_fraction']):.1f}% atravessa sem tocar) e o modelo pistonado não "
                          f"está no regime inercial" if not inertial else
                          f"St = {out['stokes']:.1f}; impactaram {100 * out['impact_fraction']:.1f}% das partículas")),
            check("ângulo de impacto dentro de (0°, 90°)", track["alpha_min"] > 0.0 and track["alpha_max"] < 90.0,
                  value=track["alpha_max"], limit=90.0,
                  source="geometria do joelho: α = arcsin(|v·n|/|v|) na parede do extradorso; 0° seria rasante e 90° "
                         "normal à parede, e a partícula que entra com o fluido bate de esguelha",
                  reason=f"α de {track['alpha_min']:.1f}° a {track['alpha_max']:.1f}°, mediana "
                         f"{out['alpha_median_deg']:.1f}°"),
            check("ângulo mediano de impacto = ângulo efetivo da DNV §8.4 (15 %)",
                  None if not inertial else abs(out["alpha_median_deg"] / scr["alpha_deg"] - 1.0) <= 0.15,
                  value=out["alpha_median_deg"], limit=1.15 * scr["alpha_deg"],
                  source=DNV + " §8.4: α = arctan(1/√(2R/D)) é o ângulo de corte que a norma usa na curva; " + ANNEX,
                  reason=f"mediana medida {out['alpha_median_deg']:.1f}° contra {scr['alpha_deg']:.1f}° de "
                         f"arctan(1/√(2R/D)) com R/D = {p['r_over_d']:.2f}"),
            check("taxa cresce com a velocidade como U^n (expoente da norma)", scaling, value=sc["ratio"],
                  limit=sc["speed"] ** p["n_dnv"],
                  source=DNV + " eq. 7.1 (ER = K V^n F(α)): a massa removida por segundo é K V^n F(α) por massa de "
                  "areia, então subir a velocidade por um fator s multiplica a taxa por s^n",
                  reason=f"a {sc['speed']:.1f}× a velocidade, a massa removida por segundo subiu {sc['ratio']:.4f}× "
                         f"contra {sc['speed'] ** p['n_dnv']:.4f}× de s^n ({sc['n_particles']} partículas); a "
                         f"distribuição de impactos também muda com a velocidade, o que desloca o expoente aparente "
                         f"para {math.log(sc['ratio']) / math.log(sc['speed']):.2f}"),
            check("pico na metade externa do joelho", abs(theta) <= 45.0, value=out["peak_theta_deg"], limit=45.0,
                  source="geometria do joelho: θ = 0 é o extradorso, para onde a areia é jogada ao virar a curva",
                  reason=f"pico a {out['peak_arc_deg']:.0f}° do arco e {out['peak_theta_deg']:.0f}° na "
                         f"circunferência (0° = extradorso)"),
            check("posição do pico dentro do arco do joelho", 0.0 < out["peak_arc_deg"] < 90.0,
                  value=out["peak_arc_deg"], limit=90.0,
                  source="geometria do joelho: o arco corre de 0° (entrada) a 90° (saída)"),
            check("pico do mapa amostrado (média 5 x 5 células)", cell["window_impacts"] >= 50,
                  value=cell["window_impacts"], limit=50,
                  source="estatística do pico: o valor lido é a média de uma janela de células, e uma janela com "
                         "poucos impactos leria ruído de amostragem como erosão",
                  reason=f"{cell['window_impacts']} impactos na janela do pico com {p['n_particles']} partículas"),
        ]

    def decide(self, p, out):
        wall_loss = out["wall_loss_mm"]
        allow = p["t_allow"] * 1000.0
        ok = wall_loss <= allow and out["remaining_wall_mm"] > 0
        reasons = [f"taxa de pico {out['rate_peak_mm_year']:.3f} mm/ano no arco {out['peak_arc_deg']:.0f}° "
                   f"(θ = {out['peak_theta_deg']:.0f}°, 0° = extradorso)",
                   f"perda de parede em {p['years']:g} anos: {wall_loss:.2f} mm contra a sobrespessura admissível de "
                   f"{allow:.2f} mm; parede remanescente {out['remaining_wall_mm']:.2f} mm de "
                   f"{p['t_wall'] * 1000:.2f} mm",
                   f"vida no pico até consumir a sobrespessura admissível: {out['life_years']:.1f} anos"]
        if out["life_years"] < p["years"]:
            reasons.append("a sobrespessura se esgota antes do período avaliado: antecipe a inspeção por ultrassom "
                           f"no extradorso ({out['life_years']:.1f} anos)")
        if out["rate_dnv_mm_year"] > 0 and abs(out["rate_peak_mm_year"] / out["rate_dnv_mm_year"] - 1.0) > \
                TOL_SCREENING:
            reasons.append(f"atenção: o rastreamento ({out['rate_peak_mm_year']:.3f} mm/ano) diverge mais de "
                           f"{100 * TOL_SCREENING:.0f} % da triagem fechada da norma "
                           f"({out['rate_dnv_mm_year']:.3f} mm/ano)")
        if not ok:
            reasons.append("não atende: a perda de parede no período passa da sobrespessura admissível")
        return {"verdict": "OK" if ok else "NOT OK", "reasons": reasons}

    # ----------------------------------------------------------------------------------- validation
    def references(self):
        """A referência é a outra rota do mesmo modelo: a triagem fechada da §8.4 da norma, avaliada num ponto. Não é
        um ensaio — é a formulação do próprio DNV-RP-O501 (K, n e F(α) da mesma norma) na forma fechada, e o
        rastreamento do preset tem de reproduzi-la dentro de ``TOL_SCREENING``."""
        v = self.defaults()
        scr = self._screening(v)
        return [Reference(
            name=f"triagem fechada do DNV-RP-O501 §8.4 (R/D = {v['r_over_d']:g}, D = {1e3 * v['d_inner']:.0f} mm, "
                 "U = 15 m/s, 20 kg/dia de areia de 150 µm)",
            params={k: v[k] for k in ("d_inner", "r_over_d", "u_gas", "sand_rate", "d_particle", "rho_particle",
                                      "rho_gas", "mu_gas", "k_dnv", "n_dnv", "rho_wall")},
            output="rate_peak_mm_year", value=scr["rate_mm_year"], unit="mm/a", kind="standard",
            source=DNV + " §8.4 (α = arctan(1/√(2R/D)), G, C1 = 2,5): é o modelo do código na forma fechada, não um "
                   "ensaio; as duas rotas concordam dentro de 35 % porque a triagem troca a distribuição de ângulos "
                   "de impacto por um ângulo efetivo único", tol=TOL_SCREENING)]

    # ---------------------------------------------------------------------------------------- extras
    def variants(self):
        return {"base (R/D 1,5)": {}, "raio longo R/D 3": {"r_over_d": 3.0}, "curva 5D": {"r_over_d": 5.0},
                "areia em dobro": {"sand_rate": 40.0 / 86400.0}, "gás a 25 m/s": {"u_gas": 25.0}}

    def scene(self, result):
        from ..viz3d import Scene, Surface, tube
        p, F = result.params, result.fields
        r_pipe = p["d_inner"] / 2.0
        parts = [Surface("bend", F["bend_vertices"], F["bend_faces"], "steel",
                         {"wall_loss": F["wall_loss"], "rate": F["rate"]})]
        length = 3.5 * p["d_inner"]
        n = 8
        for name, path in (("inlet", np.c_[np.full(n, p["r_over_d"] * r_pipe), np.linspace(-length, 0.0, n),
                                           np.zeros(n)]),
                           ("outlet", np.c_[np.linspace(-length, 0.0, n), np.full(n, p["r_over_d"] * r_pipe),
                                            np.zeros(n)])):
            V, Fc, _ = tube(path, r_pipe, n_around=48)
            parts.append(Surface(name, V, Fc, "steel", {"wall_loss": np.zeros(len(V)), "rate": np.zeros(len(V))}))
        return Scene(parts, title=self.title,
                     labels={"wall_loss": f"perda de parede em {p['years']:g} anos [mm]",
                             "rate": "taxa de erosão [mm/ano]"})
