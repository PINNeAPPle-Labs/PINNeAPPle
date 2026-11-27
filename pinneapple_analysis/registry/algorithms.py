"""The registry of the algorithms this deployment actually runs, with their references and their verification.

An engineering result is only as defensible as the algorithm behind it, and an algorithm is only as defensible as what
it is: who published it, what it requires of the problem, where it runs, and whether *this* repository ever checked it
against a known answer. That is what this module holds — for the solvers, the surrogate methods, the meshing and the
verification itself, one record each, written next to the code that calls them.

Two rules keep it honest:

* **an entry exists only for something that runs here.** The registry is not a catalogue of everything PINNeAPPle
  could wrap; it is the list of what a result may have come out of.
* **``verification`` is a statement about this repository, not about the literature.** ``verified`` means there is a
  test that compares the implementation with the cited answer and a band it must stay inside; ``not_evaluated`` means
  nobody has done that yet, and it says so instead of implying otherwise.

The records are data (:data:`REGISTRY`), so the Studio can publish them and a report can cite them without importing
anything heavy.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

VERIFIED = "verified"          # um teste deste repositório compara com a resposta citada, dentro de uma banda
NOT_EVALUATED = "not_evaluated"  # ninguém fez essa comparação ainda


@dataclass(frozen=True)
class Reference:
    """A published source: what it is, who wrote it, and where to find it."""
    kind: str                       # "paper" | "book" | "standard" | "documentation"
    citation: str
    doi: str = ""
    url: str = ""


@dataclass(frozen=True)
class Algorithm:
    """One algorithm, as the registry states it."""
    id: str
    name: str
    category: str
    implementation: str
    summary: str
    requirements: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, str] = field(default_factory=dict)
    references: tuple[Reference, ...] = ()
    verification: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "references": [asdict(r) for r in self.references]}


def _ref(kind: str, citation: str, doi: str = "", url: str = "") -> Reference:
    return Reference(kind=kind, citation=citation, doi=doi, url=url)


#: os algoritmos que este deployment executa, cada um com a sua fonte e o seu estado de verificação
REGISTRY: tuple[Algorithm, ...] = (
    Algorithm(
        id="hagen_poiseuille",
        name="Hagen–Poiseuille (laminar pipe)",
        category="analytical",
        implementation="pinneapple_intent.families.laminar_pipe (analytic rung)",
        summary="Queda de pressão de escoamento laminar plenamente desenvolvido em tubo circular: Δp = 128 μ L Q / (π D⁴).",
        requirements={"regime": "laminar (Re < 2300)", "geometry": "tubo circular, comprimento desenvolvido"},
        hardware={"cpu": "sim", "gpu": "não"},
        references=(
            _ref("paper", "Hagen, G. H. L. (1839). Über die Bewegung des Wassers in engen cylindrischen Röhren. "
                          "Annalen der Physik 46, 423–442.", "10.1002/andp.18391220304"),
            _ref("paper", "Poiseuille, J. L. M. (1840). Recherches expérimentales sur le mouvement des liquides dans "
                          "les tubes de très petits diamètres. Comptes Rendus 11, 961–967."),
        ),
        verification={"status": VERIFIED,
                      "how": "o degrau analítico da família laminar_pipe é comparado com a forma fechada e com o degrau "
                             "numérico na mesma rodada; a decisão do registro de confiança só aprova com os dois",
                      "where": "tests/test_intent_study.py"},
    ),
    Algorithm(
        id="series_resistance_wall",
        name="Resistência térmica em série (parede com convecção)",
        category="analytical",
        implementation="pinneapple_intent.families.heat_wall (analytic rung)",
        summary="Condução unidimensional estacionária numa parede com película convectiva nas duas faces: "
                "q = ΔT / (1/h_i + L/k + 1/h_o).",
        requirements={"regime": "estacionário, 1-D, propriedades constantes", "geometry": "parede plana"},
        hardware={"cpu": "sim", "gpu": "não"},
        references=(
            _ref("book", "Incropera, F. P. et al. Fundamentals of Heat and Mass Transfer, 7th ed. §3.1–3.2."),
        ),
        verification={"status": VERIFIED,
                      "how": "a face fria calculada é comparada com a forma fechada e com o solver numérico "
                             "bidimensional na mesma rodada",
                      "where": "tests/test_intent_study.py"},
    ),
    Algorithm(
        id="blasius_plate",
        name="Camada limite de Blasius (placa plana)",
        category="analytical",
        implementation="app.aero.plate2d (referência da rodada de CFD externo)",
        summary="Solução de similaridade da camada limite laminar sobre placa plana; o coeficiente médio de atrito é "
                "Cf = 1,328/√Re_L.",
        requirements={"regime": "laminar, gradiente de pressão nulo", "geometry": "placa plana lisa"},
        hardware={"cpu": "sim", "gpu": "não"},
        references=(
            _ref("paper", "Blasius, H. (1908). Grenzschichten in Flüssigkeiten mit kleiner Reibung. "
                          "Zeitschrift für Mathematik und Physik 56, 1–37."),
            _ref("book", "Schlichting, H. & Gersten, K. Boundary-Layer Theory, 8th ed., §7.2."),
        ),
        verification={"status": VERIFIED,
                      "how": "o cd medido pelo OpenFOAM é comparado com 1,328/√Re_L dentro da banda do próprio caso "
                             "(10 % em Re 4e4); o desvio medido é 5,94 %",
                      "where": "engineering-api:tests/test_aero_plate2d.py, engineering-api:tests/test_aero_figura.py"},
    ),
    Algorithm(
        id="vlm",
        name="Método dos vórtices (vortex lattice / linha sustentadora)",
        category="numerical",
        implementation="pinneapple_design.aero.vlm.solve",
        summary="Distribuição de sustentação de uma asa finita por vórtices em faixas; a referência do caso da "
                "aeronave, faixa a faixa.",
        requirements={"regime": "incompressível, invíscido, pequenos ângulos", "geometry": "asa finita descrita por "
                                                                                       "estações"},
        hardware={"cpu": "sim", "gpu": "não"},
        references=(
            _ref("book", "Katz, J. & Plotkin, A. Low-Speed Aerodynamics, 2nd ed., ch. 12–13 (vortex lattice)."),
            _ref("paper", "Prandtl, L. (1918). Tragflügeltheorie. Nachrichten der Gesellschaft der Wissenschaften "
                          "zu Göttingen."),
        ),
        verification={"status": VERIFIED,
                      "how": "a distribuição do vórtice é o alvo do substituto do caso da aeronave; o erro fora do "
                             "ajuste medido é 3,25e-07 em 40 faixas",
                      "where": "engineering-api:tests/test_aero_aeronave.py"},
    ),
    Algorithm(
        id="ridge",
        name="Regressão ridge (polinomial)",
        category="surrogate",
        implementation="pinneapple_analysis.surrogate (method 'ridge')",
        summary="Ajuste linear com penalidade L2; no estudo do substituto é um dos métodos concorrentes, escolhido "
                "pelo erro fora do ajuste.",
        requirements={"dados": "amostras com entrada e saída numéricas", "n": "≥ 2 pontos por coeficiente"},
        hardware={"cpu": "sim", "gpu": "não"},
        references=(
            _ref("paper", "Hoerl, A. E. & Kennard, R. W. (1970). Ridge regression: biased estimation for "
                          "nonorthogonal problems. Technometrics 12(1), 55–67.", "10.1080/00401706.1970.10488634"),
        ),
        verification={"status": VERIFIED,
                      "how": "o método só é publicado se o erro dele for medido nas amostras deixadas fora do ajuste "
                             "(partição 'edge'), e é esse número que a tela mostra",
                      "where": "tests/test_surrogate_library.py"},
    ),
    Algorithm(
        id="gaussian_process",
        name="Processo gaussiano (regressão)",
        category="surrogate",
        implementation="pinneapple_analysis.surrogate (method 'gp')",
        summary="Regressão bayesiana não paramétrica; concorre com a ridge no mesmo estudo e é escolhido pelo mesmo "
                "critério — o erro fora do ajuste.",
        requirements={"dados": "amostras com entrada e saída numéricas", "custo": "O(n³) no número de amostras"},
        hardware={"cpu": "sim", "gpu": "não"},
        references=(
            _ref("book", "Rasmussen, C. E. & Williams, C. K. I. (2006). Gaussian Processes for Machine Learning. "
                         "MIT Press."),
        ),
        verification={"status": VERIFIED,
                      "how": "mesmo critério da ridge: erro medido nas amostras fora do ajuste, na mesma partição",
                      "where": "tests/test_surrogate_library.py"},
    ),
    Algorithm(
        id="simple_foam",
        name="SIMPLE (simpleFoam)",
        category="solver",
        implementation="OpenFOAM v2606 — simpleFoam (app.aero.plate2d escreve o deck)",
        summary="Acoplamento pressão–velocidade para escoamento incompressível estacionário; resolve o caso da placa "
                "plana.",
        requirements={"regime": "incompressível, estacionário", "malha": "volumes finitos"},
        hardware={"cpu": "sim (MPI)", "gpu": "não"},
        references=(
            _ref("paper", "Patankar, S. V. & Spalding, D. B. (1972). A calculation procedure for heat, mass and "
                          "momentum transfer in three-dimensional parabolic flows. Int. J. Heat Mass Transfer 15(10), "
                          "1787–1806.", "10.1016/0017-9310(72)90054-3"),
            _ref("documentation", "OpenFOAM v2606 — User Guide, §3 (simpleFoam) e §4.5 (fvSolution).",
                 url="https://doc.openfoam.com/"),
        ),
        verification={"status": VERIFIED,
                      "how": "a rodada só é aprovada com os resíduos abaixo do limite declarado e com o número "
                             "comparado à referência de Blasius; as duas coisas ficam no registro da rodada",
                      "where": "engineering-api:tests/test_aero_plate2d.py"},
    ),
    Algorithm(
        id="block_mesh",
        name="blockMesh",
        category="meshing",
        implementation="OpenFOAM v2606 — blockMesh (app.aero.plate2d.block_mesh_dict)",
        summary="Malha hexaédrica por blocos, ortogonal por construção no caso da placa: o número depende da física, "
                "não de um compromisso de qualidade de malha.",
        requirements={"geometria": "domínio em blocos", "consistência": "mesmo grading nas faces compartilhadas"},
        hardware={"cpu": "sim", "gpu": "não"},
        references=(
            _ref("documentation", "OpenFOAM v2606 — blockMesh, no User Guide.",
                 url="https://doc.openfoam.com/"),
        ),
        verification={"status": VERIFIED,
                      "how": "o caso confere a malha antes de resolver (contagens e grading derivados de um lugar só) "
                             "e a rodada falha com a mensagem do blockMesh se ele não construir",
                      "where": "engineering-api:tests/test_aero_plate2d.py"},
    ),
    Algorithm(
        id="gmsh",
        name="Gmsh (geração de malha)",
        category="meshing",
        implementation="pinneapple_design.geometry.mesh.generate",
        summary="Malhador por trás da geração de malha da biblioteca (2-D e 3-D, com camada limite opcional).",
        requirements={"geometria": "primitivas ou CAD importado", "dependência": "gmsh instalado"},
        hardware={"cpu": "sim", "gpu": "não"},
        references=(
            _ref("paper", "Geuzaine, C. & Remacle, J.-F. (2009). Gmsh: a three-dimensional finite element mesh "
                          "generator with built-in pre- and post-processing facilities. Int. J. Numer. Meth. Engng "
                          "79(11), 1309–1331.", "10.1002/nme.2579"),
        ),
        verification={"status": NOT_EVALUATED,
                      "how": "a malha gerada é usada, mas não há um teste que compare a qualidade dela com uma "
                             "referência publicada",
                      "where": ""},
    ),
    Algorithm(
        id="meshless_cloud",
        name="Método sem malha (nuvem de pontos)",
        category="meshing",
        implementation="pinneapple_condense.meshless.point_cloud",
        summary="Discretização por nuvem de pontos, para quando a malha convencional é o gargalo.",
        requirements={"geometria": "pacote de camadas", "parâmetro": "espaçamento h"},
        hardware={"cpu": "sim", "gpu": "opcional"},
        references=(
            _ref("paper", "Belytschko, T. et al. (1996). Meshless methods: an overview and recent developments. "
                          "Comput. Methods Appl. Mech. Engrg 139, 3–47.", "10.1016/S0045-7825(96)01078-X"),
        ),
        verification={"status": NOT_EVALUATED,
                      "how": "sem comparação registrada com caso de referência publicado",
                      "where": ""},
    ),
    Algorithm(
        id="richardson",
        name="Extrapolação de Richardson / independência de malha",
        category="verification",
        implementation="pinneapple_analysis.verification.convergence",
        summary="Ordem observada e extrapolação a partir de três malhas; é a base do estudo de independência de "
                "discretização.",
        requirements={"malhas": "três níveis, refino constante", "solução": "nas três"},
        hardware={"cpu": "sim", "gpu": "não"},
        references=(
            _ref("paper", "Richardson, L. F. (1911). The approximate arithmetical solution by finite differences. "
                          "Phil. Trans. R. Soc. A 210, 307–357.", "10.1098/rsta.1911.0009"),
            _ref("standard", "ASME V&V 20-2009 — Standard for Verification and Validation in Computational Fluid "
                             "Dynamics and Heat Transfer, §3 (grid convergence index)."),
            _ref("paper", "Roache, P. J. (1994). Perspective: a method for uniform reporting of grid refinement "
                          "studies. J. Fluids Eng. 116(3), 405–413.", "10.1115/1.2910291"),
        ),
        verification={"status": VERIFIED,
                      "how": "a ordem observada e a extrapolação são conferidas em testes com séries de erro "
                             "conhecidas",
                      "where": "tests/test_grid_convergence.py"},
    ),
    Algorithm(
        id="solver_selection",
        name="Seleção de família de solver",
        category="verification",
        implementation="pinneapple_analysis.verification.solver_orchestration.select_solver_family",
        summary="Recomenda a família de solver pela matemática do problema e pelo que está instalado e medido nesta "
                "máquina — não por preferência.",
        requirements={"entrada": "tipo de equação, simetria, tamanho, hardware"},
        hardware={"cpu": "sim", "gpu": "depende do backend"},
        references=(
            _ref("book", "Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd ed. SIAM, ch. 6–9."),
        ),
        verification={"status": VERIFIED,
                      "how": "a escolha é confrontada com o que a máquina tem instalado e com o que foi medido nela, em vez "
                             "de preferência",
                      "where": "tests/pinneapple_analysis/test_verification_solver_orchestration.py"},
    ),
)


def listar(categoria: str = "", busca: str = "") -> list[dict[str, Any]]:
    """The registry as data, optionally narrowed by category and by a word in the name or the summary."""
    termo = (busca or "").strip().lower()
    out = []
    for a in REGISTRY:
        if categoria and a.category != categoria:
            continue
        if termo and termo not in (a.name + " " + a.summary + " " + a.implementation).lower():
            continue
        out.append(a.to_dict())
    return out


def obter(algoritmo_id: str) -> dict[str, Any] | None:
    """One entry by id, or ``None`` — the caller decides whether a missing id is a 404."""
    for a in REGISTRY:
        if a.id == algoritmo_id:
            return a.to_dict()
    return None


def categorias() -> list[str]:
    """The categories present, in the registry's own order."""
    vistos: list[str] = []
    for a in REGISTRY:
        if a.category not in vistos:
            vistos.append(a.category)
    return vistos


def lacunas() -> Iterable[dict[str, Any]]:
    """The entries whose verification is not done here — read as the registry's own to-do list, not as a claim."""
    return (a.to_dict() for a in REGISTRY if a.verification.get("status") != VERIFIED)
