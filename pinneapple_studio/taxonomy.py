"""The navigations of the catalog: by task, by physics area, by product and by maturity.

Each tag comes from an explicit rule (words in the module path, the name, the docstring or the Portuguese prose),
so a tool can be in several tasks and areas, and the rule that put it there is shown (:func:`tags` returns the
matched words). Labels are in Portuguese and English.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class Tag:
    id: str
    pt: str
    en: str
    pattern: re.Pattern


def _t(id_: str, pt: str, en: str, words: str) -> Tag:
    return Tag(id_, pt, en, re.compile(rf"(?<![a-z0-9])(?:{words})", re.I))


TASKS = (
    _t("simulate", "Simular", "Simulate", r"simulat|simul|solver|solve\b|solve_|fdm|fem\b|finite.?(element|volume|"
       r"difference)|lbm|lattice|openfoam|calculix|time.?step|integrat|propagat|multibody|resolv"),
    _t("train", "Treinar", "Train", r"train|trainer|training|treino|treinar|fit\b|fit_|optimizer|epoch|loss\b|"
       r"loss_|curriculum|fine.?tun|hpo|hyperparam"),
    _t("infer", "Inferir / prever", "Infer / predict", r"predict|inference|infer\b|previs|forecast|zoo|serving|"
       r"deploy|onnx|export_model"),
    _t("verify", "Verificar / validar", "Verify / validate", r"validat|verif|trust|check|audit|guardrail|residual|"
       r"conservation|manufactured|benchmark|lint|seal|evidence|confian|valida"),
    _t("design", "Projetar / otimizar", "Design / optimise", r"design|optimi[sz]|otimiz|projet|nsga|pareto|"
       r"topology|shape|inverse.?design|sizing|dimension"),
    _t("data", "Analisar dados", "Analyse data", r"dataset|data\b|data_|dados|table|csv|parquet|hdf5|zarr|io\b|"
       r"loader|preprocess|feature|statistic|signal|spectr|piv|image|video|series"),
    _t("uq", "Incerteza (UQ)", "Uncertainty (UQ)", r"uncertain|incertez|monte.?carlo|bayes|ensemble|tolerance|"
       r"toler|confidence|interval|sobol|sensitiv|sensibil|margin|stochastic|probabil|drift|calibrat"),
    _t("inverse", "Problema inverso", "Inverse problem", r"inverse|inverso|identif|discover|parameter.?estim|"
       r"recover|assimilat|kalman|enkf|sindy|pde.?find|estimat"),
    _t("rom", "ROM / surrogate", "ROM / surrogate", r"surrogate|rom\b|rom_|reduced.?order|pod\b|pod_|dmd|"
       r"condens|operator|fno|deeponet|emulat|metamodel|kriging|gaussian.?process|gp\b"),
    _t("twin", "Gêmeo digital", "Digital twin", r"twin|gemeo|gêmeo|monitor|sensor|anomal|streaming|online|"
       r"real.?time|telemetr"),
    _t("export", "Exportar / relatar", "Export / report", r"export|report|relat|usd|omniverse|vtk|stl\b|step\b|"
       r"pdf|html|figure|plot|render|dashboard|viz|visual|blender|paraview"),
)

AREAS = (
    _t("cfd", "CFD / fluidos", "CFD / fluids", r"cfd|fluid|navier|stokes|flow_|flows?\b|turbul|rans|les\b|"
       r"openfoam|lbm|lattice.?boltz|aero|airfoil|drag|lift|pipe|nozzle|water.?hammer|hydraul|escoamento|fluido|"
       r"burgers|vortex|wake|reynolds|free.?surface|sph\b"),
    _t("thermal", "Térmica", "Thermal", r"heat|thermal|termic|térmic|temperat|conduct|convect|radiat|fin\b|fins\b|"
       r"heatsink|heat.?sink|cooling|resfri|calor|enthalp|nusselt"),
    _t("structural", "Estrutural / sólidos", "Structural / solids", r"struct|solid|elastic|stress|strain|beam|"
       r"plate|shell|fatigue|fadiga|calculix|frame|truss|buckl|fracture|contact|cantilever|estrutur|tensao|tensão|"
       r"brick|bridge|deform|vibrat|modal"),
    _t("em", "Eletromagnetismo", "Electromagnetics", r"electro|eletro|maxwell|magnet|antenna|antena|fdtd|"
       r"dielectric|permittiv|motor|machine|inductan|mie\b|wave.?guide|reflector|rf\b|microwave|coil|hall"),
    _t("bio", "Bio / saúde", "Bio / health", r"bio|protein|pdb|molec|drug|pharma|farmac|transdermal|patch|dose|"
       r"cell\b|tissue|blood|cardio|medical|medic|enzyme|genom"),
    _t("chem", "Química / reações", "Chemistry / reactions", r"chem|quim|químic|reaction|reacti|kinetic|"
       r"combust|species|catalys|catalis|arrhenius|adsor|electrochem|battery|bateria"),
    _t("dynamics", "Sistemas dinâmicos / controle", "Dynamical systems / control", r"dynamic|dinamic|dinâmic|"
       r"ode\b|ode_|control|pid\b|mpc|kalman|pendul|oscillat|chaos|lorenz|vehicle|veicul|rover|flight|aircraft|"
       r"robot|trajector|orbit|rigid.?body|multibody|suspension"),
    _t("geo", "Geo / clima / oceano", "Geo / climate / ocean", r"geo|climat|clima|ocean|enso|nino|niño|weather|"
       r"atmos|river|hydrolog|seism|soil|solo\b|terra|groundwater|usgs|sentinel|ndvi"),
    _t("astro", "Astro / espaço", "Astro / space", r"astro|orbit|satell|conjunct|rocket|space|espac|solar|"
       r"planet|lunar|black.?hole|j2\b"),
    _t("quantum", "Quântica", "Quantum", r"quantum|quant|qubit|schrod|hamiltonian"),
    _t("acoustics", "Acústica", "Acoustics", r"acoust|acust|sound|helmholtz|reflex|loudspeaker|audio"),
    _t("geometry", "Geometria / CAD / malha", "Geometry / CAD / mesh", r"geometr|cad\b|cad_|mesh|malha|stl|step\b|"
       r"freecad|cadquery|csg|sdf\b|brep|point.?cloud|voxel"),
    _t("ml", "Aprendizado de máquina", "Machine learning", r"pinn|neural|network|layer|torch|model|surrogate|"
       r"operator|fno|deeponet|gnn|graph|transformer|siren|elm\b|pielm|tfc|xtfc|kan\b|learn"),
    _t("sports", "Esporte", "Sports", r"football|futebol|soccer|ball|pitch.?control|sport"),
)

PRODUCTS = (
    _t("cfd", "CFD", "CFD", r"cfd|openfoam|freecad_openfoam|external_solvers|cfd_formats|mesh|navier|rans"),
    _t("surrogate_studio", "Surrogate Studio", "Surrogate Studio", r"surrogate|zoo|rom\b|rom_|reduced|fno|"
       r"deeponet|operator|emulat|condens"),
    _t("veriphysics", "VeriPhysics", "VeriPhysics", r"veriphysics|trust|validat|verif|check|guardrail|audit|"
       r"evidence|seal|residual|conservation"),
    _t("twin", "Twin", "Twin", r"twin|monitor|sensor|anomal|drift|digital_twin|telemetr|recession"),
    _t("engineering_agent", "Engineering Agent", "Engineering Agent", r"preset|design|reason|autolab|agent|"
       r"recommend|planner|mission|intent|problemdesign|science"),
)

MATURITY = {"verified": ("Verificado", "Verified"), "tested": ("Testado", "Tested"),
            "discovered": ("Descoberto", "Discovered")}


def fold(text: str) -> str:
    """Lower case without accents (``térmica`` -> ``termica``)."""
    return "".join(c for c in unicodedata.normalize("NFKD", text.lower()) if not unicodedata.combining(c))


def tags(text: str, table: tuple[Tag, ...]) -> dict[str, list[str]]:
    """``{tag id: matched words}`` of ``text`` for a table (:data:`TASKS`, :data:`AREAS`, :data:`PRODUCTS`)."""
    t = fold(text.replace("_", " ").replace(".", " "))
    out = {}
    for tag in table:
        hits = sorted({m.group(0).lower() for m in tag.pattern.finditer(t)})
        if hits:
            out[tag.id] = hits[:5]
    return out


def labels(table: tuple[Tag, ...]) -> list[dict[str, str]]:
    return [{"id": t.id, "pt": t.pt, "en": t.en} for t in table]
