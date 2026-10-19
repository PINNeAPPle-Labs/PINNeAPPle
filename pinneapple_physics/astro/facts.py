"""Facts and curiosities about the objects in the sky, for a general audience, in English and Portuguese.

A versioned dataset (``data/facts/``): the Sun, planets, Moon, dwarf planets and major moons (NASA fact sheets), the
brightest stars (Yale BSC + Hipparcos 2007 parallaxes), the 88 constellations (IAU; areas computed from the
boundaries), the 110 Messier objects (positions from d3-celestial; distances from the literature where cited) and
the satellites the observatory tracks. Every number carries a ``source`` (a key of ``data/facts/sources.json``); the
text is written for this project in both languages (none copied).

    from pinneapple_physics.astro import facts
    c = facts.card("saturn", "pt")              # name, summary, numbers with sources and comparisons, curiosities, quiz
    facts.search("cruzeiro")                    # ids, in either language
    facts.did_you_know("en", seed=3)

Comparisons ("9.4 times the Earth's diameter", "light takes 8.3 min") are computed from the card's own numbers and
carry the source ``computed``.
"""
from __future__ import annotations

import json
import math
import os
import random
import unicodedata
from functools import lru_cache

DATA = os.path.join(os.path.dirname(__file__), "data", "facts")
LANGS = ("en", "pt")
C_KM_S = 299792.458
LY_KM = 9.4607304725808e12
AU_KM = 149597870.7
EARTH = {"diameter_km": 12756.0, "mass_kg": 5.97e24, "day_h": 24.0, "orbital_period_d": 365.25}
SATELLITE_NORAD = {"iss": 25544, "hubble": 20580, "tiangong": 48274}


def _load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def sources() -> dict:
    """Every source a number or a curiosity can cite: {key: {title, publisher, url, licence}}."""
    return _load("sources.json")


def _check_lang(lang):
    if lang not in LANGS:
        raise ValueError(f"language {lang!r}: use 'en' or 'pt'")


def _fold(s: str) -> str:
    """Lower case without accents (search in either language)."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(s).lower()) if not unicodedata.combining(c))


# ------------------------------------------------------------------------------------------------- labels
LABELS = {
    "diameter_km": ("Diameter", "Diâmetro"), "mass_kg": ("Mass", "Massa"), "density_kg_m3": ("Density", "Densidade"),
    "gravity_m_s2": ("Surface gravity", "Gravidade na superfície"), "rotation_h": ("Rotation period", "Período de rotação"),
    "day_h": ("Length of day", "Duração do dia"), "distance_sun_km": ("Mean distance from the Sun", "Distância média ao Sol"),
    "distance_earth_km": ("Mean distance from the Earth", "Distância média à Terra"),
    "distance_planet_km": ("Distance from its planet", "Distância ao planeta"),
    "orbital_period_d": ("Length of year (orbit)", "Duração do ano (órbita)"),
    "mean_temp_c": ("Mean temperature", "Temperatura média"), "moons": ("Known moons", "Luas conhecidas"),
    "luminosity_w": ("Luminosity", "Luminosidade"), "teff_k": ("Surface temperature", "Temperatura da superfície"),
    "v_mag": ("Brightness (V magnitude)", "Brilho (magnitude V)"), "spectral_type": ("Spectral type", "Tipo espectral"),
    "distance_ly": ("Distance", "Distância"), "luminosity_lsun": ("Luminosity", "Luminosidade"),
    "radius_rsun": ("Radius", "Raio"), "constellation": ("Constellation", "Constelação"),
    "size_arcmin": ("Apparent size", "Tamanho aparente"), "area_deg2": ("Area in the sky", "Área no céu"),
    "area_rank": ("Size rank (of 88)", "Posição em tamanho (de 88)"),
    "brightest_star": ("Brightest star", "Estrela mais brilhante"),
    "naked_eye_stars": ("Stars brighter than magnitude 6", "Estrelas mais brilhantes que a magnitude 6"),
    "best_month": ("Best month (high at 9 pm)", "Melhor mês (alta às 21 h)"),
    "launch": ("Launched", "Lançamento"), "length_m": ("Length", "Comprimento"),
    "pressurized_volume_m3": ("Pressurized volume", "Volume pressurizado"), "crew": ("Crew", "Tripulação"),
    "mirror_m": ("Main mirror", "Espelho principal"), "modules": ("Modules", "Módulos"),
}
KINDS = {
    "star": ("star", "estrela"), "planet": ("planet", "planeta"), "dwarf_planet": ("dwarf planet", "planeta anão"),
    "moon": ("moon", "lua"), "constellation": ("constellation", "constelação"), "satellite": ("artificial satellite", "satélite artificial"),
    "gc": ("globular cluster", "aglomerado globular"), "oc": ("open cluster", "aglomerado aberto"),
    "s": ("spiral galaxy", "galáxia espiral"), "sd": ("spiral galaxy", "galáxia espiral"), "s0": ("lenticular galaxy", "galáxia lenticular"),
    "e": ("elliptical galaxy", "galáxia elíptica"), "i": ("irregular galaxy", "galáxia irregular"),
    "pn": ("planetary nebula", "nebulosa planetária"), "snr": ("supernova remnant", "remanescente de supernova"),
    "sfr": ("star-forming nebula", "nebulosa de formação de estrelas"), "bn": ("bright nebula", "nebulosa brilhante"),
    "en": ("emission nebula", "nebulosa de emissão"), "rn": ("reflection nebula", "nebulosa de reflexão"),
    "pos": ("star cloud or asterism", "nuvem de estrelas ou asterismo"), "dn": ("dark nebula", "nebulosa escura"),
}
MONTHS = {"en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                 "November", "December"],
          "pt": ["janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro",
                 "novembro", "dezembro"]}


def _label(key, lang):
    return LABELS.get(key, (key, key))[LANGS.index(lang)]


def _kind(kind, lang):
    return KINDS.get(kind, (kind, kind))[LANGS.index(lang)]


def _fmt(x, lang, digits=3):
    """A number for prose: n significant digits, thousands separator, decimal comma in Portuguese."""
    if x == 0:
        return "0"
    d = max(0, digits - 1 - int(math.floor(math.log10(abs(x)))))
    s = f"{round(x, d):,.{d}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".") if lang == "pt" else s


def _num(value, unit, source):
    return {"value": value, "unit": unit, "source": source}


# ------------------------------------------------------------------------------------------------ objects
@lru_cache(maxsize=1)
def _solar():
    return _load("solar_system.json")["objects"]


@lru_cache(maxsize=1)
def _stars():
    return _load("stars.json")["stars"]


@lru_cache(maxsize=1)
def _messier():
    cat = {o["id"]: o for o in _load("messier_catalog.json")["objects"]}
    extra = _load("messier.json")["objects"]
    for k, v in extra.items():
        cat[k].update(v)
    return cat


@lru_cache(maxsize=1)
def _constellations():
    return _load("constellations.json")


@lru_cache(maxsize=1)
def _satellites():
    return _load("satellites.json")["objects"]


def _con_key(abbr, ra_deg=None):
    if abbr == "Ser" and ra_deg is not None:
        return "Ser1" if ra_deg % 360 < 255 else "Ser2"
    return abbr


@lru_cache(maxsize=1)
def ids() -> dict:
    """Every object with a card: {id: kind}. Ids: Solar System bodies by name ('saturn'), stars 'star-<HR>',
    Messier objects 'M31', constellations 'con-<IAU abbreviation>' ('con-Ser1', 'con-Ser2' for Serpens), satellites
    'sat-<name>' ('sat-iss')."""
    out = {k: v["kind"] for k, v in _solar().items()}
    out.update({f"star-{hr}": "star" for hr in _stars()})
    out.update({k: v["type"] for k, v in _messier().items()})
    out.update({f"con-{k}": "constellation" for k in _constellations()["constellations"]})
    out.update({f"sat-{k}": "satellite" for k in _satellites()})
    return out


def resolve(ref) -> str:
    """The card id for an id or an observatory reference: 'saturn', 'body:saturn', 'star:2061', 'HR 2061', 'M 31',
    'sat:25544', 'con:Ori'. Raises KeyError if there is no card."""
    r = str(ref).strip()
    if r in ids():
        return r
    low = r.lower().replace(" ", "")
    if low.startswith("body:"):
        low = low[5:]
    if low in ids():
        return low
    for p in ("star:", "hr"):
        if low.startswith(p) and low[len(p):].isdigit() and f"star-{int(low[len(p):])}" in ids():
            return f"star-{int(low[len(p):])}"
    if low.startswith("m") and low[1:].isdigit() and f"M{int(low[1:])}" in ids():
        return f"M{int(low[1:])}"
    if low.startswith("sat:") and low[4:].isdigit():
        for k, n in SATELLITE_NORAD.items():
            if n == int(low[4:]):
                return f"sat-{k}"
    if low.startswith("con:") and f"con-{r[4:].strip()}" in ids():
        return f"con-{r[4:].strip()}"
    raise KeyError(f"no card for {ref!r}")


# ------------------------------------------------------------------------------------------------ names
@lru_cache(maxsize=1)
def _star_catalog():
    from .stars import load_bright_stars
    return load_bright_stars()


def names(oid) -> dict:
    """{en, pt} display names of a card."""
    k = ids()[oid]
    if oid in _solar():
        return dict(_solar()[oid]["names"])
    if k == "star":
        c = _star_catalog()
        n = c.label(c.index_of(int(oid[5:])))
        return {"en": n, "pt": n}                     # IAU names in both languages
    if k == "constellation":
        from .stars import constellation_names
        return dict(constellation_names()[oid[4:]])
    if k == "satellite":
        return dict(_satellites()[oid[4:]]["names"])
    m = _messier()[oid]
    common = m.get("names")
    if common:
        return {lang: f"{oid} · {common[lang]}" for lang in LANGS}
    return {lang: f"{oid} ({m['desig']})" if m.get("desig") else oid for lang in LANGS}


# ------------------------------------------------------------------------------------------------ numbers
def _solar_numbers(oid):
    return {k: dict(v) for k, v in _solar()[oid]["numbers"].items()}


def _star_numbers(hr: int):
    from .stars import star_properties, teff_from_bv
    c = _star_catalog()
    i = c.index_of(hr)
    plx = _stars()[str(hr)]["parallax_mas"]["value"]
    d_pc = 1000.0 / plx
    out = {"v_mag": _num(float(c.v[i]), "mag", "bsc5"), "spectral_type": _num(str(c.sp[i]), "", "bsc5"),
           "distance_ly": _num(round(d_pc * 3.261563777, 4 if d_pc < 5 else 1), "ly", "hipparcos2007"),
           "constellation": _num(_con_key(c.con[i], c.ra[i]), "", "bsc5")}
    if not math.isnan(c.bv[i]):
        out["teff_k"] = _num(round(float(teff_from_bv(c.bv[i]))), "K", "computed")
        p = star_properties(c, i)
        # luminosity / radius with the Hipparcos distance (the BSC5 parallaxes are pre-Hipparcos)
        Mv = c.v[i] - 5 * math.log10(d_pc / 10.0)
        L = 10 ** (-0.4 * (Mv + p.get("bolometric_correction", _bc(out["teff_k"]["value"])) - 4.74))
        R = math.sqrt(L) * (5772.0 / out["teff_k"]["value"]) ** 2
        out["luminosity_lsun"] = _num(_sig(L, 3), "Lsun", "computed")
        out["radius_rsun"] = _num(_sig(R, 3), "Rsun", "computed")
    return out


def _sig(x, n):
    return float(f"{x:.{n}g}")


def _bc(teff):
    from .stars import bolometric_correction
    return bolometric_correction(teff)


DIST = {"ly": 1.0, "kly": 1e3, "Mly": 1e6}


def _messier_numbers(oid):
    m = _messier()[oid]
    out = {"constellation": _num(_con_key(m["constellation"], m["ra_deg"]), "", "d3celestial")}
    if m.get("v_mag") is not None:
        out["v_mag"] = _num(m["v_mag"], "mag", "d3celestial")
    if m.get("size_arcmin"):
        out["size_arcmin"] = _num(m["size_arcmin"], "arcmin", "d3celestial")
    if m.get("distance"):
        d = m["distance"]
        out["distance_ly"] = _num(d["value"], d["unit"], d["source"])
    return out


@lru_cache(maxsize=1)
def _con_stats():
    """Brightest star and number of naked-eye stars per constellation, from the Yale BSC in the library."""
    c = _star_catalog()
    best, count = {}, {}
    for i in range(len(c)):
        k = _con_key(c.con[i], c.ra[i]) if c.con[i] else None
        if not k:
            continue
        if c.v[i] <= 6.0:
            count[k] = count.get(k, 0) + 1
        if k not in best or c.v[i] < c.v[best[k]]:
            best[k] = i
    return best, count


def _con_numbers(key):
    from .stars import load_constellations
    areas = _load("constellation_areas.json")["areas"]
    whole = {k: v for k, v in areas.items() if not k.startswith("Ser")}
    whole["Ser"] = areas["Ser1"] + areas["Ser2"]                  # Serpens is one constellation for the ranking
    rank = sorted(whole, key=whole.get, reverse=True).index("Ser" if key.startswith("Ser") else key) + 1
    best, count = _con_stats()
    c = _star_catalog()
    out = {"area_deg2": _num(areas[key], "deg2", "constellation_areas"), "area_rank": _num(rank, "", "constellation_areas")}
    if key in best:
        i = best[key]
        out["brightest_star"] = _num(f"{c.label(i)} (V {float(c.v[i]):.2f})", "", "bsc5")
        out["naked_eye_stars"] = _num(count.get(key, 0), "count", "bsc5")
    label = next(x["label"] for x in load_constellations() if x["key"] == key)
    out["best_month"] = _num(_best_month(label[0]), "month", "computed")
    return out


def _best_month(ra_deg: float) -> int:
    """Month (1-12) in which a point at right ascension ``ra_deg`` crosses the meridian at about 9 pm local mean
    time: the Sun's hour angle is then 135 deg, so the meridian is at RA = RA(Sun) + 135 deg; the Sun's RA is 0 at
    the March equinox (day 79) and grows by 0.9856 deg a day."""
    from datetime import date, timedelta
    day = 79 + ((ra_deg - 135.0) % 360.0) / 0.98565
    return (date(2001, 1, 1) + timedelta(days=day % 365)).month


def _sat_numbers(key):
    return {k: dict(v) for k, v in _satellites()[key]["numbers"].items()}


def numbers(oid) -> dict:
    """The key numbers of a card: {key: {value, unit, source}}."""
    k = ids()[oid]
    if oid in _solar():
        return _solar_numbers(oid)
    if k == "star":
        return _star_numbers(int(oid[5:]))
    if k == "constellation":
        return _con_numbers(oid[4:])
    if k == "satellite":
        return _sat_numbers(oid[4:])
    return _messier_numbers(oid)


# ------------------------------------------------------------------------------------------------ comparisons
def _duration(hours, lang):
    h = abs(hours)
    if h < 48:
        return f"{_fmt(h, lang)} h"
    if h < 2 * 365.25 * 24:
        return f"{_fmt(h / 24, lang)} " + ("days" if lang == "en" else "dias")
    return f"{_fmt(h / 24 / 365.25, lang)} " + ("years" if lang == "en" else "anos")


def _light_time(km, lang):
    s = km / C_KM_S
    if s < 120:
        return f"{_fmt(s, lang)} s"
    if s < 2 * 3600:
        return f"{_fmt(s / 60, lang)} min"
    if s < 2 * 86400:
        return f"{_fmt(s / 3600, lang)} h"
    return _duration(s / 3600, lang)


def comparison(key, number, lang="en") -> str | None:
    """An everyday comparison for a number, computed from it (source: computed)."""
    _check_lang(lang)
    v, en = number["value"], lang == "en"
    if not isinstance(v, (int, float)):
        return None
    if key in ("diameter_km", "mass_kg") and v < (1e3 if key == "diameter_km" else 1e18):
        return None                                                   # spacecraft: no planetary comparison
    if key == "diameter_km":
        r = v / EARTH["diameter_km"]
        if 0.98 < r < 1.02:
            return None
        return (f"{_fmt(r, lang)} × the Earth's diameter" if en else f"{_fmt(r, lang)} × o diâmetro da Terra")
    if key == "mass_kg":
        r = v / EARTH["mass_kg"]
        if 0.98 < r < 1.02:
            return None
        return (f"{_fmt(r, lang)} × the Earth's mass" if en else f"{_fmt(r, lang)} × a massa da Terra")
    if key in ("distance_sun_km", "distance_earth_km"):
        return (f"light takes {_light_time(v, lang)}; {_fmt(v / AU_KM, lang)} au" if en
                else f"a luz leva {_light_time(v, lang)}; {_fmt(v / AU_KM, lang)} UA")
    if key == "distance_ly":
        ly = v * DIST.get(number["unit"], 1.0)
        return (f"the light we see left it {_fmt(ly, lang)} years ago" if en
                else f"a luz que vemos saiu de lá há {_fmt(ly, lang)} anos")
    if key == "day_h" or key == "rotation_h":
        return (f"{_duration(v, lang)} = {_fmt(abs(v) / 24, lang)} Earth days" if en
                else f"{_duration(v, lang)} = {_fmt(abs(v) / 24, lang)} dias terrestres") if abs(v) > 48 else None
    if key == "orbital_period_d":
        y = abs(v) / 365.25
        return (f"{_fmt(y, lang)} Earth years" if en else f"{_fmt(y, lang)} anos terrestres") if y > 1.5 else None
    if key == "gravity_m_s2":
        r = v / 9.80665
        return (f"a scale would read {_fmt(60 * r, lang)} kg for a 60 kg person" if en
                else f"uma balança marcaria {_fmt(60 * r, lang)} kg para uma pessoa de 60 kg")
    if key == "luminosity_lsun":
        return (f"{_fmt(v, lang)} times the Sun" if en else f"{_fmt(v, lang)} vezes o Sol")
    if key == "radius_rsun":
        return (f"{_fmt(v * 109.1, lang)} × the Earth's radius" if en else f"{_fmt(v * 109.1, lang)} × o raio da Terra")
    if key == "area_deg2":
        return (f"about {_fmt(v / 0.2, lang)} full Moons" if en else f"cerca de {_fmt(v / 0.2, lang)} Luas cheias")
    if key == "mean_temp_c":
        return None
    return None


# ------------------------------------------------------------------------------------------------ text
def _hand_text(oid, lang):
    if oid in _solar():
        return _solar()[oid]["text"][lang]
    if oid.startswith("star-"):
        return _stars()[oid[5:]]["text"][lang]
    if oid.startswith("sat-"):
        return _satellites()[oid[4:]]["text"][lang]
    if oid.startswith("con-"):
        return {}
    return _messier()[oid].get("text", {}).get(lang, {})


def _summary(oid, lang, nums):
    t = _hand_text(oid, lang)
    if t.get("summary"):
        return t["summary"]
    from .stars import constellation_names
    cn = constellation_names(lang)
    en = lang == "en"
    if oid.startswith("con-"):
        key = oid[4:]
        return (f"{cn[key]} is one of the 88 constellations recognised by the International Astronomical Union."
                if en else f"{cn[key]} é uma das 88 constelações reconhecidas pela União Astronômica Internacional.")
    m = _messier()[oid]
    kind = _kind(m["type"], lang)
    con = cn.get(nums["constellation"]["value"], nums["constellation"]["value"])
    art = "an" if kind[0] in "aeiou" else "a"
    if en:
        return f"{names(oid)['en']} is {art} {kind} in the constellation {con}."
    fem = kind.split()[0] in ("nebulosa", "galáxia", "nuvem")
    return f"{names(oid)['pt']} é {'uma' if fem else 'um'} {kind} na constelação de {con}."


def _computed_curiosities(oid, lang, nums):
    """Curiosities derived from the numbers (source: the number's source)."""
    en, out = lang == "en", []
    k = ids()[oid]
    if "distance_ly" in nums:
        d = nums["distance_ly"]
        ly = d["value"] * DIST.get(d["unit"], 1.0)
        from datetime import date
        y = date.today().year - ly
        when = (f"in {int(round(y))}" if y > 0 and ly < 3000 else
                f"about {_fmt(ly, lang)} years ago")
        when_pt = (f"em {int(round(y))}" if y > 0 and ly < 3000 else f"há cerca de {_fmt(ly, lang)} anos")
        out.append({"text": f"The light you see from it tonight left it {when}." if en else
                    f"A luz que você vê dela hoje à noite saiu de lá {when_pt}.", "source": d["source"]})
    if k == "star" and "luminosity_lsun" in nums:
        L = nums["luminosity_lsun"]["value"]
        out.append({"text": (f"It shines about {_fmt(L, lang)} times as brightly as the Sun." if L >= 1 else
                             f"It gives off only {_fmt(L, lang)} of the Sun's light.") if en else
                    (f"Ela brilha cerca de {_fmt(L, lang)} vezes mais que o Sol." if L >= 1 else
                     f"Ela emite só {_fmt(L, lang)} da luz do Sol."), "source": "computed"})
    if k == "star" and "teff_k" in nums:
        T = nums["teff_k"]["value"]
        col = ("blue-white", "branco-azulada") if T > 9000 else ("white", "branca") if T > 7300 else \
            ("yellow-white", "branco-amarelada") if T > 6000 else ("yellow", "amarela") if T > 5200 else \
            ("orange", "laranja") if T > 3900 else ("red-orange", "vermelho-alaranjada")
        out.append({"text": f"Its surface is at about {_fmt(T, lang, 2)} K, which is why it looks {col[0]}." if en else
                    f"A superfície está a cerca de {_fmt(T, lang, 2)} K, por isso ela parece {col[1]}.",
                    "source": "computed"})
    if k == "constellation":
        n = nums
        out.append({"text": (f"It covers {_fmt(n['area_deg2']['value'], lang, 4)} square degrees, number {n['area_rank']['value']} of 88 in size." if en else
                             f"Ocupa {_fmt(n['area_deg2']['value'], lang, 4)} graus quadrados, a {n['area_rank']['value']}ª de 88 em tamanho."),
                    "source": "constellation_areas"})
        if "naked_eye_stars" in n:
            out.append({"text": (f"On a dark night you can see about {n['naked_eye_stars']['value']} of its stars with the naked eye; the brightest is {n['brightest_star']['value']}." if en else
                                 f"Numa noite escura dá para ver cerca de {n['naked_eye_stars']['value']} das suas estrelas a olho nu; a mais brilhante é {n['brightest_star']['value']}."),
                        "source": "bsc5"})
        month = MONTHS[lang][n["best_month"]["value"] - 1]
        out.append({"text": (f"It is highest in the sky at about 9 pm in {month}." if en else
                             f"Fica mais alta no céu por volta das 21 h em {month}."), "source": "computed"})
    if k not in ("star", "constellation", "satellite") and oid not in _solar() and "size_arcmin" in nums:
        try:
            a = max(float(x) for x in str(nums["size_arcmin"]["value"]).split("x"))
            r = a / 31.0
            out.append({"text": (f"In the sky it spans about {_fmt(a, lang, 2)}′, {_fmt(r, lang, 2)} times the width of the full Moon." if en else
                                 f"No céu ela ocupa cerca de {_fmt(a, lang, 2)}′, {_fmt(r, lang, 2)} vezes a largura da Lua cheia."),
                        "source": "d3celestial"})
        except ValueError:
            pass
    if k != "constellation" and oid not in _solar() and k != "satellite":
        ra = _star_catalog().ra[_star_catalog().index_of(int(oid[5:]))] if k == "star" else _messier()[oid]["ra_deg"]
        month = MONTHS[lang][_best_month(ra) - 1]
        out.append({"text": (f"From most places it is highest in the evening sky (around 9 pm) in {month}." if en else
                             f"Na maioria dos lugares ela fica mais alta no céu da noite (por volta das 21 h) em {month}."),
                    "source": "computed"})
    if k not in ("star", "constellation", "satellite") and oid not in _solar() and "v_mag" in nums:
        v = nums["v_mag"]["value"]
        see = (("You can see it with the naked eye under a dark sky.", "Dá para vê-la a olho nu num céu escuro.") if v <= 5.5 else
               ("Binoculars show it on a dark night.", "Um binóculo mostra numa noite escura.") if v <= 8.5 else
               ("You need a telescope to see it.", "É preciso um telescópio para vê-la."))
        out.append({"text": see[0] if en else see[1], "source": "d3celestial"})
    return out


def _history(oid, lang, nums):
    t = _hand_text(oid, lang)
    if t.get("history"):
        return t["history"]
    en = lang == "en"
    if oid.startswith("con-"):
        c = _constellations()
        return ("It was " if en else "Foi ") + c["origins"][c["constellations"][oid[4:]]][lang] + \
            ("; its boundaries were fixed by the IAU in 1930." if en else "; os limites foram fixados pela UAI em 1930.")
    n = int(oid[1:])
    if n <= 45:
        when = ("Charles Messier listed it in the first part of his catalogue, published in 1774." if en else
                "Charles Messier o listou na primeira parte do seu catálogo, publicada em 1774.")
    elif n <= 68:
        when = ("Charles Messier added it to his catalogue in the edition published in 1780." if en else
                "Charles Messier o incluiu no catálogo na edição publicada em 1780.")
    elif n <= 103:
        when = ("It entered Messier's final list, published in 1781." if en else
                "Entrou na lista final de Messier, publicada em 1781.")
    else:
        when = ("Messier or Méchain observed it, but it was added to the catalogue only in the 20th century, from their notes." if en else
                "Messier ou Méchain o observaram, mas ele só entrou no catálogo no século XX, a partir das anotações deles.")
    why = (" Messier listed fuzzy objects so that comet hunters like him would not mistake them for comets." if en else
           " Messier listava objetos difusos para que caçadores de cometas como ele não os confundissem com cometas.")
    return when + why


# ------------------------------------------------------------------------------------------------ quiz
def quiz(oid, lang="en", seed: int = 0, n: int = 3) -> list[dict]:
    """A short multiple-choice quiz from the card's numbers: [{question, options, answer (index), source}]."""
    _check_lang(lang)
    oid = resolve(oid)
    rng = random.Random(f"{oid}-{seed}")
    nums, name, en = numbers(oid), names(oid)[lang], lang == "en"
    qs = []

    def add(question, right, wrong, source):
        opts = [right] + [w for w in wrong if w != right][:3]
        rng.shuffle(opts)
        qs.append({"question": question, "options": opts, "answer": opts.index(right), "source": source})

    kind = ids()[oid]
    kinds = ["planet", "dwarf_planet", "moon", "star", "constellation", "satellite", "gc", "oc", "s", "e", "pn", "snr", "sfr"]
    add(f"What kind of object is {name}?" if en else f"Que tipo de objeto é {name}?", _kind(kind, lang),
        [_kind(k, lang) for k in rng.sample([k for k in kinds if _kind(k, lang) != _kind(kind, lang)], 3)], "computed")
    if "launch" in nums:
        y = int(str(nums["launch"]["value"])[:4])
        add(f"In which year was {name} launched?" if en else f"Em que ano {name} foi lançado?", str(y),
            [str(y + d) for d in rng.sample([-15, -9, -4, 5, 11], 3)], nums["launch"]["source"])
    if "moons" in nums:
        m = nums["moons"]["value"]
        add(f"How many known moons does {name} have?" if en else f"Quantas luas conhecidas {name} tem?", str(m),
            [str(x) for x in sorted({0, 1, 2, 5, 16, 28, 95, 274} - {m}, key=lambda x: abs(x - m))[:3]], nums["moons"]["source"])
    if "orbital_period_d" in nums and ids()[oid] != "moon":
        y = nums["orbital_period_d"]["value"] / 365.25
        right = _duration(y * 365.25 * 24, lang)
        wrong = [_duration(y * f * 365.25 * 24, lang) for f in (0.25, 3.0, 10.0)]
        add(f"How long is a year on {name}?" if en else f"Quanto dura um ano em {name}?", right, wrong,
            nums["orbital_period_d"]["source"])
    if "distance_ly" in nums:
        d = nums["distance_ly"]
        ly = d["value"] * DIST.get(d["unit"], 1.0)
        fmt = lambda x: f"{_fmt(x, lang, 2)} " + ("light-years" if en else "anos-luz")  # noqa: E731
        add(f"About how far away is {name}?" if en else f"A que distância, mais ou menos, está {name}?", fmt(ly),
            [fmt(ly * f) for f in (0.01, 0.1, 30.0)], d["source"])
    if "constellation" in nums and ids()[oid] != "constellation":
        from .stars import constellation_names
        cn = constellation_names(lang)
        right_key = nums["constellation"]["value"]
        others = [k for k in sorted(cn) if k != right_key]
        add(f"In which constellation is {name}?" if en else f"Em que constelação fica {name}?", cn[right_key],
            [cn[k] for k in rng.sample(others, 3)], nums["constellation"]["source"])
    if "area_rank" in nums:
        r = nums["area_rank"]["value"]
        add(f"Of the 88 constellations, where does {name} rank in size?" if en else
            f"Entre as 88 constelações, qual é a posição de {name} em tamanho?", f"{r}",
            [f"{x}" for x in rng.sample([i for i in range(1, 89) if abs(i - r) > 5], 3)], nums["area_rank"]["source"])
    if "diameter_km" in nums:
        r = nums["diameter_km"]["value"] / EARTH["diameter_km"]
        if abs(r - 1) > 0.02:
            f = lambda x: _fmt(x, lang, 2) + "×"  # noqa: E731
            add(f"How wide is {name} compared with the Earth?" if en else f"Qual é a largura de {name} comparada à da Terra?",
                f(r), [f(r * k) for k in (0.1, 3.0, 10.0)], nums["diameter_km"]["source"])
    return qs[1:n + 1] if len(qs) > n else qs[:n]       # the kind question only when there are few others


# ------------------------------------------------------------------------------------------------ cards
def card(ref, lang="en") -> dict:
    """The info card of an object in ``lang``: name, kind, summary, numbers (value, unit, label, comparison,
    source), three to five curiosities with sources, a short history, missions, a quiz and the sources cited."""
    _check_lang(lang)
    oid = resolve(ref)
    nums = numbers(oid)
    rows = []
    for key, num in nums.items():
        rows.append({"key": key, "label": _label(key, lang), **num, "comparison": comparison(key, num, lang)})
    t = _hand_text(oid, lang)
    curios = list(t.get("curiosities", []))
    for c in _computed_curiosities(oid, lang, nums):
        if len(curios) >= 5:
            break
        curios.append(c)
    kind = ids()[oid]
    out = {"id": oid, "kind": kind, "kind_label": _kind(kind, lang), "name": names(oid)[lang], "names": names(oid),
           "lang": lang, "summary": _summary(oid, lang, nums), "numbers": rows, "curiosities": curios[:5],
           "history": _history(oid, lang, nums), "missions": list((_solar().get(oid) or _satellites().get(oid[4:], {})
                                                                   if not oid.startswith(("star-", "con-", "M")) else {}).get("missions", [])),
           "quiz": quiz(oid, lang)}
    used = {r["source"] for r in rows} | {c["source"] for c in curios} | {q["source"] for q in out["quiz"]}
    out["sources"] = {k: sources()[k] for k in sorted(used)}
    if oid in _solar():
        out["observe"] = oid if oid not in ("earth",) else None
        out["parent"] = _solar()[oid].get("parent")
    elif kind == "star":
        out["hr"] = int(oid[5:])
    elif kind == "satellite":
        out["norad"] = SATELLITE_NORAD[oid[4:]]
    elif kind != "constellation":
        m = _messier()[oid]
        out["ra_deg"], out["dec_deg"] = m["ra_deg"], m["dec_deg"]
    return out


def list_cards(kind: str | None = None, lang="en") -> list[dict]:
    """[{id, kind, name}] of every card (or of one kind: 'planet', 'star', 'constellation', 'messier', ...)."""
    _check_lang(lang)
    out = []
    for oid, k in ids().items():
        group = "messier" if oid.startswith("M") and oid[1:].isdigit() else k
        if kind in (None, k, group):
            out.append({"id": oid, "kind": k, "group": group, "name": names(oid)[lang]})
    return out


def search(query: str, lang: str | None = None, limit: int = 12) -> list[dict]:
    """Cards whose name (in English or Portuguese), id or designation matches ``query`` (accents and case ignored);
    prefix matches first."""
    q = _fold(query).strip()
    if not q:
        return []
    hits = []
    for oid in ids():
        n = names(oid)
        keys = {_fold(n["en"]), _fold(n["pt"]), _fold(oid), _fold(oid.replace("-", " "))}
        if oid in _messier() and _messier()[oid].get("desig"):
            keys.add(_fold(_messier()[oid]["desig"]))
        if oid.startswith("star-"):
            keys.add(f"hr {oid[5:]}")
        score = min((0 if k.startswith(q) else 1 if f" {q}" in f" {k}" else 2 if q in k else 9) for k in keys)
        if score < 9:
            hits.append((score, len(n["en"]), oid))
    hits.sort()
    return [{"id": oid, "kind": ids()[oid], "name": names(oid)[lang or "en"], "names": names(oid)}
            for _, _, oid in hits[:limit]]


def did_you_know(lang="en", seed: int | None = None) -> dict:
    """A random curiosity from the hand-written cards, for a start-up tip: {id, name, text, source}."""
    _check_lang(lang)
    pool = []
    for oid in list(_solar()) + [f"sat-{k}" for k in _satellites()]:
        for c in _hand_text(oid, lang).get("curiosities", []):
            pool.append((oid, c))
    oid, c = random.Random(seed).choice(pool)
    return {"id": oid, "name": names(oid)[lang], "text": c["text"], "source": c["source"],
            "source_title": sources()[c["source"]]["title"]}


def validate() -> list[str]:
    """Problems in the dataset (an empty list means every card has a source per number, text in both languages,
    three to five curiosities, a history and known sources)."""
    bad = []
    src = sources()
    for oid in ids():
        for lang in LANGS:
            try:
                c = card(oid, lang)
            except Exception as exc:  # noqa: BLE001
                bad.append(f"{oid}/{lang}: {exc}")
                continue
            if not c["summary"] or not c["history"]:
                bad.append(f"{oid}/{lang}: missing summary or history")
            if not 3 <= len(c["curiosities"]) <= 5:
                bad.append(f"{oid}/{lang}: {len(c['curiosities'])} curiosities")
            for r in c["numbers"]:
                if r["source"] not in src:
                    bad.append(f"{oid}: number {r['key']} has unknown source {r['source']!r}")
            for x in c["curiosities"]:
                if x["source"] not in src or not x["text"]:
                    bad.append(f"{oid}/{lang}: curiosity without a known source")
            if not c["numbers"]:
                bad.append(f"{oid}: no numbers")
    return bad


# ------------------------------------------------------------------------------------------------ tonight
def position(ref):
    """J2000 (ICRS) RA/Dec in degrees of a fixed object (star, Messier object, constellation label point), or None
    for Solar System bodies and satellites, whose positions come from the ephemeris / TLE."""
    oid = resolve(ref)
    k = ids()[oid]
    if k == "star":
        c = _star_catalog()
        i = c.index_of(int(oid[5:]))
        return float(c.ra[i]), float(c.dec[i])
    if k == "constellation":
        from .stars import load_constellations
        lab = next(x["label"] for x in load_constellations() if x["key"] == oid[4:])
        return float(lab[0]), float(lab[1])
    if oid in _messier():
        return _messier()[oid]["ra_deg"], _messier()[oid]["dec_deg"]
    return None


def tonight(ref, observer, t, hours: float = 24.0, step_min: float = 10.0, eph=None) -> dict | None:
    """How to see an object from ``observer`` in the ``hours`` after ``t``: altitude now, rise, culmination (highest
    point) and set (the next rise only if it is below the horizon now), the highest altitude while the sky is dark (Sun below -12 deg) and when, and whether it never
    rises or never sets. None for satellites (their passes come from the TLE: satellites.find_passes)."""
    import numpy as np

    from .astrometry import apparent, star_apparent
    from .timescales import Time
    oid = resolve(ref)
    if ids()[oid] == "satellite" or oid == "earth":
        return None
    jd0 = float(np.ravel(t.utc)[0])
    jds = jd0 + np.arange(0, hours * 60 + step_min / 2, step_min) / 1440.0
    T = Time(jds)
    if oid in _solar():
        alt = apparent(oid, T, observer, eph).alt_apparent
    else:
        ra, dec = position(oid)
        alt = np.array([star_apparent(np.array([ra]), np.array([dec]), Time(j), observer)[4][0] for j in jds])
    sun = apparent("sun", T, observer, eph).alt
    up = alt > 0

    def crossing(i):          # linear interpolation of the horizon crossing between samples i and i + 1
        f = alt[i] / (alt[i] - alt[i + 1])
        return Time(jds[i] + f * (jds[i + 1] - jds[i])).iso()

    rises = [crossing(i) for i in range(len(alt) - 1) if not up[i] and up[i + 1]]
    sets = [crossing(i) for i in range(len(alt) - 1) if up[i] and not up[i + 1]]
    k = int(np.argmax(alt))
    dark = sun < -12
    out = {"alt_now": float(alt[0]), "above_now": bool(up[0]), "rise": rises[0] if rises and not up[0] else None,
           "set": sets[0] if sets else None, "culmination": Time(jds[k]).iso(), "max_alt": float(alt[k]),
           "never_rises": bool(not up.any()), "never_sets": bool(up.all()), "dark_best": None}
    if (dark & up).any():
        j = int(np.argmax(np.where(dark & up, alt, -99)))
        out["dark_best"] = {"time": Time(jds[j]).iso(), "alt": float(alt[j])}
    return out
