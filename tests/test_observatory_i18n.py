"""Observatory interface language (#721): the English and Portuguese message catalogues have the same keys and
placeholders, every key the page uses exists, and no user-visible text is hard-coded in static/*.js or index.html."""
from __future__ import annotations

import html
import json
import os
import re

import pytest

STATIC = os.path.join(os.path.dirname(__file__), "..", "apps", "observatory", "observatory", "static")
CAT = {lang: json.load(open(os.path.join(STATIC, "i18n", f"{lang}.json"), encoding="utf-8")) for lang in ("en", "pt")}
JS = sorted(f for f in os.listdir(STATIC) if f.endswith(".js"))

# Tokens that are the same in both languages: units, symbols, catalogue designations, proper names, code words.
ALLOW = {
    "km", "nm", "pm", "mas", "mm", "pc", "mag", "kpc", "Mpc", "µas", "AU", "UA", "HR", "HD", "NGC", "IC", "NORAD", "TLE",
    "UTC", "LST", "JPL", "DE421", "SGP4", "PINN", "PINNeAPPle", "EHT", "IAU", "BSC5", "ISS", "Kp",
    "F10", "Ap", "CD", "Teff", "M_V", "M87", "Sgr", "GM", "RA", "Dec", "rgb", "px", "deg", "lst", "RK4", "MK",
    "Hα", "ISCO", "NEO", "NRLMSIS", "SBDB", "DSS2", "CDS", "hips2fits", "Open", "Meteo", "NASA", "SIMBAD", "Yoshida",
    "leapfrog", "yoshida4", "rk4", "Wien", "Messier", "English", "Português", "EN", "PT", "CC", "BY", "Open-Meteo",
    "rel",
}
# Messages that may read the same in both languages: place names, references, and a few cognates.
SAME_IN_BOTH = ("city.", "valref.", "twilight.civil", "object.magnitude")
CODE_WORDS = re.compile(r"^(?:[a-z]+[A-Z]\w*|[a-z_]+_[a-z_]+|#[\w-]+|\.[\w-]+|[\w-]+\.(?:js|json|jpg|png|css)|https?|"
                        r"rgba?|hsla?|var|px|em|vh|vw|none|auto|block|flex|inline|absolute)$")


def _js_literals(src: str):
    """String and template literals of a JS source: (text with ${...} removed, the 60 characters before it)."""
    out, i, n = [], 0, len(src)
    prev_sig = ""

    def template(j):
        parts, buf = [], []
        while j < n:
            c = src[j]
            if c == "\\":
                buf.append(src[j:j + 2])
                j += 2
            elif c == "`":
                parts.append("".join(buf))
                return " ".join(parts), j + 1
            elif c == "$" and src[j + 1:j + 2] == "{":
                parts.append("".join(buf))
                buf = []
                j = expression(j + 2)
                parts.append("\x00")
            else:
                buf.append(c)
                j += 1
        raise ValueError("unterminated template literal")

    def expression(j):                      # skip to the matching }, collecting nested literals
        depth = 1
        while j < n and depth:
            c = src[j]
            if c in "'\"":
                j = string(j)
            elif c == "`":
                start = j
                text, j = template(j + 1)
                out.append((text, src[max(0, start - 60):start]))
            elif c == "{":
                depth += 1
                j += 1
            elif c == "}":
                depth -= 1
                j += 1
            else:
                j += 1
        return j

    def string(j):
        q, k = src[j], j + 1
        while src[k] != q:
            k += 2 if src[k] == "\\" else 1
        out.append((src[j + 1:k], src[max(0, j - 60):j]))
        return k + 1

    while i < n:
        c = src[i]
        if src.startswith("//", i):
            i = src.find("\n", i)
            i = n if i < 0 else i
        elif src.startswith("/*", i):
            i = src.index("*/", i) + 2
        elif c in "'\"":
            i = string(i)
            prev_sig = "a"
        elif c == "`":
            start = i
            text, i = template(i + 1)
            out.append((text, src[max(0, start - 60):start]))
            prev_sig = "a"
        elif c == "/" and (prev_sig in "(,=:[!&|?{};+-*%<>~^" or re.search(r"\b(return|typeof|case)\s*$", src[:i])):
            k = i + 1                                   # a regular expression literal
            in_class = False
            while k < n and (src[k] != "/" or in_class):
                if src[k] == "\\":
                    k += 1
                elif src[k] == "[":
                    in_class = True
                elif src[k] == "]":
                    in_class = False
                k += 1
            i = k + 1
            while i < n and src[i].isalpha():
                i += 1
            prev_sig = "a"
        else:
            if not c.isspace():
                prev_sig = c if not (c.isalnum() or c in "_$.)]") else "a"
            i += 1
    return out


def _visible_text(lit: str) -> tuple[str, bool]:
    """Text a user would read from a literal: HTML text nodes and title/placeholder/aria-label values."""
    has_tag = bool(re.search(r"<\s*/?\s*[a-zA-Z][\w-]*[\s>/]", lit))
    attrs = " ".join(re.findall(r'(?:title|placeholder|aria-label|alt)="([^"$\x00]*)"', lit))
    text = re.sub(r"<[^<>]*>", " ", lit) if has_tag else lit
    text = html.unescape(text.replace("\x00", " "))
    return (text + " " + attrs).strip(), has_tag


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-zµÀ-ÖØ-öø-ÿ][\wµÀ-ÖØ-öø-ÿ'’-]*", text) if not w.isdigit()]


# literals that are never interface text: catalogue keys, URLs and paths, file names, IANA time zones
NOT_TEXT = re.compile(r"^(?:[a-z]+(?:\.[\w-]+)+|/[^\s]*|[\w-]+\.(?:js|json|jpg|png|css|html)|"
                      r"(?:Africa|America|Asia|Atlantic|Australia|Europe|Pacific|Indian|Antarctica)/[A-Za-z_]+)$")


def _prose(text: str, has_tag: bool) -> list[str]:
    """Words that look like interface text (not code, units or symbols)."""
    words = [w for w in _words(text) if w not in ALLOW and not CODE_WORDS.match(w) and len(w) > 1]
    if not words:
        return []
    if has_tag:
        return words
    stripped = text.strip()
    # plain strings: flag phrases ("No passes above") and capitalised words ("Moon"), not identifiers ("pointerdown")
    if re.search(r"[A-Za-z]{2,}[ ,.:;!?…]+[A-Za-z]{2,}", stripped) or re.match(r"^[A-Z][a-zà-ÿ]{2,}\b", stripped):
        return words
    return []


SKIP_CONTEXT = re.compile(r"(?:\bt|tOr|has|querySelector(?:All)?|\$|getElementById|addEventListener|"
                          r"removeEventListener|classList\.\w+|getAttribute|setAttribute|dataset|import|from|"
                          r"fetch|api|load|loader\.load|new THREE\.Color|Color|set|setStyle|matchMedia|"
                          r"setItem|getItem|createElement|Error|startsWith|includes|split|replace|test|"
                          r"getContext|toggle|Event|CustomEvent)\s*\(\s*$|\bfrom\s*$|\bimport\s*$|\bcase\s*$|===?\s*$|!==?\s*$|\.(?:font|filter|textAlign|fillStyle|strokeStyle)\s*=\s*$")


def _is_shader(text: str) -> bool:
    return "gl_" in text or "void main" in text or "uniform " in text or "varying " in text


def test_catalogues_have_the_same_keys_and_placeholders():
    en, pt = CAT["en"], CAT["pt"]
    assert set(en) - set(pt) == set(), f"missing in pt.json: {sorted(set(en) - set(pt))}"
    assert set(pt) - set(en) == set(), f"missing in en.json: {sorted(set(pt) - set(en))}"
    ph = lambda s: sorted(re.findall(r"\{(\w+)\}", s))  # noqa: E731
    bad = [k for k in en if ph(en[k]) != ph(pt[k])]
    assert not bad, f"placeholders differ between en and pt: {bad}"
    empty = [k for lang in CAT for k, v in CAT[lang].items() if not str(v).strip()]
    assert not empty, f"empty messages: {empty}"


def test_portuguese_is_translated():
    """Messages identical in both catalogues must be language-neutral (numbers, symbols, proper names)."""
    same = [k for k, v in CAT["en"].items() if v == CAT["pt"][k] and len(_prose(v, False)) > 0
            and not k.startswith(SAME_IN_BOTH)]
    assert not same, f"untranslated in pt.json: {same}"


def test_every_key_used_exists():
    used = set()
    for f in JS:
        src = open(os.path.join(STATIC, f), encoding="utf-8").read()
        used |= set(re.findall(r"\b(?:t|tOr)\(\s*[\"']([\w.\-]+)[\"']\s*[,)]", src))
    page = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
    used |= set(re.findall(r'data-i18n(?:-title|-placeholder|-aria)?="([\w.\-]+)"', page))
    missing = sorted(k for k in used if k not in CAT["en"])
    assert not missing, f"keys used but not in the catalogues: {missing}"


@pytest.mark.parametrize("name", JS)
def test_no_hard_coded_text_in_scripts(name):
    src = open(os.path.join(STATIC, name), encoding="utf-8").read()
    bad = []
    for lit, before in _js_literals(src):
        if _is_shader(lit) or SKIP_CONTEXT.search(before) or NOT_TEXT.match(re.sub(r"\s*\x00\s*", "", lit)):
            continue
        text, has_tag = _visible_text(lit)
        words = _prose(text, has_tag)
        if words:
            bad.append(f"{lit[:70]!r} -> {words[:4]}")
    assert not bad, f"{name}: user-visible text outside the catalogues (use t(...)):\n" + "\n".join(bad)


def test_no_hard_coded_text_in_page():
    page = open(os.path.join(STATIC, "index.html"), encoding="utf-8").read()
    body = page[page.index("<body"):]
    body = re.sub(r"<script\b.*?</script>", "", body, flags=re.S)
    bad = []
    for m in re.finditer(r"<(\w+)([^<>]*)>([^<]*)", body):
        tag, attrs, text = m.groups()
        if _prose(html.unescape(text), True) and "data-i18n=" not in attrs:
            bad.append(f"<{tag}{attrs[:40]}> {text.strip()[:40]!r}")
        for a in ("title", "placeholder", "aria-label"):
            v = re.search(rf'\b{a}="([^"]*)"', attrs)
            key = {"title": "data-i18n-title", "placeholder": "data-i18n-placeholder", "aria-label": "data-i18n-aria"}[a]
            if v and _prose(v.group(1), True) and key not in attrs:
                bad.append(f"<{tag}> {a}={v.group(1)!r}")
    assert not bad, "index.html: text without a data-i18n key:\n" + "\n".join(bad)
    head = page[:page.index("<body")]
    assert 'name="description"' in head and "<title>" in head       # replaced at start-up by meta.* keys


def test_lint_catches_hard_coded_text():
    """The lint itself: an English phrase, a capitalised word, a tag's text are flagged; keys and code are not."""
    src = 'const a = "No passes above 10°"; el.innerHTML = `<b>Moon</b> ${x}`; toast("Saved");' \
          'const k = t("tonight.title"); el.classList.add("on"); const s = $("#tab-object p");'
    flagged = []
    for lit, before in _js_literals(src):
        if SKIP_CONTEXT.search(before):
            continue
        text, tag = _visible_text(lit)
        if _prose(text, tag):
            flagged.append(lit)
    assert flagged == ["No passes above 10°", "<b>Moon</b>  \x00 ", "Saved"]


def test_constellation_names_in_both_languages():
    from pinneapple_physics.astro.stars import constellation_names, load_constellations
    pt, en = constellation_names("pt"), constellation_names("en")
    assert len(pt) == len(en) == 89                  # 88 constellations, Serpens in two parts
    assert pt["Cru"] == "Cruzeiro do Sul" and en["Cru"] == "Crux" and pt["Sco"] == "Escorpião"
    cons = load_constellations()
    keys = [c["key"] for c in cons]
    assert len(keys) == len(set(keys)) == 89 and {"Ser1", "Ser2"} <= set(keys)
    ser = {c["key"]: c for c in cons if c["id"] == "Ser"}
    assert ser["Ser1"]["names"]["en"] == "Serpens Caput" and ser["Ser2"]["names"]["pt"] == "Cauda da Serpente"
    assert ser["Ser1"]["label"] != ser["Ser2"]["label"]
    with pytest.raises(ValueError):
        constellation_names("fr")
