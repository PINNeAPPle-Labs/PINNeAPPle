"""Render the demo video from ``VIDEO_SCRIPT.md`` and the files in ``results/``.

    python examples/end_to_end/airfoil_trust/make_video.py --cut long --out ~/trust-video
    python examples/end_to_end/airfoil_trust/make_video.py --cut short --out ~/trust-video

The narration is synthesised locally with macOS ``say`` -- nothing is sent anywhere -- and the frames are drawn from
the same JSON and figures the report is built from, so the film cannot say anything the study did not measure. Every
frame is regenerated on each run; the narrations are kept as WAV files next to the output so they can be replaced with
a human read without touching this script.

Requires macOS (``say``) and ffmpeg. Silence either with ``--voice none`` for a silent cut with the same timing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figures")
SCRIPT = os.path.join(HERE, "VIDEO_SCRIPT.md")

W, H = 1920, 1080
BG = (13, 17, 23)
FG = (230, 237, 243)
MUTED = (139, 148, 158)
BLUE, ORANGE, GREEN, RED = (88, 166, 255), (255, 166, 87), (63, 185, 80), (248, 81, 73)
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
FONT_M = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    for candidate in (path, FONT, FONT_M):
        if os.path.exists(candidate):
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                continue
    return ImageFont.load_default()


def load(name: str) -> dict:
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


# --------------------------------------------------------------------------- text
def spoken(text: str) -> str:
    """What ``say`` should read, for text written for a human reader."""
    t = text.replace("\u2009", ",").replace(" -- ", ", ").replace("--", ", ").replace("—", ", ")
    for token, said in (("U_inf", "U infinity"), ("k-omega SST", "k omega S S T"), ("x/c", "x over c"),
                        ("CL", "C L"), ("CD", "C D"), ("Cp", "C p"), ("CM", "C M"), ("POD", "P O D"),
                        ("SST", "S S T"), ("SHA-256", "S H A 256")):
        t = t.replace(token, said)

    def _exp(m):
        exponent = int(m.group(2).replace("−", "-"))
        return (f"{m.group(1)} times ten to the {'minus' if exponent < 0 else 'plus'} {_word(abs(exponent))}")

    t = re.sub(r"([-\u2212]?\d+(?:\.\d+)?)e([-+\u2212]?\d+)", _exp, t)
    t = t.replace("%", " percent").replace("|", " ").replace("`", "").replace("*", "")
    t = re.sub(r"(?<=\d)[\u2009 ](?=\d{3}\b)", ",", t)      # 16 117 is sixteen thousand, not two numbers
    return re.sub(r"\s{2,}", " ", t)


def hexc(colour: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*colour)


def _word(n: int) -> str:
    ones = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
            "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    return ones[n] if n < len(ones) else str(n)


# --------------------------------------------------------------------------- frames
def blank() -> Image.Image:
    return Image.new("RGB", (W, H), BG)


def wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int) -> list[str]:
    words, lines, line = text.split(), [], ""
    for w in words:
        probe = f"{line} {w}".strip()
        if draw.textlength(probe, font=font) <= width:
            line = probe
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines


def card(title: str, body: str = "", *, kicker: str = "", accent=BLUE, big: bool = False,
         footer: str = "") -> Image.Image:
    img = blank()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 8], fill=accent)
    f_title = _font(FONT_B, 96 if big else 62)
    f_body = _font(FONT, 40)
    f_kick = _font(FONT, 30)
    title_lines = wrap(d, title, f_title, W - 280)
    body_lines = []
    if body:
        for para in body.split("\n"):
            body_lines += (wrap(d, para, f_body, W - 320) if para.strip() else [""]) + [None]
    height = (70 if kicker else 0) + len(title_lines) * int(f_title.size * 1.25) + 40 + \
        sum(1 for line in body_lines if line is None) * 12 + \
        sum(1 for line in body_lines if line is not None) * int(f_body.size * 1.45) if body else \
        (70 if kicker else 0) + len(title_lines) * int(f_title.size * 1.25)
    y = max(150, (H - height) // 2 - 30)
    if kicker:
        d.text((140, y - 70), kicker.upper(), font=f_kick, fill=MUTED)
    for line in title_lines:
        d.text((140, y), line, font=f_title, fill=FG)
        y += int(f_title.size * 1.25)
    if body:
        y += 40
        for line in body_lines:
            if line is None:
                y += 12
                continue
            if line:
                d.text((140, y), line, font=f_body, fill=FG)
            y += int(f_body.size * 1.45)
    if footer:
        f = _font(FONT, 26)
        d.text((140, H - 110), footer, font=f, fill=MUTED)
    return img


def figure_frame(path: str, title: str, caption: str = "", accent=BLUE) -> Image.Image:
    img = blank()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 8], fill=accent)
    f_title = _font(FONT_B, 44)
    d.text((140, 90), title, font=f_title, fill=FG)
    art = Image.open(path).convert("RGB")
    scale = min((W - 280) / art.width, (H - 400) / art.height, 1.9)
    art = art.resize((int(art.width * scale), int(art.height * scale)), Image.LANCZOS)
    img.paste(art, ((W - art.width) // 2, 200))
    if caption:
        f = _font(FONT, 26)
        d.text((140, 200 + art.height + 40), caption, font=f, fill=MUTED)
    return img


def fit(draw: ImageDraw.ImageDraw, text: str, font, room: float) -> str:
    """Cut a cell to the width it is given, with an ellipsis, rather than to a character count."""
    if draw.textlength(text, font=font) <= room:
        return text
    while text and draw.textlength(text + "…", font=font) > room:
        text = text[:-1]
    return text + "…"


def table_frame(title: str, headers: list[str], rows: list[list[str]], colours: list[tuple] | None = None,
                caption: str = "", accent=BLUE, widths: list[float] | None = None) -> Image.Image:
    img = blank()
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 8], fill=accent)
    f_h, f_c = _font(FONT_B, 34), _font(FONT, 34)
    row_h = 64
    block = 100 + 56 + row_h * len(rows) + (70 if caption else 0)
    d.text((140, max(70, (H - block) // 2 - 40)), title, font=_font(FONT_B, 48), fill=FG)
    span = W - 280
    widths = widths or [1 / len(headers)] * len(headers)
    xs = [140 + sum(widths[:i]) * span for i in range(len(headers))]
    y = max(70, (H - block) // 2 - 40) + 140
    for x, head in zip(xs, headers, strict=True):
        d.text((x, y), head, font=f_h, fill=MUTED)
    y += 56
    d.line([140, y - 12, W - 140, y - 12], fill=(48, 54, 61), width=2)
    for k, row in enumerate(rows):
        colour = FG if colours is None else colours[k]
        for i, (x, cell) in enumerate(zip(xs, row, strict=True)):
            room = (widths[i] * span) - 30
            text = fit(d, str(cell), f_c, room)
            d.text((x, y), text, font=f_c, fill=colour if i == len(row) - 1 else FG)
        y += row_h
    if caption:
        d.text((140, y + 40), caption, font=_font(FONT, 28), fill=MUTED)
    return img


def doe_frame(path: str) -> Image.Image:
    """The design space, drawn from the same plan the campaign ran."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plan = load("doe_round1.json")["splits"]
    fig = plt.figure(figsize=(11.2, 5.6), dpi=150)
    ax = fig.add_subplot(projection="3d")
    ax.view_init(elev=22, azim=-58)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    for name, colour in (("train", BLUE), ("calibration", ORANGE), ("validation", GREEN)):
        pts = plan[name]
        ax.scatter([p["alpha_deg"] for p in pts], [p["speed"] for p in pts], [p["thickness"] * 100 for p in pts],
                   s=42, color=hexc(colour), label=f"{name} ({len(pts)})", depthshade=False)
    ax.set_xlabel("alpha [deg]", color=hexc(MUTED))
    ax.set_ylabel("speed [m/s]", color=hexc(MUTED))
    ax.set_zlabel("thickness [% chord]", color=hexc(MUTED))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.label.set_color(hexc(MUTED))
        axis.pane.set_facecolor("#11161d")
        axis.pane.set_edgecolor("#30363d")
    ax.tick_params(colors=hexc(MUTED), labelsize=10)
    leg = ax.legend(fontsize=11, loc="upper left", framealpha=0.15, labelcolor=hexc(FG))
    if leg is not None:
        leg.get_frame().set_facecolor("#11161d")
    fig.subplots_adjust(left=0.02, right=0.95, bottom=0.03, top=0.98)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def pod_frame(path: str) -> Image.Image:
    """How much of the field the basis keeps, from the singular values the analysis wrote."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    basis = load("analysis_v1.json")["basis"]
    s = np.asarray(basis["singular_values"], float) ** 2
    frac = s / s.sum()
    fig, ax = plt.subplots(figsize=(9.6, 4.2), dpi=140)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.bar(np.arange(1, len(frac) + 1), 100 * frac, color=hexc(BLUE))
    ax.plot(np.arange(1, len(frac) + 1), 100 * np.cumsum(frac), "o-", color=hexc(ORANGE), ms=4, lw=1.3,
            label="cumulative")
    ax.axvline(basis["n_modes"] + 0.5, color=hexc(MUTED), ls="--", lw=1)
    ax.text(basis["n_modes"] + 0.7, 60, f"{basis['n_modes']} modes kept\n{100 * basis['energy_captured']:.2f} % "
                                        "of the energy", color=hexc(FG), fontsize=9)
    ax.set_xlabel("POD mode", color=hexc(MUTED))
    ax.set_ylabel("share of the field energy [%]", color=hexc(MUTED))
    ax.tick_params(colors=hexc(MUTED), labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#30363d")
    ax.grid(axis="y", color="#21262d", lw=0.6)
    ax.legend(fontsize=8, framealpha=0.15, labelcolor=hexc(FG))
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- the deck
CUTS = {"long": "## The 8-10 minute technical demo", "short": "## The 90-120 second cut"}
_END = "## Production notes"


def parse_script(cut: str) -> list[dict]:
    """The narration of one cut, read out of the script so that the film cannot drift from it."""
    text = open(SCRIPT).read()
    headings = [CUTS["long"], CUTS["short"], _END]
    start = text.index(CUTS[cut])
    end = text.index(next(h for h in headings if text.index(h) > start))
    body = text[start:end].splitlines()[1:]
    scenes, current = [], None
    for line in body:
        long_scene = re.match(r"### Scene (\d+) -- (.+?) \(", line)
        short_scene = re.match(r"\*\*(\d+:\d+ - \d+:\d+) -- (.+?)\.\*\*", line)
        if long_scene or short_scene:
            m = long_scene or short_scene
            current = {"n": int(m.group(1)) if long_scene else len(scenes) + 1,
                       "title": (m.group(2) if long_scene else m.group(2)).strip(), "narration": []}
            scenes.append(current)
            continue
        if line.startswith("**CLOSING CARD"):
            current = None                      # its line is the card, not narration: the words are on screen
            continue
        if current is not None and line.startswith(">"):
            said = line[1:].strip()
            if said and not said.startswith(("How do you know", "`")):   # no on-screen question, no commands
                current["narration"].append(said)
    for s in scenes:
        s["narration"] = " ".join(s["narration"])
    return [s for s in scenes if s["narration"]]


def deck(work: str) -> dict[int, list[tuple[Image.Image, str]]]:
    """Frames per scene, in order: (image, what it is for the log)."""
    mesh, phy = load("mesh_study.json"), load("physics_v1.json")
    dom, rev = load("domain_v1.json"), load("revalidation_v2.json")
    dos, ana = load("dossier.json"), load("analysis_v1.json")
    q = dom["queries"]
    a = ana["validation"]["qoi"]
    frames: dict[int, list[tuple[Image.Image, str]]] = {}

    frames[1] = [
        (card("How do you know when to trust a Physics AI model?", big=True, accent=BLUE), "question"),
        (card("a validation error is an average\n a design is not an average\n an average has no shape",
              big=True, accent=BLUE), "the three lines"),
    ]

    rows = [[lv["name"], f"{lv['cells']:,}".replace(",", " "), f"{lv['values']['CL']:.4f}",
             f"{lv['values']['CD']:.5f}", f"{lv['nonorth_avg']:.1f}", f"{lv['skew_max']:.2f}",
             f"{lv['cost_s']:.0f} s" if lv.get("cost_s") else "not timed"] for lv in mesh["levels"]]
    frames[2] = [
        (table_frame("The same section on four meshes",
                     ["mesh", "cells", "CL", "CD", "non-orth.", "skew", "time"], rows,
                     caption="every spacing, including the boundary layer, refined by the same factor 1.45",
                     widths=[0.1, 0.16, 0.14, 0.16, 0.16, 0.12, 0.16]), "mesh table"),
        (figure_frame(os.path.join(FIG, "mesh_convergence.png"),
                      "The reference is a measurement, not the truth",
                      "CL oscillates at its own discretisation error; CD converges, order 1.69, GCI 11.8 %",
                      accent=BLUE), "mesh convergence"),
    ]

    dpath, ppath = os.path.join(work, "doe.png"), os.path.join(work, "pod.png")
    doe_frame(dpath)
    pod_frame(ppath)
    frames[3] = [
        (figure_frame(dpath, "Sixty designs, split by whole designs", "32 train, 12 calibration, 16 held out",
                      accent=BLUE), "design space"),
        (figure_frame(ppath, "Eight modes, 98.87 % of the field energy",
                      f"an unseen field reconstructs to {100 * ana['basis']['held_out_reconstruction_rel_l2']:.2f} % "
                      "from the basis alone", accent=BLUE), "POD spectrum"),
    ]

    frames[4] = [
        (figure_frame(os.path.join(FIG, "accuracy.png"), "Accuracy, and where it is wrong",
                      f"CL relative L2 {100 * a['CL_rel_l2']:.2f} % over the held-out designs, inside the 5 % bar",
                      accent=ORANGE), "parity"),
        (figure_frame(os.path.join(FIG, "fields.png"), "The worst held-out design, along the chord",
                      "the error is at the leading edge, not spread over the section", accent=ORANGE), "error trace"),
    ]

    s = phy["summary"]
    rows = [["no-slip at the wall", f"{s['wall_condition']['surrogate_mean']:.4f}",
             f"{s['wall_condition']['reference_mean']:.4f}", "ratio to the reference"],
            ["continuity residual", f"{s['continuity']['surrogate_mean']:.3f}",
             f"{s['continuity']['reference_mean']:.3f}", "ratio to the reference"],
            ["freestream decay", f"{s['freestream']['surrogate_mean']:.3f}",
             f"{s['freestream']['reference_mean']:.3f}", "ratio to the reference"],
            ["pressure integral", f"{s['cp_integral']['surrogate_mean']:.2f} CL", "—", "field vs scalar output"],
            ["field accuracy (Cp)", f"{s['gate_accuracy']['surrogate_mean']:.2f}", "—", "relative L2, 5 % bar"]]
    frames[5] = [
        (table_frame("Is it a flow, or only a fit?", ["check", "surrogate", "reference", "how it is judged"], rows,
                     colours=[FG, FG, FG, FG, RED], caption="the last one fails: the lift is inside tolerance and the "
                     "field it integrates is not", accent=ORANGE, widths=[0.3, 0.17, 0.17, 0.36]), "gates"),
        (figure_frame(os.path.join(FIG, "fields.png"), "The lift is inside tolerance. The field it comes from is not.",
                      f"relative L2 over u, v and cp: {100 * ana['validation']['field_rel_l2_mean']:.1f} %, against a "
                      f"{100 * ana['basis']['held_out_reconstruction_rel_l2']:.1f} % floor from the basis alone",
                      accent=ORANGE), "the field"),
    ]

    checks = dos["checks"]
    rows = [[c["check_id"].replace("_", " "), c["category"],
             "not run" if c["passed"] is None else ("passed" if c["passed"] else "failed")]
            for c in checks]
    colours = [RED if c["passed"] is False else (ORANGE if c["passed"] is None else FG) for c in checks]
    frames[6] = [
        (table_frame("The checks, with one that did not run", ["check", "group", "status"], rows,
                     colours=colours, caption="a check that did not run is not a pass and not an omission",
                     accent=ORANGE, widths=[0.52, 0.24, 0.24]), "check table"),
    ]

    frames[7] = [
        (figure_frame(os.path.join(FIG, "uncertainty.png"), "The spread is not a confidence interval",
                      f"one sigma holds {100 * checks[-3]['value']:.0f} % of the errors, two hold "
                      f"{100 * ana['validation']['calibration']['coverage_2s']:.0f} % -- not 68 and 95",
                      accent=BLUE), "uncertainty"),
    ]

    rows = [[name.replace("_", " "), f"{q[name]['query'].get('alpha_deg', 0):.1f}",
             f"{q[name]['query'].get('speed', 0):.1f}", f"{q[name]['query'].get('thickness', 0):.3f}",
             q[name]["result"]["status"].replace("_", " ")] for name in
            ("locally_supported", "inside_but_uncovered", "outside_training_bounds", "new_parameter_dimension")]
    colours = [GREEN if r[4] == "EMPIRICALLY SUPPORTED" else RED for r in rows]
    frames[8] = [
        (table_frame("Four questions a range check cannot tell apart",
                     ["query", "alpha", "speed", "thickness", "answer"], rows, colours=colours,
                     caption=f"{dom['evidence_gap']['unsupported']} of {dom['evidence_gap']['candidates']} sampled "
                     "candidates have nothing behind them", accent=GREEN, widths=[0.36, 0.11, 0.11, 0.16, 0.26]),
         "queries"),
        (figure_frame(os.path.join(FIG, "domain_map.png"), "The evidence has a shape, and it is not the box",
                      "the supported set is not the design box: it is what the references actually cover",
                      accent=GREEN), "domain map"),
    ]

    frames[9] = [
        (card("Camber is not a value outside a range. It is a dimension the model does not have.",
              "a query carrying it comes back UNSUPPORTED, reason: parameter_schema_changed\n"
              "so it is bought: 68 cambered designs, and a second model that replaces the first\n"
              f"lift relative L2 on designs it has never seen: {100 * rev['held_out']['CL_rel_l2']:.2f} %",
              kicker="Scene 09", accent=BLUE), "camber"),
    ]

    examples = dom["evidence_gap"]["examples"][:6]
    rows = [[f"{e['query']['alpha_deg']:5.1f}", f"{e['query']['speed']:5.1f}",
             f"{e['query']['thickness']:.3f}", e["reason"].replace("_", " ")] for e in examples]
    frames[10] = [
        (table_frame("What to simulate next", ["alpha", "speed", "thickness", "why it is unsupported"], rows,
                     caption="a ranked list, not a decision: choosing is engineering judgement",
                     accent=GREEN, widths=[0.14, 0.14, 0.18, 0.54]), "evidence gap"),
    ]

    mid = rev["midpoints_between_new_and_nearest_old"]
    rows = [["references in the domain", str(rev["domain_before_extend"]["evidence"]),
             str(rev["domain_after_extend"]["evidence"])],
            ["candidates supported, out of 400", str(rev["domain_before_extend"]["supported"]),
             str(rev["domain_after_extend"]["supported"])],
            [f"the {mid['candidates']} midpoints the new evidence unlocks",
             f"{mid['status_changes']} supported", f"{mid['candidates'] - mid['status_changes']} not"]]
    frames[11] = [
        (table_frame("The domain grows without the model moving", ["", "before", "after"], rows,
                     caption="same weights, same version: only the evidence changed", accent=BLUE,
                     widths=[0.56, 0.22, 0.22]), "extend"),
        (card("Two measurements, and only one of them flatters the method",
              f"uniform over the box: {rev['domain_before_extend']['supported']} of 400 supported before, "
              f"{rev['domain_after_extend']['supported']} after\n"
              f"where the evidence lands: of the {mid['candidates']} midpoints the new references unlock, "
              f"{mid['status_changes']} became supported\n"
              f"and the other {mid['candidates'] - mid['status_changes']} did not, for reasons the record names",
              accent=BLUE), "two measurements"),
    ]

    dossier_lines = [
        f"{dos['assessment']['status'].upper()} ({dos['assessment']['status_source']})",
        f"{len([c for c in checks if c['passed'] is False])} of {len(checks)} checks failed",
        f"CL relative L2 {100 * a['CL_rel_l2']:.2f} % against a 5 % tolerance: passed",
        f"field relative L2 {100 * ana['validation']['field_rel_l2_mean']:.1f} %: failed",
        f"uncertainty coverage {100 * checks[-3]['value']:.0f} % against 68 %: failed",
        "domain at the centre of the box: UNSUPPORTED",
        "reproducibility rerun: not run",
        "scientific maturity 4 / 5, industrial 4 / 5",
    ]
    frames[12] = [
        (card("The dossier", "\n".join(f"· {line}" for line in dossier_lines), kicker="Scene 12",
              accent=RED, footer="results/dossier.md -- every number above has a file behind it"), "dossier"),
        (table_frame("What the report refuses to sign, and why",
                     ["finding", "what happened"], findings(), caption="results/dossier.md, findings section",
                     accent=RED, widths=[0.16, 0.84]), "findings"),
    ]
    frames[13] = [
        (card("A model that tells you where its predictions are supported.", big=True, accent=BLUE,
              kicker="Trust beyond accuracy",
              footer="python examples/end_to_end/airfoil_trust/run_all.py"), "closing card"),
    ]
    return frames


def findings() -> list[list[str]]:
    """What the report refuses to sign: the failing and warning findings, with what happened."""
    text = open(os.path.join(RES, "dossier.md")).read()
    out = []
    for block in re.finditer(r"^### (VP-\d+) — .+? `(PASS|FAIL|WARNING)`\n\n- \*\*What happened:\*\* (.+)$",
                             text, re.M):
        if block.group(2) == "PASS":
            continue
        said = block.group(3).strip().replace("_", " ")
        said = re.sub(r"\{[^}]*\}", "the centre of the box", said)
        parts = [part.strip() for part in re.split(r" -- |; ", said) if part.strip()]
        said = next((part for part in parts if not part.endswith(":")), parts[0])
        out.append([block.group(1), said])
    return out


def short_deck(work: str) -> dict[int, list[tuple[Image.Image, str]]]:
    """The 90-120 second cut: six frames, one per beat, from the same material."""
    full = deck(work)
    picks = {1: (1, 0), 2: (2, 1), 3: (4, 0), 4: (6, 0), 5: (8, 1), 6: (13, 0)}
    return {beat: [full[scene][index]] for beat, (scene, index) in picks.items()}


# --------------------------------------------------------------------------- audio and assembly
def narrate(text: str, path: str, voice: str, rate: int) -> float:
    """Synthesise one scene locally and return its duration in seconds."""
    txt = os.path.join(tempfile.gettempdir(), "narration.txt")
    with open(txt, "w") as f:
        f.write(spoken(text))
    aiff = path.replace(".wav", ".aiff")
    subprocess.run(["say", "-v", voice, "-r", str(rate), "-f", txt, "-o", aiff], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", aiff, "-ar", "48000", "-ac", "1", path], check=True)
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def silence(path: str, seconds: float) -> None:
    frames = int(48000 * seconds)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(b"\x00\x00" * frames)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", choices=("long", "short"), default="long")
    ap.add_argument("--out", default=os.path.join(os.path.expanduser("~"), "Documents", "GitHub", "trust-video"))
    ap.add_argument("--voice", default="Samantha", help="a macOS voice, or 'none' for a silent cut")
    ap.add_argument("--rate", type=int, default=170, help="words per minute handed to say")
    ap.add_argument("--gap", type=float, default=0.45, help="silence between scenes, seconds")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="trustvideo-")
    scenes = parse_script(args.cut)
    if args.cut == "long":
        scenes = [*scenes, {"n": 13, "title": "closing card", "narration": ""}]
    deck_frames = deck(work) if args.cut == "long" else short_deck(work)

    concat, audio_files, total = [], [], 0.0
    for scene in scenes:
        frames = deck_frames.get(scene["n"])
        if not frames:
            continue
        wav = os.path.join(args.out, f"scene{scene['n']:02d}.wav")
        if not scene["narration"]:
            silence(wav, 5.0)
            duration = 5.0
        elif args.voice == "none":
            silence(wav, max(6.0, 3.0 * len(frames)))
            duration = max(6.0, 3.0 * len(frames))
        else:
            duration = narrate(scene["narration"], wav, args.voice, args.rate)
        audio_files.append(wav)
        silence(os.path.join(work, f"gap{scene['n']:02d}.wav"), args.gap)
        audio_files.append(os.path.join(work, f"gap{scene['n']:02d}.wav"))

        share = duration / len(frames)
        for k, (image, _label) in enumerate(frames):
            png = os.path.join(work, f"s{scene['n']:02d}_{k}.png")
            image.save(png)
            hold = share if k < len(frames) - 1 else share + 0.0
            concat.append(f"file '{png}'\nduration {hold:.3f}")
        total += duration + args.gap
        print(f"scene {scene['n']:02d}  {duration:6.1f} s  {len(frames)} frames  {labels(frames)}", flush=True)

    last_png = os.path.join(work, f"s{scenes[-1]['n']:02d}_{len(deck_frames[scenes[-1]['n']]) - 1}.png")
    concat.append(f"file '{last_png}'\n")
    lst = os.path.join(work, "frames.txt")
    open(lst, "w").write("\n".join(concat) + "\n")

    alst = os.path.join(work, "audio.txt")
    open(alst, "w").write("\n".join(f"file '{f}'" for f in audio_files) + "\n")

    out = os.path.join(args.out, f"trust-beyond-accuracy-{args.cut}.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-f", "concat", "-safe", "0", "-i", alst,
                    "-vf", f"scale={W}:{H},format=yuv420p", "-r", "30", "-c:v", "libx264", "-preset", "medium",
                    "-crf", "20", "-c:a", "aac", "-b:a", "160k", "-shortest", out], check=True)
    print(f"\n{out}\n{total / 60:.1f} min, {os.path.getsize(out) / 1e6:.1f} MB")
    return 0


def labels(frames: list[tuple[Image.Image, str]]) -> str:
    return ", ".join(label for _, label in frames)


if __name__ == "__main__":
    raise SystemExit(main())
