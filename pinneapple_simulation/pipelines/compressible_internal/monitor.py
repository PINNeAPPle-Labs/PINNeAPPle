"""A live local page for a study in progress: what is running, how far, and what it already says.

The page is :mod:`pinneapple_autolab.dashboard` — the library's dashboard, with its four areas, its
live ``/state.json`` and its replay — fed with the *session files that module already reads*:
``ledger.json`` (one entry per solve), ``events.jsonl`` (what started and what ended),
``snapshot.json`` and ``summary.json``. Nothing about the dashboard changes here; what this module
does is turn a folder of OpenFOAM cases into that session:

* each ``*/solve_*`` directory is a run: iterations done against the target, the last residual of
  every equation, whether it converged or went to ``nan``, and — for a solve that finished — the
  indicators from its ``csv/indicadores.csv``;
* the run that is still writing its log is the one shown as running, so the page says what is
  happening *now*, not what happened;
* the figures the study writes appear in the page as soon as they exist (the page is rebuilt when
  the set of PNGs changes), and the study's ``resumo.json`` becomes the summary when it lands.

    python -m pinneapple_simulation.pipelines.compressible_internal.monitor /tmp/cfd08_work --port 8130

This is a monitor, not a result: a case marked ``falhou`` or with residual ``nan`` is shown that way,
and the page never turns a run that has not finished into a finished one.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
from typing import Any

__all__ = ["scan_study", "write_session", "build", "serve", "main"]

_EQUATIONS = ("p", "Ux", "Uy", "Uz", "h", "k", "omega")


def _case_dirs(folder: str) -> list[str]:
    """Every directory under ``folder`` that holds a solver log — one run per solve."""
    out = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in ("csv", "polyMesh", "constant", "system", "_blender")]
        for f in files:
            if f.startswith("log.rhoSimpleFoam") or f.startswith("log.rhoPimpleFoam"):
                out.append(root)
                break
    return sorted(out)


def _read_log(path: str) -> dict[str, Any]:
    """Iterations, last residual per equation, and how the run ended — from the log alone."""
    iters = 0
    residuals: dict[str, float] = {}
    fatal = ""
    converged = False
    last_time = None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("Time = "):
                    iters += 1
                    last_time = line.split("=")[1].strip()
                elif "Initial residual" in line and "Solving for" in line:
                    m = re.search(r"Solving for (\w+), Initial residual = ([-+0-9.eE]+)", line)
                    if m:
                        try:
                            residuals[m.group(1)] = float(m.group(2))
                        except ValueError:
                            pass
                elif "SIMPLE solution converged" in line:
                    converged = True
                elif "FOAM FATAL" in line:
                    fatal = line.strip()
    except OSError:
        return {"iteracoes": 0, "residuos": {}, "convergiu": False, "erro": "log ausente"}
    bad = any(v != v for v in residuals.values())          # nan
    return {"iteracoes": iters, "residuos": residuals, "convergiu": converged,
            "ultimo_tempo": last_time, "nan": bad, "fatal": fatal}


def _indicators(case: str) -> dict[str, float]:
    path = os.path.join(case, "csv", "indicadores.csv")
    if not os.path.exists(path):
        return {}
    out: dict[str, float] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    out[row["grandeza"]] = float(row["valor"])
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        return {}
    return out


def _target_iterations(case: str) -> int:
    path = os.path.join(case, "system", "controlDict")
    try:
        with open(path, encoding="utf-8") as f:
            m = re.search(r"^\s*endTime\s+(\d+)\s*;", f.read(), flags=re.M)
        return int(m.group(1)) if m else 0
    except OSError:
        return 0


def _label(case: str) -> tuple[str, dict[str, Any]]:
    """A stable id and the parameters a run was started with, read off its folder name."""
    parent = os.path.basename(os.path.dirname(case)) or os.path.basename(case)
    leaf = os.path.basename(case)
    m = re.fullmatch(r"v([0-9.]+)", parent)
    params: dict[str, Any] = {"pasta": parent, "solve": leaf}
    if m:
        params["abertura_mm"] = float(m.group(1))
    return f"{parent}/{leaf}", params


def scan_study(folder: str, *, tail: int = 200) -> dict[str, Any]:
    """The session state of a study folder, in the shape the dashboard's ``live_state`` reads."""
    runs = []
    events: list[dict[str, Any]] = []
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for case in _case_dirs(folder):
        rid, params = _label(case)
        log = next((os.path.join(case, f) for f in sorted(os.listdir(case))
                    if f.startswith("log.rho")), None)
        info = _read_log(log) if log else {"iteracoes": 0, "residuos": {}, "convergiu": False}
        alvo = _target_iterations(case)
        ind = _indicators(case)
        done = bool(info.get("convergiu")) or (alvo and info["iteracoes"] >= alvo) or \
            bool(info.get("fatal")) or bool(ind)
        ok = None if not done else (not info.get("nan") and not info.get("fatal")
                                   and (info.get("convergiu") or bool(ind)))
        outputs = {k: ind.get(k) for k in ("mdot_kg_s", "p_in_bar", "dp_bar", "T_out_K", "dT_K",
                                           "Ma_max", "Kv_m3_h") if k in ind}
        outputs.update({"iteracoes": info["iteracoes"], "iteracoes_alvo": alvo,
                        "residuo_p": info["residuos"].get("p"),
                        "residuo_Ux": info["residuos"].get("Ux")})
        if ind:
            outputs.pop("residuo_p", None)
        runs.append({"id": rid, "purpose": f"abertura {params.get('abertura_mm', '?')} mm",
                     "ok": ok, "params": params, "outputs": outputs, "cost_s": None,
                     "meta": {"rodando": not done, "convergiu": bool(info.get("convergiu")),
                              "iteracoes": info["iteracoes"], "alvo": alvo,
                              "residuos": {k: info["residuos"].get(k) for k in _EQUATIONS
                                           if k in info["residuos"]},
                              "nan": bool(info.get("nan")), "fatal": info.get("fatal") or "",
                              "caso": case}})
        events.append({"time": now, "stage": "simulate", "running": not done,
                       "message": f"{rid}: {info['iteracoes']}/{alvo} iterações",
                       "params": params, "purpose": f"abertura {params.get('abertura_mm', '?')} mm"})
    running = next((r for r in runs if r["meta"]["rodando"]), None)
    finished = bool(runs) and not running
    status = {"phase": "deliver" if finished else "simulate",
              "running": {"id": running["id"], "params": running["params"], "since": now,
                          "purpose": running["purpose"]} if running else None,
              "training": False,
              "failed": [r["id"] for r in runs if r["ok"] is False],
              "finished": finished,
              "last_event": events[-1] if events else None}
    summary = None
    resumo = os.path.join(folder, "resumo.json")
    if os.path.exists(resumo):
        try:
            with open(resumo, encoding="utf-8") as f:
                summary = json.load(f)
        except (OSError, ValueError):
            summary = None
    ledger = {"runs": runs, "budget": {"planned": len(runs), "used": len(runs)},
              "environment": {"pasta": folder, "atualizado": now,
                              "solver": "rhoSimpleFoam (OpenFOAM v2606)",
                              "gás": "metano, Peng-Robinson"}}
    return {"ledger": ledger, "events": events[-tail:], "snapshot": {"atualizado": now},
            "summary": summary, "status": status}


def write_session(folder: str) -> dict[str, Any]:
    """Write the session files the dashboard's ``live_state`` reads. Returns the state."""
    state = scan_study(folder)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "ledger.json"), "w", encoding="utf-8") as f:
        json.dump(state["ledger"], f, ensure_ascii=False, indent=1)
    with open(os.path.join(folder, "snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(state["snapshot"], f, ensure_ascii=False, indent=1)
    if state["summary"]:
        with open(os.path.join(folder, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(state["summary"], f, ensure_ascii=False, indent=1)
    lines = [json.dumps(e, ensure_ascii=False) for e in state["events"]]
    with open(os.path.join(folder, "events.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return state


def _figures(folder: str) -> dict[str, str]:
    out = {}
    for root, _dirs, files in os.walk(folder):
        for f in sorted(files):
            if f.endswith(".png") and not f.startswith("frame_"):
                out[os.path.splitext(f)[0]] = os.path.join(root, f)
    return out


#: o que a página mostra: um estudo de passagem não tem design space nem arrasto, tem figuras de
#: campo e indicadores — declarados aqui para o dashboard não cair no template de campanha
_VIEWS = {
    "geo": ["geometria_3d"],
    "phy": ["corte", "perfil_axial", "render_blender", "render_blender_p_bar", "render_blender_T_K",
            "render_blender_U_m_s"],
    "labels": {"geometria_3d": "domínio 3-D contra o CAD",
               "corte": "plano meridiano: p, Ma e T",
               "perfil_axial": "perfil ao longo do eixo (p, T, Ma)",
               "render_blender": "render 3-D, número de Mach",
               "render_blender_p_bar": "render 3-D, pressão estática",
               "render_blender_T_K": "render 3-D, temperatura",
               "render_blender_U_m_s": "render 3-D, velocidade"},
    "geo_sub": "geometria: domínio construído e conferido contra o STEP",
    "phy_sub": "campos da solução",
}

_OUTPUTS = [{"key": "mdot_kg_s", "label": "vazão mássica", "unit": "kg/s"},
            {"key": "p_in_bar", "label": "pressão na entrada", "unit": "bar"},
            {"key": "dp_bar", "label": "queda de pressão", "unit": "bar"},
            {"key": "T_out_K", "label": "temperatura na saída", "unit": "K"},
            {"key": "dT_K", "label": "variação de temperatura", "unit": "K"},
            {"key": "Ma_max", "label": "número de Mach máximo", "unit": "-"},
            {"key": "Kv_m3_h", "label": "coeficiente de vazão Kv", "unit": "m³/h"}]


def build(folder: str, out_dir: str, *, title: str = "") -> str:
    """Rebuild the page with the current state and the figures that exist now."""
    from pinneapple_autolab.dashboard import build_dashboard
    write_session(folder)
    scenes = _scenes(folder, out_dir)
    keys = set(_figures(folder))
    view = dict(_VIEWS)
    view["geo"] = [k for k in _VIEWS["geo"] if k in keys]
    view["phy"] = [k for k in _VIEWS["phy"] if k in keys]
    return build_dashboard(folder, out_dir, live=True, title=title or "PINNeAPPle — simulações",
                           figures=_figures(folder), views=view, outputs=_OUTPUTS, viewer_paths=scenes)


def _scenes(folder: str, out_dir: str = "") -> dict:
    """One interactive 3-D scene per run, for the page to rotate and zoom.

    A run whose case saved several time directories becomes an **animated** scene: the page gets a
    time slider, which is the "ver segundo por segundo" the figures cannot give.
    """
    from .post import read_fields
    from .visual import build_scene_from_case

    out: list = []
    for case in _case_dirs(folder):
        try:
            # ``0`` é o campo inicial: numa série longa ele fica a dezenas de ms do resto e o
            # controle de tempo vira um salto — a animação começa no primeiro instante gravado
            tempos = sorted((d for d in os.listdir(case) if d[0].isdigit()
                             and os.path.isdir(os.path.join(case, d))), key=float)
            if len(tempos) > 3:
                tempos = [t for t in tempos if float(t) > 0.0]
            leituras = [read_fields(case, time_dir=t) for t in tempos]
            # a chave vira nome de arquivo do visualizador: sem separador de caminho
            chave = os.path.basename(os.path.dirname(case)) + "-" + os.path.basename(case)
            sc = build_scene_from_case(case, leituras, f"resultado — {chave}",
                                       times=[float(t) for t in tempos] if len(leituras) > 1 else None)
            # o dashboard mostra cenas do Studio; o Scene do twin3d converte com to_studio
            # o visualizador do próprio twin3d é o que entende tempo: ele exporta scene.json + glb
            # + fields.bin (uma série por campo) e uma página com o controle de tempo
            destino = os.path.join(out_dir or "", f"viewer_{chave}")
            sc.export(destino, with_viewer=True)
            out.append((len(tempos), chave, f"viewer_{chave}/index.html"))
        except Exception:                                  # noqa: BLE001
            continue
    # a série temporal primeiro: é a que a página abre
    out.sort(key=lambda x: -x[0])
    return {k: p for _n, k, p in out}


def serve(folder: str, out_dir: str | None = None, *, port: int = 8130, host: str = "127.0.0.1",
          title: str = "", block: bool = True, interval: float = 2.0):
    """Serve the live page: the state is re-read from the case folders every request.

    A small thread keeps ``ledger.json``/``snapshot.json``/``summary.json`` up to date and rebuilds
    the page when a new figure appears, so the tab shows the run that is writing its log right now.
    """
    out_dir = out_dir or os.path.join(folder, "painel")
    os.makedirs(out_dir, exist_ok=True)
    build(folder, out_dir, title=title)

    def writer() -> None:
        seen = set()
        while True:
            try:
                write_session(folder)
                figs = set(_figures(folder))
                if figs - seen:
                    seen = figs
                    build(folder, out_dir, title=title)
            except Exception:
                pass
            time.sleep(interval)

    threading.Thread(target=writer, daemon=True).start()
    return _servir(folder, out_dir, port=port, host=host, block=block)


def _servir(folder: str, out_dir: str, *, port: int, host: str, block: bool):
    """The page the dashboard built, with this module's state behind ``/state.json``.

    The autolab's own ``serve`` re-derives the state with ``live_state``, which counts a run as
    finished the moment its id is in the ledger — so a solve still writing its log could never be
    shown as running. The page is the library's; only the state behind it is this module's.
    """
    import http.server
    import json as _json
    import socketserver

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=out_dir, **k)

        def do_GET(self):  # noqa: N802 - http.server naming
            if self.path.split("?")[0] == "/state.json":
                body = _json.dumps(scan_study(folder), ensure_ascii=False, default=float).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def log_message(self, *a):        # keep the terminal quiet; the page is the output
            pass

    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.TCPServer((host, port), H)
    if block:
        httpd.serve_forever()
    else:
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="painel local de um estudo de passagem compressível")
    ap.add_argument("folder", help="pasta do estudo (onde ficam os casos do OpenFOAM)")
    ap.add_argument("--out-dir", default=None, help="onde escrever a página (padrão: <folder>/painel)")
    ap.add_argument("--port", type=int, default=8130)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--title", default="")
    a = ap.parse_args(argv)
    srv = serve(a.folder, a.out_dir, port=a.port, host=a.host, title=a.title, block=False)
    print(f"painel em http://{a.host}:{a.port}  (Ctrl+C para parar)", flush=True)
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        srv.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
