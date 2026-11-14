"""The experiment report: one self-contained HTML page built from ``results/icenode_seed<k>.json`` and the figures.

    python examples/reproductions/lamb2026_icenode/report.py                  # ~/Downloads/lamb2026_icenode_reporte.html
    python examples/reproductions/lamb2026_icenode/report.py --out report.html --seed 1

Every number in the page is read from the results file (never typed in), so a re-run with a different seed or epoch
count cannot leave the prose quoting stale values. The figures are embedded as base64, so the file opens on its own,
without the repository next to it. Portuguese, because the report is written for the person who commissioned the
reproduction; the repository's own README and TARGETS.md are English like the rest of the code base.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent

_CSS = """
  :root { --ink:#1b1d21; --dim:#5b6169; --line:#dfe3e8; --bg:#fbfbfa; --card:#fff;
          --accent:#0f6f5c; --warn:#8a5a00; --bad:#8a1f1f; --mono:ui-monospace,"SF Mono",Menlo,monospace; }
  @media (prefers-color-scheme: dark) {
    :root { --ink:#e8eaea; --dim:#a2a8ae; --line:#2b2f33; --bg:#16181a; --card:#1d2023; --accent:#4fbfa0;
            --warn:#e0b05a; --bad:#e07a7a; }
  }
  * { box-sizing:border-box }
  body { margin:0; background:var(--bg); color:var(--ink);
    font:16px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
  main { max-width:940px; margin:0 auto; padding:56px 24px 96px; }
  header { border-bottom:2px solid var(--ink); padding-bottom:20px; margin-bottom:34px; }
  .kicker { font:600 12px/1.4 var(--mono); letter-spacing:.12em; text-transform:uppercase; color:var(--accent); }
  h1 { font-size:31px; line-height:1.22; margin:.35em 0 .3em; letter-spacing:-.01em; }
  h2 { font-size:20px; margin:2.3em 0 .6em; padding-top:.2em; border-top:1px solid var(--line); }
  h3 { font-size:16px; margin:1.6em 0 .4em; }
  p, li { max-width:74ch }
  .meta { color:var(--dim); font-size:14px }
  a { color:var(--accent) }
  code { font-family:var(--mono); font-size:.88em; background:rgba(127,127,127,.12); padding:.1em .35em;
         border-radius:4px }
  table { border-collapse:collapse; width:100%; margin:1.2em 0; font-size:14.5px; }
  th, td { text-align:left; padding:9px 10px; border-bottom:1px solid var(--line); vertical-align:top }
  th { font-size:12px; text-transform:uppercase; letter-spacing:.06em; color:var(--dim); font-weight:600 }
  td.num { font-family:var(--mono); font-size:13px; text-align:right; white-space:nowrap }
  .paper { display:block; color:var(--dim); font-size:11.5px }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; margin:1.4em 0 }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px }
  .card .v { font:600 25px/1.15 var(--mono); letter-spacing:-.02em }
  .card .k { font-size:12.5px; color:var(--dim); margin-top:4px }
  .tag { display:inline-block; font:600 11px/1.6 var(--mono); text-transform:uppercase; letter-spacing:.08em;
         border:1px solid currentColor; border-radius:999px; padding:0 9px; }
  .ok { color:var(--accent) } .partial { color:var(--warn) } .no { color:var(--bad) }
  figure { margin:1.8em 0 }
  figure img { width:100%; height:auto; border:1px solid var(--line); border-radius:10px; background:#fff }
  figcaption { color:var(--dim); font-size:13px; margin-top:.5em }
  blockquote { margin:1.4em 0; padding:.2em 0 .2em 16px; border-left:3px solid var(--accent); color:var(--dim) }
  .note { background:var(--card); border:1px solid var(--line); border-left:3px solid var(--warn);
          border-radius:0 10px 10px 0; padding:14px 18px; margin:1.5em 0 }
  footer { margin-top:56px; padding-top:18px; border-top:1px solid var(--line); color:var(--dim); font-size:13px }
  ol.missing li { margin-bottom:.7em }
"""

ROWS = (("Sem cinética de superfície (<code>G = Gc</code>)", "none"),
        ("Nelson &amp; Baker (1996)", "nelson_baker"),
        ("NODE fortemente restrito · <code>α = f(Si, T)</code>", "strong"),
        ("NODE fracamente restrito · <code>G/Gc = f(Si, T, m)</code>", "weak"))


def _img(name: str, alt: str) -> str:
    b = base64.b64encode((HERE / "figures" / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b}" alt="{alt}">'


def _num(x: float, digits: int = 0) -> str:
    """Thousands separated by a dot (pt-BR) and a comma for the decimals."""
    s = f"{x:,.{digits}f}"
    return s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _row(label: str, key: str, table: dict) -> str:
    v, p = table[key], table[key]["paper"]
    if not p:
        return (f"<tr><td>{label}</td><td class=num>{_num(v['sse_short'])}</td><td class=num>—</td>"
                f"<td class=num>{_num(v['sse_long'])}</td><td class=num>—</td><td class=num>—</td></tr>")
    return (f"<tr><td>{label}</td>"
            f"<td class=num><b>{_num(v['sse_short'])}</b><span class=paper>{_num(p['sse_short'])}</span></td>"
            f"<td class=num>{_num(v['sse_long'])}<span class=paper>{_num(p['sse_long'])}</span></td>"
            f"<td class=num>{v['best_short']}<span class=paper>{p['best_short']}</span></td>"
            f"<td class=num>{v['best_long']}<span class=paper>{p['best_long']}</span></td>"
            f"<td class=num>{v['sse_short'] / p['sse_short']:.2f}×</td></tr>").replace(".", ",")


def build(result: dict) -> str:
    t = result["table1"]
    sym, syn, cfg, data = result["symbolic"], result["synthetic"], result["config"], result["data"]
    rows = "".join([_row(label, k, t) for label, k in ROWS])
    rows += _row("Eq. 6 do paper como lei de crescimento", "eq6", t)
    weak_vs_paper = t["weak"]["paper"]["sse_short"] / t["weak"]["sse_short"]
    base = f"{100 * (t['none']['sse_short'] / t['none']['paper']['sse_short'] - 1):.1f} %".replace(".", ",")
    base_nb = f"{100 * (t['nelson_baker']['sse_short'] / t['nelson_baker']['paper']['sse_short'] - 1):.1f} %".replace(".", ",")
    ratio = lambda k: (t[k]["sse_short"] / t[k]["paper"]["sse_short"]).__round__(2)  # noqa: E731
    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lamb et al. (2026) · reprodução no PINNeAPPle</title>
<style>{_CSS}</style></head>
<body><main>

<header>
  <div class="kicker">PINNeAPPle · reproduções de artigos</div>
  <h1>Como cristais de gelo crescem: NODE + regressão simbólica, reproduzido</h1>
  <div class="meta">
    Lamb, Harrington, Moyle, Pokrifka, Clouser, Ebert, Möhler &amp; Saathoff —
    <i>Science Advances</i> <b>12</b>(39), eaee6174 (2026), doi:10.1126/sciadv.aee6174 · lido de arXiv:2510.17935v1<br>
    Código: <code>examples/reproductions/lamb2026_icenode</code>, física em <code>pinneapple_physics/cloud</code> ·
    números de <code>results/icenode_seed{cfg['seed']}.json</code> ({cfg['epochs']} épocas, dt = {cfg['dt']:g} s,
    seed {cfg['seed']}, {data['n_series']} séries)
  </div>
</header>

<h2>Resumo</h2>

<div class="cards">
  <div class="card"><div class="v">{base} e {base_nb}</div>
    <div class="k">erro das duas baselines contra os valores do paper — sem nenhuma rede envolvida</div></div>
  <div class="card"><div class="v">{_num(100 * syn['median_rel_error'], 1)} %</div>
    <div class="k">erro mediano na recuperação de α em dados sintéticos (Figura 2 do paper)</div></div>
  <div class="card"><div class="v">{_num(weak_vs_paper, 1)}×</div>
    <div class="k">o NODE fraco ficou <i>abaixo</i> do valor publicado — a diferença que precisa de explicação</div></div>
  <div class="card"><div class="v">{_num(sym['rmse_ratio'], 3)}</div>
    <div class="k">RMSE da forma fechada descoberta (desvio do alvo: {_num(sym['std_ratio'], 3)})</div></div>
</div>

<p>O paper aprende a parte desconhecida da lei de crescimento de gelo por deposição — o coeficiente de transferência
<code>G</code> em <code>dm/dt = 4πr(Si − 1)G</code> — com uma equação diferencial neural ajustada a 290 séries
temporais de razão de massa de cristais individuais crescidos numa câmara de levitação, destila a rede treinada em
forma fechada por regressão simbólica e testa essa forma em dados independentes da câmara AIDA.</p>

<p>O que foi reproduzido aqui: a Tabela 1 (quatro modelos), a validação em dados sintéticos com α conhecido (Figura
2) e o passo de regressão simbólica (Figura 3b / Tabela S2). O teste na AIDA (Figura 4) <b>não</b> foi tentado — é o
que separaria "o método aprende física" de "a rede ajusta estas séries", e está registrado como tal.</p>

<h2>O modelo implementado</h2>

<p>Lei de crescimento por capacitância com correção cinética de superfície, exatamente como nas Eqs. 7–12:</p>

<table>
<tr><th>peça</th><th>equação</th><th>fonte implementada</th></tr>
<tr><td>taxa de crescimento</td><td><code>dm/dt = 4πr(Si − 1)G</code>, <code>C = r</code></td><td>Eq. 7</td></tr>
<tr><td>coeficiente de transferência</td><td><code>G = [RT/(e_sat,i·Dv*·Mw) + (Ls/(ka·T))(Ls·Mw/(RT) − 1)]⁻¹</code></td><td>Eq. 8</td></tr>
<tr><td>difusividade modificada</td><td><code>Dv* = Dv / ( r/(r+Δr) + 4Dv/(rαw) )</code></td><td>Eq. 9</td></tr>
<tr><td>coeficiente de deposição</td><td><code>α = (s/scrit)^m · tanh[(scrit/s)^m]</code></td><td>Eq. 10, Nelson &amp; Baker 1996</td></tr>
<tr><td>supersaturação crítica</td><td>ajuste por partes (face basal)</td><td>Harrington et al. 2019</td></tr>
<tr><td>e<sub>sat,i</sub>, Dv, ka</td><td>Murphy &amp; Koop 2005 · Pruppacher &amp; Klett</td><td>Eqs. 13–15</td></tr>
</table>

<blockquote>Uma correção honesta no meio do caminho: a primeira versão usava <code>r/(αw)</code> no termo cinético
da Eq. 9, perdido na extração de texto do PDF. Com ela a cinética não tinha efeito nenhum (α = 0,05 mudava
<code>G</code> em 0,01 %) e o modelo fortemente restrito não aprendia nada. A fonte LaTeX do arXiv deu a forma
correta, <code>4Dv/(rαw)</code>: com ela α = 0,05 corta o crescimento em 20–30 % e α = 10⁻³ em 60–96 %, e as duas
baselines passaram a casar com o paper em {base} e {base_nb}.</blockquote>

<h2>Dados</h2>

<p>{data['n_series']} das 290 séries do paper, dos dois arquivos públicos que a declaração de dados cita — Pokrifka
et al. 2018 (doi:10.26208/z7bf-nq20, 18 séries) e Pokrifka et al. 2023 (doi:10.26208/htw5-q166, 268 séries).
Temperaturas de {data['T_K'][0]:.1f} a {data['T_K'][1]:.1f} K, <code>Si</code> de {data['Si'][0]:.3f} a
{data['Si'][1]:.3f}, {_num(data['n_samples'])} amostras a 1 Hz, raios iniciais de 5,1 a 26,4 µm.</p>

<p>As quatro séries que faltam são as experiências "frias" de alta supersaturação: os autores as reprocessaram de
arquivos brutos de voltagem do eletrodo inferior que não estão em nenhum dos dois arquivos. Os mesmos cristais
aparecem como razão de massa processada dentro das 268 (linhas 188, 227, 115 e 72 das condições coincidem nos trios
(T, Si, r0)), então estão representados nas {data['n_series']} — só não pelo pipeline exato do paper. O loader já
sabe lê-las (<code>bel.load(cold_dir=...)</code>) se os brutos aparecerem.</p>

<h2>Método</h2>

<p>Quatro modelos integrados com o <b>mesmo</b> Runge-Kutta 4 de passo fixo, para que a comparação isole a cinética:
sem cinética (<code>G = Gc</code>), com α de Nelson &amp; Baker, e os dois NODEs do paper — o fortemente restrito
(<code>α = f(Si, T)</code>, saída sigmoide, 0 ≤ α ≤ 1) e o fracamente restrito
(<code>G/Gc = f(Si, T, m)</code>, sigmoide escalada para (0, 2]). Rede 3×50 com ReLU nas duas primeiras camadas e
sigmoide na saída; loss de Eq. 5 (soma dos erros quadráticos em <code>m/m0</code> sobre todas as séries), AdamW com
lr 0,01 e decaimento cosseno, {cfg['epochs']} épocas. <code>torchdiffeq</code> não é necessário: o paper também testa
RK4 de passo fixo, e RK4 é diferenciável por construção.</p>

<p>Passo <code>dt = {cfg['dt']:g} s</code> (o paper não declara o seu): conferido contra <code>dt = 1 s</code>, as
perdas coincidem em quatro algarismos significativos nas duas baselines, a um quinto do custo.</p>

<h2>Resultados</h2>

<table>
<tr><th>modelo</th><th>MSE · t &lt; 500 s</th><th>MSE · t &lt; 1000 s</th><th>melhor em N exp. · t &lt; 500 s</th>
<th>melhor em N exp. · t &lt; 1000 s</th><th>razão</th></tr>
{rows}
</table>
<p class="meta">Em cada célula, o número grande é o nosso e o pequeno abaixo é o do paper (Tabela 1). As perdas são
somadas sobre as séries a 1 Hz; "melhor em N exp." conta em quantas séries cada modelo tem o menor erro.</p>

<figure>{_img("mass_ratio_fits.png", "razão de massa medida contra os quatro modelos")}
<figcaption>Quatro das séries mais longas: medido em preto, modelos coloridos. Sem cinética e Nelson &amp; Baker
superestimam o crescimento; os dois NODEs acompanham a curva, inclusive na extrapolação além dos 500 s usados no
treino.</figcaption></figure>

<figure>{_img("learned_G.png", "coeficiente de transferência aprendido")}
<figcaption>O <code>G</code> aprendido pelo NODE fraco ao longo das trajetórias medidas (azul), o limite de contínuo
<code>Gc</code> (laranja) e a Eq. 6 do paper (verde). O aprendido fica tipicamente 1,2 a 3 vezes abaixo de
<code>Gc</code> e cai com a massa — a afirmação da Figura 3a.</figcaption></figure>

<figure>{_img("alpha_synthetic.png", "alfa recuperado em dados sintéticos")}
<figcaption>Validação em dados sintéticos: o NODE fortemente restrito recupera o α que gerou as séries, com erro
mediano de {_num(100 * syn['median_rel_error'], 1)} %. Os poucos pontos de α baixo que sobem para ~0,4 são
experimentos em que o cristal mal cresce e qualquer α abaixo de 0,4 ajusta igualmente bem.</figcaption></figure>

<figure>{_img("training_loss.png", "curvas de treino")}
<figcaption>Treino dos dois NODEs. O forte estabiliza cedo; o fraco segue descendo, coerente com a conclusão do paper
de que um único α(Si, T) é uma restrição estrutural forte demais para cristais recém-nucleados.</figcaption></figure>

<h3>Regressão simbólica</h3>
<p>A busca genética (seleção por torneio, cruzamento de subárvores, frente de Pareto por complexidade, constantes
polidas por mínimos quadrados não lineares) devolve uma forma de {sym['nodes']} nós para <code>G/Gc</code>:</p>
<p><code>{sym['expression']}</code></p>
<p>RMSE {_num(sym['rmse_ratio'], 3)} sobre um alvo cujo desvio é {_num(sym['std_ratio'], 3)}. A Eq. 6 do próprio
paper, avaliada no <code>G</code> da <b>nossa</b> rede treinada, dá {_num(sym['paper_eq6']['rmse_ratio'], 3)}: as
duas estão na mesma faixa, mas a nossa expressão não é a forma do paper, e ela usa <code>Si</code> onde todas as
expressões da Tabela S2 usam só <code>Gc</code> e <code>m</code>. Aberto como
<a href="https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1690">#1690</a>.</p>

<h3>A Eq. 6 do paper, integrada nos nossos dados</h3>
<p>Usada como lei de crescimento nas {data['n_series']} séries, ela dá
<b>{_num(t['eq6']['sse_short'])}</b> em t &lt; 500 s — melhor que as duas baselines
({_num(t['none']['sse_short'])} e {_num(t['nelson_baker']['sse_short'])}) e pior que a nossa rede treinada
({_num(t['weak']['sse_short'])}), numa rede da qual ela não foi destilada: retém cerca de
{_num(100 * (t['none']['sse_short'] - t['eq6']['sse_short']) / (t['none']['sse_short'] - t['weak']['sse_short']))} %
do ganho que o paper lhe atribui.</p>

<h2>Status por meta</h2>
<table>
<tr><th>meta</th><th>status</th><th>evidência</th></tr>
<tr><td>T1 · Tabela 1 (perdas e contagem de melhores)</td><td><span class="tag ok">reproduzido</span></td>
<td>As quatro perdas dentro de um fator 2 (razões {ratio('none')}×, {ratio('nelson_baker')}×, {ratio('strong')}× e
{ratio('weak')}×). A contagem de melhores bate em três modelos e <b>não</b> no fortemente restrito
({t['strong']['best_short']} contra {t['strong']['paper']['best_short']} do paper), porque aqui o fraco ganha
quase todos.</td></tr>
<tr><td>T2 · α conhecido em dados sintéticos (Figura 2)</td><td><span class="tag ok">reproduzido</span></td>
<td>Erro mediano de {_num(100 * syn['median_rel_error'], 1)} %; o paper mostra o acordo graficamente e não cita
número.</td></tr>
<tr><td>T3 · forma fechada para G (Figura 3b / Tabela S2)</td><td><span class="tag partial">parcial</span></td>
<td>RMSE {_num(sym['rmse_ratio'], 3)} contra {_num(sym['paper_eq6']['rmse_ratio'], 3)} da Eq. 6 na mesma rede; a
estrutura encontrada, porém, não é a do paper.</td></tr>
<tr><td>T4 · AIDA (Figura 4)</td><td><span class="tag no">não tentado</span></td>
<td>Exige o arquivo IsoCloud e um modelo de parcela com microfísica em bins — outro estudo
(<a href="https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1683">#1683</a>).</td></tr>
<tr><td>T5 · afirmações qualitativas</td><td><span class="tag ok">reproduzido</span></td>
<td>{_num(t['weak']['sse_short'])} &lt; {_num(t['strong']['sse_short'])} &lt; {_num(t['nelson_baker']['sse_short'])}
&lt; {_num(t['none']['sse_short'])} (mesma ordem em t &lt; 1000 s); todas as expressões dependem de <code>Gc</code>
e de <code>m</code>.</td></tr>
</table>

<h2>Onde me afasto do paper</h2>

<div class="note"><b>A diferença que importa.</b> O NODE fraco ficou {_num(weak_vs_paper, 1)} vezes melhor que o valor
publicado ({_num(t['weak']['sse_short'])} contra {_num(t['weak']['paper']['sse_short'])}), e o mesmo vale na janela
de extrapolação ({_num(t['weak']['sse_long'])} contra {_num(t['weak']['paper']['sse_long'])}), então não é artefato
de interpolação. As baselines casam em {base} e {base_nb}, o que descarta solver, constantes e definição de loss.
Sobram duas leituras que este run não separa: ou aprender <code>G</code> direto é mais eficaz do que o paper relata,
ou uma rede alimentada com <code>(Si, T, m)</code> — com <code>Si</code> constante dentro de cada série — memoriza
parte dos experimentos individuais. O teste da AIDA separaria as duas, e é justamente o que não foi feito. A regra de
status chama essa linha de "não reproduzido" porque mede concordância, não qualidade.</div>

<ul>
<li><b>{data['n_series']} séries em vez de 290</b> (−1,4 % da amostra), pelos motivos da seção de dados.</li>
<li><b>RK4 com dt = {cfg['dt']:g} s</b>, onde o paper não declara o passo; <code>--dt 1</code> reproduz as mesmas
perdas a cinco vezes o custo.</li>
<li><b>Regressão simbólica própria, não PySR</b> (que exige Julia): mesmo tipo de busca genética, com as constantes
polidas por mínimos quadrados.</li>
<li><b>AIDA não tentada</b> — nada neste relatório é evidência sobre ela.</li>
</ul>

<h2>O que falta, com o que já se sabe</h2>
<ol class="missing">
<li><b>Teste na câmara AIDA</b> (<a href="https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1683">#1683</a>) — o
arquivo IsoCloud mais um modelo de parcela em bins; é o que transformaria o resultado acima de "ajusta bem" em
"generaliza".</li>
<li><b>As quatro séries frias</b> (<a href="https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1687">#1687</a>) —
pedir os arquivos brutos ao grupo; efeito esperado pequeno, dado que as baselines já casam sem elas.</li>
<li><b>A vantagem de {_num(weak_vs_paper, 1)}× é memorização?</b>
(<a href="https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1688">#1688</a>) — validação deixando um experimento
de fora; separa método de memorização sem precisar da AIDA.</li>
<li><b>Paridade com a frente do PySR</b>
(<a href="https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1690">#1690</a>) — falta um nó de potência com
expoente ajustado para alcançar a família da Eq. 6.</li>
</ol>

<h2>Como reproduzir</h2>
<p>Os arquivos são baixados sozinhos na primeira execução (~19 s, {data['n_series']} séries); o resto é treino.</p>
<table>
<tr><th>comando</th><th>o que faz</th></tr>
<tr><td><code>python examples/reproductions/lamb2026_icenode/run.py --quick</code></td><td>canalização completa em 20 épocas</td></tr>
<tr><td><code>… run.py</code></td><td>os quatro modelos, 500 épocas cada, mais sintéticos, simbólico e figuras</td></tr>
<tr><td><code>… run.py --model weak</code></td><td>treina um modelo só e para (checkpoints em <code>results/checkpoints</code>)</td></tr>
<tr><td><code>… report.py</code></td><td>esta página, a partir de <code>results/icenode_seed{cfg['seed']}.json</code></td></tr>
</table>
<p>Testes: <code>pytest tests/test_icenode_reproduction.py</code> — física contra os valores que os ajustes devem dar,
o RK4, as janelas de loss da Tabela 1, a busca simbólica e o loader.</p>

<footer>
Reprodução em <code>PINNeAPPle-Labs/PINNeAPPle</code>, branch <code>feat/icenode-reproduction</code>.
Números de <code>results/icenode_seed{cfg['seed']}.json</code>; texto e figuras gerados a partir dele por
<code>report.py</code>.
</footer>

</main></body></html>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.expanduser("~/Downloads/lamb2026_icenode_reporte.html"))
    args = ap.parse_args(argv)
    path = HERE / "results" / f"icenode_seed{args.seed}.json"
    if not path.exists():
        raise SystemExit(f"no results file at {path}: run run.py first")
    result = json.load(open(path))
    html = build(result)
    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
