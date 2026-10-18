# Adding an experiment / Como adicionar um experimento

Every experiment in the repository is described by one manifest, `experiment.json`, next to its scripts or results.
The Lab curation (A–F seal and next steps), the experiments site and the Flow Studio all read it through one loader,
`pinneapple_lab.manifest.load`. You rarely write it by hand: the paths below write it for you.

## English

### 1. A new example

```bash
python -m pinneapple_lab scaffold my_study          # examples/my_study: run.py, README.md, experiment.json
python examples/my_study/run.py                     # writes the manifest, registers it in the Lab, curates
python -m pinneapple_lab seal --explain manifest/examples/my_study    # its letter and what it lacks for the next
```

`run.py` is wrapped in `pinneapple_lab.experiment(...)`:

```python
from pinneapple_lab import experiment

with experiment("my_study", title="...", domain="structural mechanics", methods=["FEM"], params={"n": 64}) as ex:
    ...
    ex.log_metric("Kt", kt, unit="", how="max von Mises / nominal stress at the hole edge")
    ex.check("kt_vs_peterson", value=kt, reference=3.03, rtol=0.03, kind="reference", label="Kt vs Peterson")
    fig.savefig("examples/my_study/results/stress.png")      # recorded automatically
    ex.add_file("examples/my_study/results/wake.mp4")       # movies, viewers, data
```

On exit it records the parameters, the git commit, the environment and the wall time, writes `experiment.json`,
imports it into the Lab as the item `manifest/<id>` (its checks are graded A–F with next steps, like any Lab run)
and re-curates. An exception inside the block is recorded (the item is graded F) and re-raised. `register=False`
writes the manifest only. The decorator form `@experiment("name", ...)` passes the capture to the function.

### 2. Other ways an experiment is captured

| path | what writes the manifest |
|---|---|
| Lab experiments (`pinneapple_lab.run`, `sweep`) | the runner writes `experiment.json` into every run folder; older run folders are read from `run.json` |
| `pinneapple_core.workflows` runs | `WorkflowRunner.run` writes `runs/<id>/experiment.json` (step outputs that are numbers become metrics) |
| `pp.solve(...)` | `solution.to_experiment("out/folder", register=True)` |
| `pp.compare(...)` | `comparison.to_experiment("out/folder", register=True)` (every metric of every method that ran) |
| Flow Studio runs | `pinneapple_lab.manifest.from_flow_run(run, out_dir)` (keys below) |
| an existing folder | `python -m pinneapple_lab register examples/<folder>` (generated from what the folder holds) |

`from_flow_run(run, out_dir)` takes `id`, `title`, `description`, `domain`, `nodes` (`[{"type": ...}]`, become the
methods), `modules`, `params`, `metrics` (`[{"name", "value", "unit", "how"}]`, measured values only; others are
dropped), `checks` (as `ctx.check` records them), `files` and `links`, and writes `out_dir/experiment.json`.

### 3. The format (`pinneapple.experiment/1`)

`schema`, `id`, `title`, `description`, `domain`, `methods`, `modules`, `params`, `scripts`, `metrics`
(`name`, `value`, `unit`, `how`), `checks`, `files` (`figures`, `videos`, `viewers`, `data`, paths relative to the
manifest's folder), `links` (`guide`, `script`, `pr`), `references`, `status` (`curated` / `not curated`), `lab`
(the Lab item it is linked to) and `provenance` (`source`, `git`, `created`, `seconds`, `environment`). See
`pinneapple_lab/manifest.py` for an example.

Rules a manifest must meet (`pinneapple_lab.manifest.validate`): every metric has a finite measured value, a unit
field (`""` for dimensionless) and says how it was measured; every listed file exists; the schema is the current one.
Nothing in a manifest is a placeholder: `register` fills only what the folder holds (title and description from the
README or the first script's docstring, modules from the imports, figures and movies, the docs page that cites the
folder, the pull requests whose merge changed it), leaves the rest empty, and marks methods found by name as
inferred. Metrics come from a Lab run when the Lab has curated the folder's scripts, else the list is empty.

### 4. The CI gate

```bash
python scripts/check_experiment_manifests.py        # also: python -m pinneapple_lab manifests --check
```

It fails when an `examples/*` folder (other than names starting with `_` or `.`, and `scratch`) has no
`experiment.json`, when a manifest lists a file that does not exist or a metric without a value, and when a Lab run
folder does not read as a valid manifest. CI runs it as a must-pass step.

### 5. Discovery

`pinneapple_lab.manifest.scan()` finds every `experiment.json` under `examples/` and every Lab run folder; a new
folder with a manifest appears with no code change. `python -m pinneapple_lab manifests` lists them.

## Português

### 1. Um exemplo novo

```bash
python -m pinneapple_lab scaffold meu_estudo         # examples/meu_estudo: run.py, README.md, experiment.json
python examples/meu_estudo/run.py                    # grava o manifesto, registra no Lab e refaz a curadoria
python -m pinneapple_lab seal --explain manifest/examples/meu_estudo    # nível e o que falta para o seguinte
```

O `run.py` usa `pinneapple_lab.experiment(...)`: `ex.log_metric(nome, valor, unit=..., how=...)` registra uma
métrica medida (com unidade e como foi medida), `ex.check(...)` uma verificação (mesmos argumentos de `ctx.check`),
e toda figura salva com `savefig` entra no manifesto; vídeos, visualizadores 3-D e dados entram com `ex.add_file`.
Ao sair, grava parâmetros, commit git, ambiente e tempo, escreve `experiment.json`, importa-o no Lab como o item
`manifest/<id>` (avaliado de A a F, com os próximos passos) e refaz a curadoria. Uma exceção dentro do bloco fica
registrada (o item recebe F) e é propagada.

### 2. Outros caminhos

Execuções do Lab (`run`, `sweep`) e de `pinneapple_core.workflows` gravam `experiment.json` na pasta da execução;
`pp.solve(...).to_experiment(pasta)` e `pp.compare(...).to_experiment(pasta)` salvam o resultado;
o Flow Studio chama `pinneapple_lab.manifest.from_flow_run(run, pasta)`; uma pasta antiga entra com
`python -m pinneapple_lab register examples/<pasta>`.

### 3. Regras

Toda métrica tem valor medido, unidade (`""` se adimensional) e diz como foi medida; todo arquivo listado existe.
O `register` só preenche o que a pasta já tem e deixa o resto vazio; nenhum número é inventado.

### 4. Verificação no CI

`python scripts/check_experiment_manifests.py` falha quando uma pasta `examples/*` não tem `experiment.json` válido,
quando um manifesto aponta para um arquivo inexistente ou uma métrica sem valor, ou quando uma execução do Lab não
é um manifesto válido. O CI roda essa verificação como etapa obrigatória.
