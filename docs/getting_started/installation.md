# Installation

## What you need

- Python 3.10 or newer.
- Nothing else to get started: `numpy`, `scipy`, `torch`, `sympy` and `matplotlib` come with the
  package.

## From PyPI

```bash
pip install pinneapple
```

Then check the machine, not just the import:

```bash
pinneapple doctor start
```

That prints one line per thing it checked and, for each missing item, the command that fixes it.

## Profiles: install for the work you intend to do

Instead of picking among the small extras, install the bundle that matches your work:

| profile | `pip install ...` | for |
|---|---|---|
| `start` | `pinneapple` | the first week: solve a preset, plot a field, train one small network |
| `cfd` | `pinneapple[cfd]` | geometry, meshing, 3-D viewing; OpenFOAM and CalculiX are separate installs |
| `research` | `pinneapple[pinn,uq,viz3d,jax_cfd]` | PINNs, uncertainty, JAX, publication figures |
| `enterprise` | `pinneapple[enterprise]` | serving, signatures, the CI linter, the hub, the MCP server |
| everything | `pinneapple[all]` | every extra the package declares |

`pinneapple doctor <profile>` checks exactly the set above, and `pinneapple doctor all` lists every
extra with the ones missing here.

Two things the Python package cannot install for you, and `doctor cfd` reports:

- **OpenFOAM** for the CFD campaigns: install it (or the ESI build) and source its environment, so
  `blockMesh` and `simpleFoam` are on `PATH`.
- **FEniCS** for the 18 presets whose reference solver is `fenics`: `pip install "pinneapple[fenics]"`.
  Without it those presets train a network instead, and `pp.card(name)` says so.

## From a checkout, for development

```bash
git clone https://github.com/barrosyan/PINNeAPPle.git
cd PINNeAPPle
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -U pip
pip install -e ".[dev]"
```

Verify the checkout:

```bash
python -c "import pinneapple as pp; pp.info()"
pytest tests/test_public_api.py tests/test_onboarding.py -q
```

## Colab

The notebooks in [`notebooks/colab/`](../../notebooks/colab/README.md) install the package in their
first cell; open one from the badge in the README.
