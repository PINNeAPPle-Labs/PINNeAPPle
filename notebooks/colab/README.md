# Colab notebooks

Six self-contained notebooks that run top to bottom on a free Google Colab GPU runtime (Runtime > Change runtime
type > T4 GPU), or on a CPU (slower). The first cell installs PINNeAPPle from PyPI (`pinneapple>=0.7.0`) and picks the
GPU when there is one; the parameter cell (`SMOKE`, epochs, grid sizes) sets how big each run is.

| Notebook | What it shows | Open |
|---|---|---|
| [01 First PINN: heat in a plate](01_first_pinn_heat.ipynb) | a Laplace problem solved with `pp.solve` by a PINN and by FEM, then its trust card: PDE residual, boundary conditions, conservation, maximum principle, reference | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PINNeAPPle-Labs/PINNeAPPle/blob/main/notebooks/colab/01_first_pinn_heat.ipynb) |
| [02 Airfoil: geometry, surrogate, validation](02_airfoil_surrogate.ipynb) | CST airfoil geometry, Cl / Cd / Cm from a trained ensemble with its spread, validated against stored OpenFOAM runs, XFOIL and thin-airfoil theory | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PINNeAPPle-Labs/PINNeAPPle/blob/main/notebooks/colab/02_airfoil_surrogate.ipynb) |
| [03 Equal-budget benchmark](03_equal_budget_benchmark.ipynb) | three PINN architectures and FEM with the same epochs, points and seeds; error against budget; trust card of the winners; `pp.benchmark` when installed | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PINNeAPPle-Labs/PINNeAPPle/blob/main/notebooks/colab/03_equal_budget_benchmark.ipynb) |
| [04 Lunar rover terramechanics + dashboard](04_lunar_rover_dashboard.ipynb) | NASA VIPER on lunar regolith (Wong-Reece wheels, multibody, deformable terrain), the testbed validation, a PINN of the wheel-soil contact and the animated dashboard as a GIF | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PINNeAPPle-Labs/PINNeAPPle/blob/main/notebooks/colab/04_lunar_rover_dashboard.ipynb) |
| [05 Verify a model](05_verify_a_model.ipynb) | the trust card catching an under-trained PINN, a harmonic field with the wrong boundary, and extrapolation, without a reference solution | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PINNeAPPle-Labs/PINNeAPPle/blob/main/notebooks/colab/05_verify_a_model.ipynb) |
| [06 Thirteen recipes](06_quickstart_ten_recipes.ipynb) | the notebook version of [`GETTING_STARTED.md`](../../GETTING_STARTED.md): install, solve and compare, the trust card, your own problem and data, an inverse problem, grid prediction and export, an architecture benchmark, uncertainty and validation, an industrial case, a digital twin, visualisation (1-D/2-D/3-D/4-D), a sentence to a live twin, and geometry variants to a verified inverse model — each cell executed, outputs kept | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PINNeAPPle-Labs/PINNeAPPle/blob/main/notebooks/colab/06_quickstart_ten_recipes.ipynb) |

Files the notebooks read (the airfoil model and CFD results, the trust-card module on releases older than the one
that ships `pp.trust_card`) come from a local checkout when there is one, otherwise from the repository on GitHub.

## Running them headless

```bash
python ../../scripts/run_colab_notebooks.py --smoke            # every notebook with tiny sizes (what CI runs)
python ../../scripts/run_colab_notebooks.py 02                 # one notebook at full size
python ../../scripts/run_colab_notebooks.py --check-clean      # committed notebooks must carry no outputs
```

The committed notebooks have their outputs cleared; `--clear` strips them after editing.

A notebook can opt out of that, with `metadata.pinneapple.keep_outputs = true`: `06_quickstart_ten_recipes.ipynb`
is the deliverable as a *tour*, so the printed numbers are the point and they stay. `--check-clean` accepts it and
says so; `--clear` leaves it alone. Re-execute it after editing:

```bash
python ../../scripts/run_colab_notebooks.py 06 --out /tmp/nb && \
  cp /tmp/nb/06_quickstart_ten_recipes.ipynb 06_quickstart_ten_recipes.ipynb
```

## Coming next

The airfoil and rover notebooks use the model zoo (`pp.zoo`) when it is installed, and fall back to the repository
files or to training on PINNeAPPle 0.7.0. The equal-budget benchmark API (`pp.benchmark`) and the geometry library
(`pp.geometries`) are being merged: the benchmark notebook already has the cell that switches to `pp.benchmark`
(marked **Switch to `pp.benchmark`**) and runs it only when the API is installed.
