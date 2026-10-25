from __future__ import annotations

from typing import Any, Dict

import numpy as np

from pinneapple_tools.benchmark_suite.bundle.loader import BundleData
from pinneapple_tools.benchmark_suite.registry import register_backend


@register_backend
class PhysicsNeMoSymBackend:
    """
    Backend integration for PhysicsNeMo Sym (formerly Modulus Sym).

    This backend builds a Navier-Stokes PINN using PhysicsNeMo Sym's
    domain, constraint, and solver abstractions. It relies entirely
    on the PhysicsNeMo internal training loop and does not guarantee
    returning a PyTorch model object.
    """

    name = "physicsnemo_sym"

    def train(self, bundle: BundleData, run_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train a steady 2D Navier-Stokes PINN using PhysicsNeMo Sym.

        Parameters
        ----------
        bundle : BundleData
            Data bundle containing collocation points, boundary points,
            and physical parameters (e.g., viscosity `nu`).
        run_cfg : Dict[str, Any]
            Configuration dictionary containing:
                - train parameters (device, max_steps, lr, weights)
                - arena sampling parameters
                - model architecture parameters

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
                - device used
                - minimal training metadata (nu, max_steps, lr)

        Notes
        -----
        This backend requires `physicsnemo.sym` to be installed.
        Training logs and detailed metrics depend on the PhysicsNeMo
        configuration and logging setup.
        """
        train_cfg = dict(run_cfg.get("train", {}))
        arena_cfg = dict(run_cfg.get("arena", {}))
        model_cfg = dict(run_cfg.get("model", {}))

        device = str(train_cfg.get("device", "cuda"))
        max_steps = int(train_cfg.get("max_steps", 20000))
        lr = float(train_cfg.get("lr", 1e-3))
        weights = dict(train_cfg.get("weights", {}))
        w_pde = float(weights.get("pde", 1.0))
        w_bc = float(weights.get("bc", 10.0))

        n_collocation = int(arena_cfg.get("n_collocation", 4096))
        n_boundary = int(arena_cfg.get("n_boundary", 2048))

        nr_layers = int(model_cfg.get("nr_layers", 6))
        layer_size = int(model_cfg.get("layer_size", 256))
        activation = str(model_cfg.get("activation", "tanh"))

        nu = float(bundle.manifest["nu"])

        try:
            # nvidia-physicsnemo.sym 2.4.0 API (ModulusConfig no longer exists; configs are composed with hydra)
            import torch
            from physicsnemo.sym.domain import Domain
            from physicsnemo.sym.domain.constraint import PointwiseConstraint
            from physicsnemo.sym.eq.pdes.navier_stokes import NavierStokes
            from physicsnemo.sym.hydra.utils import compose
            from physicsnemo.sym.key import Key
            from physicsnemo.sym.models.activation import Activation
            from physicsnemo.sym.models.fully_connected import FullyConnectedArch
            from physicsnemo.sym.solver import Solver
        except ImportError as e:
            raise ImportError(
                "the PhysicsNeMo-Sym backend needs nvidia-physicsnemo.sym (tested with 2.4.0): "
                "pip install nvidia-physicsnemo.sym (it builds a CUDA extension; see "
                "benchmarks/vs_frameworks/REPORT.md for a CPU-only install). "
                f"Import error: {type(e).__name__}: {e}"
            ) from e

        # sample
        rng = np.random.default_rng(0)

        def _sample_df(df, n):
            """
            Randomly sample n rows from a dataframe without replacement.

            If n exceeds dataframe length, sampling is clipped to
            the available number of rows.
            """
            n = min(int(n), len(df))
            if n <= 0:
                return df.iloc[:0]
            idx = rng.integers(0, len(df), size=n, endpoint=False)
            return df.iloc[idx]

        col = _sample_df(bundle.points_collocation, n_collocation)
        bnd = _sample_df(bundle.points_boundary, n_boundary)

        input_keys = [Key("x"), Key("y")]
        output_keys = [Key("u"), Key("v"), Key("p")]

        ns = NavierStokes(nu=nu, rho=1.0, dim=2, time=False)

        net = FullyConnectedArch(
            input_keys=input_keys,
            output_keys=output_keys,
            nr_layers=nr_layers,
            layer_size=layer_size,
            activation_fn=Activation[activation.upper()],
        )

        nodes = ns.make_nodes() + [net.make_node(name="flow_net")]

        domain = Domain()

        def _pointwise(*, nodes, invar, outvar, batch_size, lambda_weighting):
            """PointwiseConstraint.from_numpy with float32 arrays and per-point weights (2.4.0 indexes every weight
            per point, so a scalar weight is broadcast)."""
            n = len(next(iter(invar.values())))
            return PointwiseConstraint.from_numpy(
                nodes=nodes, invar={k: np.asarray(v, dtype=np.float32) for k, v in invar.items()},
                outvar={k: np.asarray(v, dtype=np.float32) for k, v in outvar.items()}, batch_size=batch_size,
                lambda_weighting={k: np.full((n, 1), float(w), dtype=np.float32) for k, w in lambda_weighting.items()})

        # interior constraint
        invar_interior = {"x": col["x"].to_numpy().reshape(-1, 1), "y": col["y"].to_numpy().reshape(-1, 1)}
        outvar_interior = {
            "continuity": np.zeros((len(col), 1), dtype=np.float32),
            "momentum_x": np.zeros((len(col), 1), dtype=np.float32),
            "momentum_y": np.zeros((len(col), 1), dtype=np.float32),
        }
        interior = _pointwise(
            nodes=nodes,
            invar=invar_interior,
            outvar=outvar_interior,
            batch_size=min(len(col), n_collocation),
            lambda_weighting={
                "continuity": w_pde,
                "momentum_x": w_pde,
                "momentum_y": w_pde,
            },
        )
        domain.add_constraint(interior, "interior")

        # boundary constraints helper
        def _mask(region: str):
            """
            Filter boundary dataframe by region label.
            """
            return bnd[bnd["region"].astype(str) == region]

        # inlet u=1 v=0
        inlet = _mask("inlet")
        if len(inlet) > 0:
            invar = {"x": inlet["x"].to_numpy().reshape(-1, 1), "y": inlet["y"].to_numpy().reshape(-1, 1)}
            outvar = {"u": np.ones((len(inlet), 1), dtype=np.float32), "v": np.zeros((len(inlet), 1), dtype=np.float32)}
            c = _pointwise(
                nodes=nodes,
                invar=invar,
                outvar=outvar,
                batch_size=min(len(inlet), n_boundary),
                lambda_weighting={"u": w_bc, "v": w_bc},
            )
            domain.add_constraint(c, "bc_inlet")

        # walls, obstacle no-slip
        for reg, name in (("walls", "bc_walls"), ("obstacle", "bc_obstacle")):
            df = _mask(reg)
            if len(df) > 0:
                invar = {"x": df["x"].to_numpy().reshape(-1, 1), "y": df["y"].to_numpy().reshape(-1, 1)}
                outvar = {"u": np.zeros((len(df), 1), dtype=np.float32), "v": np.zeros((len(df), 1), dtype=np.float32)}
                c = _pointwise(
                    nodes=nodes,
                    invar=invar,
                    outvar=outvar,
                    batch_size=min(len(df), n_boundary),
                    lambda_weighting={"u": w_bc, "v": w_bc},
                )
                domain.add_constraint(c, name)

        # outlet p=0
        outlet = _mask("outlet")
        if len(outlet) > 0:
            invar = {"x": outlet["x"].to_numpy().reshape(-1, 1), "y": outlet["y"].to_numpy().reshape(-1, 1)}
            outvar = {"p": np.zeros((len(outlet), 1), dtype=np.float32)}
            c = _pointwise(
                nodes=nodes,
                invar=invar,
                outvar=outvar,
                batch_size=min(len(outlet), n_boundary),
                lambda_weighting={"p": w_bc},
            )
            domain.add_constraint(c, "bc_outlet")

        # config: PhysicsNeMo-Sym's registered default plus the groups its examples select
        import tempfile
        out_dir = str(run_cfg.get("output_dir") or tempfile.mkdtemp(prefix="physicsnemo_sym_"))
        overrides = ["+optimizer=adam", "+scheduler=tf_exponential_lr", "+loss=sum", f"optimizer.lr={lr}",
                     f"training.max_steps={max_steps}", f"network_dir={out_dir}", f"device={device}",
                     "training.rec_results_freq=1000000", "training.rec_constraint_freq=1000000",
                     "training.save_network_freq=1000000", "training.summary_freq=1000000"]
        if not device.startswith("cuda"):
            overrides.append("cuda_graphs=false")      # the default (true) crashes without a GPU driver
        cfg = compose(config_name="physicsnemo_default", config_path=None, overrides=overrides)

        solver = Solver(cfg, domain)
        solver.solve()

        net.eval()
        p0 = next(net.parameters())

        def predict_fn(X: np.ndarray) -> np.ndarray:
            X = np.asarray(X, dtype=np.float32)
            with torch.no_grad():
                inv = {k: torch.as_tensor(X[:, i:i + 1], device=p0.device) for i, k in enumerate(("x", "y"))}
                out = net(inv)
            return np.concatenate([out[k].detach().cpu().numpy() for k in ("u", "v", "p")], axis=1)

        return {
            "device": device,
            "model": net,
            "predict_fn": predict_fn,
            "metrics": {
                "nu": float(nu),
                "max_steps": float(max_steps),
                "lr": float(lr),
            },
        }
