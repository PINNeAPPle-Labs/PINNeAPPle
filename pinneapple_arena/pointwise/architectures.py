"""
Architecture catalog of the PINNeAPPle Arena: ``ARCHITECTURES_CATALOG: dict[str, ArchitectureInfo]``, descriptions
of the PINNeAPPle registry models the arena can select (``pp.list_models()`` / ``pp.build_model()``).

The catalog id is the registry key, except for nine entries renamed in the arena (``REGISTRY_KEYS``:
``symm_*`` -> ``noether_*``, ``icosahedral_gnn`` -> ``graphcast``); ``registry_key(arch_id)`` gives the key to
build. Whether an entry really builds for pointwise data is not a property of the catalog: ``catalog_status()``
asks the registry, and ``pinneapple_arena.pointwise.training.instantiate_model`` reports the ones that do not.
"""
from __future__ import annotations

from pinneapple_arena.pointwise.schema import ArchitectureFamily, ArchitectureInfo

# ===========================================================================
# PINN family
# ===========================================================================

vanilla_pinn = ArchitectureInfo(
    id="vanilla_pinn",
    name="Vanilla PINN",
    family=ArchitectureFamily.PINN,
    description=(
        "Standard Physics-Informed Neural Network: a fully-connected MLP whose loss "
        "combines PDE residuals, boundary conditions, and initial conditions evaluated "
        "via automatic differentiation. The canonical baseline for all PINN comparisons."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Elliptic PDEs (Laplace, Poisson)", "Parabolic PDEs (heat equation)", "Burgers / wave / KdV", "Algorithm benchmarking"],
    parameters={"hidden_dim": 128, "n_layers": 4, "activation": "tanh", "in_dim": 2, "out_dim": 1},
    paper_reference="Raissi, Perdikaris & Karniadakis (2019) — Journal of Computational Physics 378, 686-707.",
)

inverse_pinn = ArchitectureInfo(
    id="inverse_pinn",
    name="Inverse PINN",
    family=ArchitectureFamily.PINN,
    description=(
        "PINN extended for inverse problems: unknown PDE parameters (e.g., viscosity, "
        "diffusivity) are learnable nn.Parameters trained jointly with the network "
        "weights using sparse measurement data plus physics residual constraints."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    inverse_only=True,
    best_for=["Parameter identification from sparse data", "Coefficient estimation", "Data assimilation", "System identification"],
    parameters={"hidden_dim": 128, "n_layers": 5, "activation": "tanh"},
    paper_reference="Raissi, Perdikaris & Karniadakis (2019) — Journal of Computational Physics 378, 686-707.",
)

pielm = ArchitectureInfo(
    id="pielm",
    name="PIELM",
    family=ArchitectureFamily.PINN,
    description=(
        "Physics-Informed Extreme Learning Machine: output-layer weights are computed "
        "analytically via least squares — no back-propagation through hidden layers — "
        "giving dramatically faster training than full-gradient PINNs."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Rapid solution screening", "Smooth elliptic PDEs", "Laplace / Poisson / linear elasticity", "Real-time training"],
    parameters={"hidden_dim": 512, "n_layers": 1, "activation": "tanh", "solver": "lstsq"},
    paper_reference="Dwivedi & Srinivasan (2020) — Neurocomputing 391, 96-118.",
)

pinn_lstm = ArchitectureInfo(
    id="pinn_lstm",
    name="PINN-LSTM",
    family=ArchitectureFamily.PINN,
    description=(
        "PINN variant where spatial MLP layers are replaced by LSTM units, enabling "
        "long-range temporal dependencies without explicit time-marching. Processes the "
        "time coordinate as a sequence, improving accuracy for strongly time-dependent PDEs."
    ),
    supports_physics_loss=True,
    supports_3d=False,
    best_for=["Time-dependent PDEs over long horizons", "Burgers / wave / heat equations", "Chaotic dynamical systems", "Sequential physics problems"],
    parameters={"hidden_dim": 128, "n_layers": 2, "lstm_hidden": 128, "dropout": 0.0},
    paper_reference="Heydari, Thompson & Mehmood (2019) — arXiv:1912.12355.",
)

pinnsformer = ArchitectureInfo(
    id="pinnsformer",
    name="PINNsFormer",
    family=ArchitectureFamily.PINN,
    description=(
        "Transformer-based PINN encoding (x, t) coordinate tokens via multi-head "
        "self-attention. Global correlations across the spatio-temporal domain are "
        "captured without positional inductive biases."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Multi-scale spatio-temporal problems", "Navier-Stokes with complex flow patterns", "Problems requiring global spatial context", "High-dimensional PDEs"],
    parameters={"d_model": 128, "n_heads": 4, "n_layers": 4, "ffn_dim": 256, "dropout": 0.1, "activation": "gelu"},
    paper_reference="Zheng et al. (2023) — arXiv:2307.11833.",
)

vpinn = ArchitectureInfo(
    id="vpinn",
    name="VPINN",
    family=ArchitectureFamily.PINN,
    description=(
        "Variational PINN reformulates the PDE residual in weak (variational) form "
        "using test functions, relaxing smoothness requirements. The weak-form loss is "
        "computed via numerical quadrature over sub-domains."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Problems with discontinuities or sharp gradients", "Advection-dominated problems", "Elasticity (variational principle is natural)", "Galerkin formulation domains"],
    parameters={"hidden_dim": 128, "n_layers": 4, "activation": "tanh", "n_test_functions": 10, "quadrature_order": 5},
    paper_reference="Kharazmi, Zhang & Karniadakis (2021) — CMAME 374, 113547.",
)

xpinn = ArchitectureInfo(
    id="xpinn",
    name="XPINN",
    family=ArchitectureFamily.PINN,
    description=(
        "Extended PINN with domain decomposition: the domain is partitioned into "
        "non-overlapping sub-domains each handled by a separate sub-network. Interface "
        "continuity conditions are enforced explicitly. Enables parallelism and handles "
        "complex multi-region domains."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Complex multi-region domains", "Problems with material interfaces", "Large-scale parallelizable simulations", "Multi-physics coupling"],
    parameters={"hidden_dim": 64, "n_layers": 3, "activation": "tanh", "n_subdomains": 4, "interface_weight": 100.0},
    paper_reference="Jagtap & Karniadakis (2020) — CiCP 28(5), 2002-2041.",
)

xtfc = ArchitectureInfo(
    id="xtfc",
    name="X-TFC",
    family=ArchitectureFamily.PINN,
    description=(
        "Extreme Theory of Functional Connections: combines the TFC constrained "
        "expression framework with extreme learning machine efficiency. Boundary and "
        "initial conditions are satisfied by construction via constrained expressions, "
        "and the free parameters are solved analytically."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["BVPs with hard BC enforcement", "Fast PDE solving with TFC structure", "Smooth ODEs / PDEs", "Real-time embedded solving"],
    parameters={"hidden_dim": 100, "n_layers": 1, "activation": "tanh", "solver": "lstsq"},
    paper_reference="De Florio, Schiassi & Furfaro (2022) — arXiv:2205.14012.",
)

modified_mlp = ArchitectureInfo(
    id="modified_mlp",
    name="Modified MLP",
    family=ArchitectureFamily.PINN,
    description=(
        "Enhanced MLP with Random Fourier Feature (RFF) input embedding and "
        "highway U/V gating connections between layers (Wang et al. 2022) to mitigate "
        "gradient pathologies. Significantly outperforms vanilla PINN on high-frequency "
        "and multi-scale problems."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["High-frequency PDEs (Helmholtz, wave)", "Multi-scale problems", "Navier-Stokes at moderate Re", "Problems where vanilla PINN fails"],
    parameters={"hidden_dim": 128, "n_layers": 5, "activation": "tanh", "n_fourier": 64, "sigma_fourier": 1.0},
    paper_reference="Wang, Wang & Perdikaris (2022) — Journal of Scientific Computing 92(2).",
)


# ===========================================================================
# Neural Operator family
# ===========================================================================

fno = ArchitectureInfo(
    id="fno",
    name="Fourier Neural Operator",
    family=ArchitectureFamily.NEURAL_OPERATOR,
    description=(
        "FNO learns mappings between function spaces using global spectral convolutions "
        "in Fourier space. A learned linear transform is applied to low-frequency Fourier "
        "modes, enabling resolution-invariant operator learning. State-of-the-art for "
        "Navier-Stokes and turbulence operator learning."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Navier-Stokes operator learning", "Climate/weather prediction", "Periodic or quasi-periodic domains", "Multi-resolution PDE families"],
    parameters={"in_channels": 1, "out_channels": 1, "hidden_channels": 64, "n_fourier_layers": 4, "modes": 16, "activation": "gelu"},
    paper_reference="Li et al. (2021) — ICLR 2021.",
)

gno = ArchitectureInfo(
    id="gno",
    name="Graph Neural Operator",
    family=ArchitectureFamily.NEURAL_OPERATOR,
    description=(
        "Graph Neural Operator generalises neural operators to non-uniform and unstructured "
        "discretisations by representing the domain as a graph. Kernel integration is "
        "approximated via message passing on the graph structure."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Unstructured mesh operator learning", "Irregular domain PDEs", "Non-uniform point clouds", "Geometry-varying PDE families"],
    parameters={"in_channels": 1, "out_channels": 1, "hidden_dim": 64, "n_layers": 4},
    paper_reference="Li et al. (2020) — arXiv:2003.03485.",
)

deeponet = ArchitectureInfo(
    id="deeponet",
    name="DeepONet",
    family=ArchitectureFamily.NEURAL_OPERATOR,
    description=(
        "Deep Operator Network learns a mapping between function spaces with a branch "
        "network (encodes input functions) and a trunk network (encodes query locations). "
        "Once trained, evaluates the operator at arbitrary query points in a single "
        "forward pass."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Parametric PDE families", "Operator learning from data", "Fast many-query applications (design/UQ)", "Solution operators for ODEs/PDEs"],
    parameters={"branch_hidden": [128, 128, 128], "trunk_hidden": [128, 128, 128], "p": 128, "activation": "relu"},
    paper_reference="Lu et al. (2021) — Nature Machine Intelligence 3, 218-229.",
)

multiscale_deeponet = ArchitectureInfo(
    id="multiscale_deeponet",
    name="Multiscale DeepONet",
    family=ArchitectureFamily.NEURAL_OPERATOR,
    description=(
        "Extends DeepONet with multiple branch networks operating at different "
        "spatial/temporal scales, enabling accurate representation of multi-scale physics. "
        "Outputs from each scale branch are combined via learnable aggregation."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Multi-scale fluid dynamics", "Turbulence with energy cascade", "Composite material homogenisation", "Multi-resolution geophysical problems"],
    parameters={"branch_networks": 3, "scale_factors": [1, 4, 16], "p": 128, "trunk_hidden": [128, 128, 128]},
    paper_reference="Jin, Meng & Lu (2022) — SIAM J. Scientific Computing 44(6).",
)

pino = ArchitectureInfo(
    id="pino",
    name="Physics-Informed Neural Operator",
    family=ArchitectureFamily.NEURAL_OPERATOR,
    description=(
        "Physics-Informed Neural Operator (PINO) combines FNO-style operator learning "
        "with PINN-style PDE residual constraints during training. Can be trained on "
        "data plus physics, achieving better generalisation than either approach alone."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Navier-Stokes with physics supervision", "Operator learning with limited data", "Turbulence modelling with physics constraints", "Scientific ML for PDEs"],
    parameters={"in_channels": 1, "out_channels": 1, "hidden_channels": 64, "modes": 16, "physics_weight": 1.0},
    paper_reference="Li et al. (2021) — arXiv:2111.03794.",
)

uno = ArchitectureInfo(
    id="uno",
    name="U-NO (Universal Neural Operator)",
    family=ArchitectureFamily.NEURAL_OPERATOR,
    description=(
        "Universal Neural Operator combines FNO spectral convolutions with a U-Net "
        "encoder-decoder architecture. Skip connections preserve fine-scale information "
        "while the bottleneck captures global structure."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Multi-scale PDE operator learning", "Image-based physics simulation", "Turbulent flow super-resolution", "Encoder-decoder physics problems"],
    parameters={"in_channels": 1, "out_channels": 1, "hidden_channels": 64, "modes": 16, "n_scales": 3},
    paper_reference="Rahman, Ross & Azizzadenesheli (2022) — arXiv:2204.11127.",
)

ts_fno = ArchitectureInfo(
    id="ts_fno",
    name="Time-Stepping FNO",
    family=ArchitectureFamily.NEURAL_OPERATOR,
    description=(
        "Time-stepping variant of FNO that auto-regressively advances the solution "
        "forward in time. Each step uses a full FNO to map the current field to the "
        "next, enabling long rollouts for unsteady PDE problems."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Unsteady fluid dynamics roll-outs", "Turbulence forecasting", "Climate emulation", "Time-series operator learning"],
    parameters={"in_channels": 1, "out_channels": 1, "hidden_channels": 64, "modes": 16, "n_steps": 10},
    paper_reference="Li et al. (2022) — arXiv:2202.11214.",
)

afno = ArchitectureInfo(
    id="afno",
    name="Adaptive FNO",
    family=ArchitectureFamily.NEURAL_OPERATOR,
    description=(
        "Adaptive Fourier Neural Operator replaces fixed-frequency Fourier layers with "
        "an attention mechanism over Fourier modes. This allows the network to adaptively "
        "weight different spectral components depending on the input function."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Adaptive spectral feature learning", "Turbulence with non-stationary spectra", "Weather modelling", "PDEs with irregular frequency content"],
    parameters={"in_channels": 1, "out_channels": 1, "hidden_channels": 64, "modes": 16, "n_heads": 8},
    paper_reference="Guibas et al. (2021) — arXiv:2111.13587.",
)


# ===========================================================================
# Transformer family
# ===========================================================================

transformer = ArchitectureInfo(
    id="transformer",
    name="Transformer",
    family=ArchitectureFamily.TRANSFORMER,
    description=(
        "Standard encoder-only Transformer applied to physics time-series. Tokenises "
        "spatial or temporal snapshots and uses multi-head self-attention to capture "
        "long-range dependencies in PDE solutions."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Sequence-to-sequence PDE forecasting", "Multi-variate time-series physics", "Long-range dependencies in simulation data", "Attention-based surrogate modelling"],
    parameters={"d_model": 128, "n_heads": 8, "n_layers": 6, "ffn_dim": 512, "dropout": 0.1},
    paper_reference="Vaswani et al. (2017) — NeurIPS 2017.",
)

informer = ArchitectureInfo(
    id="informer",
    name="Informer",
    family=ArchitectureFamily.TRANSFORMER,
    description=(
        "Informer introduces ProbSparse attention and a distilling operation to reduce "
        "the O(n²) cost of full self-attention to O(n log n), enabling efficient "
        "long-sequence time-series forecasting for physics applications."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Long-horizon time-series forecasting", "High-frequency sensor data", "Efficient attention for long sequences", "Physics-informed signal processing"],
    parameters={"d_model": 512, "n_heads": 8, "n_layers": 3, "ffn_dim": 2048, "dropout": 0.05, "attn": "prob"},
    paper_reference="Zhou et al. (2021) — AAAI 2021.",
)

autoformer = ArchitectureInfo(
    id="autoformer",
    name="Autoformer",
    family=ArchitectureFamily.TRANSFORMER,
    description=(
        "Autoformer decomposes time-series into trend and seasonality components inside "
        "the Transformer architecture using an auto-correlation mechanism. Excels at "
        "periodic physics signals such as oscillatory PDEs or seasonal climate data."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Periodic / seasonal physics signals", "Long-term weather and climate forecasting", "Oscillatory ODE solutions", "Trend + seasonality decomposition"],
    parameters={"d_model": 512, "n_heads": 8, "n_layers": 3, "moving_avg": 25},
    paper_reference="Wu et al. (2021) — NeurIPS 2021.",
)

fedformer = ArchitectureInfo(
    id="fedformer",
    name="FEDformer",
    family=ArchitectureFamily.TRANSFORMER,
    description=(
        "Frequency-Enhanced Decomposed Transformer applies Fourier or wavelet frequency "
        "selection inside the attention mechanism, combining spectral methods with "
        "Transformer sequence modelling for physics time-series."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Physics signals with dominant frequency components", "Seasonal climate modelling", "Spectral-aware surrogate modelling", "Long-range sequence prediction"],
    parameters={"d_model": 512, "n_heads": 8, "n_layers": 3, "modes": 64, "version": "Fourier"},
    paper_reference="Zhou et al. (2022) — ICML 2022.",
)

tft = ArchitectureInfo(
    id="tft",
    name="Temporal Fusion Transformer",
    family=ArchitectureFamily.TRANSFORMER,
    description=(
        "Temporal Fusion Transformer combines LSTM encoder with multi-head attention "
        "for interpretable multi-horizon forecasting. Variable selection networks "
        "automatically select relevant physical inputs at each time step."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Interpretable multi-step physics forecasting", "Multi-variate sensor fusion", "Physics with known exogenous inputs", "Production-ready time-series ML"],
    parameters={"d_model": 64, "n_heads": 4, "n_layers": 2, "hidden_continuous_size": 8, "dropout": 0.1},
    paper_reference="Lim et al. (2021) — International Journal of Forecasting 37(4).",
)

timesnet = ArchitectureInfo(
    id="timesnet",
    name="TimesNet",
    family=ArchitectureFamily.TRANSFORMER,
    description=(
        "TimesNet transforms 1-D time-series into 2-D space using the Fast Fourier "
        "Transform to identify dominant periods, then applies 2-D convolutions to "
        "capture intra- and inter-period variation in physics signals."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Periodic physics time-series", "Multi-periodicity signal modelling", "Oscillatory PDE solution forecasting", "Frequency-domain feature extraction"],
    parameters={"d_model": 64, "n_layers": 3, "top_k_periods": 5, "conv_hidden": 64},
    paper_reference="Wu et al. (2023) — ICLR 2023.",
)

symm_transformer = ArchitectureInfo(
    id="symm_transformer",
    name="Conservation Transformer",
    family=ArchitectureFamily.TRANSFORMER,
    description=(
        "Symmetry-informed Transformer that embeds physical conservation laws (energy, "
        "momentum, mass) as structural attention biases derived from continuous symmetry "
        "groups. Enforces conservation constraints by construction, yielding physically "
        "consistent predictions without additional penalty terms."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Conservation-law constrained physics", "Symmetry-preserving neural surrogates", "Mechanics with symmetry groups", "Physics-consistent Transformer modelling"],
    parameters={"d_model": 256, "n_heads": 8, "n_layers": 6, "symmetry_group": "SO3"},
    paper_reference="Liu et al. (2023) — arXiv:2306.03891.",
)

symm_transolver = ArchitectureInfo(
    id="symm_transolver",
    name="Conservation Transolver",
    family=ArchitectureFamily.TRANSFORMER,
    description=(
        "Solver-Transformer hybrid that enforces conservation symmetries inside a "
        "Transolver-style physics-embedded attention mechanism. Integrates conservation "
        "laws (energy, momentum) directly into the attention computation, making it "
        "suitable for unsteady conservation-law PDEs."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Unsteady conservation-law PDEs", "Multi-physics Transformer surrogates", "Symmetry-aware fluid mechanics", "Energy-momentum conserving surrogates"],
    parameters={"d_model": 256, "n_heads": 8, "n_layers": 6},
    paper_reference="Liu et al. (2023) — arXiv:2306.03891.",
)

symm_aero_transformer = ArchitectureInfo(
    id="symm_aero_transformer",
    name="Conservation Aero Transformer",
    family=ArchitectureFamily.TRANSFORMER,
    description=(
        "Conservation-aware Transformer specialised for aerodynamic applications. "
        "Embeds aerodynamic conservation laws (lift, drag, mass flux) directly into "
        "the attention mechanism via symmetry-group biases, enabling physically "
        "consistent flow field predictions around aircraft bodies."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Aerodynamic flow surrogate modelling", "Airfoil / wing pressure prediction", "Lift-drag constraint satisfaction", "CFD data-driven surrogates"],
    parameters={"d_model": 256, "n_heads": 8, "n_layers": 6, "aero_constraints": True},
    paper_reference="Liu et al. (2023) — arXiv:2306.03891.",
)

symm_aero_transolver = ArchitectureInfo(
    id="symm_aero_transolver",
    name="Conservation Aero Transolver",
    family=ArchitectureFamily.TRANSFORMER,
    description=(
        "Combines the conservation aero attention structure with Transolver-style "
        "physics-embedded attention for aerodynamic problems with unsteady flow regimes. "
        "Supports aerodynamic conservation constraints (mass flux, lift, drag) "
        "inside the solver-Transformer hybrid."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Unsteady aerodynamic flow", "Flutter and aeroelasticity surrogates", "High-Re flow prediction", "Aero-conservation constrained learning"],
    parameters={"d_model": 256, "n_heads": 8, "n_layers": 6},
    paper_reference="Liu et al. (2023) — arXiv:2306.03891.",
)


# ===========================================================================
# Recurrent family
# ===========================================================================

lstm = ArchitectureInfo(
    id="lstm",
    name="LSTM",
    family=ArchitectureFamily.RECURRENT,
    description=(
        "Long Short-Term Memory network for sequential physics data. Gating mechanisms "
        "control information flow over long sequences, enabling the network to learn "
        "temporal dependencies in PDE solutions or sensor streams."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Temporal PDE solution prediction", "Physics time-series forecasting", "Dynamical system identification", "Sequential sensor data modelling"],
    parameters={"hidden_dim": 128, "n_layers": 2, "dropout": 0.0, "bidirectional": False},
    paper_reference="Hochreiter & Schmidhuber (1997) — Neural Computation 9(8).",
)

gru = ArchitectureInfo(
    id="gru",
    name="GRU",
    family=ArchitectureFamily.RECURRENT,
    description=(
        "Gated Recurrent Unit: a simplified recurrent architecture with fewer parameters "
        "than LSTM but comparable performance on many physics time-series tasks. Update "
        "and reset gates control state evolution."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Physics time-series forecasting", "Lightweight sequence modelling", "Embedded real-time prediction", "Dynamical system surrogate"],
    parameters={"hidden_dim": 128, "n_layers": 2, "dropout": 0.0},
    paper_reference="Cho et al. (2014) — EMNLP 2014.",
)

bidirectional_lstm = ArchitectureInfo(
    id="bidirectional_lstm",
    name="Bidirectional LSTM",
    family=ArchitectureFamily.RECURRENT,
    description=(
        "Bidirectional LSTM processes sequences in both forward and backward directions, "
        "capturing context from both past and future time steps. Useful for offline "
        "physics analysis where the entire trajectory is available."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Offline trajectory analysis", "Physics signal classification", "Bidirectional temporal context", "Full-trajectory surrogate fitting"],
    parameters={"hidden_dim": 128, "n_layers": 2, "dropout": 0.1},
    paper_reference="Schuster & Paliwal (1997) — IEEE Transactions on Signal Processing.",
)

bidirectional_gru = ArchitectureInfo(
    id="bidirectional_gru",
    name="Bidirectional GRU",
    family=ArchitectureFamily.RECURRENT,
    description=(
        "Lightweight bidirectional RNN using GRU cells. Captures past and future "
        "temporal context with fewer parameters than Bi-LSTM, suited for resource- "
        "constrained offline physics analysis."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Efficient bidirectional temporal modelling", "Physics classification tasks", "Lightweight offline surrogate", "Sequence labelling for PDE data"],
    parameters={"hidden_dim": 64, "n_layers": 2, "dropout": 0.0},
    paper_reference="Cho et al. (2014) — EMNLP 2014.",
)

seq2seq_rnn = ArchitectureInfo(
    id="seq2seq_rnn",
    name="Seq2Seq RNN",
    family=ArchitectureFamily.RECURRENT,
    description=(
        "Sequence-to-sequence RNN with encoder and decoder LSTM networks. Encodes the "
        "input PDE trajectory into a context vector and decodes it into a future "
        "prediction horizon, optionally with attention."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Multi-step ahead PDE forecasting", "Physics trajectory extrapolation", "Variable-length input/output sequences", "Control-input conditioned prediction"],
    parameters={"hidden_dim": 128, "n_layers": 2, "encoder_type": "lstm", "decoder_type": "lstm", "use_attention": True},
    paper_reference="Sutskever, Vinyals & Le (2014) — NeurIPS 2014.",
)

ode_rnn = ArchitectureInfo(
    id="ode_rnn",
    name="ODE-RNN",
    family=ArchitectureFamily.RECURRENT,
    description=(
        "ODE-RNN combines a standard RNN with a continuous-time Neural ODE between "
        "observations. The ODE evolves the hidden state continuously between irregular "
        "time stamps, making it suitable for non-uniform physics sensor data."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Irregularly sampled physics measurements", "Latent continuous dynamics", "Non-uniform time-stamp trajectories", "Missing data in dynamical systems"],
    parameters={"hidden_dim": 64, "n_layers": 2, "solver": "dopri5", "rtol": 1e-4, "atol": 1e-6},
    paper_reference="Rubanova, Chen & Duvenaud (2019) — NeurIPS 2019.",
)

bayesian_rnn = ArchitectureInfo(
    id="bayesian_rnn",
    name="Bayesian RNN",
    family=ArchitectureFamily.RECURRENT,
    description=(
        "Bayesian RNN places a variational posterior over LSTM weights using dropout "
        "as a Bayesian approximation. Produces calibrated uncertainty estimates on "
        "physics time-series predictions."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Uncertainty quantification in time-series", "Safety-critical physics forecasting", "Sparse data regimes", "Probabilistic simulation surrogates"],
    parameters={"hidden_dim": 128, "n_layers": 2, "dropout": 0.2, "n_mc_samples": 50},
    paper_reference="Gal & Ghahramani (2016) — ICML 2016.",
)


# ===========================================================================
# Convolutional family
# ===========================================================================

conv1d = ArchitectureInfo(
    id="conv1d",
    name="1-D CNN",
    family=ArchitectureFamily.CONVOLUTIONAL,
    description=(
        "1-D convolutional neural network for temporal or 1-D spatial physics data. "
        "Applies learned filters along the time or space axis to extract local patterns "
        "in PDE solutions or sensor signals."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["1-D PDE solution regression", "Temporal physics signal classification", "Local-feature extraction from time-series", "Acoustic or seismic signal processing"],
    parameters={"n_filters": 64, "kernel_size": 3, "n_layers": 4, "activation": "relu", "pooling": "global_avg"},
    paper_reference="LeCun et al. (1989) — Connection Science.",
)

conv2d = ArchitectureInfo(
    id="conv2d",
    name="2-D CNN",
    family=ArchitectureFamily.CONVOLUTIONAL,
    description=(
        "2-D convolutional neural network for field-level physics simulation data on "
        "structured grids. Captures spatial correlations in 2-D PDE solutions such as "
        "velocity/pressure fields or temperature maps."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["2-D PDE solution prediction on grids", "Pressure / velocity field regression", "Structured grid physics surrogates", "Image-to-field mapping"],
    parameters={"n_filters": 64, "kernel_size": 3, "n_layers": 6, "activation": "relu"},
    paper_reference="LeCun et al. (1998) — Proceedings of the IEEE.",
)

conv3d = ArchitectureInfo(
    id="conv3d",
    name="3-D CNN",
    family=ArchitectureFamily.CONVOLUTIONAL,
    description=(
        "3-D convolutional neural network for volumetric or spatio-temporal physics "
        "simulation data on structured 3-D grids. Captures 3-D spatial correlations "
        "in flow fields, stress distributions, or temperature volumes."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["3-D CFD/FEM field prediction on grids", "Volumetric stress / temperature regression", "Spatio-temporal 3-D PDE surrogates", "MRI / CT physics data processing"],
    parameters={"n_filters": 32, "kernel_size": 3, "n_layers": 4, "activation": "relu"},
    paper_reference="Ji et al. (2013) — IEEE TPAMI.",
)


# ===========================================================================
# Continuous (dynamics) family
# ===========================================================================

neural_ode = ArchitectureInfo(
    id="neural_ode",
    name="Neural ODE",
    family=ArchitectureFamily.CONTINUOUS,
    description=(
        "Neural ODE replaces discrete residual blocks with a continuous-time ODE solver: "
        "dz/dt = f_θ(z, t). The ODE is solved with an adaptive-step solver, enabling "
        "theoretically exact continuous-time dynamics without time-step discretisation."
    ),
    supports_physics_loss=True,
    supports_3d=False,
    best_for=["Temporal ODE system identification", "Irregular time-series regression", "Dynamical system discovery", "Latent continuous dynamics modelling"],
    parameters={"latent_dim": 64, "hidden_dim": 128, "n_layers": 3, "activation": "tanh", "solver": "dopri5", "rtol": 1e-4, "atol": 1e-6},
    paper_reference="Chen et al. (2018) — NeurIPS 2018.",
)

latent_ode = ArchitectureInfo(
    id="latent_ode",
    name="Latent ODE",
    family=ArchitectureFamily.CONTINUOUS,
    description=(
        "Latent ODE combines a variational autoencoder with a Neural ODE to model "
        "stochastic continuous-time dynamics in a learned latent space. Useful for "
        "irregularly observed physics systems with uncertainty."
    ),
    supports_physics_loss=True,
    supports_3d=False,
    best_for=["Stochastic dynamical systems", "Irregularly sampled ODE trajectories", "Uncertainty-aware physics dynamics", "Generative model of PDE solutions"],
    parameters={"latent_dim": 32, "hidden_dim": 64, "n_layers": 2, "solver": "dopri5"},
    paper_reference="Rubanova, Chen & Duvenaud (2019) — NeurIPS 2019.",
)

neural_cde = ArchitectureInfo(
    id="neural_cde",
    name="Neural CDE",
    family=ArchitectureFamily.CONTINUOUS,
    description=(
        "Neural Controlled Differential Equation extends Neural ODE to controlled "
        "systems: dz = f_θ(z) dX(t), where X(t) is an input path. Handles "
        "irregularly sampled and missing physics observations naturally."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Controlled dynamical systems", "Irregularly sampled time-series", "Sensor fusion with missing data", "Continuous recurrent sequence modelling"],
    parameters={"hidden_dim": 64, "n_layers": 3, "interpolation": "cubic"},
    paper_reference="Kidger et al. (2020) — NeurIPS 2020.",
)

neural_sde = ArchitectureInfo(
    id="neural_sde",
    name="Neural SDE",
    family=ArchitectureFamily.CONTINUOUS,
    description=(
        "Neural Stochastic Differential Equation models continuous-time stochastic "
        "dynamics: dX_t = f_θ(X_t, t)dt + g_φ(X_t, t)dW_t. Both drift and diffusion "
        "networks are learned, enabling principled uncertainty quantification."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Stochastic physics (turbulence, Langevin dynamics)", "Financial derivatives under uncertainty", "Stochastic optimal control", "UQ in dynamical systems"],
    parameters={"latent_dim": 32, "drift_hidden": [64, 64], "diffusion_hidden": [32, 32], "noise_type": "diagonal", "sde_type": "ito"},
    paper_reference="Li et al. (2020) — AISTATS 2020.",
)

hamiltonian_nn = ArchitectureInfo(
    id="hamiltonian_nn",
    name="Hamiltonian Neural Network",
    family=ArchitectureFamily.CONTINUOUS,
    description=(
        "Learns a Hamiltonian H(q, p) from trajectory data, enforcing energy conservation "
        "by construction. Equations of motion are derived analytically from H via "
        "automatic differentiation, guaranteeing symplectic structure."
    ),
    supports_physics_loss=True,
    supports_3d=False,
    best_for=["Conservative mechanical systems", "Celestial mechanics / orbital dynamics", "Molecular dynamics", "Symplectic integrators for Hamiltonian systems"],
    parameters={"hidden_dim": 128, "n_layers": 3, "activation": "tanh", "input_type": "canonical"},
    paper_reference="Greydanus, Dzamba & Yosinski (2019) — NeurIPS 2019.",
)

symplectic_ode_net = ArchitectureInfo(
    id="symplectic_ode_net",
    name="Symplectic ODE-Net",
    family=ArchitectureFamily.CONTINUOUS,
    description=(
        "Combines a Neural ODE with a learnable symplectic integrator, preserving "
        "phase-space volume (Liouville's theorem). Guarantees long-time stability for "
        "Hamiltonian systems without energy drift."
    ),
    supports_physics_loss=True,
    supports_3d=False,
    best_for=["Long-horizon Hamiltonian dynamics", "Quantum circuit simulation", "Plasma particle tracking", "Rigid-body dynamics"],
    parameters={"hidden_dim": 128, "n_layers": 3, "activation": "tanh", "integrator": "leapfrog", "n_steps": 10},
    paper_reference="Zhong, Han & Brikman (2020) — ICLR 2020.",
)

symplectic_rnn = ArchitectureInfo(
    id="symplectic_rnn",
    name="Symplectic RNN",
    family=ArchitectureFamily.CONTINUOUS,
    description=(
        "Recurrent architecture with symplectic weight matrices ensuring volume "
        "preservation in phase space at each discrete time step. Designed for "
        "discrete-time modelling of Hamiltonian systems from trajectory data."
    ),
    supports_physics_loss=True,
    supports_3d=False,
    best_for=["Discrete Hamiltonian system identification", "Long-horizon mechanical prediction", "Conservative system emulation", "Energy-drift-free dynamics"],
    parameters={"hidden_dim": 64, "n_layers": 2, "integrator": "stormer_verlet"},
    paper_reference="Chen et al. (2020) — arXiv:2007.12246.",
)

neural_gp = ArchitectureInfo(
    id="neural_gp",
    name="Neural Gaussian Process",
    family=ArchitectureFamily.CONTINUOUS,
    description=(
        "Neural Gaussian Process uses deep neural network features as the mean and "
        "kernel functions of a Gaussian Process, combining the expressive power of "
        "neural networks with principled probabilistic inference."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Small-data physics regression with uncertainty", "Bayesian PDE surrogates", "Active learning for PDE sampling", "Probabilistic PDE solution operators"],
    parameters={"hidden_dim": 64, "n_layers": 3, "kernel": "rbf", "n_inducing": 100},
    paper_reference="Wilson et al. (2016) — ICML 2016.",
)

deep_state_space_model = ArchitectureInfo(
    id="deep_state_space_model",
    name="Deep State Space Model",
    family=ArchitectureFamily.CONTINUOUS,
    description=(
        "Deep state space model with a latent Markov state and neural network "
        "emission/transition functions. Suitable for Kalman-filter-like physics "
        "inference with non-linear dynamics."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Latent physics dynamics modelling", "Non-linear Kalman filtering", "State estimation from noisy sensors", "Time-series decomposition into latent states"],
    parameters={"latent_dim": 32, "emission_hidden": [64, 64], "transition_hidden": [64, 64]},
    paper_reference="Rangapuram et al. (2018) — NeurIPS 2018.",
)


# ===========================================================================
# Graph family
# ===========================================================================

gnn = ArchitectureInfo(
    id="gnn",
    name="Graph Neural Network",
    family=ArchitectureFamily.GRAPH,
    description=(
        "Message-passing GNN for physics simulation on unstructured meshes. Each node "
        "aggregates information from its neighbours via learned edge functions, naturally "
        "suited to variable-resolution meshes and irregular geometries."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Unstructured mesh simulation", "Particle-based fluid dynamics (SPH)", "Graph-structured physical systems", "Irregular domain geometries"],
    parameters={"node_dim": 128, "edge_dim": 64, "n_message_passing": 6, "activation": "relu", "aggregation": "mean"},
    paper_reference="Gilmer et al. (2017) — ICML 2017.",
)

equivariant_gnn = ArchitectureInfo(
    id="equivariant_gnn",
    name="Equivariant GNN",
    family=ArchitectureFamily.GRAPH,
    description=(
        "E(n)-Equivariant GNN where node and edge features transform consistently under "
        "rotations, reflections, and translations. Predictions are invariant/equivariant "
        "to rigid-body transformations of the input configuration."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Molecular dynamics / force fields", "Crystal property prediction", "3-D protein structure modelling", "Robotics and rigid-body physics"],
    parameters={"hidden_dim": 64, "n_layers": 5, "max_radius": 5.0, "n_basis": 10, "equivariance": "E3"},
    paper_reference="Satorras, Hoogeboom & Welling (2021) — ICML 2021.",
)

graph_neural_ode = ArchitectureInfo(
    id="graph_neural_ode",
    name="Graph Neural ODE",
    family=ArchitectureFamily.GRAPH,
    description=(
        "Graph Neural ODE couples a GNN with a continuous-time ODE to model spatially "
        "distributed dynamical systems on graphs. The GNN computes spatial interactions "
        "and the ODE evolves the node states continuously in time."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Spatially distributed dynamical systems", "Reaction-diffusion on complex graphs", "Multi-agent physics simulation", "Continuous graph dynamics"],
    parameters={"hidden_dim": 64, "n_message_passing": 4, "solver": "dopri5"},
    paper_reference="Poli et al. (2019) — arXiv:1911.07532.",
)

spatiotemporal_gnn = ArchitectureInfo(
    id="spatiotemporal_gnn",
    name="Spatiotemporal GNN",
    family=ArchitectureFamily.GRAPH,
    description=(
        "Spatiotemporal GNN jointly models spatial graph interactions and temporal "
        "dynamics by interleaving GNN message-passing layers with temporal convolution "
        "or recurrent blocks. Used for traffic flow, weather, and fluid simulations."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Traffic / flow network forecasting", "Weather station data on spatial graphs", "Sensor network physics prediction", "Dynamic mesh simulation surrogates"],
    parameters={"node_dim": 64, "n_message_passing": 6, "temporal": "lstm", "n_time_layers": 2},
    paper_reference="Yu, Yin & Zhu (2018) — IJCAI 2018.",
)

icosahedral_gnn = ArchitectureInfo(
    id="icosahedral_gnn",
    name="Multi-Scale Icosahedral GNN",
    family=ArchitectureFamily.GRAPH,
    description=(
        "Multi-scale graph neural network for weather and climate emulation that encodes "
        "the global grid onto a multi-resolution icosahedral mesh, performs message "
        "passing at each level, and decodes back to the original grid. Designed for "
        "large-scale atmospheric physics and earth-system surrogate modelling."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Global weather / climate forecasting", "Large-scale atmospheric physics emulation", "Multi-scale spatial physics on spheres", "Earth system model surrogates"],
    parameters={"mesh_size": 5, "hidden_dim": 512, "n_message_passing": 16},
    paper_reference="Lam et al. (2023) — Science 382(6677).",
)

mesh_graph_net = ArchitectureInfo(
    id="mesh_graph_net",
    name="MeshGraphNet",
    family=ArchitectureFamily.GRAPH,
    description=(
        "Graph network that builds a learned graph on the simulation mesh where edges "
        "represent mesh connectivity. Message passing propagates physics over many steps, "
        "enabling high-accuracy surrogate modelling for CFD and solid mechanics on "
        "arbitrary unstructured mesh topologies."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["FEM / CFD surrogate on complex meshes", "Structural mechanics data-driven surrogate", "Fluid dynamics on unstructured meshes", "Learning from FEM/CFD data"],
    parameters={"node_dim": 128, "edge_dim": 128, "n_message_passing": 15, "aggregation": "sum"},
    paper_reference="Pfaff et al. (2021) — ICLR 2021.",
)


# ===========================================================================
# Autoencoder family
# ===========================================================================

dense_ae = ArchitectureInfo(
    id="dense_ae",
    name="Dense Autoencoder",
    family=ArchitectureFamily.AUTOENCODER,
    description=(
        "Fully-connected autoencoder that compresses high-dimensional PDE snapshots into "
        "a low-dimensional latent code and reconstructs them. Useful for non-linear "
        "dimensionality reduction of simulation data."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Non-linear dimensionality reduction of simulation fields", "ROM basis learning", "Data compression for parametric PDEs", "Latent space interpolation"],
    parameters={"latent_dim": 32, "encoder_hidden": [256, 128], "decoder_hidden": [128, 256], "activation": "relu"},
    paper_reference="Hinton & Salakhutdinov (2006) — Science 313(5786).",
)

autoencoder_2d = ArchitectureInfo(
    id="autoencoder_2d",
    name="2-D Convolutional Autoencoder",
    family=ArchitectureFamily.AUTOENCODER,
    description=(
        "Convolutional autoencoder for 2-D physics field data. Encoder uses strided "
        "conv layers to compress spatial fields; decoder uses transposed convolutions "
        "to reconstruct. Preserves spatial correlations better than dense AE."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["2-D PDE field compression", "Snapshot-based ROM for 2-D flows", "Convolutional latent space for fields", "Data-driven 2-D PDE basis extraction"],
    parameters={"latent_dim": 64, "n_filters": 64, "n_encoder_layers": 4, "activation": "relu"},
    paper_reference="Kingma & Welling (2013) — arXiv:1312.6114.",
)

vae = ArchitectureInfo(
    id="vae",
    name="Variational Autoencoder",
    family=ArchitectureFamily.AUTOENCODER,
    description=(
        "Variational Autoencoder learns a probabilistic latent representation with a "
        "Gaussian prior. The ELBO loss trains encoder and decoder jointly. The latent "
        "space is smooth, enabling interpolation and generation of new PDE solutions."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Generative modelling of PDE solutions", "Smooth latent space interpolation", "Uncertainty quantification via sampling", "Physics data augmentation"],
    parameters={"latent_dim": 32, "encoder_hidden": [256, 128], "decoder_hidden": [128, 256], "beta": 1.0},
    paper_reference="Kingma & Welling (2013) — arXiv:1312.6114.",
)

kae = ArchitectureInfo(
    id="kae",
    name="Koopman Autoencoder",
    family=ArchitectureFamily.AUTOENCODER,
    description=(
        "Koopman Autoencoder learns an embedding in which the nonlinear dynamical system "
        "evolves linearly (Koopman operator). The autoencoder enforces that the latent "
        "space advances via a learnable linear map at each time step."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Nonlinear dynamical system linearisation", "Stable long-horizon prediction", "Koopman-based control", "Eigenfunction decomposition of dynamics"],
    parameters={"latent_dim": 64, "encoder_hidden": [128, 64], "decoder_hidden": [64, 128], "n_koopman_modes": 32},
    paper_reference="Lusch, Kutz & Brunton (2018) — Nature Communications 9(1).",
)

pi_koopman_ae = ArchitectureInfo(
    id="pi_koopman_ae",
    name="Physics-Informed Koopman AE",
    family=ArchitectureFamily.AUTOENCODER,
    description=(
        "Physics-Informed Koopman Autoencoder augments the Koopman AE loss with PDE "
        "residual constraints in the latent space, ensuring the linear evolution "
        "respects the underlying physical equations."
    ),
    supports_physics_loss=True,
    supports_3d=False,
    best_for=["Physics-constrained Koopman decomposition", "Interpretable nonlinear dynamics", "Data-efficient PDE-constrained ROM", "Hybrid data/physics latent modelling"],
    parameters={"latent_dim": 64, "n_koopman_modes": 32, "physics_weight": 1.0},
    paper_reference="Pan & Duraisamy (2020) — arXiv:2005.04676.",
)

ae_rom_hybrid = ArchitectureInfo(
    id="ae_rom_hybrid",
    name="AE-ROM Hybrid",
    family=ArchitectureFamily.AUTOENCODER,
    description=(
        "Hybrid model combining a convolutional autoencoder for nonlinear dimensionality "
        "reduction with a classical ROM (e.g., Galerkin projection) in the latent "
        "space. Bridges deep learning and physics-based ROMs."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Hybrid data-physics ROM", "Parametric PDE families", "Online/offline ROM paradigm", "Multi-query PDE applications"],
    parameters={"latent_dim": 32, "rom_type": "galerkin", "n_filters": 64},
    paper_reference="Fresca, Dede & Manzoni (2021) — SIAM Journal on Scientific Computing.",
)

rom_hybrid = ArchitectureInfo(
    id="rom_hybrid",
    name="ROM Hybrid",
    family=ArchitectureFamily.AUTOENCODER,
    description=(
        "General ROM hybrid architecture combining a neural network encoder with a "
        "physics-based ROM solver in the latent space. The ROM governs latent state "
        "evolution while the decoder reconstructs high-fidelity fields."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Physics-based latent dynamics", "Parameterised PDE emulation", "Multi-fidelity surrogate modelling", "Reduced-basis method augmentation"],
    parameters={"latent_dim": 64, "decoder_hidden": [128, 256]},
    paper_reference="Fresca, Dede & Manzoni (2021) — SIAM Journal on Scientific Computing.",
)


# ===========================================================================
# Reservoir / kernel family
# ===========================================================================

elm = ArchitectureInfo(
    id="elm",
    name="Extreme Learning Machine",
    family=ArchitectureFamily.RESERVOIR,
    description=(
        "Single hidden-layer feedforward network where input and hidden weights are "
        "set randomly and fixed. Only the output layer is trained analytically via "
        "least squares, giving very fast training for physics regression tasks."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Rapid physics regression", "Low-dimensional PDE surrogate", "Real-time signal processing", "Resource-constrained applications"],
    parameters={"hidden_dim": 1000, "activation": "sigmoid", "solver": "lstsq"},
    paper_reference="Huang, Zhu & Siew (2006) — Neurocomputing 70(1-3).",
)

rbf = ArchitectureInfo(
    id="rbf",
    name="Radial Basis Function Network",
    family=ArchitectureFamily.RESERVOIR,
    description=(
        "RBF network uses Gaussian (or other) radial kernels centred on training points "
        "as basis functions, with output weights trained via least squares. Classical "
        "approach for scattered-data interpolation of PDE solutions."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Scattered-data interpolation of PDE solutions", "Mesh-free PDE solving (Kansa method)", "Smooth function approximation", "Low-dimensional regression"],
    parameters={"n_centres": 200, "kernel": "gaussian", "epsilon": 1.0, "solver": "lstsq"},
    paper_reference="Hardy (1971) — Journal of Geophysical Research.",
)

hybrid_rbf = ArchitectureInfo(
    id="hybrid_rbf",
    name="Hybrid RBF",
    family=ArchitectureFamily.RESERVOIR,
    description=(
        "Hybrid RBF-network that combines classical radial basis functions with a "
        "trainable neural network correction layer. The RBF component provides a "
        "physics-informed prior while the neural correction captures residual features."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Physics-prior augmented regression", "Smooth field interpolation with corrections", "Semi-parametric PDE surrogates", "Hybrid classical/neural approaches"],
    parameters={"n_centres": 100, "kernel": "gaussian", "correction_hidden": [64, 64]},
    paper_reference="Buhmann (2003) — Cambridge Monographs on Applied Mathematics.",
)

esn = ArchitectureInfo(
    id="esn",
    name="Echo State Network",
    family=ArchitectureFamily.RESERVOIR,
    description=(
        "Echo State Network (ESN) is a reservoir computing approach where a large "
        "sparse recurrent reservoir is randomly initialised and fixed. Only the linear "
        "readout layer is trained, enabling very fast training for physics time-series."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Chaotic dynamical system emulation", "Fast time-series forecasting", "Lorenz / Kuramoto-Sivashinsky prediction", "Reservoir computing benchmarks"],
    parameters={"reservoir_size": 500, "spectral_radius": 0.9, "connectivity": 0.1, "input_scaling": 1.0},
    paper_reference="Jaeger (2001) — GMD Report 148.",
)

esn_rc = ArchitectureInfo(
    id="esn_rc",
    name="ESN with Ridge Regularisation",
    family=ArchitectureFamily.RESERVOIR,
    description=(
        "Echo State Network with ridge (L2) regularisation on the output weights. "
        "The regularisation improves generalisation for physics time-series where "
        "training data is noisy or limited."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Noisy physics time-series", "Regularised reservoir computing", "Short training data regimes", "Rapid chaotic attractor learning"],
    parameters={"reservoir_size": 500, "spectral_radius": 0.9, "connectivity": 0.1, "ridge_alpha": 1e-4},
    paper_reference="Jaeger (2001) — GMD Report 148.",
)

koopman = ArchitectureInfo(
    id="koopman",
    name="Koopman Operator",
    family=ArchitectureFamily.RESERVOIR,
    description=(
        "Data-driven Koopman operator approximation using Extended Dynamic Mode "
        "Decomposition (EDMD) or neural Koopman eigenfunctions. Represents nonlinear "
        "dynamics as a linear operator on an enriched observable space."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Nonlinear dynamical system analysis", "Koopman eigenvalue decomposition", "Linear prediction of nonlinear dynamics", "Control with linear Koopman model"],
    parameters={"n_observables": 100, "method": "edmd", "regularisation": 1e-6},
    paper_reference="Williams, Kevrekidis & Rowley (2015) — JNLS 25, 1307-1346.",
)


# ===========================================================================
# ROM family
# ===========================================================================

pod = ArchitectureInfo(
    id="pod",
    name="Proper Orthogonal Decomposition",
    family=ArchitectureFamily.ROM,
    description=(
        "Classical POD / PCA-based reduced-order model: computes SVD of snapshot matrix "
        "to extract dominant energy modes, then projects the PDE onto the low-dimensional "
        "basis for fast parametric evaluation."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Linear or weakly-nonlinear PDE ROMs", "Galerkin projection ROMs", "Multi-query parametric simulations", "Offline-online ROM decomposition"],
    parameters={"n_modes": 20, "centering": True},
    paper_reference="Lumley (1967) — Atmospheric Turbulence and Wave Propagation.",
)

dmd = ArchitectureInfo(
    id="dmd",
    name="Dynamic Mode Decomposition",
    family=ArchitectureFamily.ROM,
    description=(
        "Dynamic Mode Decomposition extracts spatiotemporal modes and their eigenvalues "
        "from simulation snapshots. Each DMD mode evolves at a single frequency, "
        "making it ideal for oscillatory flow fields and stability analysis."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Oscillatory flow analysis", "Stability and frequency analysis of PDEs", "Linear low-order dynamics extraction", "Data-driven modal decomposition"],
    parameters={"n_modes": 20, "svd_rank": 50, "method": "exact"},
    paper_reference="Schmid (2010) — Journal of Fluid Mechanics 656.",
)

havok = ArchitectureInfo(
    id="havok",
    name="HAVOK",
    family=ArchitectureFamily.ROM,
    description=(
        "Hankel Alternative View Of Koopman (HAVOK) constructs a Hankel matrix from a "
        "chaotic time series, performs SVD, and fits a linear forcing model in the "
        "embedded space. Identifies intermittent forcing events in chaotic attractors."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Chaotic time-series analysis", "Intermittency identification in turbulence", "Lorenz / chaotic attractor decomposition", "Data-driven linear forcing models"],
    parameters={"delay_embedding": 50, "svd_rank": 15},
    paper_reference="Brunton et al. (2017) — Nature Communications 8(1).",
)

operator_inference = ArchitectureInfo(
    id="operator_inference",
    name="Operator Inference",
    family=ArchitectureFamily.ROM,
    description=(
        "Operator Inference learns low-dimensional ODE operators (linear, quadratic, "
        "input matrices) by non-intrusive regression from simulation snapshots. Avoids "
        "intrusive modification of simulation code while recovering physics structure."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Non-intrusive ROM from simulation data", "Polynomial nonlinear ROM operators", "Structured operator learning", "Multi-query parametric simulation"],
    parameters={"n_modes": 20, "poly_order": 2, "regularisation": 1e-4},
    paper_reference="Peherstorfer & Willcox (2016) — CMAME 306.",
)

sindy = ArchitectureInfo(
    id="sindy",
    name="SINDy",
    family=ArchitectureFamily.ROM,
    description=(
        "Sparse Identification of Nonlinear Dynamics (SINDy) regresses time derivatives "
        "onto a library of candidate functions using sparse regression (LASSO/SR3). "
        "Recovers interpretable governing equations directly from data."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Interpretable equation discovery", "ODE/PDE system identification", "Sparse model selection for physics", "Governing equation extraction from data"],
    parameters={"poly_order": 3, "threshold": 0.05, "optimizer": "STLSQ", "include_sine": False},
    paper_reference="Brunton, Proctor & Kutz (2016) — PNAS 113(15).",
)

koopman_rom = ArchitectureInfo(
    id="koopman_rom",
    name="Koopman ROM",
    family=ArchitectureFamily.ROM,
    description=(
        "Koopman-based reduced-order model that lifts the nonlinear dynamics into an "
        "enriched observable space where a global linear ROM is exact. Combines Koopman "
        "theory with POD-style projection for fast parametric prediction."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Nonlinear ROM via linearisation", "Fast parametric flow prediction", "Global linear dynamics surrogate", "Koopman-based control design"],
    parameters={"n_observables": 64, "n_modes": 20, "method": "edmd"},
    paper_reference="Williams, Kevrekidis & Rowley (2015) — JNLS 25, 1307-1346.",
)

neural_rom = ArchitectureInfo(
    id="neural_rom",
    name="Neural ROM",
    family=ArchitectureFamily.ROM,
    description=(
        "Neural network-based reduced-order model that learns a nonlinear low-dimensional "
        "manifold from simulation snapshots and evolves the reduced state with a neural "
        "ODE or recurrent network. Extends classical ROMs to strongly nonlinear regimes."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Strongly nonlinear PDE ROMs", "Turbulent flow surrogates", "Multi-parameter PDE families", "Generalisation beyond linear ROMs"],
    parameters={"latent_dim": 16, "encoder_hidden": [128, 64], "decoder_hidden": [64, 128], "dynamics": "lstm"},
    paper_reference="Fresca & Manzoni (2022) — Journal of Scientific Computing.",
)

deep_uq_rom = ArchitectureInfo(
    id="deep_uq_rom",
    name="Deep UQ-ROM",
    family=ArchitectureFamily.ROM,
    description=(
        "Deep learning ROM with integrated uncertainty quantification. Combines a "
        "variational autoencoder for latent compression with Bayesian or ensemble "
        "methods for uncertainty propagation in parametric PDE predictions."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Uncertainty-aware parametric ROM", "Safety-critical PDE surrogates", "Probabilistic multi-query PDE emulation", "Bayesian model calibration"],
    parameters={"latent_dim": 32, "n_ensemble": 10, "encoder_hidden": [128, 64]},
    paper_reference="Fresca, Dede & Manzoni (2021) — SIAM Journal on Scientific Computing.",
)


# ===========================================================================
# Classical / signal-processing family
# ===========================================================================

arima = ArchitectureInfo(
    id="arima",
    name="ARIMA",
    family=ArchitectureFamily.CLASSICAL,
    description=(
        "Auto-Regressive Integrated Moving Average: classical statistical time-series "
        "model combining differencing (I), autoregression (AR), and moving average (MA) "
        "components. Baseline method for univariate physics time-series forecasting."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Univariate physics signal forecasting", "Baseline time-series benchmark", "Stationary sensor data modelling", "Low-data forecasting"],
    parameters={"p": 2, "d": 1, "q": 2},
    paper_reference="Box & Jenkins (1970) — Time Series Analysis: Forecasting and Control.",
)

var = ArchitectureInfo(
    id="var",
    name="Vector AR",
    family=ArchitectureFamily.CLASSICAL,
    description=(
        "Vector Auto-Regression: multivariate extension of AR modelling capturing "
        "linear temporal dependencies between multiple physics signals simultaneously. "
        "Frequently used as a classical baseline for multi-variate PDE data."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Multi-variate physics signal modelling", "Linear coupling between PDE fields", "Classical baseline for multi-variate forecasting", "Reduced-form dynamic modelling"],
    parameters={"p": 5},
    paper_reference="Sims (1980) — Econometrica 48(1).",
)

kalman_filter = ArchitectureInfo(
    id="kalman_filter",
    name="Kalman Filter",
    family=ArchitectureFamily.CLASSICAL,
    description=(
        "Optimal linear state estimator for linear Gaussian dynamical systems. "
        "Alternates between predict (physics model propagation) and update (sensor "
        "fusion) steps. Baseline for physics-based state estimation."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Linear physics state estimation", "Sensor fusion with linear dynamics", "Navigation and tracking", "Optimal estimation baseline"],
    parameters={"process_noise": 1e-4, "measurement_noise": 1e-2},
    paper_reference="Kalman (1960) — Journal of Basic Engineering 82(1).",
)

extended_kalman_filter = ArchitectureInfo(
    id="extended_kalman_filter",
    name="Extended Kalman Filter",
    family=ArchitectureFamily.CLASSICAL,
    description=(
        "Extended Kalman Filter linearises nonlinear dynamics via first-order Taylor "
        "expansion, enabling state estimation for mildly nonlinear physics. Computes "
        "Jacobians of the dynamics and measurement models analytically or numerically."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Mildly nonlinear state estimation", "Nonlinear physics sensor fusion", "Robotics localisation", "Parameter estimation for nonlinear ODEs"],
    parameters={"process_noise": 1e-4, "measurement_noise": 1e-2},
    paper_reference="Jazwinski (1970) — Stochastic Processes and Filtering Theory.",
)

unscented_kalman_filter = ArchitectureInfo(
    id="unscented_kalman_filter",
    name="Unscented Kalman Filter",
    family=ArchitectureFamily.CLASSICAL,
    description=(
        "Unscented Kalman Filter propagates a set of deterministic sigma points through "
        "the nonlinear dynamics, capturing the mean and covariance more accurately than "
        "EKF for strongly nonlinear physics without Jacobian computation."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Strongly nonlinear state estimation", "Nonlinear physics without Jacobians", "UQ for nonlinear dynamical systems", "Parameter estimation under uncertainty"],
    parameters={"alpha": 1e-3, "beta": 2.0, "kappa": 0.0},
    paper_reference="Julier & Uhlmann (1997) — Proc. SPIE AeroSense 3068.",
)

ensemble_kalman_filter = ArchitectureInfo(
    id="ensemble_kalman_filter",
    name="Ensemble Kalman Filter",
    family=ArchitectureFamily.CLASSICAL,
    description=(
        "Ensemble Kalman Filter (EnKF) represents the state distribution with an "
        "ensemble of particles, making it scalable to high-dimensional PDEs without "
        "Jacobians. Widely used for geophysical data assimilation."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["High-dimensional PDE data assimilation", "Geophysical state estimation", "Large-ensemble uncertainty quantification", "Real-time NWP assimilation"],
    parameters={"n_ensemble": 100, "inflation": 1.05, "localisation_radius": None},
    paper_reference="Evensen (1994) — Journal of Geophysical Research 99(C5).",
)

tcn = ArchitectureInfo(
    id="tcn",
    name="Temporal Convolutional Network",
    family=ArchitectureFamily.CLASSICAL,
    description=(
        "Temporal Convolutional Network uses dilated causal convolutions to efficiently "
        "capture long-range temporal dependencies in physics time-series. Comparable to "
        "LSTM performance with faster training due to fully-convolutional architecture."
    ),
    supports_physics_loss=False,
    supports_3d=False,
    best_for=["Causal time-series physics modelling", "Long-range temporal dependency capture", "Fast training alternative to LSTM", "Signal processing for physics data"],
    parameters={"n_filters": 64, "kernel_size": 3, "n_layers": 8, "dilation_base": 2, "dropout": 0.1},
    paper_reference="Bai, Kolter & Koltun (2018) — arXiv:1803.01271.",
)


# ===========================================================================
# Physics-Aware family
# ===========================================================================

siren = ArchitectureInfo(
    id="siren",
    name="SIREN",
    family=ArchitectureFamily.PHYSICS_AWARE,
    description=(
        "Sinusoidal Representation Network uses sin(ω₀ · Wx + b) activations in every "
        "layer. This enables accurate representation of high-frequency signals and their "
        "derivatives. Particularly well-suited for PDEs with oscillatory solutions such "
        "as wave and Helmholtz equations."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Wave and Helmholtz equations (high k)", "Acoustic / electromagnetic wave problems", "Signal representation (image, audio)", "PDEs requiring high-order accurate derivatives"],
    parameters={"hidden_dim": 256, "n_layers": 5, "omega_0": 30.0, "in_dim": 2, "out_dim": 1},
    paper_reference="Sitzmann et al. (2020) — NeurIPS 2020.",
)

hash_grid_mlp = ArchitectureInfo(
    id="hash_grid_mlp",
    name="Hash Grid MLP",
    family=ArchitectureFamily.PHYSICS_AWARE,
    description=(
        "Instant-NGP style multi-resolution hash grid encoding combined with a small "
        "MLP decoder. The hash grid provides extremely fast convergence for spatial "
        "fields, orders of magnitude faster than vanilla PINNs on 3-D problems."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Fast 3-D implicit neural field fitting", "Volumetric PDE solution approximation", "Neural radiance / density fields", "High-resolution 3-D physics surrogates"],
    parameters={"n_levels": 16, "features_per_level": 2, "log2_hashmap_size": 19, "hidden_dim": 64, "n_layers": 2},
    paper_reference="Muller et al. (2022) — ACM SIGGRAPH 2022.",
)

physics_aware_neural_network = ArchitectureInfo(
    id="physics_aware_neural_network",
    name="Physics-Aware Neural Network",
    family=ArchitectureFamily.PHYSICS_AWARE,
    description=(
        "Embeds explicit physical constraints directly into the network architecture: "
        "hard enforcement of boundary conditions via distance-function multipliers, "
        "symmetry constraints via equivariant layers, or conservation laws via output "
        "projections. Eliminates unphysical solutions by construction."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Hard BC enforcement", "Conservation-law constrained problems", "Symmetry-constrained physical systems", "High-accuracy PDE solutions"],
    parameters={"hidden_dim": 128, "n_layers": 4, "activation": "tanh", "bc_enforcement": "hard"},
    paper_reference="Sun, Liu & Wang (2020) — CMAME 361, 112732.",
)

structure_preserving_network = ArchitectureInfo(
    id="structure_preserving_network",
    name="Structure-Preserving Network",
    family=ArchitectureFamily.PHYSICS_AWARE,
    description=(
        "Enforces geometric and physical structure of the solution manifold: "
        "positive-definiteness of stress tensors, symmetry of constitutive relations, "
        "divergence-free velocity fields. Uses constrained output layers and Riemannian "
        "optimisation to guarantee physical admissibility."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Elasticity with positive-definite stress", "Divergence-free fluid fields", "Symmetric tensor-field problems", "Structure-preserving ROMs"],
    parameters={"hidden_dim": 128, "n_layers": 4, "activation": "tanh", "preserved_structures": ["symmetry", "positive_definite"]},
    paper_reference="Hernandez et al. (2021) — Journal of Computational Physics 426, 109950.",
)

bench_siren = ArchitectureInfo(
    id="bench_siren",
    name="Benchmark SIREN",
    family=ArchitectureFamily.PHYSICS_AWARE,
    description=(
        "Benchmarking variant of SIREN with standardised hyperparameters for "
        "reproducible comparison. Uses the canonical configuration from the original "
        "paper with fixed omega_0 and layer widths."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Reproducible SIREN benchmarking", "Fair comparison of implicit neural representations", "Wave equation benchmarks", "High-frequency PDE benchmarks"],
    parameters={"hidden_dim": 256, "n_layers": 5, "omega_0": 30.0},
    paper_reference="Sitzmann et al. (2020) — NeurIPS 2020.",
)

bench_fourier_mlp = ArchitectureInfo(
    id="bench_fourier_mlp",
    name="Benchmark Fourier MLP",
    family=ArchitectureFamily.PHYSICS_AWARE,
    description=(
        "Benchmarking MLP with Random Fourier Feature (RFF) input encoding for "
        "standardised comparison. Applies fixed Gaussian random features to map "
        "coordinates into a higher-dimensional spectral space before the MLP."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Spectral PINN benchmarking", "RFF encoding comparison studies", "High-frequency PDE regression baselines", "Feature-mapping ablation studies"],
    parameters={"hidden_dim": 256, "n_layers": 4, "n_fourier": 128, "sigma": 1.0},
    paper_reference="Tancik et al. (2020) — NeurIPS 2020.",
)

bench_mlp = ArchitectureInfo(
    id="bench_mlp",
    name="Benchmark MLP",
    family=ArchitectureFamily.PHYSICS_AWARE,
    description=(
        "Standard fully-connected MLP with fixed hyperparameters used as a "
        "reproducible baseline across PINNeAPPle Arena experiments. Tanh activations "
        "and Xavier initialisation following PINNeAPPle default conventions."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Baseline PINN comparison", "Ablation studies", "Architecture search starting point", "General PDE regression"],
    parameters={"hidden_dim": 128, "n_layers": 4, "activation": "tanh"},
    paper_reference="Raissi, Perdikaris & Karniadakis (2019) — Journal of Computational Physics 378.",
)

bench_res_mlp = ArchitectureInfo(
    id="bench_res_mlp",
    name="Benchmark Residual MLP",
    family=ArchitectureFamily.PHYSICS_AWARE,
    description=(
        "Benchmarking MLP with residual (skip) connections between layers. "
        "Residual connections ease gradient flow in deep networks and are used "
        "as a standardised comparison point for deeper PINN architectures."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Deep PINN baselines", "Residual connection ablation studies", "Gradient flow comparison", "Deep physics regression"],
    parameters={"hidden_dim": 128, "n_layers": 6, "activation": "tanh", "skip_connections": True},
    paper_reference="He et al. (2016) — CVPR 2016.",
)

bench_linear = ArchitectureInfo(
    id="bench_linear",
    name="Benchmark Linear Model",
    family=ArchitectureFamily.PHYSICS_AWARE,
    description=(
        "Linear model baseline for physics regression: a single linear layer mapping "
        "inputs directly to outputs via least squares. Provides the simplest possible "
        "benchmark to compare against nonlinear architectures."
    ),
    supports_physics_loss=False,
    supports_3d=True,
    best_for=["Linear physics regression baseline", "Diagnostic comparison for nonlinear models", "Least-squares PDE surrogate", "Feature importance analysis"],
    parameters={"solver": "lstsq"},
    paper_reference="N/A — standard linear regression.",
)

symm_upt = ArchitectureInfo(
    id="symm_upt",
    name="Conservation Universal Physics Transformer",
    family=ArchitectureFamily.PHYSICS_AWARE,
    description=(
        "Universal Physics Transformer with embedded conservation-law inductive biases "
        "(energy, momentum, charge) derived from continuous symmetry groups. Designed "
        "for general multi-physics operator learning where conservation guarantees are "
        "structurally embedded rather than enforced via soft penalty terms."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Multi-physics operator learning", "Conservation-law constrained surrogates", "General PDE families with symmetries", "Physics-universal Transformer modelling"],
    parameters={"d_model": 256, "n_heads": 8, "n_layers": 8, "conserved_quantities": ["energy", "momentum"]},
    paper_reference="Liu et al. (2023) — arXiv:2306.03891.",
)

symm_abupt = ArchitectureInfo(
    id="symm_abupt",
    name="Conservation Attention-Bias UPT",
    family=ArchitectureFamily.PHYSICS_AWARE,
    description=(
        "Extends the Conservation Universal Physics Transformer with attention-bias "
        "terms analytically derived from physical conservation laws and symmetry groups. "
        "The computed bias terms are added directly to attention logits, enforcing "
        "conservation without explicit Lagrange multipliers."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Symmetry-biased attention for physics", "Hard conservation via attention bias", "Multi-scale physics Transformer", "Attention-bias physics embedding"],
    parameters={"d_model": 256, "n_heads": 8, "n_layers": 8, "bias_strength": 1.0},
    paper_reference="Liu et al. (2023) — arXiv:2306.03891.",
)

symm_aero_upt = ArchitectureInfo(
    id="symm_aero_upt",
    name="Conservation Aero UPT",
    family=ArchitectureFamily.PHYSICS_AWARE,
    description=(
        "Conservation Universal Physics Transformer specialised for aerodynamic "
        "applications with embedded Kutta-Joukowski and mass-conservation constraints. "
        "Structurally encodes lift/drag physics for accurate flow field prediction "
        "around aircraft bodies and lifting surfaces."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["Aerodynamic force / field prediction", "Airfoil / wing surrogate modelling", "Lift-drag conservation constraints", "Aero-structural coupling surrogates"],
    parameters={"d_model": 256, "n_heads": 8, "n_layers": 8, "aero_constraints": True},
    paper_reference="Liu et al. (2023) — arXiv:2306.03891.",
)

symm_aero_abupt = ArchitectureInfo(
    id="symm_aero_abupt",
    name="Conservation Aero AB-UPT",
    family=ArchitectureFamily.PHYSICS_AWARE,
    description=(
        "Combines Conservation Aero UPT with attention-bias conservation terms for "
        "aerodynamic problems. Provides the most comprehensive conservation enforcement "
        "for aerodynamic physics, combining structural symmetry biases with hard "
        "conservation enforcement via analytically computed attention biasing."
    ),
    supports_physics_loss=True,
    supports_3d=True,
    best_for=["High-fidelity aerodynamic surrogate", "Hard-constraint aerodynamic conservation", "Multi-physics aero-structural problems", "Industrial CFD surrogate modelling"],
    parameters={"d_model": 256, "n_heads": 8, "n_layers": 8, "aero_constraints": True, "bias_strength": 1.0},
    paper_reference="Liu et al. (2023) — arXiv:2306.03891.",
)


# ===========================================================================
# Assembled catalog — all 93 architectures
# ===========================================================================

ARCHITECTURES_CATALOG: dict[str, ArchitectureInfo] = {
    # PINN (9)
    "vanilla_pinn": vanilla_pinn,
    "inverse_pinn": inverse_pinn,
    "pielm": pielm,
    "pinn_lstm": pinn_lstm,
    "pinnsformer": pinnsformer,
    "vpinn": vpinn,
    "xpinn": xpinn,
    "xtfc": xtfc,
    "modified_mlp": modified_mlp,
    # Neural Operator (8)
    "fno": fno,
    "gno": gno,
    "deeponet": deeponet,
    "multiscale_deeponet": multiscale_deeponet,
    "pino": pino,
    "uno": uno,
    "ts_fno": ts_fno,
    "afno": afno,
    # Transformer (10)
    "transformer": transformer,
    "informer": informer,
    "autoformer": autoformer,
    "fedformer": fedformer,
    "tft": tft,
    "timesnet": timesnet,
    "symm_transformer": symm_transformer,
    "symm_transolver": symm_transolver,
    "symm_aero_transformer": symm_aero_transformer,
    "symm_aero_transolver": symm_aero_transolver,
    # Recurrent (7)
    "lstm": lstm,
    "gru": gru,
    "bidirectional_lstm": bidirectional_lstm,
    "bidirectional_gru": bidirectional_gru,
    "seq2seq_rnn": seq2seq_rnn,
    "ode_rnn": ode_rnn,
    "bayesian_rnn": bayesian_rnn,
    # Convolutional (3)
    "conv1d": conv1d,
    "conv2d": conv2d,
    "conv3d": conv3d,
    # Continuous (10)
    "neural_ode": neural_ode,
    "latent_ode": latent_ode,
    "neural_cde": neural_cde,
    "neural_sde": neural_sde,
    "hamiltonian_nn": hamiltonian_nn,
    "symplectic_ode_net": symplectic_ode_net,
    "symplectic_rnn": symplectic_rnn,
    "neural_gp": neural_gp,
    "deep_state_space_model": deep_state_space_model,
    # Graph (6)
    "gnn": gnn,
    "equivariant_gnn": equivariant_gnn,
    "graph_neural_ode": graph_neural_ode,
    "spatiotemporal_gnn": spatiotemporal_gnn,
    "icosahedral_gnn": icosahedral_gnn,
    "mesh_graph_net": mesh_graph_net,
    # Autoencoder (7)
    "dense_ae": dense_ae,
    "autoencoder_2d": autoencoder_2d,
    "vae": vae,
    "kae": kae,
    "pi_koopman_ae": pi_koopman_ae,
    "ae_rom_hybrid": ae_rom_hybrid,
    "rom_hybrid": rom_hybrid,
    # Reservoir (6)
    "elm": elm,
    "rbf": rbf,
    "hybrid_rbf": hybrid_rbf,
    "esn": esn,
    "esn_rc": esn_rc,
    "koopman": koopman,
    # ROM (8)
    "pod": pod,
    "dmd": dmd,
    "havok": havok,
    "operator_inference": operator_inference,
    "sindy": sindy,
    "koopman_rom": koopman_rom,
    "neural_rom": neural_rom,
    "deep_uq_rom": deep_uq_rom,
    # Classical (7)
    "arima": arima,
    "var": var,
    "kalman_filter": kalman_filter,
    "extended_kalman_filter": extended_kalman_filter,
    "unscented_kalman_filter": unscented_kalman_filter,
    "ensemble_kalman_filter": ensemble_kalman_filter,
    "tcn": tcn,
    # Physics-Aware (13)
    "siren": siren,
    "hash_grid_mlp": hash_grid_mlp,
    "physics_aware_neural_network": physics_aware_neural_network,
    "structure_preserving_network": structure_preserving_network,
    "bench_siren": bench_siren,
    "bench_fourier_mlp": bench_fourier_mlp,
    "bench_mlp": bench_mlp,
    "bench_res_mlp": bench_res_mlp,
    "bench_linear": bench_linear,
    "symm_upt": symm_upt,
    "symm_abupt": symm_abupt,
    "symm_aero_upt": symm_aero_upt,
    "symm_aero_abupt": symm_aero_abupt,
}

assert len(ARCHITECTURES_CATALOG) == 93, (
    f"Expected 93 architectures, got {len(ARCHITECTURES_CATALOG)}"
)


# catalog id -> registry key, for the entries the arena renamed (the registry kept the original keys)
REGISTRY_KEYS: dict[str, str] = {
    "symm_transformer": "noether_transformer",
    "symm_transolver": "noether_transolver",
    "symm_aero_transformer": "noether_aero_transformer",
    "symm_aero_transolver": "noether_aero_transolver",
    "symm_upt": "noether_upt",
    "symm_abupt": "noether_abupt",
    "symm_aero_upt": "noether_aero_upt",
    "symm_aero_abupt": "noether_aero_abupt",
    "icosahedral_gnn": "graphcast",
}


def registry_key(arch_id: str) -> str:
    """The ``pp.build_model`` key of a catalog entry."""
    return REGISTRY_KEYS.get(arch_id, arch_id)


def catalog_status() -> dict[str, bool]:
    """``{catalog id: registered}``: whether the PINNeAPPle registry has the entry's model at all."""
    import pinneapple as pp
    registered = set(pp.list_models())
    return {a: registry_key(a) in registered for a in ARCHITECTURES_CATALOG}
