"""pinneapple_tools.nvidia -- PINNeAPPle with the NVIDIA stack (PhysicsNeMo, Omniverse / OpenUSD, Isaac Sim,
Cosmos, NIM).

PINNeAPPle's part is the physics: the problem, the reference solvers and the checks every result must pass before
anyone uses it. The NVIDIA part trains (PhysicsNeMo), shows and simulates (Omniverse, Isaac Sim), generates
(Cosmos) or answers (NIM). This package holds the glue so the examples in ``examples/nvidia`` stay thin:

``diagram``     architecture figures of a pipeline (who does what; which steps need a GPU or an API key)
``omniverse``   results as OpenUSD stages with fields as primvars, a stage validator and a read-back preview
``isaac``       Isaac Sim-ready articulations (``UsdPhysics``), their checks, and the Isaac Sim runner script
``nim``         a client for NIM microservices (OpenAI-compatible, ``nvext.guided_json``) and a physics guard
``cosmos``      clients for Cosmos world foundation models (``/v1/infer``) and Cosmos Reason, and physics
                checks of a video (ballistic fit, conservation of the tracked object)
``physicsnemo`` small PhysicsNeMo surrogates (``FullyConnected``) trained and checked the PINNeAPPle way; grid
                operators (FNO, MeshGraphNet) are in ``pinneapple_neural.physicsnemo_bridge``
``testing``     a local fake server speaking the NIM / Cosmos HTTP contracts, for tests without a key or GPU
``runinfo``     versions, hardware and timings of a run, and which NVIDIA components ran
``figures``     field panels, check charts, curves

Every NVIDIA dependency is optional: ``usd-core`` for USD, ``nvidia-physicsnemo`` for PhysicsNeMo, ``requests``
for the HTTP clients. Isaac Sim, Omniverse Kit and the Cosmos models need an NVIDIA GPU; NIM on build.nvidia.com
needs ``NVIDIA_API_KEY``.
"""
