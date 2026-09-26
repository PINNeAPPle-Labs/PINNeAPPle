"""Catalog of public Physics AI resources: datasets, CAD/geometry sets,
pretrained models and hosted benchmarks, each with a checked license.

The evaluation behind every entry (what it is, license, size, how we use
it) lives in ``PINNeAPPle-CFD/docs/7.Recursos-publicos.md`` (research of
2026-09-23) plus the PLAID and PressNet checks of 2026-09-24. This module
turns that document into something code can query and enforce: a
component that wants a resource asks the catalog first, and the same
license gate as ``pinneapple_neural._licencas`` refuses non-commercial or
unlicensed resources when ``PINNEAPPLE_COMMERCIAL_MODE=1``.

Rules, matching the research doc:

- ``commercial`` is ``"yes"`` only when the license was read and allows
  commercial use. ``"no"`` is an explicit non-commercial / non-production
  license. ``"unverified"`` covers everything else (no license file, data
  license not declared, conflicting tags) and is treated like ``"no"``:
  a missing license means all rights reserved.
- ``paper`` is filled only with a reference whose arXiv id / DOI was
  checked against the title. Unknown stays ``None``; nothing is invented.
- ``rating`` reuses the doc's classification: ``use_now``, ``product``
  (product level, after the pilot), ``reference`` (reference / internal
  benchmark only) and ``skip`` (outside the CFD scope).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pinneapple_neural._licencas import require_research_use

CHECKED_ON = "2026-09-24"

KINDS = ("dataset", "geometry", "model", "benchmark")
RATINGS = ("use_now", "product", "reference", "skip")
COMMERCIAL = ("yes", "no", "unverified")


@dataclass(frozen=True)
class Resource:
    id: str
    name: str
    kind: str
    domain: str
    summary: str
    license: str
    commercial: str
    url: str
    rating: str
    paper: Optional[str] = None
    size: Optional[str] = None
    hf_repo: Optional[Tuple[str, str]] = None  # (repo_id, "dataset" | "model")
    notes: str = ""
    tags: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def research_only(self) -> bool:
        return self.commercial != "yes"

    def license_notice(self) -> str:
        return (
            f"{self.name} is RESEARCH ONLY in PINNeAPPle: license '{self.license}' "
            f"(commercial use: {self.commercial}, checked on {CHECKED_ON}). Use it for internal "
            "benchmarks only; never to train, serve or deliver models or services to clients."
        )


def _r(**kw) -> Resource:
    kw["tags"] = tuple(kw.get("tags", ()))
    return Resource(**kw)


_RESOURCES: List[Resource] = [
    # ── Fluids / CFD datasets ────────────────────────────────────────────
    _r(id="pdebench", name="PDEBench", kind="dataset", domain="cfd",
       summary="Advection, Burgers, diffusion-reaction, Darcy 2D, shallow water, compressible NS 1D/2D/3D, "
               "incompressible NS 2D. HDF5 [b,t,x...,v]; FNO/U-Net/PINN weights (CC BY 4.0, darus-2987).",
       license="CC-BY-4.0 (data), MIT (code)", commercial="yes",
       url="https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi:10.18419/darus-2986",
       paper="Takamoto et al., PDEBench, NeurIPS 2022 Datasets & Benchmarks, arXiv:2210.07182",
       size="3.9 TB (Darcy 2D: 1.31 GB per file)", rating="use_now", tags=("benchmark", "fno")),
    _r(id="airfrans", name="AirfRANS", kind="dataset", domain="cfd",
       summary="1000 OpenFOAM v2112 incompressible RANS cases over NACA airfoils, Re 2-6e6, AoA -5..15 deg; "
               ".vtu/.vtp plus raw OpenFOAM data. Closest public mirror of our OpenFOAM pipeline.",
       license="ODbL-1.0 (data), MIT (code)", commercial="yes",
       url="https://airfrans.readthedocs.io/",
       paper="Bonnet et al., AirfRANS, NeurIPS 2022 Datasets & Benchmarks, arXiv:2212.07564",
       size="10.0 GB", rating="use_now", notes="Share-alike on published derived databases.",
       tags=("openfoam", "rans", "airfoil")),
    _r(id="flowbench", name="FlowBench", kind="dataset", domain="cfd",
       summary="10k+ 2D/3D DNS of NS with and without heat transfer around parametric/non-parametric obstacles.",
       license="CC-BY-NC-4.0", commercial="no", url="https://huggingface.co/datasets/BGLab/FlowBench",
       paper="Tali et al., FlowBench, arXiv:2409.18032", size="1.72 TB",
       hf_repo=("BGLab/FlowBench", "dataset"), rating="reference"),
    _r(id="blastnet", name="BLASTNet", kind="dataset", domain="cfd",
       summary="High-fidelity DNS/LES of reacting and non-reacting flows; 100+ pretrained weights.",
       license="CC-BY-NC-SA-4.0 (data)", commercial="no", url="https://blastnet.github.io/",
       paper="Chung et al., BLASTNet 2.0, NeurIPS 2023 Datasets & Benchmarks, arXiv:2309.13457",
       size="4.8 TB", rating="reference", tags=("combustion", "turbulence")),
    _r(id="the_well", name="The Well", kind="dataset", domain="multiphysics",
       summary="16 datasets / 15 TB: shear_flow, rayleigh_benard, turbulent_radiative_layer, "
               "euler_multi_quadrants, MHD, supernova... Uniform HDF5, HF streaming, FNO/TFNO/U-Net baselines.",
       license="CC-BY-4.0 (data on HF), BSD-3-Clause (code)", commercial="yes",
       url="https://github.com/PolymathicAI/the_well",
       paper="Ohana et al., The Well, NeurIPS 2024 Datasets & Benchmarks, arXiv:2412.00568",
       size="6.9 GB .. 5.1 TB per dataset", rating="use_now",
       notes="euler_multi_quadrants_openBC, MHD_64, turbulent_radiative_layer_3D and supernova_128 have no "
             "license tag on HF (unverified); per-model licenses of polymathic-ai/* unverified."),
    _r(id="realpdebench", name="RealPDEBench", kind="dataset", domain="cfd",
       summary="Real experimental data paired with simulation: cylinder wake, controlled cylinder, tandem FSI, "
               "NACA0025, 3D LES combustion. Models (10 architectures, 3 regimes) are CC BY 4.0.",
       license="CC-BY-NC-4.0 (data); CC-BY-4.0 (models)", commercial="no",
       url="https://realpdebench.westlake.edu.cn/", size="~922 GB", rating="reference",
       notes="Copy the sim-to-real protocol, not the data."),
    _r(id="meshgraphnets_data", name="MeshGraphNets datasets", kind="dataset", domain="cfd",
       summary="cylinder_flow, airfoil, deforming_plate, flag_*, sphere_* mesh-based simulations (TFRecord).",
       license="Apache-2.0 (repo); data license not declared", commercial="unverified",
       url="https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets",
       paper="Pfaff et al., Learning Mesh-Based Simulation with Graph Networks, ICLR 2021, arXiv:2010.03409",
       size="cylinder_flow train 13.6 GB; airfoil train 50.5 GB", rating="use_now",
       notes="Canonical check for our MeshGraphNet; internal benchmark until the data license is confirmed.",
       tags=("gnn",)),
    _r(id="drivaernet", name="DrivAerNet / DrivAerNet++", kind="dataset", domain="cfd",
       summary="8150 parametric cars: volume fields, surface Cp/WSS, point clouds, STL, coefficients.",
       license="CC-BY-NC-4.0 (data), MIT (code)", commercial="no",
       url="https://github.com/Mohamedelrefaie/DrivAerNet",
       paper="Elrefaie et al., DrivAerNet++, NeurIPS 2024 Datasets & Benchmarks, arXiv:2406.09624; "
             "DrivAerNet, arXiv:2403.08055",
       size="39 TB", rating="reference",
       notes="README forbids commercial model training. The lnkd.in link pointed to a fork (roharon/drivaernet)."),
    _r(id="drivaerml", name="DrivAerML", kind="dataset", domain="cfd",
       summary="High-fidelity road-car external aerodynamics (hybrid RANS-LES); base of DoMINO.",
       license="CC-BY-SA-4.0", commercial="yes", url="https://huggingface.co/datasets/neashton/drivaerml",
       paper="Ashton et al., DrivAerML, arXiv:2408.11969", size="31 TB",
       hf_repo=("neashton/drivaerml", "dataset"), rating="product",
       notes="Commercial alternative to DrivAerNet; share-alike."),
    # ── PLAID benchmarks (HF PLAIDcompetitions + PLAID-datasets) ─────────
    *[
        _r(id=f"plaid_{key}", name=f"PLAID {label}", kind="benchmark", domain=domain,
           summary=f"{what} Hosted benchmark with leaderboard (HF space PLAIDcompetitions/{space}); "
                   "data in the PLAID format (lib PLAID-lib/plaid, BSD-3-Clause).",
           license="CC-BY-SA-4.0", commercial="yes",
           url=f"https://huggingface.co/datasets/PLAID-datasets/{ds}",
           hf_repo=(f"PLAID-datasets/{ds}", "dataset"), rating=rating, tags=("plaid",) + extra)
        for key, label, ds, space, domain, what, rating, extra in [
            ("rotor37", "Rotor37", "Rotor37", "Rotor37Benchmark", "turbomachinery",
             "NASA Rotor 37 transonic compressor rotor, 3D.", "product", ("turbomachinery",)),
            ("vki_ls59", "VKI-LS59", "VKI-LS59", "VKILS59Benchmark", "turbomachinery",
             "VKI LS59 turbine blade cascade, 2D.", "product", ("turbomachinery",)),
            ("2d_profile", "2D_profile", "2D_profile", "2DprofileBenchmark", "cfd",
             "2D airfoil profiles.", "reference", ()),
            ("tensile2d", "Tensile2d", "Tensile2d", "Tensile2dBenchmark", "structural",
             "2D tensile test, nonlinear solid mechanics.", "reference", ()),
            ("multiscale_hyperelasticity", "2D_Multiscale_Hyperelasticity", "2D_Multiscale_Hyperelasticity",
             "2DMultiscaleHyperelasticityBenchmark", "structural",
             "2D multiscale hyperelasticity.", "reference", ()),
            ("elastoplastodynamics", "2D_ElastoPlastoDynamics", "2D_ElastoPlastoDynamics",
             "2DElastoPlastoDynamics", "structural", "2D elasto-plastic dynamics.", "reference", ()),
        ]
    ],
    _r(id="plaid_airfrans", name="PLAID AirfRANS (original / clipped / remeshed)", kind="dataset", domain="cfd",
       summary="AirfRANS converted to the PLAID format in three variants.",
       license="ODbL-1.0", commercial="yes", url="https://huggingface.co/datasets/PLAID-datasets/AirfRANS_original",
       paper="Bonnet et al., AirfRANS, arXiv:2212.07564",
       hf_repo=("PLAID-datasets/AirfRANS_original", "dataset"), rating="use_now", tags=("plaid",)),
    _r(id="posteriorbench", name="PosteriorBench", kind="benchmark", domain="inverse_problems",
       summary="Four inverse tasks (Poisson, Darcy, light transport, CCS) with high-fidelity reference "
               "posteriors; judges whether generative solvers recover the whole posterior, not only the mean. "
               "Metrics ported to pinneapple_analysis.uncertainty.posterior_metrics.",
       license="CC-BY-4.0 (data), MIT (code)", commercial="yes",
       url="https://github.com/neuraloperator/PosteriorBench",
       paper="PosteriorBench: From Point Estimates to Posterior Matching in Evaluating Generative Inverse "
             "Solvers, NeurIPS 2026, arXiv:2609.20794",
       hf_repo=("anonymousmay/PosteriorBench", "dataset"), rating="use_now", tags=("uq", "inverse")),
    _r(id="finalrev_abc_embeddings", name="ABC 1M: preview images, captions and text embeddings (finalrev)",
       kind="geometry", domain="cad",
       summary="~1M ABC parts, each with a rendered preview, a VLM caption and a text embedding (100 parquet "
               "shards, 42 GB); no geometry, only abc_id/URIs. Embedding model not published. Indexed by "
               "pinneapple_design.geometry.retrieval.load_caption_index.",
       license="none declared (ABC/Onshape origin)", commercial="unverified",
       url="https://www.finalrev.com/blog/embedding-one-million-3d-models",
       hf_repo=("daveferbear/3d-model-images-embeddings", "dataset"), size="42 GB", rating="reference"),
    _r(id="vista_ssa", name="VISTA-SSA", kind="model", domain="space",
       summary="Attention-based multi-agent RL for space-situational-awareness sensor tasking (1 to 48 sensors, "
               "up to 20k tracked objects); native C environments on PufferLib, frozen checkpoints, classical "
               "scheduling baselines. Fits PINNeAPPle-apps satellite_conjunction_screening.",
       license="MIT", commercial="yes", url="https://github.com/RocketNeurons/VISTA-SSA", rating="reference",
       notes="Linux x86_64 / WSL2 only (compiled environments)."),
    _r(id="prism_reachability", name="PRISM (reachability-intercept set model)", kind="benchmark", domain="space",
       summary="Set-based finite-time encounter feasibility under uncertainty, latency and bounded acceleration. "
               "Only the generic, civilian core (collision screening / rendezvous) is implemented, research only: "
               "pinneapple_analysis.uncertainty.reachability. No targeting or guidance.",
       license="CC-BY-4.0", commercial="yes", url="https://zenodo.org/records/22979350",
       paper="Enayati, PRISM: Predictive Reachability-Intercept Set Model for Relative Motion and Intercept "
             "Kinematics in Defensive Aerospace Engineering, Zenodo 22979350", rating="reference"),
    # ── Structural / manufacturing ───────────────────────────────────────
    _r(id="pressnet", name="PressNet", kind="dataset", domain="structural",
       summary="Press forming of a plate between two dies: 15 die shapes x 10 variations = 150 simulations, "
               "1500 steps (dt 0.01 s), coarse/medium/fine meshes; deformation + stress; GNN baseline.",
       license="none (no LICENSE file)", commercial="unverified",
       url="https://github.com/AnK-Accelerated-Komputing/PressNet",
       paper="Panta et al., PRESSNET, ASME IDETC/CIE 2025, doi:10.1115/DETC2025-163821",
       rating="reference", notes="Dataset still being released (medium mesh pending per README)."),
    _r(id="mechanical_mnist", name="Mechanical MNIST", kind="dataset", domain="structural",
       summary="70k hyperelastic (Neo-Hooke) FEA runs on 28x28 MNIST-heterogeneous domains.",
       license="CC-BY-SA-3.0 (data), MIT (code)", commercial="yes",
       url="https://github.com/elejeune11/Mechanical-MNIST", rating="skip"),
    # ── Materials / Earth (outside the CFD scope, kept for completeness) ─
    _r(id="materials_project", name="Materials Project", kind="dataset", domain="materials",
       summary="DFT material properties.", license="CC-BY-4.0", commercial="yes",
       url="https://materialsproject.org/", rating="skip"),
    _r(id="oqmd", name="OQMD", kind="dataset", domain="materials", summary="1.4M DFT materials.",
       license="CC-BY-4.0", commercial="yes", url="https://oqmd.org/", rating="skip"),
    _r(id="open_catalyst", name="Open Catalyst (OC20/OC22)", kind="dataset", domain="materials",
       summary="DFT relaxations of catalysts.", license="not declared on the home page", commercial="unverified",
       url="https://opencatalystproject.org/", rating="skip"),
    _r(id="era5", name="ERA5", kind="dataset", domain="earth", summary="ECMWF climate reanalysis.",
       license="Copernicus licence (CC-BY since 2025-07)", commercial="yes",
       url="https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5", rating="skip"),
    _r(id="usgs_earthquakes", name="USGS Earthquake Catalog", kind="dataset", domain="earth",
       summary="ComCat earthquake catalog.", license="public domain", commercial="yes",
       url="https://earthquake.usgs.gov/earthquakes/search/", rating="skip"),
    # ── Geometry / CAD ───────────────────────────────────────────────────
    _r(id="abc", name="ABC Dataset", kind="geometry", domain="cad",
       summary="1M Onshape CAD models (STEP, Parasolid, OBJ, STL, features).",
       license="Onshape Terms 1.g.ii (data), MIT (code)", commercial="unverified",
       url="https://deep-geometry.github.io/abc-dataset/",
       paper="Koch et al., ABC, CVPR 2019, arXiv:1812.06216", rating="reference"),
    _r(id="fusion360_gallery", name="Fusion 360 Gallery (+ Segmentation)", kind="geometry", domain="cad",
       summary="Construction sequences, assemblies, B-Rep segmentation (35,680 models).",
       license="Autodesk non-commercial research license", commercial="no",
       url="https://github.com/AutodeskAILab/Fusion360GalleryDataset",
       paper="Willis et al., Fusion 360 Gallery, SIGGRAPH 2021, arXiv:2010.02392", rating="reference"),
    _r(id="mfcad", name="MFCAD", kind="geometry", domain="cad",
       summary="STEP models labelled with machining features.", license="MIT", commercial="yes",
       url="https://github.com/hducg/MFCAD", rating="reference"),
    _r(id="mfcadpp", name="MFCAD++ / MFInstSeg (via AAGNet)", kind="geometry", domain="cad",
       summary="60k+ STEP with machining-feature and topology annotations.",
       license="MIT (AAGNet code); data license not declared", commercial="unverified",
       url="https://github.com/whjdark/AAGNet", rating="reference"),
    _r(id="fabricad", name="FabriCAD", kind="geometry", domain="cad",
       summary="Synthetic STEP CAD paired with manufacturing process plans.",
       license="CC-BY-NC-4.0 (data), MIT (code)", commercial="no",
       url="https://cimtt-kiel.github.io/FabriCAD/", rating="skip"),
    _r(id="ucsm", name="U-Channel (UCSM)", kind="geometry", domain="cad",
       summary="2533 U-channel sheet geometries: STEP, graph, mesh, point cloud, deep-drawing deformations.",
       license="CC-BY-4.0", commercial="yes", url="https://zenodo.org/records/15327950",
       size="17.5 GB (STEP only: 411 MB)", rating="reference"),
    _r(id="nist_step_pmi", name="NIST CAD models / STEP + PMI", kind="geometry", domain="cad",
       summary="NIST MBE PMI test models in STEP with GD&T/PMI.", license="NIST open license",
       commercial="yes", url="https://catalog.data.gov/dataset/nist-cad-models-and-step-files-with-pmi",
       rating="use_now", notes="Test cases for the STEP importer (E3)."),
    _r(id="deepcad", name="DeepCAD", kind="geometry", domain="cad",
       summary="~178k Onshape construction sequences.", license="MIT (code); data license not declared",
       commercial="unverified", url="https://github.com/rundiwu/DeepCAD",
       paper="Wu et al., DeepCAD, ICCV 2021, arXiv:2105.09492", rating="reference"),
    _r(id="cad_steps", name="CAD-Steps", kind="geometry", domain="cad",
       summary="215,096 models, ~750k intermediate STEP states (derived from DeepCAD).",
       license="CC-BY-4.0 on the card; no HF tag; Onshape origin", commercial="unverified",
       url="https://huggingface.co/datasets/amzyst1/cad-steps", hf_repo=("amzyst1/cad-steps", "dataset"),
       size="~13.5-16 GB", rating="reference"),
    _r(id="pie2f_cad", name="PIE2F-CAD", kind="geometry", domain="cad",
       summary="100k instruction <-> CAD-program pairs; 5k with render, STL and STEP (Parquet).",
       license="Apache-2.0", commercial="yes", url="https://huggingface.co/datasets/Parergon/pie2f-cad",
       hf_repo=("Parergon/pie2f-cad", "dataset"), size="3.51 GB", rating="reference",
       notes="Gated: login and contact sharing required."),
    _r(id="thingi10k", name="Thingi10K", kind="geometry", domain="mesh",
       summary="10k Thingiverse STL: 50% non-solid, 45% self-intersecting, 22% non-manifold.",
       license="Apache-2.0 (code); per-model licenses", commercial="unverified",
       url="https://github.com/Thingi10K/Thingi10K",
       paper="Zhou & Jacobson, Thingi10K, arXiv:1605.04797", rating="use_now",
       notes="Filter to CC BY / CC0 models; best public 'dirty client STL' test set for E3."),
    _r(id="shapenet", name="ShapeNet", kind="geometry", domain="mesh",
       summary="~51k annotated models in ShapeNetCore.", license="non-commercial research terms",
       commercial="no", url="https://shapenet.org/",
       paper="Chang et al., ShapeNet, arXiv:1512.03012", rating="reference",
       notes="GeoPT was pre-trained on ShapeNet geometries."),
    _r(id="objaverse", name="Objaverse (1.0 / XL)", kind="geometry", domain="mesh",
       summary="800k+ 3D objects (1.0); 10M+ in XL.", license="ODC-By-1.0 base; per-object licenses",
       commercial="unverified", url="https://objaverse.allenai.org/",
       paper="Deitke et al., Objaverse, CVPR 2023, arXiv:2212.08051", rating="product",
       notes="Only the CC BY / CC0 subset is commercially usable; filter by the per-object license."),
    _r(id="huhb3d", name="Huhb3D Industrial Topology", kind="geometry", domain="cad",
       summary="20 industrial parts, 863 labelled B-Rep faces, STEP + mesh triangles.",
       license="CC-BY-4.0 annotations, CC0 models (HF tags); card lists a commercial price",
       commercial="unverified", url="https://huggingface.co/datasets/Hgodwarrior/Huhb3D-Industrial-Topology",
       hf_repo=("Hgodwarrior/Huhb3D-Industrial-Topology", "dataset"), size="4.3 MB", rating="reference"),
    # ── Pretrained models / architectures ────────────────────────────────
    _r(id="geopt", name="GeoPT", kind="model", domain="cfd",
       summary="Lifted geometric pre-training (Transolver backbone, 8 layers); fine-tunes cars, aircraft, "
               "ships, crash with 20-60% less labelled data.",
       license="MIT (weights and downstream data); code has no license", commercial="unverified",
       url="https://github.com/Physics-Scaling/GeoPT",
       paper="Wu et al., GeoPT, ICML 2026, arXiv:2602.20399", size="15.6 MB (GeoPT_8layers.pt)",
       hf_repo=("GeoPT/GeoPT_Pretrained_Models", "model"), rating="product",
       notes="Weights are MIT but pre-training used ShapeNet (non-commercial): legal review before product use."),
    _r(id="ab_upt", name="AB-UPT", kind="model", domain="cfd",
       summary="Anchored-branched universal physics transformer for automotive CFD (DrivAerML).",
       license="CC-BY-NC-4.0 (weights); Noether framework ENPL (non-production)", commercial="no",
       url="https://github.com/Emmi-AI/anchored-branched-universal-physics-transformers",
       paper="Alkin et al., AB-UPT, arXiv:2502.09692", hf_repo=("EmmiAI/AB-UPT", "model"), rating="reference",
       notes="Already bridged in pinneapple_neural.architectures.neural_operators.noether_bridge (research only)."),
    _r(id="dpot", name="DPOT", kind="model", domain="cfd",
       summary="Auto-regressive denoising operator transformer pre-trained on FNO data, PDEBench, PDEArena, "
               "CFDBench; Tiny 7M .. Huge 1.03B.",
       license="Apache-2.0 (HF weights)", commercial="yes", url="https://huggingface.co/hzk17/DPOT",
       paper="Hao et al., DPOT, ICML 2024, arXiv:2403.03542", hf_repo=("hzk17/DPOT", "model"),
       rating="use_now", notes="The lnkd.in link (AI4Science-WestlakeU/dpot) returns 404; code is HaoZhongkai/DPOT."),
    _r(id="transolver", name="Transolver", kind="model", domain="cfd",
       summary="Physics-attention transformer on general geometries (AirfRANS, ShapeNetCar folders).",
       license="MIT", commercial="yes", url="https://github.com/thuml/Transolver",
       paper="Wu et al., Transolver, ICML 2024, arXiv:2402.02366", rating="product"),
    _r(id="domino", name="DoMINO", kind="model", domain="cfd",
       summary="Decomposable multi-scale iterative neural operator for external aero on STL.",
       license="Apache-2.0 (code in PhysicsNeMo); checkpoint under NVIDIA Community Model License",
       commercial="yes", url="https://github.com/NVIDIA/physicsnemo",
       paper="Ranade et al., DoMINO, arXiv:2501.13350", rating="product",
       notes="'yes' covers the Apache code only: train on our own data instead of using the checkpoint."),
    _r(id="meshgraphnet", name="MeshGraphNet", kind="model", domain="cfd",
       summary="Mesh GNN (TF1 reference); no official checkpoint.", license="Apache-2.0", commercial="yes",
       url="https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets",
       paper="Pfaff et al., ICLR 2021, arXiv:2010.03409", rating="use_now"),
    _r(id="oformer", name="OFormer", kind="model", domain="cfd", summary="Operator transformer; no weights.",
       license="MIT", commercial="yes", url="https://github.com/BaratiLab/OFormer",
       paper="Li et al., Transformer for PDEs' Operator Learning, TMLR, arXiv:2205.13671", rating="reference"),
    _r(id="disco", name="DISCO (neural operator splitting)", kind="model", domain="cfd",
       summary="Test-time generalization via neural operator splitting.", license="MIT", commercial="yes",
       url="https://github.com/LouisSerrano/neural-operator-splitting", rating="reference"),
    _r(id="hyena_no", name="Hyena Neural Operator", kind="model", domain="cfd",
       summary="Operator with Hyena long convolutions.", license="none (no LICENSE file)",
       commercial="unverified", url="https://github.com/Saupatil07/Hyena-Neural-Operator",
       paper="Patil et al., Hyena Neural Operator for PDEs, arXiv:2306.16524", rating="reference"),
    _r(id="rno", name="RNO (Radon Neural Operator)", kind="model", domain="porous_media",
       summary="Operator in the sinogram domain, Darcy 421^2.", license="MIT", commercial="yes",
       url="https://huggingface.co/OneScience-Group/RNO", hf_repo=("OneScience-Group/RNO", "model"),
       rating="reference"),
    _r(id="multigrid_no", name="Physics-informed multi-grid neural operator", kind="model",
       domain="porous_media", summary="Multigrid operator for porous-media flow.", license="MIT",
       commercial="yes", url="https://github.com/SuihongSong/Physics-informed_multi-grid_neural_operator",
       rating="reference"),
    _r(id="deq_no", name="DEQ Neural Operators", kind="model", domain="pde",
       summary="Deep-equilibrium operators for steady-state PDEs (NeurIPS 2023).",
       license="none (no LICENSE file)", commercial="unverified",
       url="https://github.com/risteskilab/deq-neural-operators", rating="reference"),
    _r(id="pfem", name="PFEM (pretrained FEM)", kind="model", domain="structural",
       summary="Physics-informed pre-training (Transolver) to warm-start FEM.",
       license="none (no LICENSE file)", commercial="unverified",
       url="https://github.com/yizheng-wang/PFEM", rating="reference",
       notes="Idea worth reusing: model prediction as the OpenFOAM initial guess (warm start)."),
    _r(id="fen", name="Finite Element Networks", kind="model", domain="structural",
       summary="GNN over finite elements (FEN, T-FEN).", license="MIT", commercial="yes",
       url="https://github.com/martenlienen/finite-element-networks",
       paper="Lienen & Guennemann, ICLR 2022, arXiv:2203.08852", rating="reference"),
    _r(id="realpdebench_models", name="RealPDEBench models", kind="model", domain="cfd",
       summary="10 architectures (DPOT, FNO, Transolver, DeepONet, U-Net, CNO, ...) in 3 regimes.",
       license="CC-BY-4.0", commercial="yes", url="https://huggingface.co/AI4Science-WestlakeU/RealPDEBench-models",
       hf_repo=("AI4Science-WestlakeU/RealPDEBench-models", "model"), rating="reference"),
    _r(id="lgno", name="LGNO", kind="model", domain="conservation_laws",
       summary="Local-global operator for hyperbolic conservation laws (convection, Burgers, shallow water, Euler).",
       license="MIT", commercial="yes", url="https://github.com/shanxue-w/ConservationLaws", rating="reference"),
    _r(id="pinto_kovasznay", name="PINTO (Kovasznay flow)", kind="model", domain="cfd",
       summary="Physics-informed transformer for Kovasznay flow.",
       license="other (source repo unlicensed)", commercial="unverified",
       url="https://huggingface.co/OneScience-Group/PINTO-Kovasznay-Flow",
       hf_repo=("OneScience-Group/PINTO-Kovasznay-Flow", "model"), rating="reference",
       notes="Kovasznay has a closed form: use it as our own verification case, not the weights."),
    _r(id="mantle_convection_no", name="Mantle convection neural operators", kind="model", domain="earth",
       summary="Direct/inverse mantle convection operators, Ra 1e5..1e7.", license="CC-BY-4.0", commercial="yes",
       url="https://data.caltech.edu/records/mx664-fpw98", size="8.3 GB", rating="skip"),
    _r(id="mace", name="MACE", kind="model", domain="materials",
       summary="Equivariant interatomic potential.", license="NOASSERTION (API); README badge MIT",
       commercial="unverified", url="https://github.com/ACEsuit/mace",
       paper="Batatia et al., MACE, NeurIPS 2022, arXiv:2206.07697", rating="skip"),
    _r(id="aimnet2", name="AIMNet2", kind="model", domain="materials",
       summary="Interatomic potential for molecules.", license="MIT", commercial="yes",
       url="https://github.com/isayevlab/aimnetcentral", rating="skip"),
    _r(id="sevennet", name="SevenNet", kind="model", domain="materials",
       summary="GNN interatomic potential.", license="MIT", commercial="yes",
       url="https://github.com/MDIL-SNU/SevenNet", rating="skip"),
]

_BY_ID: Dict[str, Resource] = {}
for _res in _RESOURCES:
    if _res.id in _BY_ID:
        raise ValueError(f"duplicate resource id: {_res.id}")
    if _res.kind not in KINDS or _res.rating not in RATINGS or _res.commercial not in COMMERCIAL:
        raise ValueError(f"invalid field in resource {_res.id}")
    _BY_ID[_res.id] = _res


def list_resources(
    kind: Optional[str] = None,
    domain: Optional[str] = None,
    rating: Optional[str] = None,
    commercial_only: bool = False,
) -> List[Resource]:
    """Resources filtered by kind/domain/rating; ``commercial_only`` keeps ``commercial == "yes"``."""
    out = list(_RESOURCES)
    if kind is not None:
        out = [r for r in out if r.kind == kind]
    if domain is not None:
        out = [r for r in out if r.domain == domain]
    if rating is not None:
        out = [r for r in out if r.rating == rating]
    if commercial_only:
        out = [r for r in out if not r.research_only]
    return out


def get_resource(resource_id: str) -> Resource:
    try:
        return _BY_ID[resource_id]
    except KeyError:
        raise KeyError(f"unknown resource '{resource_id}'. Known: {sorted(_BY_ID)}") from None


def require_allowed(resource_id: str) -> Resource:
    """License gate: warns for research-only resources, refuses them in commercial mode."""
    res = get_resource(resource_id)
    if res.research_only:
        require_research_use(res.name, res.license_notice())
    return res


def fetch(resource_id: str, *, allow_patterns=None, local_dir: Optional[str] = None,
          token: Optional[str] = None, revision: Optional[str] = None) -> str:
    """Download a Hugging Face-hosted resource (after the license gate); returns the local path.

    Use ``allow_patterns`` to fetch a subset: several datasets here are terabytes.
    """
    res = require_allowed(resource_id)
    if res.hf_repo is None:
        raise ValueError(
            f"{res.name} is not hosted on the Hugging Face Hub; get it from {res.url} "
            "(see PINNeAPPle-CFD/docs/7.Recursos-publicos.md section 8 for the access plan)."
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError('fetch requires the "hub" extra: pip install "pinneapple[hub]"') from e
    repo_id, repo_type = res.hf_repo
    return snapshot_download(repo_id=repo_id, repo_type=repo_type, allow_patterns=allow_patterns,
                             local_dir=local_dir, token=token, revision=revision)
