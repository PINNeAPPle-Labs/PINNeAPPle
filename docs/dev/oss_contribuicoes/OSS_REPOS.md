# Repositorios open source de ciencia, tecnologia e pesquisa

Gerado em 2026-10-09 a partir da API do GitHub. Lista completa em `oss_repos.csv` (mesma pasta).

## Como a lista foi feita

1. Organizacoes procuradas: 58 orgs confirmadas no GitHub (empresas, NASA, NOAA, laboratorios do DOE, agencias, comunidades de simulacao e ML cientifico). Total de repositorios publicos nelas: **11,393**.
2. Filtros de 'contribuivel': nao e fork, nao esta arquivado, houve push nos ultimos 12 meses, tem licenca reconhecida (SPDX; `NOASSERTION` entra marcado como licenca propria, ex.: NASA Open Source Agreement), tem aba de issues e ao menos 1 issue aberta.
3. Resultado: **2,793 repositorios** candidatos. Para cada um: issues `good first issue`/`help wanted`, PRs abertos, data do ultimo PR mesclado (proxy de quanto os mantenedores respondem) e se existe CONTRIBUTING.

## Quem nao tem codigo aberto utilizavel

- **PhysicsX** (empresa, physicsx.ai): nao ha organizacao publica no GitHub. A conta `PhysicsX` no GitHub e de outra pessoa (projetos de Jetson/Qt) e nao tem relacao. Nada a listar.
- **CIA**: nao ha organizacao oficial com codigo no GitHub (`cia` nao existe; `cia-foundation` e um projeto de TempleOS sem relacao). A agencia publica ferramentas em outros canais, nao no GitHub. Nada a listar.
- **Luminary Cloud**: 3 repositorios publicos (`tutorials` e forks do Modulus e do Open MPI). Pouco a contribuir; entra na lista pelo que existe.

## Resumo por organizacao

| Org | Repos publicos | Candidatos | Com good-first/help-wanted |
|---|---|---|---|
| NVIDIA | 815 | 308 | 35 |
| NatLabRockies | 816 | 184 | 24 |
| llnl | 676 | 157 | 19 |
| NCAR | 1228 | 148 | 8 |
| nasa | 632 | 146 | 15 |
| SciML | 228 | 142 | 25 |
| ansys | 223 | 132 | 8 |
| google-deepmind | 409 | 129 | 7 |
| NVlabs | 508 | 125 | 1 |
| sandialabs | 857 | 125 | 13 |
| google-research | 363 | 116 | 3 |
| pnnl | 553 | 115 | 2 |
| ecmwf | 172 | 106 | 7 |
| lanl | 664 | 84 | 10 |
| ACCESS-NRI | 181 | 73 | 5 |
| NOAA-EMC | 255 | 59 | 6 |
| nasa-jpl | 155 | 49 | 7 |
| NOAA-OWP | 98 | 44 | 6 |
| idaholab | 161 | 37 | 4 |
| NASA-AMMOS | 99 | 35 | 5 |
| NASA-IMPACT | 319 | 35 | 3 |
| ORNL | 177 | 34 | 2 |
| NOAA-GFDL | 78 | 33 | 10 |
| precice | 64 | 28 | 17 |
| deepmodeling | 67 | 26 | 2 |
| nasa-gcn | 52 | 25 | 1 |
| E3SM-Project | 110 | 25 | 3 |
| Unidata | 132 | 25 | 5 |
| NVIDIA-Omniverse | 91 | 24 | 2 |
| materialsproject | 58 | 21 | 1 |
| esa | 89 | 18 | 3 |
| OPM | 58 | 18 | 0 |
| Autodesk | 88 | 15 | 1 |
| NOAA-PSL | 88 | 14 | 1 |
| NOAA-GSL | 132 | 13 | 0 |
| tum-pbs | 62 | 11 | 2 |
| jax-ml | 15 | 11 | 4 |
| openmm | 28 | 11 | 1 |
| firedrakeproject | 51 | 10 | 2 |
| pangeo-data | 107 | 10 | 3 |
| ufs-community | 30 | 8 | 2 |
| FEniCS | 20 | 8 | 3 |
| ESA-PhiLab | 38 | 7 | 0 |
| noaa-oar-arl | 62 | 6 | 2 |
| NOAA-ORR-ERD | 55 | 6 | 0 |
| noaa-ocs-modeling | 36 | 5 | 0 |
| dealii | 10 | 5 | 1 |
| PolymathicAI | 9 | 5 | 0 |
| NASA-SW-VnV | 8 | 4 | 0 |
| su2code | 13 | 4 | 1 |
| mfem | 12 | 4 | 1 |
| KratosMultiphysics | 10 | 4 | 2 |
| NGSolve | 29 | 3 | 0 |
| altairengineering | 26 | 2 | 0 |
| ecmwf-lab | 31 | 1 | 0 |
| luminarycloud | 3 | 0 | 0 |
| rescale | 30 | 0 | 0 |
| opencae | 12 | 0 | 0 |

## Melhores alvos (good first issue / help wanted + CONTRIBUTING + PRs sendo mesclados)

126 repositorios. Mostrando os 120 primeiros.

| Repo | Estrelas | Lang | Licenca | Issues | GFI | HW | PRs abertos | Ultimo merge | Descricao |
|---|---|---|---|---|---|---|---|---|---|
| [idaholab/moose](https://github.com/idaholab/moose) | 2374 | C++ | LGPL-2.1 | 2853 | 51 | 0 | 168 | 2026-10-07 | Multiphysics Object Oriented Simulation Environment |
| [NVIDIA/cudf-spark](https://github.com/NVIDIA/cudf-spark) | 1009 | Scala | licenca propria (ver LICENSE) | 2053 | 43 | 0 | 70 | 2026-10-09 | NVIDIA cuDF for Apache Spark plugin - accelerate Apache Spark with GPUs |
| [jax-ml/jax](https://github.com/jax-ml/jax) | 36395 | Python | Apache-2.0 | 2623 | 2 | 35 | 866 | 2026-10-09 | Composable transformations of Python+NumPy programs: differentiate, vectorize, JIT to GPU/ |
| [NVIDIA/cudf](https://github.com/NVIDIA/cudf) | 9773 | C++ | Apache-2.0 | 1392 | 31 | 0 | 214 | 2026-09-29 | cuDF - GPU DataFrame Library  |
| [KratosMultiphysics/Kratos](https://github.com/KratosMultiphysics/Kratos) | 1373 | C++ | licenca propria (ver LICENSE) | 752 | 0 | 31 | 305 | 2026-10-09 | Kratos Multiphysics (A.K.A Kratos) is a framework for building parallel multi-disciplinary |
| [E3SM-Project/E3SM](https://github.com/E3SM-Project/E3SM) | 444 | Fortran | licenca propria (ver LICENSE) | 666 | 0 | 29 | 85 | 2026-10-09 | Energy Exascale Earth System Model source code.  NOTE:  use "maint" branches for your work |
| [NVIDIA/DALI](https://github.com/NVIDIA/DALI) | 5772 | C++ | Apache-2.0 | 218 | 2 | 24 | 40 | 2026-10-08 | A GPU-accelerated library containing highly optimized building blocks and an execution eng |
| [NVIDIA/cccl](https://github.com/NVIDIA/cccl) | 2530 | C++ | licenca propria (ver LICENSE) | 1986 | 17 | 9 | 332 | 2026-10-08 | CUDA Core Compute Libraries |
| [SciML/OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) | 692 | Julia | licenca propria (ver LICENSE) | 592 | 10 | 11 | 160 | 2026-10-09 | High performance ordinary differential equation (ODE) and differential-algebraic equation  |
| [precice/precice](https://github.com/precice/precice) | 979 | C++ | LGPL-3.0 | 241 | 15 | 5 | 38 | 2026-09-23 | A coupling library and ecosystem for partitioned multi-physics and multi-scale simulations |
| [idaholab/MontePy](https://github.com/idaholab/MontePy) | 66 | Python | MIT | 107 | 18 | 0 | 3 | 2026-10-08 | MontePy is the most user friendly Python library (API) to read, edit, and write MCNP input |
| [NVIDIA/garak](https://github.com/NVIDIA/garak) | 9508 | Python | Apache-2.0 | 478 | 15 | 0 | 208 | 2026-10-08 | the LLM vulnerability scanner |
| [google-deepmind/formal-conjectures](https://github.com/google-deepmind/formal-conjectures) | 1313 | Lean | Apache-2.0 | 1402 | 14 | 0 | 607 | 2026-10-09 | A collection of formalized statements of conjectures in Lean. |
| [NOAA-GFDL/fre-cli](https://github.com/NOAA-GFDL/fre-cli) | 4 | Python | Apache-2.0 | 140 | 9 | 5 | 26 | 2026-09-30 | Python-based command line interface for FRE (FMS Runtime Environment) to compile and run F |
| [NVIDIA/nvcf](https://github.com/NVIDIA/nvcf) | 231 | Go | Apache-2.0 | 486 | 12 | 0 | 119 | 2026-10-08 | Platform for deploying and routing GPU-accelerated inference, streaming, and batch workloa |
| [NOAA-OWP/ngen](https://github.com/NOAA-OWP/ngen) | 97 | C++ | Apache-2.0 | 203 | 9 | 3 | 59 | 2026-10-07 | Next Generation Water Modeling Engine and Framework Prototype |
| [NVIDIA/cuml](https://github.com/NVIDIA/cuml) | 5302 | Python | Apache-2.0 | 807 | 10 | 1 | 52 | 2026-10-08 | NVIDIA cuML: GPU-Accelerated Machine Learning |
| [Unidata/MetPy](https://github.com/Unidata/MetPy) | 1447 | Python | BSD-3-Clause | 393 | 11 | 0 | 69 | 2026-10-08 | MetPy is a collection of tools in Python for reading, visualizing and performing calculati |
| [precice/tutorials](https://github.com/precice/tutorials) | 143 | C | LGPL-3.0 | 69 | 6 | 4 | 23 | 2026-10-02 | Various tutorial cases for the coupling library preCICE with real solvers. These files are |
| [NOAA-OWP/inundation-mapping](https://github.com/NOAA-OWP/inundation-mapping) | 131 | Python | Apache-2.0 | 308 | 9 | 1 | 23 | 2026-10-08 | Flood inundation mapping and evaluation software configured to work with U.S. National Wat |
| [NOAA-OWP/nwm-post-processing](https://github.com/NOAA-OWP/nwm-post-processing) | 0 | Python | Apache-2.0 | 21 | 5 | 5 | 0 | 2026-09-17 | The post processing application suite for National Water Model output |
| [llnl/maestrowf](https://github.com/llnl/maestrowf) | 161 | Python | MIT | 91 | 9 | 0 | 17 | 2026-10-06 | A tool to easily orchestrate general computational workflows both locally and on supercomp |
| [sandialabs/PEAT](https://github.com/sandialabs/PEAT) | 31 | Python | GPL-3.0 | 41 | 4 | 5 | 5 | 2026-10-01 | The Process Extraction and Analysis Tool (PEAT), a Operational Technology (OT) device data |
| [NVIDIA/cuCollections](https://github.com/NVIDIA/cuCollections) | 672 | Cuda | Apache-2.0 | 65 | 8 | 0 | 14 | 2026-10-08 |  |
| [NVIDIA/cudf-spark-tools](https://github.com/NVIDIA/cudf-spark-tools) | 72 | Scala | Apache-2.0 | 306 | 8 | 0 | 7 | 2026-10-06 | User tools for Spark RAPIDS |
| [llnl/Surfactant](https://github.com/llnl/Surfactant) | 43 | Python | MIT | 75 | 2 | 6 | 19 | 2026-10-08 | Modular framework for file information extraction and dependency analysis to generate accu |
| [NVIDIA/gpu-operator](https://github.com/NVIDIA/gpu-operator) | 2920 | Go | Apache-2.0 | 99 | 6 | 1 | 55 | 2026-10-06 | NVIDIA GPU Operator creates, configures, and manages GPUs in Kubernetes |
| [nasa/cFE](https://github.com/nasa/cFE) | 526 | C | Apache-2.0 | 435 | 7 | 0 | 48 | 2026-10-08 | The Core Flight System (cFS) Core Flight Executive (cFE) |
| [ansys/pyaedt](https://github.com/ansys/pyaedt) | 403 | Python | MIT | 231 | 7 | 0 | 27 | 2026-10-09 | AEDT Python Client Package |
| [ORNL/GridKit](https://github.com/ORNL/GridKit) | 27 | C++ | licenca propria (ver LICENSE) | 83 | 3 | 4 | 16 | 2026-10-07 | Modeling framework for power systems simulations and analysis. |
| [precice/micro-manager](https://github.com/precice/micro-manager) | 24 | Python | LGPL-3.0 | 32 | 5 | 2 | 10 | 2026-10-01 | A manager tool to facilitate two-scale coupling in multi-physics simulations using preCICE |
| [sandialabs/OpenCSP](https://github.com/sandialabs/OpenCSP) | 13 | Python | licenca propria (ver LICENSE) | 91 | 7 | 0 | 0 | 2026-09-17 | Code for Concentrating Solar Power |
| [precice/aste](https://github.com/precice/aste) | 11 | Python | GPL-3.0 | 31 | 6 | 1 | 9 | 2026-07-10 | Artificial Solver Testing Environment |
| [NVIDIA/k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) | 3894 | Go | Apache-2.0 | 74 | 5 | 1 | 43 | 2026-10-08 | NVIDIA device plugin for Kubernetes |
| [llnl/RAJA](https://github.com/llnl/RAJA) | 602 | C++ | BSD-3-Clause | 197 | 0 | 6 | 37 | 2026-10-07 | RAJA Performance Portability Layer (C++) |
| [esa/torchquad](https://github.com/esa/torchquad) | 229 | Python | GPL-3.0 | 15 | 2 | 4 | 8 | 2026-07-24 | Numerical integration in arbitrary dimensions on the GPU using PyTorch / TF / JAX |
| [NVIDIA/nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) | 4592 | Go | Apache-2.0 | 67 | 1 | 4 | 38 | 2026-10-01 | Build and run containers leveraging NVIDIA GPUs |
| [nasa/cFS](https://github.com/nasa/cFS) | 1527 | C | Apache-2.0 | 105 | 4 | 1 | 12 | 2026-10-09 | The Core Flight System (cFS) |
| [NVIDIA/TensorRT-Model-Connect](https://github.com/NVIDIA/TensorRT-Model-Connect) | 274 | Python | Apache-2.0 | 254 | 5 | 0 | 88 | 2026-10-09 | From PyTorch model to end-to-end TensorRT inference experience in two commands—AI-native,  |
| [llnl/axom](https://github.com/llnl/axom) | 197 | C++ | BSD-3-Clause | 251 | 2 | 3 | 29 | 2026-10-07 | CS infrastructure components for HPC applications |
| [precice/openfoam-adapter](https://github.com/precice/openfoam-adapter) | 173 | C++ | GPL-3.0 | 50 | 4 | 1 | 8 | 2026-09-14 | OpenFOAM-preCICE adapter |
| [NOAA-GFDL/FMS](https://github.com/NOAA-GFDL/FMS) | 121 | Fortran | licenca propria (ver LICENSE) | 108 | 0 | 5 | 21 | 2026-09-28 | GFDL's Flexible Modeling System |
| [NatLabRockies/GEOPHIRES-X](https://github.com/NatLabRockies/GEOPHIRES-X) | 66 | Python | MIT | 83 | 5 | 0 | 2 | 2026-10-06 | GEOPHIRES is NREL's free and open-source geothermal techno-economic simulator.  |
| [ansys/pymotorcad](https://github.com/ansys/pymotorcad) | 35 | Python | MIT | 124 | 5 | 0 | 41 | 2026-10-07 |  |
| [NOAA-GFDL/pace](https://github.com/NOAA-GFDL/pace) | 21 | Python | Apache-2.0 | 38 | 4 | 1 | 2 | 2026-10-05 | Re-write of FV3GFS weather/climate model in Python |
| [NVIDIA/OpenShell](https://github.com/NVIDIA/OpenShell) | 15561 | Rust | Apache-2.0 | 582 | 0 | 4 | 192 | 2026-10-09 | OpenShell is the safe, private runtime for autonomous AI agents. |
| [FEniCS/dolfinx](https://github.com/FEniCS/dolfinx) | 1219 | C++ | LGPL-3.0 | 123 | 4 | 0 | 38 | 2026-10-09 | Next generation FEniCS problem solving environment for solving finite element problems |
| [nasa/osal](https://github.com/nasa/osal) | 673 | C | Apache-2.0 | 152 | 4 | 0 | 14 | 2026-10-08 | The Core Flight System (cFS) Operating System Abstraction Layer (OSAL) |
| [NatLabRockies/PVDegradationTools](https://github.com/NatLabRockies/PVDegradationTools) | 49 | Jupyter Notebook | licenca propria (ver LICENSE) | 57 | 1 | 3 | 11 | 2026-10-03 | Set of tools to calculate degradation responses and degradation related parameters for PV. |
| [NatLabRockies/routee-compass](https://github.com/NatLabRockies/routee-compass) | 29 | Rust | BSD-3-Clause | 93 | 4 | 0 | 12 | 2026-09-16 | An energy-aware vehicle routing engine |
| [ansys/pyspeos](https://github.com/ansys/pyspeos) | 23 | Python | MIT | 55 | 4 | 0 | 7 | 2026-10-08 | PySpeos is a Python library that gathers functionalities and tools based on Speos software |
| [NASA-IMPACT/veda-backend](https://github.com/NASA-IMPACT/veda-backend) | 22 | Python | licenca propria (ver LICENSE) | 80 | 3 | 1 | 17 | 2026-09-30 | Backend services for VEDA |
| [NASA-AMMOS/BSL](https://github.com/NASA-AMMOS/BSL) | 15 | C | Apache-2.0 | 26 | 4 | 0 | 1 | 2026-10-06 | Bundle Protocol Security Library (BSL) |
| [sandialabs/pytribeam](https://github.com/sandialabs/pytribeam) | 8 | Python | licenca propria (ver LICENSE) | 24 | 0 | 4 | 0 | 2026-09-30 | Automated data collection for the TriBeam microscope |
| [NOAA-OWP/nwm-coastal](https://github.com/NOAA-OWP/nwm-coastal) | 0 | Python | licenca propria (ver LICENSE) | 3 | 2 | 2 | 1 | 2026-08-10 | A collection of coastal tools designed to implement skill assessments for coastal model ru |
| [NVIDIA/NemoClaw](https://github.com/NVIDIA/NemoClaw) | 22690 | TypeScript | Apache-2.0 | 765 | 1 | 2 | 136 | 2026-10-09 | Run agents like Hermes, LangChain Deep Agents, and OpenClaw more securely inside NVIDIA Op |
| [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | 18096 | Python | licenca propria (ver LICENSE) | 1586 | 3 | 0 | 1129 | 2026-10-09 | Ongoing research training transformer models at scale |
| [NASA-AMMOS/3DTilesRendererJS](https://github.com/NASA-AMMOS/3DTilesRendererJS) | 2482 | JavaScript | Apache-2.0 | 107 | 0 | 3 | 14 | 2026-10-07 | Renderer for 3D Tiles in Javascript using three.js, Babylon.js, and r3f |
| [mfem/mfem](https://github.com/mfem/mfem) | 2259 | C++ | BSD-3-Clause | 299 | 0 | 3 | 170 | 2026-10-07 | Lightweight, general, scalable C++ library for finite element methods |
| [NVIDIA/NVSentinel](https://github.com/NVIDIA/NVSentinel) | 394 | Go | Apache-2.0 | 81 | 0 | 3 | 21 | 2026-10-09 | NVSentinel detects and remediates GPU faults on Kubernetes nodes |
| [NVIDIA/numba-cuda](https://github.com/NVIDIA/numba-cuda) | 296 | Python | BSD-2-Clause | 144 | 1 | 2 | 47 | 2026-10-07 | The CUDA target for Numba |
| [NASA-AMMOS/MMGIS](https://github.com/NASA-AMMOS/MMGIS) | 232 | JavaScript | Apache-2.0 | 100 | 3 | 0 | 65 | 2026-10-07 | Multi-Mission Geographic Information System - A Web-based Mapping and Spatial Data Infrast |
| [google-research/perch-hoplite](https://github.com/google-research/perch-hoplite) | 128 | Python | Apache-2.0 | 26 | 3 | 0 | 12 | 2026-08-12 | Tooling for agile modeling on large machine perception embedding databases. |
| [ansys/pydpf-core](https://github.com/ansys/pydpf-core) | 92 | Python | MIT | 218 | 2 | 1 | 41 | 2026-10-09 | Data Processing Framework - Python Core |
| [NVIDIA/numbast](https://github.com/NVIDIA/numbast) | 62 | Python | Apache-2.0 | 61 | 3 | 0 | 16 | 2026-10-05 | Numbast is a tool to build an automated pipeline that converts CUDA APIs into Numba bindin |
| [ansys/pydpf-post](https://github.com/ansys/pydpf-post) | 58 | Python | MIT | 48 | 3 | 0 | 10 | 2026-09-10 | Data Processing Framework - Post Processing Module |
| [idaholab/HERON](https://github.com/idaholab/HERON) | 33 | Python | Apache-2.0 | 53 | 3 | 0 | 4 | 2026-05-26 | Holistic Energy Resource Optimization Network (HERON) is a modeling toolset and plugin for |
| [sandialabs/sceptre-phenix](https://github.com/sandialabs/sceptre-phenix) | 28 | JavaScript | GPL-3.0 | 51 | 2 | 1 | 19 | 2026-10-07 | phenix is an orchestration tool and GUI for Sandia's minimega platform |
| [sandialabs/firewheel](https://github.com/sandialabs/firewheel) | 10 | Python | licenca propria (ver LICENSE) | 36 | 2 | 1 | 14 | 2026-10-05 | FIREWHEEL is an experiment orchestration tool that assists a user in building and controll |
| [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco) | 15544 | C++ | Apache-2.0 | 286 | 2 | 0 | 141 | 2026-10-09 | Multi-Joint dynamics with Contact. A general purpose physics simulator. |
| [nasa-jpl/open-source-rover](https://github.com/nasa-jpl/open-source-rover) | 9691 | HTML | Apache-2.0 | 17 | 0 | 2 | 5 | 2026-09-03 | A build-it-yourself, 6-wheel rover based on the rovers on Mars! |
| [SciML/ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) | 1693 | Julia | licenca propria (ver LICENSE) | 712 | 2 | 0 | 153 | 2026-10-06 | An acausal modeling framework for automatically parallelized scientific machine learning ( |
| [Autodesk/react-base-table](https://github.com/Autodesk/react-base-table) | 1541 | TypeScript | MIT | 89 | 0 | 2 | 18 | 2026-04-17 | A react table component to display large datasets with high performance and flexibility |
| [llnl/zfp](https://github.com/llnl/zfp) | 887 | C++ | BSD-3-Clause | 43 | 0 | 2 | 6 | 2026-09-29 | Compressed numerical arrays that support high-speed random access |
| [jax-ml/ml_dtypes](https://github.com/jax-ml/ml_dtypes) | 363 | C++ | Apache-2.0 | 43 | 0 | 2 | 13 | 2026-10-08 | A stand-alone implementation of several NumPy dtype extensions used in machine learning. |
| [NOAA-EMC/WW3](https://github.com/NOAA-EMC/WW3) | 349 | Fortran | licenca propria (ver LICENSE) | 241 | 0 | 2 | 16 | 2026-10-05 | WAVEWATCH III |
| [NVIDIA/container-canary](https://github.com/NVIDIA/container-canary) | 308 | Go | Apache-2.0 | 38 | 1 | 1 | 9 | 2026-08-18 | A tool for testing and validating container requirements against versioned manifests |
| [SciML/ReservoirComputing.jl](https://github.com/SciML/ReservoirComputing.jl) | 233 | Julia | MIT | 41 | 2 | 0 | 4 | 2026-09-29 | Reservoir computing utilities for scientific machine learning (SciML) |
| [Unidata/netcdf-java](https://github.com/Unidata/netcdf-java) | 200 | Java | BSD-3-Clause | 73 | 0 | 2 | 2 | 2026-10-01 | The Unidata netcdf-java library |
| [sandialabs/seacas](https://github.com/sandialabs/seacas) | 190 | C | licenca propria (ver LICENSE) | 37 | 0 | 2 | 13 | 2026-10-07 | The Sandia Engineering Analysis Code Access System (SEACAS) is a suite of preprocessing, p |
| [nasa/cFS-GroundSystem](https://github.com/nasa/cFS-GroundSystem) | 104 | Python | Apache-2.0 | 30 | 1 | 1 | 5 | 2026-10-08 | The Core Flight System (cFS) Ground System Lab Tool (cFS-GroundSystem) |
| [Unidata/tds](https://github.com/Unidata/tds) | 83 | Java | BSD-3-Clause | 28 | 1 | 1 | 1 | 2026-10-01 | THREDDS Data Server |
| [nasa/PSP](https://github.com/nasa/PSP) | 82 | C | Apache-2.0 | 49 | 2 | 0 | 6 | 2026-10-08 | The Core Flight System (cFS) Platform Support Package (PSP) |
| [NOAA-GFDL/MDTF-diagnostics](https://github.com/NOAA-GFDL/MDTF-diagnostics) | 81 | Jupyter Notebook | licenca propria (ver LICENSE) | 73 | 0 | 2 | 16 | 2026-09-14 | Analysis framework and collection of process-oriented diagnostics for weather and climate  |
| [NOAA-GFDL/FRE-NCtools](https://github.com/NOAA-GFDL/FRE-NCtools) | 25 | C | LGPL-3.0 | 34 | 0 | 2 | 4 | 2026-09-10 | Tools for manipulating and creating netCDF inputs for FMS managed models |
| [sandialabs/WecOptTool](https://github.com/sandialabs/WecOptTool) | 21 | Python | GPL-3.0 | 33 | 1 | 1 | 4 | 2026-09-30 | WEC Design Optimization Toolbox |
| [sandialabs/sceptre-phenix-apps](https://github.com/sandialabs/sceptre-phenix-apps) | 11 | Python | GPL-3.0 | 36 | 2 | 0 | 17 | 2026-05-14 | Apps written to work with the latest version of phenix |
| [sandialabs/sansmic](https://github.com/sandialabs/sansmic) | 4 | C++ | BSD-3-Clause | 12 | 2 | 0 | 3 | 2026-04-30 | Sandia solution mining code |
| [nasa-jpl/tos2ca-anomaly-detection](https://github.com/nasa-jpl/tos2ca-anomaly-detection) | 1 | Python | licenca propria (ver LICENSE) | 11 | 2 | 0 | 1 | 2026-08-07 | Python library that defines a phenomenon and curates data for the user |
| [pnnl/ieee-std-1815-2-test-tool](https://github.com/pnnl/ieee-std-1815-2-test-tool) | 1 | Rust | BSD-3-Clause | 39 | 2 | 0 | 2 | 2026-10-08 | Profile editor, reference control station, reference outstation, and conformance tests for |
| [ansys/grantami-jobqueue](https://github.com/ansys/grantami-jobqueue) | 0 | Python | MIT | 3 | 2 | 0 | 0 | 2026-10-07 |  |
| [nasa/fprime](https://github.com/nasa/fprime) | 11822 | C++ | Apache-2.0 | 455 | 0 | 1 | 30 | 2026-10-08 | F´ - A flight software and embedded systems framework |
| [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | 8178 | Python | Apache-2.0 | 338 | 0 | 1 | 116 | 2026-10-07 | NVIDIA Isaac GR00T N1.7 -  A Foundation Model for Generalist Robots. |
| [google-deepmind/open_spiel](https://github.com/google-deepmind/open_spiel) | 5522 | C++ | Apache-2.0 | 78 | 0 | 1 | 46 | 2026-08-31 | OpenSpiel is a collection of environments and algorithms for research in general reinforce |
| [NVIDIA/cuda-rust](https://github.com/NVIDIA/cuda-rust) | 3688 | Rust | Apache-2.0 | 158 | 1 | 0 | 48 | 2026-10-08 | NVIDIA's CUDA platform for Rust. Host runtime crates plus Tile (cutile-rs) and SIMT (cuda- |
| [NVIDIA/NeMo-Retriever](https://github.com/NVIDIA/NeMo-Retriever) | 2988 | Python | Apache-2.0 | 253 | 1 | 0 | 119 | 2026-10-08 | NeMo Retriever Library is a scalable, performance-oriented document content and metadata e |
| [google-deepmind/optax](https://github.com/google-deepmind/optax) | 2344 | Python | Apache-2.0 | 136 | 0 | 1 | 104 | 2026-10-04 | Optax is a gradient processing and optimization library for JAX. |
| [google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground) | 2244 | Python | Apache-2.0 | 111 | 0 | 1 | 44 | 2026-10-07 | An open-source library for GPU-accelerated robot learning and sim-to-real transfer. |
| [deepmodeling/deepmd-kit](https://github.com/deepmodeling/deepmd-kit) | 2054 | Python | LGPL-3.0 | 236 | 1 | 0 | 68 | 2026-10-09 | A deep learning package for many-body potential energy representation and molecular dynami |
| [google-deepmind/concordia](https://github.com/google-deepmind/concordia) | 1767 | Python | Apache-2.0 | 41 | 1 | 0 | 17 | 2026-10-01 | A library for generative social simulation |
| [nasa/spacewasm](https://github.com/nasa/spacewasm) | 1606 | Rust | Apache-2.0 | 25 | 1 | 0 | 10 | 2026-09-28 | A flight-compliant WebAssembly interpreter. |
| [NVIDIA/NVFlare](https://github.com/NVIDIA/NVFlare) | 981 | Python | Apache-2.0 | 47 | 1 | 0 | 28 | 2026-10-08 | NVIDIA Federated Learning Application Runtime Environment |
| [jax-ml/oryx](https://github.com/jax-ml/oryx) | 326 | Python | Apache-2.0 | 20 | 1 | 0 | 3 | 2026-10-07 | Oryx is a library for probabilistic programming and deep learning built on top of Jax. |
| [llnl/smith](https://github.com/llnl/smith) | 245 | C++ | BSD-3-Clause | 178 | 0 | 1 | 22 | 2026-10-08 | Smith is a high order nonlinear thermomechanical simulation code |
| [llnl/units](https://github.com/llnl/units) | 173 | C++ | BSD-3-Clause | 6 | 0 | 1 | 1 | 2026-10-01 | A run-time C++ library for working with units of measurement and conversions between them  |
| [FEniCS/basix](https://github.com/FEniCS/basix) | 148 | C++ | MIT | 49 | 1 | 0 | 10 | 2026-09-28 | FEniCSx finite element basis evaluation library |
| [ecmwf/metview-python](https://github.com/ecmwf/metview-python) | 147 | Python | Apache-2.0 | 23 | 1 | 0 | 0 | 2026-05-21 | Python interface to Metview meteorological workstation and batch system |
| [NatLabRockies/solar-data-tools](https://github.com/NatLabRockies/solar-data-tools) | 102 | Jupyter Notebook | licenca propria (ver LICENSE) | 7 | 1 | 0 | 2 | 2026-06-02 | Some data analysis tools for working with historical PV solar time-series data sets.  |
| [ORNL/ReSolve](https://github.com/ORNL/ReSolve) | 85 | C++ | licenca propria (ver LICENSE) | 32 | 1 | 0 | 2 | 2026-10-06 | Library of GPU-resident linear solvers |
| [NOAA-OWP/hydrotools](https://github.com/NOAA-OWP/hydrotools) | 70 | Python | Apache-2.0 | 17 | 1 | 0 | 1 | 2026-06-04 | Suite of tools for retrieving hydrological data and evaluating model output. |
| [nasa/sample_app](https://github.com/nasa/sample_app) | 67 | C | Apache-2.0 | 14 | 1 | 0 | 2 | 2026-10-08 | The Core Flight System (cFS) Sample App (sample_app) |
| [NVIDIA/cudf-spark-jni](https://github.com/NVIDIA/cudf-spark-jni) | 65 | Cuda | Apache-2.0 | 143 | 1 | 0 | 34 | 2026-10-09 | NVIDIA cuDF plugin JNI For Apache Spark |
| [llnl/proteus](https://github.com/llnl/proteus) | 54 | C++ | Apache-2.0 | 65 | 0 | 1 | 5 | 2026-10-08 | Programmable JIT Compilation and Optimization for C/C++ using LLVM |
| [nasa/SC](https://github.com/nasa/SC) | 47 | C | Apache-2.0 | 21 | 1 | 0 | 4 | 2026-10-08 | The Core Flight System (cFS) Stored Commands (SC) application. |
| [nasa/HK](https://github.com/nasa/HK) | 43 | C | Apache-2.0 | 15 | 1 | 0 | 3 | 2026-10-08 | The Core Flight System (cFS) Housekeeping (HK) application. |
| [google-research/tnco](https://github.com/google-research/tnco) | 36 | Python | Apache-2.0 | 3 | 0 | 1 | 1 | 2026-10-08 | TNCO is a heuristic tool that optimizes tensor network contraction paths. |
| [ansys/pyprimemesh](https://github.com/ansys/pyprimemesh) | 34 | Python | MIT | 66 | 1 | 0 | 6 | 2026-10-05 | Pythonic Meshing Client for Ansys Prime Server |
| [sandialabs/spack-manager](https://github.com/sandialabs/spack-manager) | 31 | Python | licenca propria (ver LICENSE) | 15 | 0 | 1 | 2 | 2026-07-29 | A project and machine deployment model using Spack |
| [nasa/elf2cfetbl](https://github.com/nasa/elf2cfetbl) | 28 | C | Apache-2.0 | 26 | 1 | 0 | 4 | 2026-10-08 | The Core Flight System (cFS) ELF to CFE Table Tool (elf2cfetbl) |
| [NCAR/pop-tools](https://github.com/NCAR/pop-tools) | 28 | Python | Apache-2.0 | 35 | 0 | 1 | 9 | 2026-04-02 | Tools to support analysis of POP2-CESM model solutions |

## Por categoria (repos com 30+ estrelas ou com issues para iniciantes)


### Empresas (579 de 851 candidatos)

| Org | Repo | ★ | Lang | Licenca | Issues | GFI | HW | CONTRIB. | Descricao |
|---|---|---|---|---|---|---|---|---|---|
| Autodesk | [react-base-table](https://github.com/Autodesk/react-base-table) | 1541 | TypeScript | MIT | 89 | 0 | 2 | sim | A react table component to display large datasets with high performance and flex |
| Autodesk | [Aurora](https://github.com/Autodesk/Aurora) | 625 | C++ | Apache-2.0 | 22 | 0 | 0 | sim | Real-time GPU path tracing with an OpenUSD Hydra render delegate |
| Autodesk | [XLB](https://github.com/Autodesk/XLB) | 510 | Python | licenca propria (ver LICENSE) | 14 | 0 | 0 | sim | XLB: Accelerated Lattice Boltzmann (XLB) for Physics-based ML |
| Autodesk | [arnold-usd](https://github.com/Autodesk/arnold-usd) | 284 | Python | licenca propria (ver LICENSE) | 149 | 0 | 0 | sim | Arnold components for USD |
| Autodesk | [synthesis](https://github.com/Autodesk/synthesis) | 186 | TypeScript | Apache-2.0 | 18 | 0 | 0 | sim | A Robotics Simulator for Autodesk Fusion CAD Designs |
| Autodesk | [Neon](https://github.com/Autodesk/Neon) | 70 | C++ | licenca propria (ver LICENSE) | 10 | 0 | 0 | sim | Multi-GPU Framework for Voxel Grid Computations  |
| Autodesk | [bifrost-usd](https://github.com/Autodesk/bifrost-usd) | 57 | C++ | Apache-2.0 | 3 | 0 | 0 | sim | Bifrost nodes for USD |
| Autodesk | [AutomaticComponentToolkit](https://github.com/Autodesk/AutomaticComponentToolkit) | 53 | Go | BSD-2-Clause | 45 | 0 | 0 | sim | A toolkit to automatically generate software components: abstract API, implement |
| Autodesk | [PowerShapeAndPowerMillAPI](https://github.com/Autodesk/PowerShapeAndPowerMillAPI) | 52 | C# | MIT | 3 | 0 | 0 | nao | An API for Autodesk PowerShape and PowerMill |
| Autodesk | [AutodeskMachineControlFramework](https://github.com/Autodesk/AutodeskMachineControlFramework) | 46 | C++ | licenca propria (ver LICENSE) | 6 | 0 | 0 | sim | Middleware framework to integrate CAD/CAM software with machine hardware systems |
| Autodesk | [hydra-viewport-toolbox](https://github.com/Autodesk/hydra-viewport-toolbox) | 37 | C++ | Apache-2.0 | 7 | 0 | 0 | sim | Utilities to support graphics viewports using OpenUSD Hydra |
| NVIDIA | [NemoClaw](https://github.com/NVIDIA/NemoClaw) | 22690 | TypeScript | Apache-2.0 | 765 | 1 | 2 | sim | Run agents like Hermes, LangChain Deep Agents, and OpenClaw more securely inside |
| NVIDIA | [SkillSpector](https://github.com/NVIDIA/SkillSpector) | 19754 | Python | Apache-2.0 | 145 | 0 | 0 | sim | Security scanner for AI agent skills. Detect vulnerabilities, malicious patterns |
| NVIDIA | [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | 18096 | Python | licenca propria (ver LICENSE) | 1586 | 3 | 0 | sim | Ongoing research training transformer models at scale |
| NVIDIA | [open-gpu-kernel-modules](https://github.com/NVIDIA/open-gpu-kernel-modules) | 17449 | C | licenca propria (ver LICENSE) | 575 | 0 | 0 | sim | NVIDIA Linux open GPU kernel module source |
| NVIDIA | [OpenShell](https://github.com/NVIDIA/OpenShell) | 15561 | Rust | Apache-2.0 | 582 | 0 | 4 | sim | OpenShell is the safe, private runtime for autonomous AI agents. |
| NVIDIA | [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | 14780 | Python | licenca propria (ver LICENSE) | 1597 | 0 | 0 | sim | TensorRT LLM provides users with an easy-to-use Python API to define Large Langu |
| NVIDIA | [TensorRT](https://github.com/NVIDIA/TensorRT) | 13392 | C++ | Apache-2.0 | 652 | 0 | 0 | sim | NVIDIA® TensorRT™ is an SDK for high-performance deep learning inference on NVID |
| NVIDIA | [cosmos](https://github.com/NVIDIA/cosmos) | 12000 | Jupyter Notebook | licenca propria (ver LICENSE) | 61 | 0 | 0 | sim | NVIDIA Cosmos is an open platform of world models, datasets, and tools that enab |
| NVIDIA | [personaplex](https://github.com/NVIDIA/personaplex) | 10598 | Python | MIT | 69 | 0 | 0 | nao | PersonaPlex code. |
| NVIDIA | [cutlass](https://github.com/NVIDIA/cutlass) | 10547 | C++ | licenca propria (ver LICENSE) | 781 | 2 | 4 | nao | CUDA Templates and Python DSLs for High-Performance Linear Algebra |
| NVIDIA | [cudf](https://github.com/NVIDIA/cudf) | 9773 | C++ | Apache-2.0 | 1392 | 31 | 0 | sim | cuDF - GPU DataFrame Library  |
| NVIDIA | [cuda-samples](https://github.com/NVIDIA/cuda-samples) | 9683 | C++ | licenca propria (ver LICENSE) | 138 | 0 | 0 | sim | Samples for CUDA Developers which demonstrates features in CUDA Toolkit |
| NVIDIA | [garak](https://github.com/NVIDIA/garak) | 9508 | Python | Apache-2.0 | 478 | 15 | 0 | sim | the LLM vulnerability scanner |
| NVIDIA | [apex](https://github.com/NVIDIA/apex) | 9005 | Python | BSD-3-Clause | 773 | 0 | 0 | nao | A PyTorch Extension:  Tools for easy mixed precision and distributed training in |
| NVIDIA | [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | 8178 | Python | Apache-2.0 | 338 | 0 | 1 | sim | NVIDIA Isaac GR00T N1.7 -  A Foundation Model for Generalist Robots. |
| NVIDIA | [warp](https://github.com/NVIDIA/warp) | 7179 | Python | Apache-2.0 | 368 | 0 | 0 | sim | A Python framework for GPU-accelerated simulation, robotics, and machine learnin |
| NVIDIA | [DALI](https://github.com/NVIDIA/DALI) | 5772 | C++ | Apache-2.0 | 218 | 2 | 24 | sim | A GPU-accelerated library containing highly optimized building blocks and an exe |
| NVIDIA | [cuml](https://github.com/NVIDIA/cuml) | 5302 | Python | Apache-2.0 | 807 | 10 | 1 | sim | NVIDIA cuML: GPU-Accelerated Machine Learning |
| NVIDIA | [Model-Optimizer](https://github.com/NVIDIA/Model-Optimizer) | 5250 | Python | Apache-2.0 | 471 | 0 | 0 | sim | A unified library of SOTA model optimization techniques like quantization, disti |
| NVIDIA | [nccl](https://github.com/NVIDIA/nccl) | 5147 | C++ | licenca propria (ver LICENSE) | 443 | 0 | 0 | sim | Optimized primitives for collective multi-GPU communication |
| NVIDIA | [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) | 4592 | Go | Apache-2.0 | 67 | 1 | 4 | sim | Build and run containers leveraging NVIDIA GPUs |
| NVIDIA | [GenerativeAIExamples](https://github.com/NVIDIA/GenerativeAIExamples) | 4205 | Jupyter Notebook | Apache-2.0 | 85 | 0 | 0 | nao | Generative AI reference workflows optimized for accelerated infrastructure and m |
| NVIDIA | [k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) | 3894 | Go | Apache-2.0 | 74 | 5 | 1 | sim | NVIDIA device plugin for Kubernetes |
| NVIDIA | [cuda-rust](https://github.com/NVIDIA/cuda-rust) | 3688 | Rust | Apache-2.0 | 158 | 1 | 0 | sim | NVIDIA's CUDA platform for Rust. Host runtime crates plus Tile (cutile-rs) and S |
| NVIDIA | [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) | 3571 | Python | Apache-2.0 | 364 | 0 | 0 | sim | A library for accelerating Transformer models on NVIDIA GPUs, including using 8- |
| NVIDIA | [skills](https://github.com/NVIDIA/skills) | 3547 | Python | Apache-2.0 | 17 | 0 | 0 | sim | Agent Skills for NVIDIA products — install into Claude Code, Codex, and other co |
| NVIDIA | [cuda-python](https://github.com/NVIDIA/cuda-python) | 3395 | Cython | Apache-2.0 | 299 | 0 | 0 | sim | CUDA Python: Performance meets Productivity |
| NVIDIA | [physicsnemo](https://github.com/NVIDIA/physicsnemo) | 3332 | Python | Apache-2.0 | 109 | 0 | 0 | sim | Open-source deep-learning framework for building, training, and fine-tuning deep |
| NVIDIA | [flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch) | 3291 | Python | licenca propria (ver LICENSE) | 168 | 4 | 1 | nao | Pytorch implementation of FlowNet 2.0: Evolution of Optical Flow Estimation with |
| NVIDIA | [NeMo-Retriever](https://github.com/NVIDIA/NeMo-Retriever) | 2988 | Python | Apache-2.0 | 253 | 1 | 0 | sim | NeMo Retriever Library is a scalable, performance-oriented document content and  |
| NVIDIA | [gpu-operator](https://github.com/NVIDIA/gpu-operator) | 2920 | Go | Apache-2.0 | 99 | 6 | 1 | sim | NVIDIA GPU Operator creates, configures, and manages GPUs in Kubernetes |
| NVIDIA | [NeMo-Agent-Toolkit](https://github.com/NVIDIA/NeMo-Agent-Toolkit) | 2662 | Python | Apache-2.0 | 30 | 0 | 0 | sim | The NVIDIA NeMo Agent toolkit is an open-source library for efficiently connecti |
| NVIDIA | [cccl](https://github.com/NVIDIA/cccl) | 2530 | C++ | licenca propria (ver LICENSE) | 1986 | 17 | 9 | sim | CUDA Core Compute Libraries |
| NVIDIA | [CUDALibrarySamples](https://github.com/NVIDIA/CUDALibrarySamples) | 2525 | Cuda | Apache-2.0 | 113 | 0 | 0 | sim | CUDA Library Samples |
| NVIDIA | [stdexec](https://github.com/NVIDIA/stdexec) | 2453 | C++ | Apache-2.0 | 160 | 3 | 0 | nao | `std::execution`, the standard C++ framework for asynchronous and parallel progr |
| NVIDIA | [cutile-python](https://github.com/NVIDIA/cutile-python) | 2154 | Python | licenca propria (ver LICENSE) | 31 | 0 | 0 | sim | cuTile is a programming model for writing parallel kernels for NVIDIA GPUs |
| NVIDIA | [accelerated-computing-hub](https://github.com/NVIDIA/accelerated-computing-hub) | 2039 | Jupyter Notebook | licenca propria (ver LICENSE) | 10 | 0 | 0 | sim | NVIDIA curated collection of educational resources related to general purpose GP |
| NVIDIA | [aistore](https://github.com/NVIDIA/aistore) | 1942 | Go | MIT | 12 | 0 | 0 | sim | AIStore: scalable storage for AI applications |
| NVIDIA | [dcgm-exporter](https://github.com/NVIDIA/dcgm-exporter) | 1897 | Go | Apache-2.0 | 137 | 0 | 0 | sim | NVIDIA GPU metrics exporter for Prometheus leveraging DCGM |
| NVIDIA | [nccl-tests](https://github.com/NVIDIA/nccl-tests) | 1673 | Cuda | BSD-3-Clause | 165 | 0 | 0 | nao | NCCL Tests |
| NVIDIA | [trt-samples-for-hackathon-cn](https://github.com/NVIDIA/trt-samples-for-hackathon-cn) | 1673 | Python | Apache-2.0 | 65 | 0 | 0 | nao | Simple samples for TensorRT programming |
| NVIDIA | [RULER](https://github.com/NVIDIA/RULER) | 1622 | Python | Apache-2.0 | 23 | 0 | 0 | nao | This repo contains the source code for RULER: What’s the Real Context Size of Yo |
| NVIDIA | [Personal-AI-Router](https://github.com/NVIDIA/Personal-AI-Router) | 1578 | Go | Apache-2.0 | 94 | 0 | 0 | sim | Router that virtually distributes inference across connected devices in the home |
| NVIDIA | [deepops](https://github.com/NVIDIA/deepops) | 1475 | Shell | BSD-3-Clause | 3 | 0 | 0 | sim | Tools for building GPU clusters |
| NVIDIA | [MatX](https://github.com/NVIDIA/MatX) | 1447 | C++ | BSD-3-Clause | 18 | 0 | 0 | sim | An efficient C++20 GPU numerical computing library with Python-like syntax |
| NVIDIA | [gdrcopy](https://github.com/NVIDIA/gdrcopy) | 1421 | C | MIT | 75 | 0 | 4 | nao | A fast GPU memory copy library based on NVIDIA GPUDirect RDMA technology |
| NVIDIA | [dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) | 1417 | Jupyter Notebook | Apache-2.0 | 87 | 0 | 0 | nao | Collection of step-by-step playbooks for setting up AI/ML workloads on NVIDIA DG |
| NVIDIA | [kvpress](https://github.com/NVIDIA/kvpress) | 1224 | Python | Apache-2.0 | 8 | 0 | 0 | sim | LLM KV cache compression made easy |
| NVIDIA | [earth2studio](https://github.com/NVIDIA/earth2studio) | 1192 | Python | Apache-2.0 | 58 | 0 | 0 | sim | Open-source deep-learning framework for exploring, building and deploying AI wea |
| NVIDIA | [cuda-quantum](https://github.com/NVIDIA/cuda-quantum) | 1159 | C++ | Apache-2.0 | 546 | 12 | 0 | nao | C++ and Python support for the CUDA Quantum programming model for heterogeneous  |
| NVIDIA | [libnvidia-container](https://github.com/NVIDIA/libnvidia-container) | 1134 | C | Apache-2.0 | 14 | 0 | 0 | sim | NVIDIA container runtime library |
| NVIDIA | [enroot](https://github.com/NVIDIA/enroot) | 1114 | Shell | Apache-2.0 | 104 | 0 | 0 | sim | A simple yet powerful tool to turn traditional container/OS images into unprivil |
| NVIDIA | [DreamDojo](https://github.com/NVIDIA/DreamDojo) | 1111 | Python | Apache-2.0 | 19 | 0 | 0 | nao | Official Codebase for "DreamDojo: A Generalist Robot World Model from Large-Scal |
| NVIDIA | [jetson-gpio](https://github.com/NVIDIA/jetson-gpio) | 1078 | Python | MIT | 5 | 0 | 0 | nao | A Python library that enables the use of Jetson's GPIOs |
| NVIDIA | [cuopt](https://github.com/NVIDIA/cuopt) | 1056 | Cuda | Apache-2.0 | 238 | 0 | 0 | sim | GPU accelerated decision optimization  |
| NVIDIA | [raft](https://github.com/NVIDIA/raft) | 1043 | Cuda | Apache-2.0 | 443 | 2 | 0 | nao | RAFT contains fundamental widely-used algorithms and primitives for machine lear |
| NVIDIA | [cuda-tile](https://github.com/NVIDIA/cuda-tile) | 1039 | C++ | licenca propria (ver LICENSE) | 6 | 0 | 0 | nao | CUDA Tile IR is an MLIR-based intermediate representation and compiler infrastru |
| NVIDIA | [cudf-spark](https://github.com/NVIDIA/cudf-spark) | 1009 | Scala | licenca propria (ver LICENSE) | 2053 | 43 | 0 | sim | NVIDIA cuDF for Apache Spark plugin - accelerate Apache Spark with GPUs |
| NVIDIA | [NVFlare](https://github.com/NVIDIA/NVFlare) | 981 | Python | Apache-2.0 | 47 | 1 | 0 | sim | NVIDIA Federated Learning Application Runtime Environment |
| NVIDIA | [cudnn-frontend](https://github.com/NVIDIA/cudnn-frontend) | 964 | Python | Apache-2.0 | 198 | 0 | 0 | sim | cuDNN Frontend is NVIDIA's modern, open-source entry point to the cuDNN library  |
| NVIDIA | [nvbench](https://github.com/NVIDIA/nvbench) | 935 | Cuda | Apache-2.0 | 87 | 4 | 0 | nao | CUDA Kernel Benchmarking Library |
| NVIDIA | [cuvs](https://github.com/NVIDIA/cuvs) | 864 | Cuda | Apache-2.0 | 757 | 0 | 0 | sim | cuVS - a library for vector search and clustering on the GPU |
| NVIDIA | [TileGym](https://github.com/NVIDIA/TileGym) | 824 | Python | licenca propria (ver LICENSE) | 16 | 0 | 0 | sim | Helpful kernel tutorials, examples and SKILLs for tile-based GPU programming |
| NVIDIA | [DCGM](https://github.com/NVIDIA/DCGM) | 799 | C++ | Apache-2.0 | 191 | 0 | 0 | nao | NVIDIA Data Center GPU Manager (DCGM) is a project for gathering telemetry and m |
| NVIDIA | [nvbandwidth](https://github.com/NVIDIA/nvbandwidth) | 785 | C++ | Apache-2.0 | 24 | 0 | 0 | nao | A tool for bandwidth measurements on NVIDIA GPUs. |
| NVIDIA | [torch-harmonics](https://github.com/NVIDIA/torch-harmonics) | 711 | Jupyter Notebook | BSD-3-Clause | 23 | 0 | 0 | sim | Differentiable signal processing on the sphere for PyTorch |
| NVIDIA | [cuCollections](https://github.com/NVIDIA/cuCollections) | 672 | Cuda | Apache-2.0 | 65 | 8 | 0 | sim |  |
| NVIDIA | [soma-retargeter](https://github.com/NVIDIA/soma-retargeter) | 664 | Python | Apache-2.0 | 8 | 0 | 0 | sim | SOMA BVH to humanoid robot motion retargeting library built with Newton and NVID |
| NVIDIA | [runx](https://github.com/NVIDIA/runx) | 641 | Python | BSD-3-Clause | 11 | 0 | 0 | nao | Deep Learning Experiment Management |
| NVIDIA | [GR00T-Dreams](https://github.com/NVIDIA/GR00T-Dreams) | 620 | Jupyter Notebook | Apache-2.0 | 12 | 0 | 0 | sim | DreamGen: Nvidia GEAR Lab's initiative to solve the robotics data problem using  |
| NVIDIA | [nvmath-python](https://github.com/NVIDIA/nvmath-python) | 600 | Cython | Apache-2.0 | 8 | 0 | 0 | sim | NVIDIA Math Libraries for the Python Ecosystem |
| NVIDIA | [nvshmem](https://github.com/NVIDIA/nvshmem) | 596 | C++ | Apache-2.0 | 23 | 0 | 0 | sim | NVIDIA NVSHMEM is a parallel programming interface for NVIDIA GPUs based on Open |
| NVIDIA | [TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) | 583 | Python | Apache-2.0 | 103 | 0 | 0 | sim | High-performance, light-weight C++ LLM and VLM Inference Software for Physical A |
| NVIDIA | [NVTX](https://github.com/NVIDIA/NVTX) | 562 | C++ | licenca propria (ver LICENSE) | 8 | 0 | 0 | sim | The NVIDIA® Tools Extension SDK (NVTX) is a C-based Application Programming Inte |
| NVIDIA | [cosmos-framework](https://github.com/NVIDIA/cosmos-framework) | 560 | Python | licenca propria (ver LICENSE) | 30 | 0 | 0 | sim | Our inference and training framework to run on the Cosmos Models |
| NVIDIA | [SkillEvaluator](https://github.com/NVIDIA/SkillEvaluator) | 552 | Python | Apache-2.0 | 29 | 0 | 0 | sim | Multi-tier framework for evaluating AI agent skills with quality gates, semantic |
| NVIDIA | [hpc-container-maker](https://github.com/NVIDIA/hpc-container-maker) | 519 | Python | Apache-2.0 | 12 | 0 | 0 | nao | HPC Container Maker |
| NVIDIA | [flashdreams](https://github.com/NVIDIA/flashdreams) | 512 | Python | licenca propria (ver LICENSE) | 122 | 0 | 0 | sim | high-performance inference and serving library for interactive autoregressive vi |
| NVIDIA | [NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing) | 511 | Python | Apache-2.0 | 19 | 0 | 0 | sim | NeMo text processing for ASR and TTS |
| NVIDIA | [cuQuantum](https://github.com/NVIDIA/cuQuantum) | 501 | Jupyter Notebook | BSD-3-Clause | 30 | 0 | 0 | sim | Home for cuQuantum Python & NVIDIA cuQuantum SDK C++ samples |
| NVIDIA | [cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint) | 498 | C | licenca propria (ver LICENSE) | 42 | 0 | 0 | nao | CUDA checkpoint and restore utility |
| NVIDIA | [tilus](https://github.com/NVIDIA/tilus) | 495 | Python | Apache-2.0 | 7 | 0 | 0 | sim | Tilus is a tile-level kernel programming language with explicit control over sha |
| NVIDIA | [cuopt-examples](https://github.com/NVIDIA/cuopt-examples) | 472 | Jupyter Notebook | Apache-2.0 | 28 | 0 | 0 | sim | NVIDIA cuOpt examples for decision optimization |
| NVIDIA | [pyxis](https://github.com/NVIDIA/pyxis) | 465 | C | Apache-2.0 | 32 | 0 | 0 | sim | Container plugin for Slurm Workload Manager |
| NVIDIA | [go-nvml](https://github.com/NVIDIA/go-nvml) | 458 | C | Apache-2.0 | 8 | 0 | 0 | sim | Go Bindings for the NVIDIA Management Library (NVML) |
| NVIDIA | [aicr](https://github.com/NVIDIA/aicr) | 440 | Go | Apache-2.0 | 203 | 0 | 0 | sim | Tooling for optimized, validated, and reproducible GPU-accelerated AI runtime in |
| NVIDIA | [aerial-cuda-accelerated-ran](https://github.com/NVIDIA/aerial-cuda-accelerated-ran) | 438 | C++ | licenca propria (ver LICENSE) | 17 | 0 | 0 | sim | An SDK (Software Development Kit) for building commercial-grade, AI-native, 3GPP |
| NVIDIA | [JAX-Toolbox](https://github.com/NVIDIA/JAX-Toolbox) | 432 | Python | Apache-2.0 | 50 | 0 | 0 | nao | JAX-Toolbox |
| NVIDIA | [Fuser](https://github.com/NVIDIA/Fuser) | 406 | C++ | licenca propria (ver LICENSE) | 415 | 15 | 0 | nao | A Fusion Code Generator for NVIDIA GPUs (commonly known as "nvFuser") |
| NVIDIA | [makani](https://github.com/NVIDIA/makani) | 402 | Python | licenca propria (ver LICENSE) | 3 | 0 | 0 | nao | Massively parallel training of machine-learning based weather and climate models |
| NVIDIA | [cuda-q-academic](https://github.com/NVIDIA/cuda-q-academic) | 399 | Jupyter Notebook | licenca propria (ver LICENSE) | 7 | 0 | 0 | sim | This repo contains CUDA-Q Academic materials, including self-paced Jupyter noteb |
| NVIDIA | [IsaacCapture](https://github.com/NVIDIA/IsaacCapture) | 395 | Python | Apache-2.0 | 150 | 0 | 0 | sim | The unified framework for sim & real robot teleoperation |
| NVIDIA | [NVSentinel](https://github.com/NVIDIA/NVSentinel) | 394 | Go | Apache-2.0 | 81 | 0 | 3 | sim | NVSentinel detects and remediates GPU faults on Kubernetes nodes |
| NVIDIA | [gds-nvidia-fs](https://github.com/NVIDIA/gds-nvidia-fs) | 388 | C | licenca propria (ver LICENSE) | 45 | 0 | 0 | sim | NVIDIA GPUDirect Storage Driver |
| NVIDIA | [logits-processor-zoo](https://github.com/NVIDIA/logits-processor-zoo) | 388 | Python | Apache-2.0 | 1 | 0 | 0 | sim | A collection of LogitsProcessors to customize and enhance LLM behavior for speci |
| NVIDIA | [Megatron-Energon](https://github.com/NVIDIA/Megatron-Energon) | 386 | Python | licenca propria (ver LICENSE) | 63 | 0 | 0 | sim | Megatron's multi-modal data loader |
| NVIDIA | [nvidia-settings](https://github.com/NVIDIA/nvidia-settings) | 347 | C | GPL-2.0 | 68 | 0 | 0 | nao | NVIDIA driver control panel |
| NVIDIA | [nvidia-resiliency-ext](https://github.com/NVIDIA/nvidia-resiliency-ext) | 338 | Python | licenca propria (ver LICENSE) | 50 | 0 | 0 | sim | NVIDIA Resiliency Extension is a python package for framework developers and use |
| NVIDIA | [nvidia-kaggle](https://github.com/NVIDIA/nvidia-kaggle) | 338 | Python | MIT | 6 | 0 | 0 | sim | NVIDIA Kaggle Plugin gives agents end-to-end Kaggle competition workflows throug |
| NVIDIA | [recsys-examples](https://github.com/NVIDIA/recsys-examples) | 336 | Python | licenca propria (ver LICENSE) | 57 | 0 | 0 | sim | Examples for Recommenders - easy to train and deploy on accelerated infrastructu |
| NVIDIA | [egl-wayland](https://github.com/NVIDIA/egl-wayland) | 331 | C | MIT | 50 | 0 | 0 | nao | The EGLStream-based Wayland external platform |
| NVIDIA | [structured-data-models](https://github.com/NVIDIA/structured-data-models) | 328 | Python | Apache-2.0 | 71 | 0 | 0 | sim | Foundation Models for Structured Data |
| NVIDIA | [nvtrust](https://github.com/NVIDIA/nvtrust) | 323 | Python | Apache-2.0 | 5 | 0 | 0 | nao | Ancillary open source software to support confidential computing on NVIDIA GPUs |
| NVIDIA | [SOL-ExecBench](https://github.com/NVIDIA/SOL-ExecBench) | 310 | Python | Apache-2.0 | 5 | 0 | 0 | sim | A benchmark of real-world DL kernel problems  |
| NVIDIA | [container-canary](https://github.com/NVIDIA/container-canary) | 308 | Go | Apache-2.0 | 38 | 1 | 1 | sim | A tool for testing and validating container requirements against versioned manif |
| NVIDIA | [nsight-python](https://github.com/NVIDIA/nsight-python) | 297 | Python | Apache-2.0 | 2 | 0 | 0 | nao | Nsight Python is a Python kernel profiling interface based on NVIDIA Nsight Tool |
| NVIDIA | [numba-cuda](https://github.com/NVIDIA/numba-cuda) | 296 | Python | BSD-2-Clause | 144 | 1 | 2 | sim | The CUDA target for Numba |
| NVIDIA | [kubevirt-gpu-device-plugin](https://github.com/NVIDIA/kubevirt-gpu-device-plugin) | 294 | Go | BSD-3-Clause | 29 | 0 | 0 | nao | NVIDIA k8s device plugin for Kubevirt |
| NVIDIA | [nim-anywhere](https://github.com/NVIDIA/nim-anywhere) | 281 | Python | Apache-2.0 | 26 | 0 | 0 | nao | Accelerate your Gen AI with NVIDIA NIM and NVIDIA AI Workbench |
| NVIDIA | [VisRTX](https://github.com/NVIDIA/VisRTX) | 280 | C++ | licenca propria (ver LICENSE) | 6 | 0 | 0 | nao | OptiX + GL based implementations of ANARI |
| NVIDIA | [TensorRT-Model-Connect](https://github.com/NVIDIA/TensorRT-Model-Connect) | 274 | Python | Apache-2.0 | 254 | 5 | 0 | sim | From PyTorch model to end-to-end TensorRT inference experience in two commands—A |
| NVIDIA | [DeepStream](https://github.com/NVIDIA/DeepStream) | 273 | C++ | licenca propria (ver LICENSE) | 2 | 0 | 1 | nao | NVIDIA DeepStream Monorepo: DeepStream SDK and reference apps for building GPU‑a |
| NVIDIA | [OSMO](https://github.com/NVIDIA/OSMO) | 269 | Python | Apache-2.0 | 94 | 0 | 0 | sim | The developer-first platform for scaling complex Physical AI workloads across he |
| NVIDIA | [mig-parted](https://github.com/NVIDIA/mig-parted) | 266 | Go | Apache-2.0 | 8 | 0 | 0 | sim | MIG Partition Editor for NVIDIA GPUs |
| NVIDIA | [asset-harvester](https://github.com/NVIDIA/asset-harvester) | 264 | Python | Apache-2.0 | 3 | 0 | 0 | nao | NVIDIA Asset Harvester is a generative AI model for autonomous vehicle simulatio |
| NVIDIA | [earth2mip](https://github.com/NVIDIA/earth2mip) | 257 | Python | Apache-2.0 | 28 | 0 | 0 | sim | Earth-2 Model Intercomparison Project (MIP) is a python framework that enables c |
| NVIDIA | [cuda-gdb](https://github.com/NVIDIA/cuda-gdb) | 252 | C | GPL-2.0 | 6 | 0 | 0 | nao | CUDA GDB |
| NVIDIA | [cloud-native-stack](https://github.com/NVIDIA/cloud-native-stack) | 248 | Shell | Apache-2.0 | 13 | 0 | 0 | sim | Run cloud native workloads on NVIDIA GPUs |
| NVIDIA | [G-Assist](https://github.com/NVIDIA/G-Assist) | 248 | C | licenca propria (ver LICENSE) | 17 | 0 | 0 | sim | Help shape the future of Project G-Assist |
| NVIDIA | [nvapi](https://github.com/NVIDIA/nvapi) | 246 | C | licenca propria (ver LICENSE) | 15 | 0 | 0 | nao | NVAPI is NVIDIA's core software development kit that allows direct access to NVI |
| NVIDIA | [nim-deploy](https://github.com/NVIDIA/nim-deploy) | 241 | Jupyter Notebook | Apache-2.0 | 13 | 0 | 0 | nao | A collection of YAML files, Helm Charts, Operator code, and guides to act as an  |
| NVIDIA | [nvalchemi-toolkit-ops](https://github.com/NVIDIA/nvalchemi-toolkit-ops) | 241 | Python | licenca propria (ver LICENSE) | 5 | 0 | 0 | sim | ALCHEMI Toolkit-Ops is a collection of optimized batch kernels to accelerate com |
| NVIDIA | [Deep-Learning-Accelerator-SW](https://github.com/NVIDIA/Deep-Learning-Accelerator-SW) | 238 | Python | licenca propria (ver LICENSE) | 15 | 0 | 0 | sim | NVIDIA DLA-SW, the recipes and tools for running deep learning workloads on NVID |
| NVIDIA | [nvcf](https://github.com/NVIDIA/nvcf) | 231 | Go | Apache-2.0 | 486 | 12 | 0 | sim | Platform for deploying and routing GPU-accelerated inference, streaming, and bat |
| NVIDIA | [instant-nurec](https://github.com/NVIDIA/instant-nurec) | 228 | Python | Apache-2.0 | 10 | 0 | 0 | sim | InstantNuRec: Feed-Forward 3D Gaussian Reconstruction from Driving Logs |
| NVIDIA | [NeMo-speech-data-processor](https://github.com/NVIDIA/NeMo-speech-data-processor) | 222 | Python | Apache-2.0 | 24 | 0 | 0 | sim | A toolkit for processing speech data and creating speech datasets |
| NVIDIA | [nvkind](https://github.com/NVIDIA/nvkind) | 212 | Go | Apache-2.0 | 23 | 0 | 0 | sim |  |
| NVIDIA | [ncore](https://github.com/NVIDIA/ncore) | 198 | Python | Apache-2.0 | 8 | 0 | 0 | sim | Data representations, APIs, and tools for high quality AV and robotics applicati |
| NVIDIA | [TorchFort](https://github.com/NVIDIA/TorchFort) | 197 | C++ | Apache-2.0 | 1 | 0 | 0 | sim | An Online Deep Learning Interface for HPC programs on NVIDIA GPUs |
| NVIDIA | [NeMo-Relay](https://github.com/NVIDIA/NeMo-Relay) | 192 | Rust | Apache-2.0 | 4 | 0 | 0 | sim | Multi-language agent runtime and library for execution scope management, lifecyc |
| NVIDIA | [nsight-training](https://github.com/NVIDIA/nsight-training) | 191 | C | licenca propria (ver LICENSE) | 5 | 0 | 0 | nao | Training material for Nsight developer tools |
| NVIDIA | [gpu-driver-container](https://github.com/NVIDIA/gpu-driver-container) | 190 | Shell | Apache-2.0 | 28 | 0 | 0 | sim | The NVIDIA GPU driver container allows the provisioning of the NVIDIA driver thr |
| NVIDIA | [nvalchemi-toolkit](https://github.com/NVIDIA/nvalchemi-toolkit) | 189 | Python | Apache-2.0 | 12 | 0 | 0 | sim | ALCHEMI Toolkit is a developer toolkit for accelerating training and inference f |
| NVIDIA | [cudf-spark-examples](https://github.com/NVIDIA/cudf-spark-examples) | 174 | Jupyter Notebook | Apache-2.0 | 25 | 0 | 0 | sim | A repo for all spark examples using Rapids Accelerator including ETL, ML/DL, etc |
| NVIDIA | [workbench-example-agentic-rag](https://github.com/NVIDIA/workbench-example-agentic-rag) | 172 | Jupyter Notebook | Apache-2.0 | 12 | 0 | 0 | nao | An NVIDIA AI Workbench example project for an Agentic Retrieval Augmented Genera |
| NVIDIA | [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) | 170 | C++ | Apache-2.0 | 21 | 0 | 0 | sim | NeMo-Speech.cpp is a lightweight C++ inference runtime for Speech models  |
| NVIDIA | [vgpu-device-manager](https://github.com/NVIDIA/vgpu-device-manager) | 169 | Go | Apache-2.0 | 7 | 0 | 0 | sim | NVIDIA vGPU Device Manager manages NVIDIA vGPU devices on top of Kubernetes |
| NVIDIA | [nvImageCodec](https://github.com/NVIDIA/nvImageCodec) | 161 | Jupyter Notebook | Apache-2.0 | 26 | 0 | 0 | sim | A nvImageCodec library of GPU- and CPU- accelerated codecs featuring a unified i |
| NVIDIA | [nvidia-installer](https://github.com/NVIDIA/nvidia-installer) | 159 | C | GPL-2.0 | 8 | 0 | 0 | nao | NVIDIA driver installer |
| NVIDIA | [k8s-nim-operator](https://github.com/NVIDIA/k8s-nim-operator) | 159 | Go | Apache-2.0 | 16 | 0 | 0 | sim | An Operator for deployment and maintenance of NVIDIA NIMs and NeMo microservices |
| NVIDIA | [go-dcgm](https://github.com/NVIDIA/go-dcgm) | 158 | C | Apache-2.0 | 5 | 0 | 0 | sim | Golang bindings for Nvidia Datacenter GPU Manager (DCGM) |
| NVIDIA | [physicsnemo-cfd](https://github.com/NVIDIA/physicsnemo-cfd) | 154 | Jupyter Notebook | Apache-2.0 | 1 | 0 | 0 | sim | L​ibrary for using the models trained in PhysicsNeMo in Engineering and CFD work |
| NVIDIA | [edk2-nvidia](https://github.com/NVIDIA/edk2-nvidia) | 149 | C | licenca propria (ver LICENSE) | 30 | 0 | 0 | nao | NVIDIA EDK2 platform support |
| NVIDIA | [compute-eval](https://github.com/NVIDIA/compute-eval) | 149 | Python | licenca propria (ver LICENSE) | 14 | 0 | 0 | sim | Evaluating Large Language Models for CUDA Code Generation ComputeEval is a frame |
| NVIDIA | [optix-toolkit](https://github.com/NVIDIA/optix-toolkit) | 144 | C++ | BSD-3-Clause | 4 | 0 | 0 | nao | Set of utilities supporting workflows common in GPU raytracing applications |
| NVIDIA | [ais-k8s](https://github.com/NVIDIA/ais-k8s) | 137 | Go | MIT | 1 | 0 | 0 | sim | Kubernetes Operator, Helm Charts, Ansible Playbooks, and utility scripts for lar |
| NVIDIA | [ansible-role-nvidia-driver](https://github.com/NVIDIA/ansible-role-nvidia-driver) | 134 | Python | BSD-3-Clause | 21 | 0 | 0 | nao |  |
| NVIDIA | [MagnumIO](https://github.com/NVIDIA/MagnumIO) | 125 | Python | Apache-2.0 | 11 | 0 | 0 | sim | Magnum IO community repo |
| NVIDIA | [nemoclaw-community](https://github.com/NVIDIA/nemoclaw-community) | 124 | Python | Apache-2.0 | 22 | 0 | 0 | sim | Our repository of community driven examples, showcases, and integrations |
| NVIDIA | [go-gpuallocator](https://github.com/NVIDIA/go-gpuallocator) | 123 | Go | Apache-2.0 | 9 | 0 | 0 | sim | Go Abstraction for Allocating NVIDIA GPUs with Custom Policies |
| NVIDIA | [cudaq-qec](https://github.com/NVIDIA/cudaq-qec) | 118 | C++ | licenca propria (ver LICENSE) | 78 | 1 | 0 | nao | Accelerated libraries for quantum-classical computing built on CUDA-Q.   |
| NVIDIA | [Audio2Face-3D-Training-Framework](https://github.com/NVIDIA/Audio2Face-3D-Training-Framework) | 116 | Python | Apache-2.0 | 4 | 0 | 0 | sim | Audio2Face-3D Training Framework for creating custom neural networks that genera |
| NVIDIA | [exemplar-performance](https://github.com/NVIDIA/exemplar-performance) | 115 | Python | licenca propria (ver LICENSE) | 5 | 0 | 0 | sim | Exemplar Performance provides recipes in ready-to-use templates for evaluating p |
| NVIDIA | [TensorRT-RTX](https://github.com/NVIDIA/TensorRT-RTX) | 113 |  | Apache-2.0 | 10 | 0 | 0 | sim | NVIDIA TensorRT-RTX is an SDK for high-performance AI inference on NVIDIA RTX GP |
| NVIDIA | [NeMo-Agent-Toolkit-UI](https://github.com/NVIDIA/NeMo-Agent-Toolkit-UI) | 110 | TypeScript | licenca propria (ver LICENSE) | 6 | 0 | 0 | sim | The NVIDIA NeMo Agent Toolkit UI streamlines interacting with NeMo Agent Toolkit |
| NVIDIA | [nsight-vscode-edition](https://github.com/NVIDIA/nsight-vscode-edition) | 105 | TypeScript | licenca propria (ver LICENSE) | 26 | 0 | 0 | nao | A Visual Studio Code extension for building and debugging CUDA applications. |
| NVIDIA | [cloudai](https://github.com/NVIDIA/cloudai) | 99 | Python | Apache-2.0 | 32 | 0 | 0 | sim | CloudAI Benchmark Framework |
| NVIDIA | [gpu-admin-tools](https://github.com/NVIDIA/gpu-admin-tools) | 96 | Python | MIT | 14 | 0 | 0 | nao | GPU Admin Tools. Includes Confidential Computing controls for H100, and other fu |
| NVIDIA | [simready-foundation](https://github.com/NVIDIA/simready-foundation) | 95 | Python | Apache-2.0 | 28 | 0 | 0 | sim | SimReady Foundation is a central repository for defining simulation content spec |
| NVIDIA | [xr-ai](https://github.com/NVIDIA/xr-ai) | 95 | Python | Apache-2.0 | 39 | 0 | 0 | sim | XR AI |
| NVIDIA | [srt-slurm](https://github.com/NVIDIA/srt-slurm) | 93 | Python | licenca propria (ver LICENSE) | 73 | 0 | 0 | sim |  |
| NVIDIA | [cuml-spark](https://github.com/NVIDIA/cuml-spark) | 92 | Jupyter Notebook | Apache-2.0 | 34 | 0 | 0 | sim | Spark RAPIDS MLlib – accelerate Apache Spark MLlib with GPUs |
| NVIDIA | [cuBQL](https://github.com/NVIDIA/cuBQL) | 92 | C++ | Apache-2.0 | 14 | 0 | 0 | sim | A CUDA BVH Build And Query Library |
| NVIDIA | [Ising](https://github.com/NVIDIA/Ising) | 92 |  | Apache-2.0 | 3 | 0 | 0 | nao | A model family accelerating the development of useful quantum computers |
| NVIDIA | [k8s-test-infra](https://github.com/NVIDIA/k8s-test-infra) | 91 | Go | Apache-2.0 | 144 | 0 | 0 | sim | Simulate NVIDIA infrastructure (e.g. GPU) on CPU nodes in Kubernetes :coffee:️ |
| NVIDIA | [context-aware-rag](https://github.com/NVIDIA/context-aware-rag) | 90 | Python | Apache-2.0 | 5 | 0 | 0 | sim | Context-Aware RAG library for Knowledge Graph ingestion and retrieval functions. |
| NVIDIA | [nvidia-modprobe](https://github.com/NVIDIA/nvidia-modprobe) | 85 | C | GPL-2.0 | 8 | 0 | 0 | nao | Load the NVIDIA kernel module and create NVIDIA character device files |
| NVIDIA | [clara-viz](https://github.com/NVIDIA/clara-viz) | 85 | C++ | Apache-2.0 | 5 | 0 | 0 | nao | NVIDIA Clara Viz is a platform for visualization of 2D/3D medical imaging data |
| NVIDIA | [multi-storage-client](https://github.com/NVIDIA/multi-storage-client) | 85 | Python | Apache-2.0 | 1 | 0 | 0 | sim | Unified high-performance Python client for object and file stores. |
| NVIDIA | [jaxpp](https://github.com/NVIDIA/jaxpp) | 83 | Python | Apache-2.0 | 5 | 0 | 0 | sim | JaxPP is a library for JAX that enables flexible MPMD pipeline parallelism for l |
| NVIDIA | [vagrant-swift-all-in-one](https://github.com/NVIDIA/vagrant-swift-all-in-one) | 78 | Ruby | Apache-2.0 | 6 | 0 | 0 | nao | Vagrant Swift All In One |
| NVIDIA | [harmonizer](https://github.com/NVIDIA/harmonizer) | 78 | Python | Apache-2.0 | 2 | 0 | 0 | sim | NVIDIA Harmonizer enhances Omniverse NuRec renderings through artifact correctio |
| NVIDIA | [ib-traffic-monitor](https://github.com/NVIDIA/ib-traffic-monitor) | 76 | C | Apache-2.0 | 1 | 0 | 0 | sim | A TUI-based utility for real-time monitoring of InfiniBand traffic and performan |
| NVIDIA | [nvidia-hpcg](https://github.com/NVIDIA/nvidia-hpcg) | 75 | C++ | licenca propria (ver LICENSE) | 1 | 0 | 0 | sim | NVIDIA HPCG is based on the HPCG benchmark and optimized for performance on NVID |
| NVIDIA | [halos-outside-in-safety](https://github.com/NVIDIA/halos-outside-in-safety) | 75 | C++ | Apache-2.0 | 4 | 0 | 0 | sim | NVIDIA Halos Outside-In Safety Blueprint extends robot perception beyond on-boar |
| NVIDIA | [nodewright](https://github.com/NVIDIA/nodewright) | 74 | Go | Apache-2.0 | 52 | 0 | 0 | sim | A Kubernetes Operator to manage Node OS customizations. |
| NVIDIA | [cluster-readiness-engine](https://github.com/NVIDIA/cluster-readiness-engine) | 73 | Go | Apache-2.0 | 36 | 0 | 0 | sim | NVIDIA Cluster Readiness Engine |
| NVIDIA | [cudf-spark-tools](https://github.com/NVIDIA/cudf-spark-tools) | 72 | Scala | Apache-2.0 | 306 | 8 | 0 | sim | User tools for Spark RAPIDS |
| NVIDIA | [Quantum-Calibration-Agent-Blueprint](https://github.com/NVIDIA/Quantum-Calibration-Agent-Blueprint) | 71 | TypeScript | Apache-2.0 | 2 | 0 | 0 | nao | This is a reference agent blueprint for AI-powered quantum device calibration. I |
| NVIDIA | [numba-cuda-mlir](https://github.com/NVIDIA/numba-cuda-mlir) | 69 | Python | Apache-2.0 | 72 | 0 | 0 | sim | repo for Numba-CUDA-MLIR |
| NVIDIA | [nvidia-persistenced](https://github.com/NVIDIA/nvidia-persistenced) | 68 | C | MIT | 15 | 0 | 0 | nao | NVIDIA driver persistence daemon |
| NVIDIA | [nccl-extensions](https://github.com/NVIDIA/nccl-extensions) | 68 | Cuda | licenca propria (ver LICENSE) | 15 | 0 | 0 | sim | Communication patterns for AI, built on top of NCCL device and host APIs |
| NVIDIA | [Trustworthy-AI](https://github.com/NVIDIA/Trustworthy-AI) | 67 | Python | CC0-1.0 | 1 | 0 | 0 | nao | NVIDIA’s repository for enabling trustworthy AI. |
| NVIDIA | [ACCV-Lab](https://github.com/NVIDIA/ACCV-Lab) | 66 | Python | Apache-2.0 | 5 | 0 | 0 | nao | Accelerated Computer Vision Lab (ACCV-Lab) is a systematic collection of package |
| NVIDIA | [cudf-spark-jni](https://github.com/NVIDIA/cudf-spark-jni) | 65 | Cuda | Apache-2.0 | 143 | 1 | 0 | sim | NVIDIA cuDF plugin JNI For Apache Spark |
| NVIDIA | [workbench-example-sdxl-customization](https://github.com/NVIDIA/workbench-example-sdxl-customization) | 65 | Python | Apache-2.0 | 12 | 0 | 0 | nao | An NVIDIA AI Workbench example project for customizing an SDXL model |
| NVIDIA | [nvloom](https://github.com/NVIDIA/nvloom) | 65 | C++ | Apache-2.0 | 2 | 0 | 0 | sim | nvloom is a set of tools designed to scalably test MNNVL fabrics. |
| NVIDIA | [physicsnemo-curator](https://github.com/NVIDIA/physicsnemo-curator) | 64 | Python | Apache-2.0 | 7 | 0 | 0 | sim | Accelerated ETL toolkit for building AI-ready datasets across multiple scientifi |
| NVIDIA | [nvidia-terraform-modules](https://github.com/NVIDIA/nvidia-terraform-modules) | 62 |  | Apache-2.0 | 2 | 0 | 0 | sim | Infrastructure as code for GPU accelerated managed Kubernetes clusters. |
| NVIDIA | [numbast](https://github.com/NVIDIA/numbast) | 62 | Python | Apache-2.0 | 61 | 3 | 0 | sim | Numbast is a tool to build an automated pipeline that converts CUDA APIs into Nu |
| NVIDIA | [cuCascade](https://github.com/NVIDIA/cuCascade) | 60 | C++ | Apache-2.0 | 19 | 0 | 0 | nao | GPU Memory Reservation Library |
| NVIDIA | [go-nvlib](https://github.com/NVIDIA/go-nvlib) | 58 | Go | Apache-2.0 | 4 | 0 | 0 | sim | A collection of useful Go libraries for use with NVIDIA GPU management tools |
| NVIDIA | [k8s-driver-manager](https://github.com/NVIDIA/k8s-driver-manager) | 55 | Go | Apache-2.0 | 15 | 0 | 0 | sim | The NVIDIA Driver Manager is a Kubernetes component which assist in seamless upg |
| NVIDIA | [nvbmc-docs](https://github.com/NVIDIA/nvbmc-docs) | 52 | TeX | CC-BY-4.0 | 2 | 0 | 0 | sim | Documentation for Nvidia OpenBMC stack |
| NVIDIA | [cuEmbed](https://github.com/NVIDIA/cuEmbed) | 50 | Cuda | Apache-2.0 | 5 | 0 | 0 | sim | CUDA Embedding Lookup Kernel Library |
| NVIDIA | [NeMo-Fabric](https://github.com/NVIDIA/NeMo-Fabric) | 50 | Python | Apache-2.0 | 16 | 0 | 0 | sim | NVIDIA NeMo Fabric |
| NVIDIA | [tinylinux-scripts](https://github.com/NVIDIA/tinylinux-scripts) | 49 | Shell | licenca propria (ver LICENSE) | 3 | 0 | 0 | nao | Scripts for building minimal Linux distribution for diagnostics |
| NVIDIA | [attestation-sdk](https://github.com/NVIDIA/attestation-sdk) | 48 | C++ | Apache-2.0 | 12 | 0 | 0 | sim | C++ SDK that provides resources for implementing and validating Trusted Computin |
| NVIDIA | [cudf-spark-benchmarks](https://github.com/NVIDIA/cudf-spark-benchmarks) | 47 | Python | Apache-2.0 | 48 | 0 | 0 | sim | cuDF plugin Benchmarks – benchmark sets and utilities for the NVIDIA cuDF plugin |
| NVIDIA | [OWL](https://github.com/NVIDIA/OWL) | 47 | C++ | Apache-2.0 | 3 | 0 | 0 | sim | The OptiX Wrappers Library |
| NVIDIA | [bluebazel](https://github.com/NVIDIA/bluebazel) | 46 | TypeScript | MIT | 19 | 0 | 0 | nao | Blue Bazel: A vscode extension for bazel building, running, and testing with UI |
| NVIDIA | [nvrc](https://github.com/NVIDIA/nvrc) | 46 | Rust | Apache-2.0 | 23 | 0 | 0 | sim | The NVRC project provides a Rust binary that implements a simple init system for |
| NVIDIA | [sphinx-llm](https://github.com/NVIDIA/sphinx-llm) | 45 | Python | Apache-2.0 | 20 | 0 | 0 | sim | LLM extensions for Sphinx Documentation |
| NVIDIA | [mlperf-common](https://github.com/NVIDIA/mlperf-common) | 44 | Python | Apache-2.0 | 3 | 0 | 0 | nao | NVIDIA's launch, startup, and logging scripts used by our MLPerf Training and HP |
| NVIDIA | [egl-wayland2](https://github.com/NVIDIA/egl-wayland2) | 44 | C | Apache-2.0 | 6 | 0 | 0 | sim | Dma-buf-based Wayland external platform library |
| NVIDIA | [cloud-native-docs](https://github.com/NVIDIA/cloud-native-docs) | 43 | PowerShell | Apache-2.0 | 32 | 0 | 0 | sim | Documentation repository for NVIDIA Cloud Native Technologies |
| NVIDIA | [nvidia-xconfig](https://github.com/NVIDIA/nvidia-xconfig) | 42 | C | GPL-2.0 | 3 | 0 | 0 | nao | NVIDIA xorg.conf configurator |
| NVIDIA | [barney](https://github.com/NVIDIA/barney) | 42 | C++ | Apache-2.0 | 7 | 0 | 0 | sim | A Scalable (and Optionally, Data-Parallel) ANARI Multi-GPU Path Tracer |
| NVIDIA | [physical-ai-data-factory](https://github.com/NVIDIA/physical-ai-data-factory) | 42 | Shell | licenca propria (ver LICENSE) | 1 | 0 | 0 | sim | Synthetic Data Generation Workflows for Physical AI |
| NVIDIA | [yum-packaging-precompiled-kmod](https://github.com/NVIDIA/yum-packaging-precompiled-kmod) | 41 | Shell | Apache-2.0 | 5 | 0 | 0 | sim | NVIDIA precompiled kernel module packaging for RHEL |
| NVIDIA | [Kumo-TS](https://github.com/NVIDIA/Kumo-TS) | 41 | Python | Apache-2.0 | 4 | 0 | 0 | sim | Kumo-TS (formerly NV-Tesseract) is a comprehensive time series analysis package  |
| NVIDIA | [daqiri](https://github.com/NVIDIA/daqiri) | 40 | C++ | Apache-2.0 | 34 | 0 | 0 | sim | DAQIRI connects high bandwidth streaming sensor data to the NVIDIA software ecos |
| NVIDIA | [cloudxr-lovr-sample](https://github.com/NVIDIA/cloudxr-lovr-sample) | 37 | Lua | licenca propria (ver LICENSE) | 3 | 0 | 0 | nao | Stream your LÖVR VR applications wirelessly to supported headsets using NVIDIA C |
| NVIDIA | [OpenShell-Research](https://github.com/NVIDIA/OpenShell-Research) | 37 | Python | Apache-2.0 | 2 | 0 | 0 | sim | 🧪 OpenShell's Research Journal |
| NVIDIA | [workbench-example-onboarding-project](https://github.com/NVIDIA/workbench-example-onboarding-project) | 34 | Python | Apache-2.0 | 21 | 0 | 0 | nao | An interactive tutorial project that demonstrates the capabilities of NVIDIA AI  |
| NVIDIA | [cloudxr-framework](https://github.com/NVIDIA/cloudxr-framework) | 33 | C++ | licenca propria (ver LICENSE) | 4 | 0 | 0 | nao | Swift frameworks for building client applications that connect to CloudXR server |
| NVIDIA | [spark-process](https://github.com/NVIDIA/spark-process) | 32 | Python | GFDL-1.3 | 1 | 0 | 0 | nao | A process for Ada/SPARK software to meet ISO 26262 |
| NVIDIA | [NeMo-Agent-Toolkit-Examples](https://github.com/NVIDIA/NeMo-Agent-Toolkit-Examples) | 32 | Python | Apache-2.0 | 3 | 0 | 0 | sim | Community examples utilizing NVIDIA NeMo Agent Toolkit. |
| NVIDIA | [ai-cloud-validation](https://github.com/NVIDIA/ai-cloud-validation) | 32 | Python | Apache-2.0 | 51 | 0 | 0 | sim | Validation and management tools for NVIDIA AI Cloud environments. |
| NVIDIA | [TensorRT-RTX-EP-ABI](https://github.com/NVIDIA/TensorRT-RTX-EP-ABI) | 32 | C++ | Apache-2.0 | 2 | 0 | 0 | sim | The NVIDIA TensorRT RTX Execution Provider (EP) is an inference deployment solut |
| NVIDIA | [go-ratelimit](https://github.com/NVIDIA/go-ratelimit) | 31 | Go | Apache-2.0 | 5 | 0 | 0 | sim | High-performance distributed rate limiting library for Go with Redis backend, sl |
| NVIDIA | [k8s-operator-libs](https://github.com/NVIDIA/k8s-operator-libs) | 30 | Go | Apache-2.0 | 11 | 0 | 0 | sim | A collection of useful Go libraries to ease the development of NVIDIA Operators  |
| NVIDIA | [holodeck](https://github.com/NVIDIA/holodeck) | 30 | Go | Apache-2.0 | 6 | 0 | 0 | sim | Holodeck is a project to create test environments optimised for GPU projects. |
| NVIDIA-Omniverse | [PhysX](https://github.com/NVIDIA-Omniverse/PhysX) | 4784 | C++ | BSD-3-Clause | 128 | 0 | 0 | sim | NVIDIA PhysX SDK |
| NVIDIA-Omniverse | [LearnOpenUSD](https://github.com/NVIDIA-Omniverse/LearnOpenUSD) | 316 | Python | Apache-2.0 | 15 | 0 | 0 | sim | Learning resources for Universal Scene Description (OpenUSD) including tutorials |
| NVIDIA-Omniverse | [ovrtx](https://github.com/NVIDIA-Omniverse/ovrtx) | 228 | Python | licenca propria (ver LICENSE) | 2 | 0 | 0 | nao | A C and Python library for physically accurate, real-time, sensor simulation and |
| NVIDIA-Omniverse | [usd-content-agents](https://github.com/NVIDIA-Omniverse/usd-content-agents) | 215 | Python | Apache-2.0 | 6 | 0 | 0 | sim | AI-powered agents for automating 3D content workflows using Vision-Language Mode |
| NVIDIA-Omniverse | [synthetic-data-examples](https://github.com/NVIDIA-Omniverse/synthetic-data-examples) | 177 | Jupyter Notebook | licenca propria (ver LICENSE) | 5 | 0 | 0 | nao | Synthetic Data Generation Examples |
| NVIDIA-Omniverse | [web-viewer-sample](https://github.com/NVIDIA-Omniverse/web-viewer-sample) | 149 | TypeScript | licenca propria (ver LICENSE) | 11 | 0 | 0 | nao | This sample demonstrates how a front-end client can present a streamed Omniverse |
| NVIDIA-Omniverse | [OpenUSD-plugin-samples](https://github.com/NVIDIA-Omniverse/OpenUSD-plugin-samples) | 111 | C++ | Apache-2.0 | 8 | 0 | 0 | sim | OpenUSD schema extension samples, build tools, and sample Kit extensions that us |
| NVIDIA-Omniverse | [OpenUSD-Code-Samples](https://github.com/NVIDIA-Omniverse/OpenUSD-Code-Samples) | 100 | Python | Apache-2.0 | 11 | 2 | 0 | sim | Common code snippets for OpenUSD |
| NVIDIA-Omniverse | [kit-usd-agents](https://github.com/NVIDIA-Omniverse/kit-usd-agents) | 99 | Python | Apache-2.0 | 4 | 0 | 0 | nao | The base repository to build OmniverseKit USD agents.  |
| NVIDIA-Omniverse | [usd-exchange](https://github.com/NVIDIA-Omniverse/usd-exchange) | 89 | Python | Apache-2.0 | 1 | 0 | 0 | sim | OpenUSD Exchange SDK |
| NVIDIA-Omniverse | [kit-extension-template-cpp](https://github.com/NVIDIA-Omniverse/kit-extension-template-cpp) | 88 | C++ | Apache-2.0 | 13 | 0 | 0 | nao | Omniverse Kit C++ Extension Template |
| NVIDIA-Omniverse | [PhysicalAI-SimReady-Materials](https://github.com/NVIDIA-Omniverse/PhysicalAI-SimReady-Materials) | 64 |  | licenca propria (ver LICENSE) | 1 | 0 | 0 | sim | This a open source collection of USD materials and textures using MaterialX |
| NVIDIA-Omniverse | [kit-cae](https://github.com/NVIDIA-Omniverse/kit-cae) | 64 | Python | licenca propria (ver LICENSE) | 6 | 0 | 0 | nao | Omniverse Kit sample for CAE use-cases |
| NVIDIA-Omniverse | [omniverse-labs](https://github.com/NVIDIA-Omniverse/omniverse-labs) | 55 | Python | licenca propria (ver LICENSE) | 1 | 0 | 0 | nao | Omniverse Labs is a project housing experimental Omniverse sample content |
| NVIDIA-Omniverse | [aif-pipeline-samples](https://github.com/NVIDIA-Omniverse/aif-pipeline-samples) | 46 | Python | MIT | 4 | 0 | 0 | sim | Sample scripts and presets for creating SimReady USD assets in NVIDIA Omniverse, |
| NVIDIA-Omniverse | [usdsearch-client](https://github.com/NVIDIA-Omniverse/usdsearch-client) | 17 | Python | licenca propria (ver LICENSE) | 3 | 0 | 1 | nao | Client library for USD Search and Asset Graph Search API’s |
| NVlabs | [instant-ngp](https://github.com/NVlabs/instant-ngp) | 17568 | Cuda | licenca propria (ver LICENSE) | 503 | 0 | 0 | nao | Instant neural graphics primitives: lightning fast NeRF and more |
| NVlabs | [Sana](https://github.com/NVlabs/Sana) | 9239 | Python | Apache-2.0 | 143 | 0 | 0 | nao | SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transforme |
| NVlabs | [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) | 4539 | C++ | licenca propria (ver LICENSE) | 252 | 0 | 0 | nao | Lightning fast C++/CUDA neural network framework |
| NVlabs | [VILA](https://github.com/NVlabs/VILA) | 3864 | Python | Apache-2.0 | 78 | 0 | 0 | nao | VILA is a family of state-of-the-art vision language models (VLMs) for diverse m |
| NVlabs | [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) | 3717 | Python | licenca propria (ver LICENSE) | 72 | 0 | 0 | sim | Welcome to GR00T Whole-Body Control (WBC)! This is a unified platform for develo |
| NVlabs | [Eagle](https://github.com/NVlabs/Eagle) | 3694 | Python | Apache-2.0 | 66 | 0 | 0 | sim | Eagle: Frontier Vision-Language Models with Data-Centric Strategies |
| NVlabs | [FoundationPose](https://github.com/NVlabs/FoundationPose) | 3624 | Python | licenca propria (ver LICENSE) | 147 | 0 | 0 | nao | [CVPR 2024 Highlight] FoundationPose: Unified 6D Pose Estimation and Tracking of |
| NVlabs | [SoL-Pi](https://github.com/NVlabs/SoL-Pi) | 3400 | TypeScript | MIT | 76 | 0 | 0 | sim | SoL-Pi: Scaling Auto-Research Loops for Efficient Agent Harnesses |
| NVlabs | [FoundationStereo](https://github.com/NVlabs/FoundationStereo) | 2939 | Python | licenca propria (ver LICENSE) | 90 | 0 | 0 | nao | [CVPR 2025 Best Paper Nomination] FoundationStereo: Zero-Shot Stereo Matching |
| NVlabs | [LongLive](https://github.com/NVlabs/LongLive) | 2720 | Python | Apache-2.0 | 1 | 0 | 0 | sim | Long Video Gen / World Model / World Action Model (Embodied) Infrastructure |
| NVlabs | [ProtoMotions](https://github.com/NVlabs/ProtoMotions) | 2430 | Python | Apache-2.0 | 14 | 0 | 0 | sim | ProtoMotions is a GPU-accelerated simulation and learning framework for training |
| NVlabs | [nvdiffrec](https://github.com/NVlabs/nvdiffrec) | 2298 | Python | licenca propria (ver LICENSE) | 72 | 0 | 0 | nao | Official code for the CVPR 2022 (oral) paper "Extracting Triangular 3D Models, M |
| NVlabs | [MambaVision](https://github.com/NVlabs/MambaVision) | 2236 | Python | licenca propria (ver LICENSE) | 21 | 0 | 0 | nao | [CVPR 2025] Official PyTorch Implementation of MambaVision: A Hybrid Mamba-Trans |
| NVlabs | [alpamayo](https://github.com/NVlabs/alpamayo) | 2035 | Python | Apache-2.0 | 66 | 0 | 0 | sim | NVIDIA Alpamayo 1 Nano is an open 10B reasoning VLA model for autonomous vehicle |
| NVlabs | [RADIO](https://github.com/NVlabs/RADIO) | 1968 | Python | licenca propria (ver LICENSE) | 58 | 0 | 0 | nao | Official repository for "AM-RADIO: Reduce All Domains Into One" |
| NVlabs | [nvdiffrast](https://github.com/NVlabs/nvdiffrast) | 1927 | C++ | licenca propria (ver LICENSE) | 27 | 0 | 0 | nao | Nvdiffrast - Modular Primitives for High-Performance Differentiable Rendering |
| NVlabs | [curobo](https://github.com/NVlabs/curobo) | 1897 | Python | Apache-2.0 | 28 | 0 | 0 | sim | CUDA Accelerated Robot Library |
| NVlabs | [sionna](https://github.com/NVlabs/sionna) | 1630 | Jupyter Notebook | licenca propria (ver LICENSE) | 13 | 0 | 0 | sim | Sionna: An Open-Source Library for Research on Communication Systems |
| NVlabs | [Fast-FoundationStereo](https://github.com/NVlabs/Fast-FoundationStereo) | 1520 | Python | licenca propria (ver LICENSE) | 35 | 0 | 0 | nao | [CVPR 2026] Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching |
| NVlabs | [BundleSDF](https://github.com/NVlabs/BundleSDF) | 1418 | Python | licenca propria (ver LICENSE) | 29 | 0 | 0 | nao | [CVPR 2023] BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown Ob |
| NVlabs | [kda](https://github.com/NVlabs/kda) | 1330 |  | licenca propria (ver LICENSE) | 3 | 0 | 0 | sim | Kernel Design Agents (KDA) is a agent-centric workflow to write high-performance |
| NVlabs | [alpasim](https://github.com/NVlabs/alpasim) | 1258 | Python | Apache-2.0 | 19 | 0 | 0 | sim | AlpaSim is an open-source autonomous vehicle simulation platform designed for de |
| NVlabs | [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) | 1095 | Python | Apache-2.0 | 34 | 0 | 0 | sim | Official implementation of "Fast-dLLM: Training-free Acceleration of Diffusion L |
| NVlabs | [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT) | 1085 | Python | Apache-2.0 | 13 | 0 | 0 | nao | [ICLR 2026 Oral] DiffusionNFT: Online Diffusion Reinforcement with Forward Proce |
| NVlabs | [FastGen](https://github.com/NVlabs/FastGen) | 1022 | Python | Apache-2.0 | 7 | 0 | 0 | sim | NVIDIA FastGen: Fast Generation from Diffusion Models |
| NVlabs | [DoRA](https://github.com/NVlabs/DoRA) | 1001 | Python | licenca propria (ver LICENSE) | 21 | 0 | 0 | nao | [ICML2024 (Oral)] Official PyTorch implementation of DoRA: Weight-Decomposed Low |
| NVlabs | [PixelDiT](https://github.com/NVlabs/PixelDiT) | 985 | Python | licenca propria (ver LICENSE) | 12 | 0 | 0 | nao | [CVPR 2026 Best Paper Finalist & NeurIPS 2026] |
| NVlabs | [cosmos-policy](https://github.com/NVlabs/cosmos-policy) | 884 | Python | Apache-2.0 | 18 | 0 | 0 | nao | Cosmos Policy |
| NVlabs | [rcm](https://github.com/NVlabs/rcm) | 816 | Python | Apache-2.0 | 20 | 0 | 0 | nao | rCM & Causal-rCM: Leading and Unified Algorithms/Infrastructures for Bidirection |
| NVlabs | [LongSplat](https://github.com/NVlabs/LongSplat) | 812 | Python | licenca propria (ver LICENSE) | 20 | 0 | 0 | nao | [ICCV 2025] LongSplat: Robust Unposed 3D Gaussian Splatting for Casual Long Vide |
| NVlabs | [SOMA-X](https://github.com/NVlabs/SOMA-X) | 804 | Python | Apache-2.0 | 4 | 0 | 0 | nao | SOMA: Unifying Parametric Human Body Models  |
| NVlabs | [ToolOrchestra](https://github.com/NVlabs/ToolOrchestra) | 762 | Python | Apache-2.0 | 19 | 0 | 0 | sim | ToolOrchestra is an end-to-end RL training framework for orchestrating tools and |
| NVlabs | [OmniVinci](https://github.com/NVlabs/OmniVinci) | 683 | Python | Apache-2.0 | 8 | 0 | 0 | sim | OmniVinci is an omni-modal LLM for joint understanding of vision, audio, and lan |
| NVlabs | [GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet) | 678 | Python | licenca propria (ver LICENSE) | 8 | 0 | 0 | nao | [ICLR 2025] Official PyTorch Implementation of Gated Delta Networks: Improving M |
| NVlabs | [flip](https://github.com/NVlabs/flip) | 652 | C++ | licenca propria (ver LICENSE) | 2 | 0 | 0 | nao | A tool for visualizing and communicating the errors in rendered images. |
| NVlabs | [OmniDrive](https://github.com/NVlabs/OmniDrive) | 646 | Python | licenca propria (ver LICENSE) | 84 | 0 | 0 | nao |  |
| NVlabs | [vibetensor](https://github.com/NVlabs/vibetensor) | 641 | Python | Apache-2.0 | 3 | 0 | 0 | sim | Our first fully AI generated deep learning system |
| NVlabs | [GraspGen](https://github.com/NVlabs/GraspGen) | 597 | Python | licenca propria (ver LICENSE) | 16 | 0 | 1 | nao | Official repo for GraspGen: A Diffusion-based Framework for 6-DOF Grasping |
| NVlabs | [GRAIL](https://github.com/NVlabs/GRAIL) | 558 | Python | licenca propria (ver LICENSE) | 6 | 0 | 0 | nao | 🤖 GRAIL is a fully digital data-generation pipeline that synthesizes humanoid lo |
| NVlabs | [RoboLab](https://github.com/NVlabs/RoboLab) | 552 | Python | Apache-2.0 | 11 | 0 | 0 | sim | A simulation benchmarking platform for evaluating generalist robot policies |
| NVlabs | [PointWorld](https://github.com/NVlabs/PointWorld) | 539 | Python | Apache-2.0 | 5 | 0 | 0 | sim | PointWorld: Scaling 3D World Models for In-The-Wild Robotic Manipulation |
| NVlabs | [timeloop](https://github.com/NVlabs/timeloop) | 527 | C++ | BSD-3-Clause | 71 | 0 | 0 | nao | Timeloop performs modeling, mapping and code-generation for tensor algebra workl |
| NVlabs | [QeRL](https://github.com/NVlabs/QeRL) | 523 | Python | Apache-2.0 | 10 | 0 | 0 | nao | [ICLR 2026]QeRL enables RL for 32B LLMs on a single H100 GPU. |
| NVlabs | [GENMO](https://github.com/NVlabs/GENMO) | 522 | Python | licenca propria (ver LICENSE) | 21 | 0 | 0 | sim |  |
| NVlabs | [GDPO](https://github.com/NVlabs/GDPO) | 512 | Python | Apache-2.0 | 6 | 0 | 0 | nao | Official implementation of GDPO: Group reward-Decoupled Normalization Policy Opt |
| NVlabs | [vla0](https://github.com/NVlabs/vla0) | 494 | Python | licenca propria (ver LICENSE) | 14 | 0 | 0 | nao | VLA-0: Building State-of-the-Art VLAs with Zero Modification |
| NVlabs | [GEM-X](https://github.com/NVlabs/GEM-X) | 472 | Python | Apache-2.0 | 17 | 0 | 0 | sim | Monocular whole-body 3D human pose estimation using the SOMA body model |
| NVlabs | [trajdata](https://github.com/NVlabs/trajdata) | 457 | Python | Apache-2.0 | 22 | 0 | 0 | nao | A unified interface to many trajectory forecasting datasets. |
| NVlabs | [AnyFlow](https://github.com/NVlabs/AnyFlow) | 441 | Python | Apache-2.0 | 6 | 0 | 0 | nao | Flow Map OPD for AnyStep Video Diffusion |
| NVlabs | [GR00T-VisualSim2Real](https://github.com/NVlabs/GR00T-VisualSim2Real) | 424 | Python | Apache-2.0 | 5 | 0 | 0 | sim | GR00T-VisualSim2Real: Open-source sim-to-real framework for humanoid visual loco |
| NVlabs | [sage](https://github.com/NVlabs/sage) | 418 | Python | Apache-2.0 | 5 | 0 | 0 | sim | Official Code Release of SAGE: Scalable Agentic 3D Scene Generation for Embodied |
| NVlabs | [SimFoundry](https://github.com/NVlabs/SimFoundry) | 413 | Python | Apache-2.0 | 2 | 0 | 0 | sim | Modular and Automated Scene Generation for Policy Learning and Evaluation |
| NVlabs | [nvdiffrecmc](https://github.com/NVlabs/nvdiffrecmc) | 410 | C | licenca propria (ver LICENSE) | 16 | 0 | 0 | nao | Official code for the NeurIPS 2022 paper "Shape, Light, and Material Decompositi |
| NVlabs | [GLAMR](https://github.com/NVlabs/GLAMR) | 391 | Python | licenca propria (ver LICENSE) | 38 | 0 | 0 | nao | [CVPR 2022 Oral] Official PyTorch Implementation of "GLAMR: Global Occlusion-Awa |
| NVlabs | [CuTe](https://github.com/NVlabs/CuTe) | 368 | Python | Apache-2.0 | 3 | 0 | 0 | sim | Reference implementation and examples of the CuTe Layout representation and alge |
| NVlabs | [physical_ai_av](https://github.com/NVlabs/physical_ai_av) | 366 | Python | MIT | 14 | 0 | 0 | nao | Devkit and documentation for the NVIDIA Physical AI Autonomous Vehicles Dataset |
| NVlabs | [alpamayo1.5](https://github.com/NVlabs/alpamayo1.5) | 366 | Python | Apache-2.0 | 11 | 0 | 0 | sim | NVIDIA Alpamayo 1.5 Nano is an open 10B reasoning VLA model for autonomous vehic |
| NVlabs | [GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2) | 328 | Python | licenca propria (ver LICENSE) | 3 | 0 | 0 | nao | Official PyTorch Implementation of Gated DeltaNet-2: Decoupling Erase and Write  |
| NVlabs | [AutoGaze](https://github.com/NVlabs/AutoGaze) | 309 | Python | Apache-2.0 | 4 | 0 | 0 | nao | AutoGaze automatically removes redundant patches in a video, reducing #tokens in |
| NVlabs | [matchlib](https://github.com/NVlabs/matchlib) | 303 | C++ | licenca propria (ver LICENSE) | 1 | 0 | 0 | nao | SystemC/C++ library of commonly-used hardware functions and components for HLS. |
| NVlabs | [GTRS](https://github.com/NVlabs/GTRS) | 290 | Python | Apache-2.0 | 6 | 0 | 0 | sim |  |
| NVlabs | [dexmimicgen](https://github.com/NVlabs/dexmimicgen) | 288 | Python | licenca propria (ver LICENSE) | 16 | 0 | 0 | nao | This code corresponds to simulation environments used as part of the DexMimicGen |
| NVlabs | [parrot](https://github.com/NVlabs/parrot) | 288 | Cuda | licenca propria (ver LICENSE) | 11 | 0 | 0 | sim | Parrot is an array fusion GPU library built on NVIDIA's CCCL libaries (Thrust/CU |
| NVlabs | [alpamayo2](https://github.com/NVlabs/alpamayo2) | 269 | Python | Apache-2.0 | 4 | 0 | 0 | sim | NVIDIA Alpamayo 2 Super is an open 34B multi-task foundation model designed to s |
| NVlabs | [RLP](https://github.com/NVlabs/RLP) | 254 |  | licenca propria (ver LICENSE) | 4 | 0 | 0 | nao | [ICLR 2026] Official PyTorch Implementation of RLP: Reinforcement as a Pretraini |
| NVlabs | [ASPIRE](https://github.com/NVlabs/ASPIRE) | 254 | Python | Apache-2.0 | 9 | 0 | 0 | sim | ASPIRE: Agentic /Skills Discovery for Robotics |
| NVlabs | [CGBN](https://github.com/NVlabs/CGBN) | 253 | Cuda | licenca propria (ver LICENSE) | 22 | 0 | 0 | nao | CGBN:   CUDA Accelerated Multiple Precision Arithmetic (Big Num) using Cooperati |
| NVlabs | [GraspGenX](https://github.com/NVlabs/GraspGenX) | 245 | Python | licenca propria (ver LICENSE) | 8 | 0 | 0 | sim | Code Release for GraspGenX |
| NVlabs | [LSM](https://github.com/NVlabs/LSM) | 240 | Python | licenca propria (ver LICENSE) | 9 | 0 | 0 | nao | [NeurIPS'24] Large Spatial Model: End-to-end Unposed Images to Semantic 3D |
| NVlabs | [cvdp_benchmark](https://github.com/NVlabs/cvdp_benchmark) | 236 | Python | Apache-2.0 | 26 | 0 | 0 | sim |  |
| NVlabs | [WarpConvNet](https://github.com/NVlabs/WarpConvNet) | 231 | Python | Apache-2.0 | 12 | 0 | 0 | sim | Make your wildest 3D ConvNet dream architectures come true |
| NVlabs | [CARI4D](https://github.com/NVlabs/CARI4D) | 226 | Python | licenca propria (ver LICENSE) | 2 | 0 | 0 | nao | [CVPR 2026] CARI4D: Category Agnostic 4D Reconstruction of Human-Object Interact |
| NVlabs | [PS3](https://github.com/NVlabs/PS3) | 225 | Python | Apache-2.0 | 6 | 0 | 0 | nao | Scaling Vision Pre-Training to 4K Resolution |
| NVlabs | [MobilityGen](https://github.com/NVlabs/MobilityGen) | 213 | Python | Apache-2.0 | 56 | 0 | 0 | nao | Data Generation Pipeline for Mobility |
| NVlabs | [alpamayo-recipes](https://github.com/NVlabs/alpamayo-recipes) | 198 | Python | Apache-2.0 | 7 | 0 | 0 | sim | Developer Hub for NVIDIA Alpamayo, containing ready-to-use recipes for fine-tuni |
| NVlabs | [queen](https://github.com/NVlabs/queen) | 188 | C++ | licenca propria (ver LICENSE) | 6 | 0 | 0 | nao | Official PyTorch implementation of QUEEN: QUantized Efficient ENcoding of Dynami |
| NVlabs | [sionna-rt](https://github.com/NVlabs/sionna-rt) | 164 | Jupyter Notebook | licenca propria (ver LICENSE) | 11 | 0 | 0 | sim | Sionna RT: The Ray Tracing Package of Sionna |
| NVlabs | [X-MOBILITY](https://github.com/NVlabs/X-MOBILITY) | 161 | Python | Apache-2.0 | 9 | 0 | 0 | sim | X-MOBILITY |
| NVlabs | [DiSECt](https://github.com/NVlabs/DiSECt) | 148 | Python | licenca propria (ver LICENSE) | 2 | 0 | 0 | nao | Differentiable Cutting Simulator |
| NVlabs | [COMPASS](https://github.com/NVlabs/COMPASS) | 145 | Python | Apache-2.0 | 3 | 0 | 0 | sim | Cross-embOdiment Mobility Policy via ResiduAl RL and Skill Synthesis |
| NVlabs | [IsaacLabEureka](https://github.com/NVlabs/IsaacLabEureka) | 143 | Python | Apache-2.0 | 6 | 0 | 0 | nao |  |
| NVlabs | [cuTAMP](https://github.com/NVlabs/cuTAMP) | 135 | Python | licenca propria (ver LICENSE) | 1 | 0 | 0 | nao | cuTAMP: Differentiable GPU-Parallelized Task and Motion Planning |
| NVlabs | [VideoITG](https://github.com/NVlabs/VideoITG) | 133 | Python | Apache-2.0 | 2 | 0 | 0 | sim | [CVPR 2026 Highlight] VideoITG: Multimodal Video Understanding with Instructed T |
| NVlabs | [SOLAR](https://github.com/NVlabs/SOLAR) | 131 | Python | Apache-2.0 | 5 | 0 | 0 | nao | Speed of Light Analysis for ML Model Runtime |
| NVlabs | [vt-refine](https://github.com/NVlabs/vt-refine) | 130 | Dockerfile | licenca propria (ver LICENSE) | 2 | 0 | 0 | nao | Official code release for CoRL'25 paper: VT-Refine: Learning Bimanual Assembly w |
| NVlabs | [DDO](https://github.com/NVlabs/DDO) | 125 | Python | licenca propria (ver LICENSE) | 5 | 0 | 0 | nao | [ICML 2025 Spotlight] Direct Discriminative Optimization: Reinforcing Diffusion/ |
| NVlabs | [cBottle](https://github.com/NVlabs/cBottle) | 110 | Python | Apache-2.0 | 10 | 0 | 0 | sim | A generative foundation model for the kilometer-scale atmosphere |
| NVlabs | [neural_rx](https://github.com/NVlabs/neural_rx) | 106 | Jupyter Notebook | licenca propria (ver LICENSE) | 1 | 0 | 0 | nao | Real-Time Inference of 5G NR Multi-user MIMO Neural Receivers |
| NVlabs | [sionna-rk](https://github.com/NVlabs/sionna-rk) | 105 | Jupyter Notebook | licenca propria (ver LICENSE) | 4 | 0 | 0 | sim | Sionna Research Kit: A GPU-Accelerated Research Platform for AI-RAN |
| NVlabs | [vla-perf](https://github.com/NVlabs/vla-perf) | 96 | Python | licenca propria (ver LICENSE) | 2 | 0 | 0 | sim | A performance analysis tool for VLA models |
| NVlabs | [HANDAL](https://github.com/NVlabs/HANDAL) | 93 | Python | licenca propria (ver LICENSE) | 2 | 0 | 0 | nao | HANDAL Dataset and Pipeline |
| NVlabs | [Nemotron-Labs-Diffusion](https://github.com/NVlabs/Nemotron-Labs-Diffusion) | 93 | Python | licenca propria (ver LICENSE) | 3 | 0 | 0 | nao |  |
| NVlabs | [Mosaic3D](https://github.com/NVlabs/Mosaic3D) | 91 | Python | Apache-2.0 | 7 | 0 | 0 | nao | [CVPR 2025] Mosaic3D: Foundation Dataset and Model for Open-vocabulary 3D Segmen |
| NVlabs | [LoRWeB](https://github.com/NVlabs/LoRWeB) | 78 | Python | licenca propria (ver LICENSE) | 1 | 0 | 0 | nao | We propose a novel modular framework that learns to dynamically mix low-rank ada |
| NVlabs | [L4P](https://github.com/NVlabs/L4P) | 76 | Python | licenca propria (ver LICENSE) | 2 | 0 | 0 | nao | (3DV 2026 Oral) L4P -- a feed-forward foundational model designed for multiple l |
| NVlabs | [SparDA](https://github.com/NVlabs/SparDA) | 74 | Python | Apache-2.0 | 1 | 0 | 0 | nao | Sparse Decoupled Attention for Efficient Long-Context LLM Inference |
| NVlabs | [scal3r](https://github.com/NVlabs/scal3r) | 69 | Python | licenca propria (ver LICENSE) | 2 | 0 | 0 | nao | Prompt tune geometry reconstruction models for long-sequence. |
| NVlabs | [object_centric_diffusion](https://github.com/NVlabs/object_centric_diffusion) | 65 | Python | licenca propria (ver LICENSE) | 5 | 0 | 0 | nao | SPOT: SE(3) Pose Trajectory Diffusion for Object-Centric Manipulation |
| NVlabs | [collab-sim](https://github.com/NVlabs/collab-sim) | 64 | Python | licenca propria (ver LICENSE) | 1 | 0 | 0 | nao | Research package for MPC VR teleop with Isaac Sim and cuRobo |
| NVlabs | [KernelBlaster](https://github.com/NVlabs/KernelBlaster) | 63 | Cuda | Apache-2.0 | 11 | 0 | 0 | sim | A framework for in context learning for code optimization |
| NVlabs | [protcomposer](https://github.com/NVlabs/protcomposer) | 62 | Python | Apache-2.0 | 6 | 0 | 0 | nao |  |
| NVlabs | [blade](https://github.com/NVlabs/blade) | 59 | Python | licenca propria (ver LICENSE) | 1 | 0 | 0 | nao | Official PyTorch implementation of BLADE: Single-view Body Mesh Estimation throu |
| NVlabs | [VoLoAgent](https://github.com/NVlabs/VoLoAgent) | 56 | Python | Apache-2.0 | 3 | 0 | 0 | sim |  |
| NVlabs | [earth2grid](https://github.com/NVlabs/earth2grid) | 55 | Python | Apache-2.0 | 12 | 0 | 0 | sim |  |
| NVlabs | [layerdenoise](https://github.com/NVlabs/layerdenoise) | 54 | Python | licenca propria (ver LICENSE) | 1 | 0 | 0 | nao | Neural Denoising with Layer Embeddings |
| NVlabs | [Cypress](https://github.com/NVlabs/Cypress) | 53 | C++ | Apache-2.0 | 2 | 0 | 0 | sim |  |
| NVlabs | [EoRA](https://github.com/NVlabs/EoRA) | 50 | Python | licenca propria (ver LICENSE) | 5 | 0 | 0 | nao | [ICLRW'26] EoRA: Fine-tuning-free Compensation for Compressed LLM with Eigenspac |
| NVlabs | [SoftMimicGen](https://github.com/NVlabs/SoftMimicGen) | 44 | Python | Apache-2.0 | 1 | 0 | 0 | sim | SoftMimicGen |
| NVlabs | [sionna-rt-gui](https://github.com/NVlabs/sionna-rt-gui) | 42 | Python | licenca propria (ver LICENSE) | 5 | 0 | 0 | sim |  |
| NVlabs | [GL0AM](https://github.com/NVlabs/GL0AM) | 33 | Python | licenca propria (ver LICENSE) | 2 | 0 | 0 | nao | GL0AM GPU Accelerated Gate Level Logic Simulator |
| NVlabs | [cuHPX](https://github.com/NVlabs/cuHPX) | 31 | Python | Apache-2.0 | 5 | 0 | 0 | sim | GPU accelerated utilities for data on HEALPix grids |
| NVlabs | [RoboVoLo](https://github.com/NVlabs/RoboVoLo) | 31 | Python | licenca propria (ver LICENSE) | 2 | 0 | 0 | sim |  |
| ansys | [pymapdl](https://github.com/ansys/pymapdl) | 521 | Python | MIT | 222 | 0 | 0 | sim | A Python client library for Ansys MAPDL |
| ansys | [pyfluent](https://github.com/ansys/pyfluent) | 518 | Python | MIT | 180 | 0 | 0 | sim | Pythonic interface to Ansys Fluent |
| ansys | [pyaedt](https://github.com/ansys/pyaedt) | 403 | Python | MIT | 231 | 7 | 0 | sim | AEDT Python Client Package |
| ansys | [pyansys](https://github.com/ansys/pyansys) | 294 | Python | MIT | 12 | 0 | 0 | nao | Delivering PyAnsys libraries as a bundle |
| ansys | [pyansys-geometry](https://github.com/ansys/pyansys-geometry) | 97 | Python | MIT | 34 | 0 | 0 | sim | A Python wrapper for Ansys Geometry Services |
| ansys | [pydpf-core](https://github.com/ansys/pydpf-core) | 92 | Python | MIT | 218 | 2 | 1 | sim | Data Processing Framework - Python Core |
| ansys | [pymechanical](https://github.com/ansys/pymechanical) | 85 | Python | MIT | 68 | 0 | 0 | sim | Pythonic interface to Ansys Mechanical ™ |
| ansys | [pydyna](https://github.com/ansys/pydyna) | 83 | Python | MIT | 20 | 0 | 0 | sim | Python interface to the LS-DYNA solver |
| ansys | [optical-automation](https://github.com/ansys/optical-automation) | 68 | Python | MIT | 13 | 0 | 0 | nao | Optical Automation Framework |
| ansys | [pymapdl-reader](https://github.com/ansys/pymapdl-reader) | 60 | Python | MIT | 35 | 0 | 0 | sim | A legacy Python library to read MAPDL result files (MAPDL 14.5 and later) |
| ansys | [pydpf-post](https://github.com/ansys/pydpf-post) | 58 | Python | MIT | 48 | 3 | 0 | sim | Data Processing Framework - Post Processing Module |
| ansys | [example-data](https://github.com/ansys/example-data) | 56 | KFramework | MIT | 7 | 0 | 0 | sim | This repository contains example datasets for PyAnsys projects. |
| ansys | [pyfluent-visualization](https://github.com/ansys/pyfluent-visualization) | 56 | Python | MIT | 55 | 0 | 0 | sim | Visualize Ansys Fluent simulations using Python |
| ansys | [pylumerical](https://github.com/ansys/pylumerical) | 51 | Python | MIT | 13 | 0 | 0 | sim | Python API for Ansys Lumerical |
| ansys | [python-installer-qt-gui](https://github.com/ansys/python-installer-qt-gui) | 45 | Python | MIT | 18 | 0 | 0 | sim | Python QT app for installing Python |
| ansys | [pyaedt-examples](https://github.com/ansys/pyaedt-examples) | 44 |  | MIT | 38 | 0 | 0 | sim | PyAEDT Application Examples |
| ansys | [pyansys-dev-guide](https://github.com/ansys/pyansys-dev-guide) | 41 |  | MIT | 16 | 0 | 0 | sim | PyAnsys Project Developer's Guide |
| ansys | [pytwin](https://github.com/ansys/pytwin) | 37 | Python | MIT | 6 | 0 | 0 | sim | Ansys Digital Twin repository |
| ansys | [pystk](https://github.com/ansys/pystk) | 37 | Python | MIT | 27 | 0 | 0 | sim | The next generation Python API for STK |
| ansys | [pymotorcad](https://github.com/ansys/pymotorcad) | 35 | Python | MIT | 124 | 5 | 0 | sim |  |
| ansys | [pyprimemesh](https://github.com/ansys/pyprimemesh) | 34 | Python | MIT | 66 | 1 | 0 | sim | Pythonic Meshing Client for Ansys Prime Server |
| ansys | [ansys-sphinx-theme](https://github.com/ansys/ansys-sphinx-theme) | 33 | Python | Apache-2.0 | 12 | 0 | 0 | sim | Ansys corporate theme for Sphinx pages |
| ansys | [pyedb](https://github.com/ansys/pyedb) | 30 | Python | MIT | 10 | 0 | 0 | sim | pyedb is a Python library to use the EDB client library. |
| ansys | [pyspeos](https://github.com/ansys/pyspeos) | 23 | Python | MIT | 55 | 4 | 0 | sim | PySpeos is a Python library that gathers functionalities and tools based on Speo |
| ansys | [ansys-templates](https://github.com/ansys/ansys-templates) | 10 | Python | MIT | 52 | 2 | 0 | sim | A tool for creating new projects according to Ansys guidelines |
| ansys | [grantami-jobqueue](https://github.com/ansys/grantami-jobqueue) | 0 | Python | MIT | 3 | 2 | 0 | sim |  |
| google-deepmind | [mujoco](https://github.com/google-deepmind/mujoco) | 15544 | C++ | Apache-2.0 | 286 | 2 | 0 | sim | Multi-Joint dynamics with Contact. A general purpose physics simulator. |
| google-deepmind | [deepmind-research](https://github.com/google-deepmind/deepmind-research) | 15226 | Jupyter Notebook | Apache-2.0 | 359 | 0 | 0 | sim | This repository contains implementations and illustrative code to accompany Deep |
| google-deepmind | [alphafold](https://github.com/google-deepmind/alphafold) | 14875 | Python | Apache-2.0 | 309 | 0 | 0 | sim | Open source code for AlphaFold 2. |
| google-deepmind | [sonnet](https://github.com/google-deepmind/sonnet) | 9971 | Python | Apache-2.0 | 44 | 0 | 0 | sim | TensorFlow-based neural network library |
| google-deepmind | [alphafold3](https://github.com/google-deepmind/alphafold3) | 8617 | Python | Apache-2.0 | 23 | 1 | 0 | sim | AlphaFold 3 inference pipeline. |
| google-deepmind | [weathernext](https://github.com/google-deepmind/weathernext) | 7702 | Python | Apache-2.0 | 81 | 0 | 0 | sim |  |
| google-deepmind | [gemma](https://github.com/google-deepmind/gemma) | 5775 | Python | Apache-2.0 | 330 | 0 | 0 | sim | Gemma open-weight LLM library, from Google DeepMind |
| google-deepmind | [open_spiel](https://github.com/google-deepmind/open_spiel) | 5522 | C++ | Apache-2.0 | 78 | 0 | 1 | sim | OpenSpiel is a collection of environments and algorithms for research in general |
| google-deepmind | [alphageometry](https://github.com/google-deepmind/alphageometry) | 4898 | Python | Apache-2.0 | 140 | 0 | 0 | sim |  |
| google-deepmind | [dm_control](https://github.com/google-deepmind/dm_control) | 4707 | Python | Apache-2.0 | 140 | 0 | 0 | sim | Google DeepMind's software stack for physics-based simulation and Reinforcement  |
| google-deepmind | [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie) | 4162 | Python | licenca propria (ver LICENSE) | 49 | 0 | 0 | sim | A collection of high-quality models for the MuJoCo physics engine, curated by Go |
| google-deepmind | [acme](https://github.com/google-deepmind/acme) | 4075 | Python | Apache-2.0 | 104 | 0 | 0 | sim | A library of reinforcement learning components and agents |
| google-deepmind | [mctx](https://github.com/google-deepmind/mctx) | 2671 | Python | Apache-2.0 | 14 | 0 | 0 | sim | Monte Carlo tree search in JAX |
| google-deepmind | [optax](https://github.com/google-deepmind/optax) | 2344 | Python | Apache-2.0 | 136 | 0 | 1 | sim | Optax is a gradient processing and optimization library for JAX. |
| google-deepmind | [mujoco_playground](https://github.com/google-deepmind/mujoco_playground) | 2244 | Python | Apache-2.0 | 111 | 0 | 1 | sim | An open-source library for GPU-accelerated robot learning and sim-to-real transf |
| google-deepmind | [tapnet](https://github.com/google-deepmind/tapnet) | 2197 | Jupyter Notebook | Apache-2.0 | 33 | 0 | 0 | sim | Tracking Any Point (TAP) |
| google-deepmind | [alphagenome](https://github.com/google-deepmind/alphagenome) | 2183 | Python | Apache-2.0 | 9 | 0 | 0 | sim | This API provides programmatic access to the AlphaGenome model developed by Goog |
| google-deepmind | [open_x_embodiment](https://github.com/google-deepmind/open_x_embodiment) | 2058 | Jupyter Notebook | Apache-2.0 | 56 | 0 | 0 | sim |  |
| google-deepmind | [concordia](https://github.com/google-deepmind/concordia) | 1767 | Python | Apache-2.0 | 41 | 1 | 0 | sim | A library for generative social simulation |
| google-deepmind | [mujoco_mpc](https://github.com/google-deepmind/mujoco_mpc) | 1744 | C++ | Apache-2.0 | 58 | 0 | 0 | sim | Real-time behaviour synthesis with MuJoCo, using Predictive Control |
| google-deepmind | [bsuite](https://github.com/google-deepmind/bsuite) | 1562 | Python | Apache-2.0 | 22 | 0 | 0 | sim | bsuite is a collection of carefully-designed experiments that investigate core c |
| google-deepmind | [mujoco_warp](https://github.com/google-deepmind/mujoco_warp) | 1535 | Python | Apache-2.0 | 58 | 0 | 0 | sim | GPU-optimized version of the MuJoCo physics simulator, designed for NVIDIA hardw |
| google-deepmind | [rlax](https://github.com/google-deepmind/rlax) | 1448 | Python | Apache-2.0 | 38 | 0 | 0 | sim |  |
| google-deepmind | [formal-conjectures](https://github.com/google-deepmind/formal-conjectures) | 1313 | Lean | Apache-2.0 | 1402 | 14 | 0 | sim | A collection of formalized statements of conjectures in Lean. |
| google-deepmind | [android_env](https://github.com/google-deepmind/android_env) | 1247 | Python | Apache-2.0 | 14 | 0 | 0 | sim | RL research on Android devices. |
| google-deepmind | [materials_discovery](https://github.com/google-deepmind/materials_discovery) | 1244 | Jupyter Notebook | Apache-2.0 | 33 | 0 | 0 | sim |  |
| google-deepmind | [synthid-text](https://github.com/google-deepmind/synthid-text) | 1149 | Python | Apache-2.0 | 21 | 0 | 0 | sim |  |
| google-deepmind | [tree](https://github.com/google-deepmind/tree) | 1026 | Python | Apache-2.0 | 47 | 0 | 0 | sim | tree is a library for working with nested data structures |
| google-deepmind | [chex](https://github.com/google-deepmind/chex) | 957 | Python | Apache-2.0 | 79 | 0 | 0 | sim |  |
| google-deepmind | [xmanager](https://github.com/google-deepmind/xmanager) | 922 | Python | Apache-2.0 | 37 | 0 | 0 | sim | A platform for managing machine learning experiments |
| google-deepmind | [alphagenome_research](https://github.com/google-deepmind/alphagenome_research) | 896 | Python | Apache-2.0 | 4 | 0 | 0 | sim | Research code accompanying AlphaGenome  |
| google-deepmind | [meltingpot](https://github.com/google-deepmind/meltingpot) | 887 | Python | Apache-2.0 | 116 | 0 | 0 | sim | A suite of test scenarios for multi-agent reinforcement learning. |
| google-deepmind | [ferminet](https://github.com/google-deepmind/ferminet) | 854 | Python | Apache-2.0 | 4 | 0 | 0 | sim | An implementation of the Fermionic Neural Network for ab-initio electronic struc |
| google-deepmind | [superhuman](https://github.com/google-deepmind/superhuman) | 805 | Lean | Apache-2.0 | 22 | 0 | 0 | sim |  |
| google-deepmind | [reverb](https://github.com/google-deepmind/reverb) | 792 | C++ | Apache-2.0 | 43 | 0 | 0 | sim | Reverb is an efficient and easy-to-use data storage and transport system designe |
| google-deepmind | [disco_rl](https://github.com/google-deepmind/disco_rl) | 739 | Python | Apache-2.0 | 6 | 0 | 0 | sim | Accompanying code for "Discovering State-of-the-art Reinforcement Algorithms" Na |
| google-deepmind | [torax](https://github.com/google-deepmind/torax) | 734 | Python | licenca propria (ver LICENSE) | 131 | 0 | 0 | sim | TORAX: Tokamak transport simulation in JAX |
| google-deepmind | [long-form-factuality](https://github.com/google-deepmind/long-form-factuality) | 692 | Python | licenca propria (ver LICENSE) | 28 | 0 | 0 | sim | Benchmarking long-form factuality in large language models. Original code for ou |
| google-deepmind | [recurrentgemma](https://github.com/google-deepmind/recurrentgemma) | 685 | Python | Apache-2.0 | 5 | 0 | 0 | sim | Open weights language model from Google DeepMind, based on Griffin. |
| google-deepmind | [distrax](https://github.com/google-deepmind/distrax) | 657 | Python | Apache-2.0 | 78 | 0 | 0 | sim |  |
| google-deepmind | [gemini-robotics-sdk](https://github.com/google-deepmind/gemini-robotics-sdk) | 612 | Python | Apache-2.0 | 19 | 0 | 0 | sim |  |
| google-deepmind | [simply](https://github.com/google-deepmind/simply) | 580 | Python | Apache-2.0 | 18 | 0 | 0 | sim | Minimal and scalable research codebase in JAX, designed for rapid iteration on f |
| google-deepmind | [clrs](https://github.com/google-deepmind/clrs) | 547 | Jupyter Notebook | Apache-2.0 | 11 | 0 | 0 | sim |  |
| google-deepmind | [dqn_zoo](https://github.com/google-deepmind/dqn_zoo) | 514 | Python | Apache-2.0 | 2 | 0 | 0 | sim | DQN Zoo is a collection of reference implementations of reinforcement learning a |
| google-deepmind | [treescope](https://github.com/google-deepmind/treescope) | 476 | Python | Apache-2.0 | 19 | 0 | 0 | sim | An interactive HTML pretty-printer for machine learning research in IPython note |
| google-deepmind | [dm_pix](https://github.com/google-deepmind/dm_pix) | 451 | Python | Apache-2.0 | 13 | 0 | 0 | sim | PIX is an image processing library in JAX, for JAX. |
| google-deepmind | [lab2d](https://github.com/google-deepmind/lab2d) | 444 | C++ | Apache-2.0 | 4 | 0 | 0 | sim | A customisable 2D platform for agent-based AI research |
| google-deepmind | [videoprism](https://github.com/google-deepmind/videoprism) | 397 | Python | Apache-2.0 | 12 | 0 | 0 | sim | Official repository for "VideoPrism: A Foundational Visual Encoder for Video Und |
| google-deepmind | [representations4d](https://github.com/google-deepmind/representations4d) | 372 | Jupyter Notebook | Apache-2.0 | 9 | 0 | 0 | sim | Foundation models for 4D spatial and temporal vision tasks. |
| google-deepmind | [physics-IQ-benchmark](https://github.com/google-deepmind/physics-IQ-benchmark) | 369 | Python | licenca propria (ver LICENSE) | 2 | 0 | 0 | sim | Benchmarking physical understanding in generative video models |
| google-deepmind | [regress-lm](https://github.com/google-deepmind/regress-lm) | 361 | Python | Apache-2.0 | 12 | 0 | 0 | sim | Library for sequence-to-sequence numeric prediction, applicable to any tokenizab |
| google-deepmind | [kfac-jax](https://github.com/google-deepmind/kfac-jax) | 331 | Python | Apache-2.0 | 23 | 0 | 0 | sim | Second Order Optimization and Curvature Estimation with K-FAC in JAX. |
| google-deepmind | [enn](https://github.com/google-deepmind/enn) | 324 | Python | Apache-2.0 | 17 | 0 | 0 | sim |  |
| google-deepmind | [ai-foundations](https://github.com/google-deepmind/ai-foundations) | 316 | Jupyter Notebook | Apache-2.0 | 20 | 0 | 0 | sim |  |
| google-deepmind | [alphaproof-nexus-results](https://github.com/google-deepmind/alphaproof-nexus-results) | 304 | Lean | Apache-2.0 | 1 | 0 | 0 | sim | Lean math proofs generated by AlphaProof Nexus and accompanying natural language |
| google-deepmind | [alphaevolve_results](https://github.com/google-deepmind/alphaevolve_results) | 302 | Jupyter Notebook | Apache-2.0 | 1 | 0 | 0 | sim |  |
| google-deepmind | [multi_object_datasets](https://github.com/google-deepmind/multi_object_datasets) | 290 | Python | Apache-2.0 | 10 | 0 | 0 | sim | Multi-object image datasets with ground-truth segmentation masks and generative  |
| google-deepmind | [onetwo](https://github.com/google-deepmind/onetwo) | 275 | Python | Apache-2.0 | 5 | 0 | 0 | sim |  |
| google-deepmind | [loft](https://github.com/google-deepmind/loft) | 238 | Python | Apache-2.0 | 1 | 0 | 0 | sim | LOFT: A 1 Million+ Token Long-Context Benchmark |
| google-deepmind | [alphaevolve_repository_of_problems](https://github.com/google-deepmind/alphaevolve_repository_of_problems) | 238 | Jupyter Notebook | Apache-2.0 | 10 | 0 | 0 | sim |  |
| google-deepmind | [predictingthepast](https://github.com/google-deepmind/predictingthepast) | 216 | Python | Apache-2.0 | 3 | 0 | 0 | sim |  |
| google-deepmind | [jax_privacy](https://github.com/google-deepmind/jax_privacy) | 199 | Python | Apache-2.0 | 26 | 0 | 0 | sim | Algorithms for Privacy-Preserving Machine Learning in JAX |
| google-deepmind | [neural_testbed](https://github.com/google-deepmind/neural_testbed) | 192 | Jupyter Notebook | Apache-2.0 | 3 | 0 | 0 | sim |  |
| google-deepmind | [jeo](https://github.com/google-deepmind/jeo) | 168 | Python | Apache-2.0 | 1 | 0 | 0 | sim | Jeo: Jax model training lib for Earth Observation |
| google-deepmind | [mishax](https://github.com/google-deepmind/mishax) | 158 | Python | Apache-2.0 | 3 | 0 | 0 | sim |  |
| google-deepmind | [transformer_grammars](https://github.com/google-deepmind/transformer_grammars) | 142 | Python | Apache-2.0 | 17 | 0 | 0 | sim | Transformer Grammars: Augmenting Transformer Language Models with Syntactic Indu |
| google-deepmind | [envlogger](https://github.com/google-deepmind/envlogger) | 136 | Python | Apache-2.0 | 2 | 0 | 0 | sim | A tool for recording RL trajectories. |
| google-deepmind | [geeflow](https://github.com/google-deepmind/geeflow) | 132 | Python | Apache-2.0 | 2 | 0 | 0 | sim | GeeFlow - generate and process large-scale geospatial datasets with Google Earth |
| google-deepmind | [tf2jax](https://github.com/google-deepmind/tf2jax) | 125 | Python | Apache-2.0 | 24 | 0 | 0 | sim |  |
| google-deepmind | [forest_typology](https://github.com/google-deepmind/forest_typology) | 116 | Jupyter Notebook | Apache-2.0 | 1 | 0 | 0 | sim | Datasets to protect Earth's forests and biodiversity |
| google-deepmind | [game_arena](https://github.com/google-deepmind/game_arena) | 116 | Python | Apache-2.0 | 4 | 0 | 0 | sim |  |
| google-deepmind | [dm_env_rpc](https://github.com/google-deepmind/dm_env_rpc) | 115 | Python | Apache-2.0 | 2 | 0 | 0 | sim | A networking protocol for agent-environment communication |
| google-deepmind | [c3_neural_compression](https://github.com/google-deepmind/c3_neural_compression) | 104 | Python | Apache-2.0 | 7 | 0 | 0 | sim |  |
| google-deepmind | [alphageometry2](https://github.com/google-deepmind/alphageometry2) | 102 | Python | Apache-2.0 | 2 | 0 | 0 | sim | AlphaGeometry2 symbolic engine (DDAR) with examples |
| google-deepmind | [neptune](https://github.com/google-deepmind/neptune) | 99 |  | Apache-2.0 | 5 | 0 | 0 | sim |  |
| google-deepmind | [gemma_penzai](https://github.com/google-deepmind/gemma_penzai) | 98 | Jupyter Notebook | Apache-2.0 | 1 | 0 | 0 | sim | A JAX Research Toolkit for Visualizing, Manipulating, and Understanding Gemma Mo |
| google-deepmind | [amplio](https://github.com/google-deepmind/amplio) | 98 | Go | Apache-2.0 | 2 | 0 | 0 | sim | Amplio: A Lightweight Agent Harness for Robust and Long-Horizon Runs |
| google-deepmind | [pushworld](https://github.com/google-deepmind/pushworld) | 96 | Python | Apache-2.0 | 8 | 0 | 0 | sim | PushWorld: A benchmark for manipulation planning with tools and movable obstacle |
| google-deepmind | [distribution_shift_framework](https://github.com/google-deepmind/distribution_shift_framework) | 90 | Python | Apache-2.0 | 15 | 0 | 0 | sim | This repository contains the code of the distribution shift framework presented  |
| google-deepmind | [nuclease_design](https://github.com/google-deepmind/nuclease_design) | 84 | Jupyter Notebook | Apache-2.0 | 1 | 0 | 0 | sim | ML-guided enzyme engineering |
| google-deepmind | [dks](https://github.com/google-deepmind/dks) | 82 | Python | Apache-2.0 | 3 | 0 | 0 | sim | Multi-framework implementation of Deep Kernel Shaping and Tailored Activation Tr |
| google-deepmind | [alignet](https://github.com/google-deepmind/alignet) | 80 | Python | Apache-2.0 | 4 | 0 | 0 | sim |  |
| google-deepmind | [dangerous-capability-evaluations](https://github.com/google-deepmind/dangerous-capability-evaluations) | 75 | Python | Apache-2.0 | 26 | 0 | 0 | sim |  |
| google-deepmind | [action_piece](https://github.com/google-deepmind/action_piece) | 71 | Python | Apache-2.0 | 4 | 0 | 0 | sim |  |
| google-deepmind | [bagz](https://github.com/google-deepmind/bagz) | 70 | C++ | Apache-2.0 | 9 | 0 | 0 | sim | Bagz is a format for storing a sequence of string records. It supports per-recor |
| google-deepmind | [calm](https://github.com/google-deepmind/calm) | 62 | Python | Apache-2.0 | 5 | 0 | 0 | sim |  |
| google-deepmind | [brave](https://github.com/google-deepmind/brave) | 52 | Python | Apache-2.0 | 4 | 0 | 0 | sim | A JAX implementation of Broaden Your Views for Self-Supervised Video Learning, o |
| google-deepmind | [xarray_jax](https://github.com/google-deepmind/xarray_jax) | 52 | Python | Apache-2.0 | 1 | 0 | 0 | sim |  |
| google-deepmind | [personality_in_llms](https://github.com/google-deepmind/personality_in_llms) | 50 | Jupyter Notebook | Apache-2.0 | 3 | 0 | 0 | sim |  |
| google-deepmind | [csuite](https://github.com/google-deepmind/csuite) | 49 | Python | Apache-2.0 | 5 | 0 | 0 | sim |  |
| google-deepmind | [disentangled_rnns](https://github.com/google-deepmind/disentangled_rnns) | 46 | Python | Apache-2.0 | 12 | 0 | 0 | sim | Fit DisRNNs to behavioral and neural data, interpret the fits, and discover new  |
| google-deepmind | [questbench](https://github.com/google-deepmind/questbench) | 44 | Python | Apache-2.0 | 1 | 0 | 0 | sim | QuestBench: Can LLMs ask the right question to acquire information in reasoning  |
| google-deepmind | [fusion_surrogates](https://github.com/google-deepmind/fusion_surrogates) | 44 | Python | Apache-2.0 | 3 | 0 | 0 | sim | A library of surrogate transport models for tokamak fusion. |
| google-deepmind | [proeval](https://github.com/google-deepmind/proeval) | 44 | Python | Apache-2.0 | 1 | 0 | 0 | sim | GenAI evaluation framework, optimized for 100x lower cost 🚀. |
| google-deepmind | [hierarchical_perceiver](https://github.com/google-deepmind/hierarchical_perceiver) | 37 | Python | Apache-2.0 | 5 | 0 | 0 | sim |  |
| google-deepmind | [formal-imo](https://github.com/google-deepmind/formal-imo) | 36 | Lean | Apache-2.0 | 1 | 0 | 0 | sim | Lean formalizations of IMO problem statements |
| google-deepmind | [fancyflags](https://github.com/google-deepmind/fancyflags) | 35 | Python | Apache-2.0 | 1 | 0 | 0 | sim | A Python library for defining structured command-line flags. |
| google-deepmind | [trecvit](https://github.com/google-deepmind/trecvit) | 33 | Python | Apache-2.0 | 2 | 0 | 0 | sim |  |
| google-research | [google-research](https://github.com/google-research/google-research) | 38887 | Jupyter Notebook | Apache-2.0 | 1997 | 0 | 0 | sim | Google Research |
| google-research | [timesfm](https://github.com/google-research/timesfm) | 34169 | Python | Apache-2.0 | 264 | 0 | 0 | nao | TimesFM (Time Series Foundation Model) is a pretrained time-series foundation mo |
| google-research | [vision_transformer](https://github.com/google-research/vision_transformer) | 12736 | Jupyter Notebook | Apache-2.0 | 136 | 0 | 0 | sim |  |
| google-research | [arxiv-latex-cleaner](https://github.com/google-research/arxiv-latex-cleaner) | 7083 | Python | Apache-2.0 | 42 | 0 | 0 | sim | arXiv LaTeX Cleaner: Easily clean the LaTeX code of your paper to submit to arXi |
| google-research | [text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer) | 6555 | Python | Apache-2.0 | 110 | 0 | 0 | sim | Code for the paper "Exploring the Limits of Transfer Learning with a Unified Tex |
| google-research | [scenic](https://github.com/google-research/scenic) | 3842 | Python | Apache-2.0 | 311 | 0 | 0 | sim | Scenic: A Jax Library for Computer Vision Research and Beyond |
| google-research | [t5x](https://github.com/google-research/t5x) | 3001 | Python | Apache-2.0 | 179 | 0 | 0 | sim |  |
| google-research | [kubric](https://github.com/google-research/kubric) | 2834 | Jupyter Notebook | Apache-2.0 | 78 | 0 | 0 | sim | A data generation pipeline for creating semi-realistic synthetic multi-object vi |
| google-research | [tabfm](https://github.com/google-research/tabfm) | 2718 | Python | Apache-2.0 | 50 | 0 | 0 | sim | TabFM (Tabular Foundation Model) is a pretrained tabular foundation model develo |
| google-research | [language](https://github.com/google-research/language) | 1814 | Python | Apache-2.0 | 124 | 0 | 0 | sim | Shared repository for open-sourced projects from the Google AI Language team. |
| google-research | [circuit_training](https://github.com/google-research/circuit_training) | 1726 | Python | Apache-2.0 | 39 | 0 | 0 | sim |  |
| google-research | [dex-lang](https://github.com/google-research/dex-lang) | 1702 | Haskell | BSD-3-Clause | 139 | 4 | 0 | sim | Research language for array processing in the Haskell/ML family |
| google-research | [FLAN](https://github.com/google-research/FLAN) | 1575 | Python | Apache-2.0 | 44 | 0 | 0 | sim |  |
| google-research | [rrsi](https://github.com/google-research/rrsi) | 1398 | Python | Apache-2.0 | 11 | 0 | 0 | sim |  |
| google-research | [morph-net](https://github.com/google-research/morph-net) | 1041 | Python | Apache-2.0 | 23 | 0 | 0 | sim | Fast & Simple Resource-Constrained Learning of Deep Network Structure |
| google-research | [inksight](https://github.com/google-research/inksight) | 1012 | Jupyter Notebook | Apache-2.0 | 18 | 0 | 0 | sim |  |
| google-research | [augmix](https://github.com/google-research/augmix) | 990 | Python | Apache-2.0 | 7 | 0 | 0 | nao | AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty |
| google-research | [android_world](https://github.com/google-research/android_world) | 947 | Python | Apache-2.0 | 53 | 0 | 0 | sim | AndroidWorld is an environment and benchmark for autonomous agents |
| google-research | [federated](https://github.com/google-research/federated) | 764 | Python | Apache-2.0 | 17 | 0 | 0 | sim | A collection of Google research projects related to Federated Learning and Feder |
| google-research | [jax3d](https://github.com/google-research/jax3d) | 761 | Python | Apache-2.0 | 46 | 0 | 0 | sim |  |
| google-research | [vmoe](https://github.com/google-research/vmoe) | 729 | Jupyter Notebook | Apache-2.0 | 29 | 0 | 0 | sim |  |
| google-research | [pix2struct](https://github.com/google-research/pix2struct) | 691 | Python | Apache-2.0 | 27 | 0 | 0 | sim |  |
| google-research | [sam](https://github.com/google-research/sam) | 647 | Python | Apache-2.0 | 29 | 0 | 0 | nao |  |
| google-research | [envharness](https://github.com/google-research/envharness) | 639 | Python | Apache-2.0 | 1 | 0 | 0 | sim |  |
| google-research | [weatherbench2](https://github.com/google-research/weatherbench2) | 638 | Python | Apache-2.0 | 97 | 0 | 0 | nao | A benchmark for the next generation of data-driven global weather models. |
| google-research | [reasoning-bank](https://github.com/google-research/reasoning-bank) | 611 | Python | Apache-2.0 | 33 | 0 | 0 | sim |  |
| google-research | [lasertagger](https://github.com/google-research/lasertagger) | 603 | Python | Apache-2.0 | 18 | 0 | 0 | sim |  |
| google-research | [papervizagent](https://github.com/google-research/papervizagent) | 512 | Python | Apache-2.0 | 7 | 0 | 0 | sim |  |
| google-research | [arco-era5](https://github.com/google-research/arco-era5) | 508 | Python | Apache-2.0 | 22 | 0 | 0 | sim | Recipes for reproducing Analysis-Ready & Cloud Optimized (ARCO) ERA5 datasets. |
| google-research | [robustness_metrics](https://github.com/google-research/robustness_metrics) | 473 | Jupyter Notebook | Apache-2.0 | 23 | 0 | 0 | sim |  |
| google-research | [self-organising-systems](https://github.com/google-research/self-organising-systems) | 442 | Jupyter Notebook | Apache-2.0 | 6 | 0 | 0 | nao |  |
| google-research | [population-dynamics](https://github.com/google-research/population-dynamics) | 431 | Jupyter Notebook | Apache-2.0 | 1 | 0 | 0 | sim | PDFM Embeddings: location-based vectors for geo-spatial analysis. |
| google-research | [perch](https://github.com/google-research/perch) | 397 | Python | Apache-2.0 | 24 | 0 | 0 | sim |  |
| google-research | [tf-slim](https://github.com/google-research/tf-slim) | 374 | Python | Apache-2.0 | 13 | 0 | 0 | sim |  |
| google-research | [language-table](https://github.com/google-research/language-table) | 368 | Jupyter Notebook | Apache-2.0 | 39 | 0 | 0 | sim | Suite of human-collected datasets and a multi-task continuous control benchmark  |
| google-research | [task_adaptation](https://github.com/google-research/task_adaptation) | 355 | Python | Apache-2.0 | 17 | 0 | 0 | sim |  |
| google-research | [flood-forecasting](https://github.com/google-research/flood-forecasting) | 352 | Python | Apache-2.0 | 2 | 0 | 0 | sim |  |
| google-research | [era](https://github.com/google-research/era) | 343 | Jupyter Notebook | Apache-2.0 | 2 | 0 | 0 | sim | Code associated with the paper An AI system to help scientists write expert-leve |
| google-research | [kauldron](https://github.com/google-research/kauldron) | 307 | Python | Apache-2.0 | 80 | 0 | 0 | sim | Modular, scalable library to train ML models |
| google-research | [lm-extraction-benchmark](https://github.com/google-research/lm-extraction-benchmark) | 306 | Python | Apache-2.0 | 15 | 0 | 0 | nao |  |
| google-research | [sparf](https://github.com/google-research/sparf) | 301 | Python | Apache-2.0 | 9 | 0 | 0 | nao | This is the official code release for SPARF: Neural Radiance Fields from Sparse  |
| google-research | [falken](https://github.com/google-research/falken) | 275 | Python | Apache-2.0 | 41 | 0 | 0 | sim | Falken provides developers with a service that allows them to train AI that can  |
| google-research | [weatherbenchX](https://github.com/google-research/weatherbenchX) | 260 | Python | Apache-2.0 | 109 | 0 | 0 | sim | A modular framework for evaluating weather forecasts |
| google-research | [perceiver-ar](https://github.com/google-research/perceiver-ar) | 258 | Python | Apache-2.0 | 31 | 0 | 0 | sim |  |
| google-research | [optformer](https://github.com/google-research/optformer) | 243 | Python | Apache-2.0 | 8 | 0 | 0 | sim |  |
| google-research | [cascades](https://github.com/google-research/cascades) | 229 | Python | Apache-2.0 | 3 | 0 | 0 | sim | Python library which enables complex compositions of language models such as scr |
| google-research | [pointdit](https://github.com/google-research/pointdit) | 228 | Python | Apache-2.0 | 1 | 0 | 0 | sim | [ICML'26] PointDiT: Pixel-Space Diffusion for Monocular Geometry Estimation |
| google-research | [nested-transformer](https://github.com/google-research/nested-transformer) | 204 | Jupyter Notebook | Apache-2.0 | 8 | 0 | 0 | sim | Nested Hierarchical Transformer https://arxiv.org/pdf/2105.12723.pdf |
| google-research | [composed_image_retrieval](https://github.com/google-research/composed_image_retrieval) | 200 | Shell | Apache-2.0 | 15 | 0 | 0 | sim |  |
| google-research | [proteinfer](https://github.com/google-research/proteinfer) | 196 | Jupyter Notebook | Apache-2.0 | 18 | 0 | 0 | sim | Deep networks for protein functional inference |
| google-research | [mood-board-search](https://github.com/google-research/mood-board-search) | 179 | Jupyter Notebook | Apache-2.0 | 39 | 0 | 0 | nao |  |
| google-research | [visu3d](https://github.com/google-research/visu3d) | 170 | Python | Apache-2.0 | 13 | 0 | 0 | sim | 3d without friction (Torch, TF, Jax, Numpy) |
| google-research | [skai](https://github.com/google-research/skai) | 166 | Python | Apache-2.0 | 29 | 0 | 0 | nao | SKAI is a machine learning based tool for performing automatic building damage a |
| google-research | [pathdreamer](https://github.com/google-research/pathdreamer) | 163 | Jupyter Notebook | Apache-2.0 | 13 | 0 | 0 | sim |  |
| google-research | [metricx](https://github.com/google-research/metricx) | 152 | Python | Apache-2.0 | 19 | 0 | 0 | sim |  |
| google-research | [semivl](https://github.com/google-research/semivl) | 148 | Python | Apache-2.0 | 10 | 0 | 0 | sim | [ECCV'24] Official Implementation of SemiVL: Semi-Supervised Semantic Segmentati |
| google-research | [spherical-cnn](https://github.com/google-research/spherical-cnn) | 145 | Python | Apache-2.0 | 6 | 0 | 0 | sim |  |
| google-research | [mt-metrics-eval](https://github.com/google-research/mt-metrics-eval) | 140 | Python | Apache-2.0 | 14 | 0 | 0 | sim | Tools for evaluating the performance of MT metrics on data from recent WMT metri |
| google-research | [neural-structural-optimization](https://github.com/google-research/neural-structural-optimization) | 130 | Jupyter Notebook | Apache-2.0 | 3 | 0 | 0 | sim | Neural reparameterization improves structural optimization |
| google-research | [perch-hoplite](https://github.com/google-research/perch-hoplite) | 128 | Python | Apache-2.0 | 26 | 3 | 0 | sim | Tooling for agile modeling on large machine perception embedding databases. |
| google-research | [e3x](https://github.com/google-research/e3x) | 126 | Python | Apache-2.0 | 5 | 0 | 0 | sim | E3x is a JAX library for constructing efficient E(3)-equivariant deep learning a |
| google-research | [corenet](https://github.com/google-research/corenet) | 121 | Python | Apache-2.0 | 7 | 0 | 0 | sim | CoReNet is a technique for joint multi-object 3D reconstruction from a single RG |
| google-research | [dice_rl](https://github.com/google-research/dice_rl) | 113 | Python | Apache-2.0 | 9 | 0 | 0 | sim |  |
| google-research | [hyperbo](https://github.com/google-research/hyperbo) | 106 | Python | Apache-2.0 | 11 | 0 | 0 | sim | Pre-trained Gaussian processes for Bayesian optimization |
| google-research | [project-guideline](https://github.com/google-research/project-guideline) | 106 | C++ | Apache-2.0 | 13 | 0 | 0 | sim | Project Guideline is a research project that leverages on-device ML to enable pe |
| google-research | [crest](https://github.com/google-research/crest) | 102 | Python | Apache-2.0 | 6 | 0 | 0 | sim | Repo for CReST: A Class-Rebalancing Self-Training Framework for Imbalanced Semi- |
| google-research | [sofima](https://github.com/google-research/sofima) | 102 | Jupyter Notebook | Apache-2.0 | 15 | 0 | 0 | nao | Scalable Optical Flow-based Image Montaging and Alignment |
| google-research | [true](https://github.com/google-research/true) | 93 | Python | Apache-2.0 | 17 | 0 | 0 | nao | Code and data accompanying the paper "TRUE: Re-evaluating Factual Consistency Ev |
| google-research | [swirl-dynamics](https://github.com/google-research/swirl-dynamics) | 87 | Jupyter Notebook | licenca propria (ver LICENSE) | 19 | 0 | 0 | sim | Swirl-Dynamics is a python repository that provides implementations of models, b |
| google-research | [zapbench](https://github.com/google-research/zapbench) | 80 | Python | Apache-2.0 | 9 | 0 | 0 | sim | The Zebrafish Activity Prediction Benchmark measures progress on the problem of  |
| google-research | [lanistr](https://github.com/google-research/lanistr) | 77 | Python | licenca propria (ver LICENSE) | 11 | 0 | 0 | sim |  |
| google-research | [swirl-lm](https://github.com/google-research/swirl-lm) | 75 | Python | Apache-2.0 | 4 | 0 | 0 | sim |  |
| google-research | [mseb](https://github.com/google-research/mseb) | 71 | Python | Apache-2.0 | 1 | 0 | 0 | sim |  |
| google-research | [connectomics](https://github.com/google-research/connectomics) | 67 | Python | Apache-2.0 | 27 | 0 | 0 | sim |  |
| google-research | [agent-based-epidemic-sim](https://github.com/google-research/agent-based-epidemic-sim) | 65 | Jupyter Notebook | Apache-2.0 | 17 | 0 | 0 | sim |  |
| google-research | [veriharness](https://github.com/google-research/veriharness) | 61 | Python | Apache-2.0 | 2 | 0 | 0 | sim |  |
| google-research | [structured-additive-IR](https://github.com/google-research/structured-additive-IR) | 60 | C++ | Apache-2.0 | 21 | 0 | 0 | sim |  |
| google-research | [openfst](https://github.com/google-research/openfst) | 60 | C++ | Apache-2.0 | 9 | 0 | 0 | sim | Finite-state Transducer (FST) Library. |
| google-research | [babelcode](https://github.com/google-research/babelcode) | 56 | Python | Apache-2.0 | 4 | 0 | 0 | sim |  |
| google-research | [spade_anomaly_detection](https://github.com/google-research/spade_anomaly_detection) | 55 | Python | Apache-2.0 | 5 | 0 | 0 | sim | Semi-supervised anomaly detection method |
| google-research | [nisaba](https://github.com/google-research/nisaba) | 54 | Python | Apache-2.0 | 36 | 0 | 0 | sim | Finite-state script normalization and processing utilities |
| google-research | [dataclass_array](https://github.com/google-research/dataclass_array) | 54 | Python | Apache-2.0 | 4 | 0 | 0 | sim | Dataclasses manipulated as numpy arrays (with batching, reshape, slicing,...) |
| google-research | [mozolm](https://github.com/google-research/mozolm) | 52 | C++ | Apache-2.0 | 7 | 0 | 0 | sim | MozoLM: A language model (LM) serving library |
| google-research | [MapTrace](https://github.com/google-research/MapTrace) | 52 | Python | Apache-2.0 | 1 | 0 | 0 | nao |  |
| google-research | [last](https://github.com/google-research/last) | 48 | Python | Apache-2.0 | 2 | 0 | 0 | sim | A JAX library for building lattice-based speech transducer models |
| google-research | [DP-FTRL](https://github.com/google-research/DP-FTRL) | 39 | Python | Apache-2.0 | 5 | 0 | 0 | sim | DP-FTRL from "Practical and Private (Deep) Learning without Sampling or Shufflin |
| google-research | [agentic-visualization](https://github.com/google-research/agentic-visualization) | 39 | Python | Apache-2.0 | 2 | 0 | 0 | sim | CoDA is a multi-agent framework that turns natural language queries into publica |
| google-research | [m2svid](https://github.com/google-research/m2svid) | 38 | Python | Apache-2.0 | 4 | 0 | 0 | sim |  This is the official code release for “M2SVid: End-to-End Inpainting and Refine |
| google-research | [raksha](https://github.com/google-research/raksha) | 37 | C++ | Apache-2.0 | 104 | 0 | 0 | sim |  |
| google-research | [tnco](https://github.com/google-research/tnco) | 36 | Python | Apache-2.0 | 3 | 0 | 1 | sim | TNCO is a heuristic tool that optimizes tensor network contraction paths. |
| google-research | [understanding-curricula](https://github.com/google-research/understanding-curricula) | 34 | Python | Apache-2.0 | 2 | 0 | 0 | sim |  |
| google-research | [fool-me-twice](https://github.com/google-research/fool-me-twice) | 34 | JavaScript | Apache-2.0 | 49 | 0 | 0 | sim | Game code and data for Fool Me Twice: Entailment from Wikipedia Gamification htt |
| google-research | [precondition](https://github.com/google-research/precondition) | 34 | Jupyter Notebook | Apache-2.0 | 9 | 0 | 0 | sim |  |

### NASA (110 de 294 candidatos)

| Org | Repo | ★ | Lang | Licenca | Issues | GFI | HW | CONTRIB. | Descricao |
|---|---|---|---|---|---|---|---|---|---|
| NASA-AMMOS | [3DTilesRendererJS](https://github.com/NASA-AMMOS/3DTilesRendererJS) | 2482 | JavaScript | Apache-2.0 | 107 | 0 | 3 | sim | Renderer for 3D Tiles in Javascript using three.js, Babylon.js, and r3f |
| NASA-AMMOS | [MMGIS](https://github.com/NASA-AMMOS/MMGIS) | 232 | JavaScript | Apache-2.0 | 100 | 3 | 0 | sim | Multi-Mission Geographic Information System - A Web-based Mapping and Spatial Da |
| NASA-AMMOS | [plandev](https://github.com/NASA-AMMOS/plandev) | 129 | Java | MIT | 232 | 0 | 0 | sim | PlanDev - A software framework for modeling spacecraft. |
| NASA-AMMOS | [VICAR](https://github.com/NASA-AMMOS/VICAR) | 57 | C | Apache-2.0 | 4 | 0 | 0 | sim | VICAR, which stands for Video Image Communication And Retrieval, is a general pu |
| NASA-AMMOS | [AIT-Core](https://github.com/NASA-AMMOS/AIT-Core) | 56 | Python | MIT | 109 | 2 | 0 | nao |  |
| NASA-AMMOS | [plandev-ui](https://github.com/NASA-AMMOS/plandev-ui) | 42 | Svelte | MIT | 319 | 0 | 0 | sim | The client application for PlanDev |
| NASA-AMMOS | [Landform](https://github.com/NASA-AMMOS/Landform) | 40 | C# | licenca propria (ver LICENSE) | 1 | 0 | 0 | nao | Terrain mesh processing toolkit that can generate 3DTiles datasets |
| NASA-AMMOS | [slim](https://github.com/NASA-AMMOS/slim) | 36 | JavaScript | Apache-2.0 | 58 | 0 | 0 | sim | Software Lifecycle Improvement & Modernization |
| NASA-AMMOS | [AIT-GUI](https://github.com/NASA-AMMOS/AIT-GUI) | 32 | JavaScript | MIT | 70 | 0 | 0 | nao |  |
| NASA-AMMOS | [LithoSphere](https://github.com/NASA-AMMOS/LithoSphere) | 27 | TypeScript | licenca propria (ver LICENSE) | 10 | 0 | 1 | nao | A Tiled 3D Planetary Web-Based GIS JavaScript Library |
| NASA-AMMOS | [BSL](https://github.com/NASA-AMMOS/BSL) | 15 | C | Apache-2.0 | 26 | 4 | 0 | sim | Bundle Protocol Security Library (BSL) |
| NASA-IMPACT | [Prithvi-WxC](https://github.com/NASA-IMPACT/Prithvi-WxC) | 204 | Python | MIT | 11 | 0 | 0 | nao | Implementation of the Prithvi WxC Foundation Model and Downstream Tasks |
| NASA-IMPACT | [Surya](https://github.com/NASA-IMPACT/Surya) | 179 | Jupyter Notebook | Apache-2.0 | 6 | 0 | 0 | nao | Implementation of the Surya Foundation Model and Downstream Tasks for Heliophysi |
| NASA-IMPACT | [veda-ui](https://github.com/NASA-IMPACT/veda-ui) | 45 | TypeScript | licenca propria (ver LICENSE) | 130 | 6 | 0 | nao | Frontend for the Dashboard Evolution project |
| NASA-IMPACT | [veda-config](https://github.com/NASA-IMPACT/veda-config) | 35 | MDX | licenca propria (ver LICENSE) | 46 | 0 | 0 | nao | Configuration template for the Dashboard Evolution Project |
| NASA-IMPACT | [veda-backend](https://github.com/NASA-IMPACT/veda-backend) | 22 | Python | licenca propria (ver LICENSE) | 80 | 3 | 1 | sim | Backend services for VEDA |
| NASA-IMPACT | [veda-docs](https://github.com/NASA-IMPACT/veda-docs) | 10 | Jupyter Notebook | Apache-2.0 | 51 | 2 | 0 | nao | Documentation for the VEDA Project |
| NASA-SW-VnV | [ikos](https://github.com/NASA-SW-VnV/ikos) | 3171 | C++ | licenca propria (ver LICENSE) | 51 | 0 | 0 | nao | Static analyzer for C/C++ based on the theory of Abstract Interpretation. |
| NASA-SW-VnV | [fret](https://github.com/NASA-SW-VnV/fret) | 486 | JavaScript | licenca propria (ver LICENSE) | 35 | 0 | 0 | nao | A framework for the elicitation, specification, formalization and analysis of re |
| NASA-SW-VnV | [CoCoSim](https://github.com/NASA-SW-VnV/CoCoSim) | 62 | MATLAB | licenca propria (ver LICENSE) | 1 | 0 | 0 | nao | Automated Analysis Framework for Simulink/Stateflow models. |
| nasa | [openmct](https://github.com/nasa/openmct) | 13161 | JavaScript | licenca propria (ver LICENSE) | 1090 | 0 | 0 | sim | A web based mission control framework.  |
| nasa | [fprime](https://github.com/nasa/fprime) | 11822 | C++ | Apache-2.0 | 455 | 0 | 1 | sim | F´ - A flight software and embedded systems framework |
| nasa | [spacewasm](https://github.com/nasa/spacewasm) | 1606 | Rust | Apache-2.0 | 25 | 1 | 0 | sim | A flight-compliant WebAssembly interpreter. |
| nasa | [cFS](https://github.com/nasa/cFS) | 1527 | C | Apache-2.0 | 105 | 4 | 1 | sim | The Core Flight System (cFS) |
| nasa | [astrobee](https://github.com/nasa/astrobee) | 1421 | C++ | Apache-2.0 | 32 | 0 | 0 | nao | NASA Astrobee Robot Software |
| nasa | [apod-api](https://github.com/nasa/apod-api) | 1091 | Python | Apache-2.0 | 20 | 0 | 0 | nao | Astronomy Picture of the Day API service |
| nasa | [earthdata-search](https://github.com/nasa/earthdata-search) | 826 | JavaScript | licenca propria (ver LICENSE) | 16 | 0 | 0 | sim | Earthdata Search is a web application developed by NASA EOSDIS to enable data di |
| nasa | [osal](https://github.com/nasa/osal) | 673 | C | Apache-2.0 | 152 | 4 | 0 | sim | The Core Flight System (cFS) Operating System Abstraction Layer (OSAL) |
| nasa | [nos3](https://github.com/nasa/nos3) | 639 | C | licenca propria (ver LICENSE) | 71 | 0 | 0 | sim | NASA Operational Simulator for Space Systems |
| nasa | [ogma](https://github.com/nasa/ogma) | 584 | Haskell | Apache-2.0 | 8 | 0 | 0 | nao | Generator of runtime monitors for flight and robotics applications. |
| nasa | [cFE](https://github.com/nasa/cFE) | 526 | C | Apache-2.0 | 435 | 7 | 0 | sim | The Core Flight System (cFS) Core Flight Executive (cFE) |
| nasa | [Common-Metadata-Repository](https://github.com/nasa/Common-Metadata-Repository) | 399 | Clojure | Apache-2.0 | 19 | 0 | 0 | sim | The Common Metadata Repository (CMR) is a high-performance, high-quality, contin |
| nasa | [cumulus](https://github.com/nasa/cumulus) | 306 | JavaScript | licenca propria (ver LICENSE) | 48 | 0 | 0 | sim | Cumulus Framework + Cumulus API |
| nasa | [nasa-latex-docs](https://github.com/nasa/nasa-latex-docs) | 274 | TeX | MIT | 10 | 0 | 0 | nao | An easy and convenient package to create technical LaTeX documents. |
| nasa | [delta](https://github.com/nasa/delta) | 231 | Python | Apache-2.0 | 7 | 0 | 0 | nao | Deep Learning for Satellite Imagery |
| nasa | [cea](https://github.com/nasa/cea) | 210 | Fortran | Apache-2.0 | 33 | 0 | 0 | sim | CEA computes the equilibrium composition of mixtures via free-energy minimizatio |
| nasa | [CrisisMappingToolkit](https://github.com/nasa/CrisisMappingToolkit) | 206 | Python | Apache-2.0 | 7 | 0 | 0 | nao | NASA Ames Crisis Mapping Toolkit |
| nasa | [icarous](https://github.com/nasa/icarous) | 179 | C | licenca propria (ver LICENSE) | 5 | 0 | 0 | nao | ICAROUS  is a software architecture for the development of UAS applications |
| nasa | [trick](https://github.com/nasa/trick) | 176 | C++ | licenca propria (ver LICENSE) | 111 | 0 | 0 | sim | Trick Simulation Environment.  Trick provides a common set of simulation capabil |
| nasa | [CryptoLib](https://github.com/nasa/CryptoLib) | 173 | C | licenca propria (ver LICENSE) | 78 | 0 | 0 | nao | Provide a software-only solution using the CCSDS Space Data Link Security Protoc |
| nasa | [CompDam_DGD](https://github.com/nasa/CompDam_DGD) | 167 | Fortran | licenca propria (ver LICENSE) | 5 | 0 | 0 | nao |  |
| nasa | [HDTN](https://github.com/nasa/HDTN) | 150 | C++ | Apache-2.0 | 36 | 0 | 0 | nao | High-rate Delay Tolerant Network (HDTN) Software |
| nasa | [HLS-Data-Resources](https://github.com/nasa/HLS-Data-Resources) | 149 | Jupyter Notebook | Apache-2.0 | 1 | 0 | 0 | nao | This repository provides guides, short how-tos, and tutorials to help users acce |
| nasa | [SMCPy](https://github.com/nasa/SMCPy) | 137 | Python | licenca propria (ver LICENSE) | 6 | 0 | 0 | nao | Python module for uncertainty quantification using a parallel sequential Monte C |
| nasa | [progpy](https://github.com/nasa/progpy) | 135 | Python | licenca propria (ver LICENSE) | 100 | 1 | 0 | nao | The NASA Prognostic Python Packages is a Python framework focused on defining an |
| nasa | [GMAT](https://github.com/nasa/GMAT) | 124 | C++ | Apache-2.0 | 10 | 0 | 0 | nao | The General Mission Analysis Tool (GMAT) is the world's leading enterprise, mult |
| nasa | [fpp](https://github.com/nasa/fpp) | 121 | Scala | Apache-2.0 | 95 | 0 | 0 | sim | F Prime Prime: A modeling language for F Prime |
| nasa | [CF](https://github.com/nasa/CF) | 120 | C | Apache-2.0 | 47 | 0 | 0 | sim | The Core Flight System (cFS) CFDP application. |
| nasa | [NDAS](https://github.com/nasa/NDAS) | 109 | LabVIEW | licenca propria (ver LICENSE) | 4 | 0 | 0 | nao | NASA Data Acquisition System (NDAS) is an integrated suite of applications that  |
| nasa | [mmt](https://github.com/nasa/mmt) | 106 | JavaScript | Apache-2.0 | 18 | 0 | 0 | sim | NASA's Metadata Management Tool. |
| nasa | [cFS-GroundSystem](https://github.com/nasa/cFS-GroundSystem) | 104 | Python | Apache-2.0 | 30 | 1 | 1 | sim | The Core Flight System (cFS) Ground System Lab Tool (cFS-GroundSystem) |
| nasa | [harmony](https://github.com/nasa/harmony) | 99 | TypeScript | licenca propria (ver LICENSE) | 4 | 0 | 0 | nao | Application for providing services for Earth observation data in the cloud using |
| nasa | [NASA-Acronyms](https://github.com/nasa/NASA-Acronyms) | 93 | JavaScript | MIT | 10 | 0 | 0 | nao |  |
| nasa | [IDF](https://github.com/nasa/IDF) | 91 | C++ | licenca propria (ver LICENSE) | 1 | 0 | 0 | nao |  |
| nasa | [PSP](https://github.com/nasa/PSP) | 82 | C | Apache-2.0 | 49 | 2 | 0 | sim | The Core Flight System (cFS) Platform Support Package (PSP) |
| nasa | [TrickHLA](https://github.com/nasa/TrickHLA) | 80 | C++ | licenca propria (ver LICENSE) | 9 | 1 | 0 | nao | TrickHLA: An IEEE 1516 High Level Architecture (HLA) Simulation Interoperability |
| nasa | [LPDAAC-Data-Resources](https://github.com/nasa/LPDAAC-Data-Resources) | 78 | Jupyter Notebook | Apache-2.0 | 2 | 0 | 0 | sim | This repository is a place to find data user resources that demonstrate how to u |
| nasa | [LHASA](https://github.com/nasa/LHASA) | 77 | Python | licenca propria (ver LICENSE) | 2 | 0 | 0 | nao | Landslide Hazard Analysis for Situational Awareness |
| nasa | [bingo](https://github.com/nasa/bingo) | 75 | Python | Apache-2.0 | 7 | 0 | 0 | nao |  |
| nasa | [gunns](https://github.com/nasa/gunns) | 74 | C++ | licenca propria (ver LICENSE) | 19 | 0 | 0 | nao | The NASA General-Use Nodal Network Solver (GUNNS) software |
| nasa | [harmony-py](https://github.com/nasa/harmony-py) | 71 | Python | licenca propria (ver LICENSE) | 8 | 0 | 0 | sim | Python client library for working with NASA’s Earth observing system data using  |
| nasa | [UQPCE](https://github.com/nasa/UQPCE) | 70 | Python | licenca propria (ver LICENSE) | 4 | 0 | 0 | nao | Uncertainty Quantification using Polynomial Chaos Expansion (UQPCE) is an open-s |
| nasa | [cumulus-dashboard](https://github.com/nasa/cumulus-dashboard) | 69 | JavaScript | licenca propria (ver LICENSE) | 23 | 0 | 0 | sim | Cumulus API Dashboard |
| nasa | [cmr-stac](https://github.com/nasa/cmr-stac) | 68 | TypeScript | licenca propria (ver LICENSE) | 30 | 0 | 0 | nao |  |
| nasa | [sample_app](https://github.com/nasa/sample_app) | 67 | C | Apache-2.0 | 14 | 1 | 0 | sim | The Core Flight System (cFS) Sample App (sample_app) |
| nasa | [condor](https://github.com/nasa/condor) | 63 | Python | licenca propria (ver LICENSE) | 37 | 0 | 0 | nao | NASA's Condor is a framework for mathematical modeling of engineering systems in |
| nasa | [Kamodo](https://github.com/nasa/Kamodo) | 60 | Jupyter Notebook | licenca propria (ver LICENSE) | 6 | 0 | 0 | nao | The CCMC Kamodo Analysis Suite is a NASA open-source Python package designed for |
| nasa | [earthdata-download](https://github.com/nasa/earthdata-download) | 60 | TypeScript | licenca propria (ver LICENSE) | 6 | 0 | 0 | sim | Download your Earth science data with only one click |
| nasa | [fmdtools](https://github.com/nasa/fmdtools) | 59 | HTML | licenca propria (ver LICENSE) | 6 | 0 | 0 | nao | System Resilience Modelling, Simulation, and Assessment in Python |
| nasa | [vscode-pvs](https://github.com/nasa/vscode-pvs) | 54 | TypeScript | licenca propria (ver LICENSE) | 4 | 0 | 0 | nao | LAR-19642-1: Visual Studio Code Extension for PVS |
| nasa | [hermes](https://github.com/nasa/hermes) | 54 | TypeScript | Apache-2.0 | 40 | 0 | 0 | nao | A lightweight extensible spacecraft commanding & telemetry processing framework |
| nasa | [HyperCP](https://github.com/nasa/HyperCP) | 53 | Python | licenca propria (ver LICENSE) | 8 | 0 | 0 | nao |  |
| nasa | [scrub](https://github.com/nasa/scrub) | 50 | Python | Apache-2.0 | 19 | 0 | 0 | nao | SCRUB is a platform for orchestration and aggregation of static code analysis to |
| nasa | [jeod](https://github.com/nasa/jeod) | 49 | C++ | licenca propria (ver LICENSE) | 10 | 0 | 0 | nao |  |
| nasa | [DS](https://github.com/nasa/DS) | 48 | C | Apache-2.0 | 18 | 0 | 0 | sim | The Core Flight System (cFS) Data Storage (DS) application. |
| nasa | [SC](https://github.com/nasa/SC) | 47 | C | Apache-2.0 | 21 | 1 | 0 | sim | The Core Flight System (cFS) Stored Commands (SC) application. |
| nasa | [HS](https://github.com/nasa/HS) | 46 | C | Apache-2.0 | 10 | 0 | 0 | sim | The Core Flight System (cFS) Health and Safety (HS) application. |
| nasa | [FM](https://github.com/nasa/FM) | 45 | C | Apache-2.0 | 18 | 0 | 0 | sim | The Core Flight System (cFS) File Manager (FM) application. |
| nasa | [HK](https://github.com/nasa/HK) | 43 | C | Apache-2.0 | 15 | 1 | 0 | sim | The Core Flight System (cFS) Housekeeping (HK) application. |
| nasa | [SBN](https://github.com/nasa/SBN) | 43 | C | Apache-2.0 | 37 | 0 | 1 | nao |  |
| nasa | [EdsLib](https://github.com/nasa/EdsLib) | 42 | C | Apache-2.0 | 11 | 0 | 0 | nao | CCSDS SOIS Electronic Data Sheet Tool and Library |
| nasa | [fprime-tools](https://github.com/nasa/fprime-tools) | 39 | Python | Apache-2.0 | 2 | 0 | 0 | nao | F´ Python tooling and helpers. |
| nasa | [abaverify](https://github.com/nasa/abaverify) | 38 | Python | licenca propria (ver LICENSE) | 1 | 0 | 0 | nao |  |
| nasa | [AppEEARS-Data-Resources](https://github.com/nasa/AppEEARS-Data-Resources) | 37 | Jupyter Notebook | Apache-2.0 | 1 | 0 | 0 | sim |  This repository provides resources and tutorials to help users work with AppEEA |
| nasa | [CS](https://github.com/nasa/CS) | 36 | C | Apache-2.0 | 33 | 0 | 0 | sim | The Core Flight System (cFS) Checksum (CS) application. |
| nasa | [LC](https://github.com/nasa/LC) | 35 | C | Apache-2.0 | 17 | 0 | 0 | sim | The Core Flight System (cFS) Limit Checker (LC) application. |
| nasa | [bplib](https://github.com/nasa/bplib) | 35 | C | Apache-2.0 | 49 | 0 | 0 | sim |  |
| nasa | [MM](https://github.com/nasa/MM) | 34 | C | Apache-2.0 | 23 | 0 | 0 | sim | The Core Flight System (cFS) Memory Manager (MM) application. |
| nasa | [cape](https://github.com/nasa/cape) | 34 | Python | licenca propria (ver LICENSE) | 6 | 0 | 0 | nao | Computational Aerosciences Productivity & Execution |
| nasa | [ncompare](https://github.com/nasa/ncompare) | 34 | Python | Apache-2.0 | 22 | 0 | 0 | sim | Compare the structure of two netCDF (or HDF5) files |
| nasa | [koviz](https://github.com/nasa/koviz) | 30 | C++ | licenca propria (ver LICENSE) | 2 | 0 | 0 | nao | Koviz is a Trick simulation data plotting, visualization and analysis tool |
| nasa | [python_cmr](https://github.com/nasa/python_cmr) | 29 | Python | MIT | 28 | 3 | 0 | nao | Python library for querying the common metadata repository. |
| nasa | [elf2cfetbl](https://github.com/nasa/elf2cfetbl) | 28 | C | Apache-2.0 | 26 | 1 | 0 | sim | The Core Flight System (cFS) ELF to CFE Table Tool (elf2cfetbl) |
| nasa-gcn | [gcn.nasa.gov](https://github.com/nasa-gcn/gcn.nasa.gov) | 225 | TypeScript | licenca propria (ver LICENSE) | 139 | 2 | 0 | nao | General Coordinates Network (GCN): NASA’s Next Generation Time-Domain and Multim |
| nasa-jpl | [open-source-rover](https://github.com/nasa-jpl/open-source-rover) | 9691 | HTML | Apache-2.0 | 17 | 0 | 2 | sim | A build-it-yourself, 6-wheel rover based on the rovers on Mars! |
| nasa-jpl | [rosa](https://github.com/nasa-jpl/rosa) | 1655 | Python | Apache-2.0 | 5 | 0 | 0 | sim | ROSA 🤖 is an AI Agent designed to interact with ROS1- and ROS2-based robotics sy |
| nasa-jpl | [osr-rover-code](https://github.com/nasa-jpl/osr-rover-code) | 541 | Python | Apache-2.0 | 20 | 2 | 5 | nao | Code that runs on the Open Source Rover |
| nasa-jpl | [autoRIFT](https://github.com/nasa-jpl/autoRIFT) | 272 | Python | Apache-2.0 | 22 | 0 | 2 | nao | A Python module of a fast and intelligent algorithm for finding the pixel displa |
| nasa-jpl | [visual-perception-engine](https://github.com/nasa-jpl/visual-perception-engine) | 198 | Python | Apache-2.0 | 2 | 0 | 0 | sim | Visual Perception Engine: fast and flexible framework designed to run multiple p |
| nasa-jpl | [ION-DTN](https://github.com/nasa-jpl/ION-DTN) | 146 | C | licenca propria (ver LICENSE) | 24 | 0 | 0 | nao | The Interplanetary Overlay Network (ION) is NASA's open-source software implemen |
| nasa-jpl | [nebula2-wildos](https://github.com/nasa-jpl/nebula2-wildos) | 146 | Python | Apache-2.0 | 2 | 0 | 0 | nao | WildOS: Open-Vocabulary Object Search in the Wild |
| nasa-jpl | [explorer-1](https://github.com/nasa-jpl/explorer-1) | 77 | Vue | MIT | 30 | 0 | 0 | sim | JPL's Design System |
| nasa-jpl | [fastcat](https://github.com/nasa-jpl/fastcat) | 69 | C++ | Apache-2.0 | 6 | 0 | 0 | nao | C++ EtherCAT Device Command & Control Library |
| nasa-jpl | [its_live](https://github.com/nasa-jpl/its_live) | 55 | Python | MIT | 2 | 0 | 0 | nao | Tool repository for accessing and working with ITS_LIVE data. |
| nasa-jpl | [honeycomb](https://github.com/nasa-jpl/honeycomb) | 37 | TypeScript | Apache-2.0 | 3 | 0 | 0 | nao | A 3D Visualization Framework built for Robotics |
| nasa-jpl | [FlightView](https://github.com/nasa-jpl/FlightView) | 30 | C++ | GPL-3.0 | 2 | 0 | 0 | nao | Real-time tools for Imaging Spectroscopy Data |
| nasa-jpl | [jsd](https://github.com/nasa-jpl/jsd) | 26 | C | Apache-2.0 | 18 | 0 | 1 | nao | Just SOEM Drivers |
| nasa-jpl | [tos2ca-anomaly-detection](https://github.com/nasa-jpl/tos2ca-anomaly-detection) | 1 | Python | licenca propria (ver LICENSE) | 11 | 2 | 0 | sim | Python library that defines a phenomenon and curates data for the user |
| nasa-jpl | [dtnnr](https://github.com/nasa-jpl/dtnnr) | 1 | Python | Apache-2.0 | 33 | 2 | 3 | nao | Delay-Tolerant Networking Node Registry |
| nasa-jpl | [tsn-time-agreement](https://github.com/nasa-jpl/tsn-time-agreement) | 0 | C++ | Apache-2.0 | 16 | 0 | 1 | nao |  |

### NOAA (44 de 188 candidatos)

| Org | Repo | ★ | Lang | Licenca | Issues | GFI | HW | CONTRIB. | Descricao |
|---|---|---|---|---|---|---|---|---|---|
| NOAA-EMC | [WW3](https://github.com/NOAA-EMC/WW3) | 349 | Fortran | licenca propria (ver LICENSE) | 241 | 0 | 2 | sim | WAVEWATCH III |
| NOAA-EMC | [global-workflow](https://github.com/NOAA-EMC/global-workflow) | 102 | Shell | LGPL-3.0 | 244 | 0 | 7 | nao | Global Superstructure/Workflow supporting the Global, Global Ensemble, and Seaso |
| NOAA-EMC | [GSI](https://github.com/NOAA-EMC/GSI) | 86 | Fortran | LGPL-3.0 | 24 | 0 | 0 | nao | Gridpoint Statistical Interpolation |
| NOAA-EMC | [NCEPLIBS-bufr](https://github.com/NOAA-EMC/NCEPLIBS-bufr) | 62 | Fortran | licenca propria (ver LICENSE) | 12 | 0 | 0 | nao | The NCEPLIBS-bufr library contains routines and utilites for working with the WM |
| NOAA-EMC | [NCEPLIBS](https://github.com/NOAA-EMC/NCEPLIBS) | 47 | Python | licenca propria (ver LICENSE) | 24 | 0 | 0 | nao | Top level repo containing submodules for NCEPLIBS and associated dependencies fo |
| NOAA-EMC | [ufsatm](https://github.com/NOAA-EMC/ufsatm) | 45 | Fortran | licenca propria (ver LICENSE) | 66 | 0 | 0 | nao |  |
| NOAA-EMC | [UPP](https://github.com/NOAA-EMC/UPP) | 42 | Fortran | licenca propria (ver LICENSE) | 26 | 0 | 0 | nao |  |
| NOAA-EMC | [hpc-stack](https://github.com/NOAA-EMC/hpc-stack) | 32 | Shell | LGPL-2.1 | 41 | 0 | 0 | nao | Create a software stack for HPC's |
| NOAA-EMC | [GDASApp](https://github.com/NOAA-EMC/GDASApp) | 25 | Jinja | LGPL-2.1 | 163 | 0 | 1 | nao | Global Data Assimilation System Application |
| NOAA-EMC | [AQM](https://github.com/NOAA-EMC/AQM) | 5 | Fortran | GPL-3.0 | 10 | 0 | 1 | nao |  |
| NOAA-EMC | [HELM](https://github.com/NOAA-EMC/HELM) | 2 | C++ | CC0-1.0 | 7 | 0 | 1 | nao | A unique set of libraries needed for ESM |
| NOAA-EMC | [ocean_ice_evaluation](https://github.com/NOAA-EMC/ocean_ice_evaluation) | 0 | Python | CC0-1.0 | 2 | 1 | 0 | nao |  |
| NOAA-GFDL | [FMS](https://github.com/NOAA-GFDL/FMS) | 121 | Fortran | licenca propria (ver LICENSE) | 108 | 0 | 5 | sim | GFDL's Flexible Modeling System |
| NOAA-GFDL | [MOM6-examples](https://github.com/NOAA-GFDL/MOM6-examples) | 99 | Jupyter Notebook | licenca propria (ver LICENSE) | 65 | 0 | 1 | nao | Example configurations for MOM6 and SIS2 |
| NOAA-GFDL | [GFDL_atmos_cubed_sphere](https://github.com/NOAA-GFDL/GFDL_atmos_cubed_sphere) | 82 | Fortran | licenca propria (ver LICENSE) | 27 | 0 | 0 | nao | The GFDL atmos_cubed_sphere dynamical core code |
| NOAA-GFDL | [MDTF-diagnostics](https://github.com/NOAA-GFDL/MDTF-diagnostics) | 81 | Jupyter Notebook | licenca propria (ver LICENSE) | 73 | 0 | 2 | sim | Analysis framework and collection of process-oriented diagnostics for weather an |
| NOAA-GFDL | [FRE-NCtools](https://github.com/NOAA-GFDL/FRE-NCtools) | 25 | C | LGPL-3.0 | 34 | 0 | 2 | sim | Tools for manipulating and creating netCDF inputs for FMS managed models |
| NOAA-GFDL | [pace](https://github.com/NOAA-GFDL/pace) | 21 | Python | Apache-2.0 | 38 | 4 | 1 | sim | Re-write of FV3GFS weather/climate model in Python |
| NOAA-GFDL | [NDSL](https://github.com/NOAA-GFDL/NDSL) | 11 | Python | Apache-2.0 | 53 | 2 | 0 | nao | NOAA NASA Domain Specific Language middleware layer |
| NOAA-GFDL | [pyFV3](https://github.com/NOAA-GFDL/pyFV3) | 7 | Python | Apache-2.0 | 16 | 0 | 1 | nao | Python version of FV3 dynamical core |
| NOAA-GFDL | [fre-cli](https://github.com/NOAA-GFDL/fre-cli) | 4 | Python | Apache-2.0 | 140 | 9 | 5 | sim | Python-based command line interface for FRE (FMS Runtime Environment) to compile |
| NOAA-GFDL | [pySHiELD](https://github.com/NOAA-GFDL/pySHiELD) | 4 | Python | Apache-2.0 | 9 | 3 | 0 | nao | Python versions of the SHiELD physics |
| NOAA-GFDL | [fre-postprocess-workflow](https://github.com/NOAA-GFDL/fre-postprocess-workflow) | 3 | Python | Apache-2.0 | 36 | 2 | 10 | nao | Code to generate, describe, validate, and configure scientific workflows within  |
| NOAA-ORR-ERD | [PyGnome](https://github.com/NOAA-ORR-ERD/PyGnome) | 77 | C | licenca propria (ver LICENSE) | 12 | 0 | 0 | nao | The General NOAA Operational Modeling Environment |
| NOAA-ORR-ERD | [gridded](https://github.com/NOAA-ORR-ERD/gridded) | 71 | Python | CC0-1.0 | 70 | 0 | 0 | nao | A single API for accessing / working with gridded model results on multiple grid |
| NOAA-OWP | [inundation-mapping](https://github.com/NOAA-OWP/inundation-mapping) | 131 | Python | Apache-2.0 | 308 | 9 | 1 | sim | Flood inundation mapping and evaluation software configured to work with U.S. Na |
| NOAA-OWP | [ngen](https://github.com/NOAA-OWP/ngen) | 97 | C++ | Apache-2.0 | 203 | 9 | 3 | sim | Next Generation Water Modeling Engine and Framework Prototype |
| NOAA-OWP | [hydrotools](https://github.com/NOAA-OWP/hydrotools) | 70 | Python | Apache-2.0 | 17 | 1 | 0 | sim | Suite of tools for retrieving hydrological data and evaluating model output. |
| NOAA-OWP | [t-route](https://github.com/NOAA-OWP/t-route) | 56 | Python | Apache-2.0 | 173 | 8 | 1 | nao | Tree based hydrologic and hydraulic routing |
| NOAA-OWP | [hydrofabric_v2.2](https://github.com/NOAA-OWP/hydrofabric_v2.2) | 40 | QML | licenca propria (ver LICENSE) | 74 | 0 | 0 | nao | DEPRECATED: Former hydrofabric meta-package |
| NOAA-OWP | [topmodel](https://github.com/NOAA-OWP/topmodel) | 33 | C | Apache-2.0 | 11 | 0 | 0 | sim | Extending TOPMODEL, a rainfall-runoff model, to BMI (Basic Model Interface).  |
| NOAA-OWP | [nwm-post-processing](https://github.com/NOAA-OWP/nwm-post-processing) | 0 | Python | Apache-2.0 | 21 | 5 | 5 | sim | The post processing application suite for National Water Model output |
| NOAA-OWP | [nwm-coastal](https://github.com/NOAA-OWP/nwm-coastal) | 0 | Python | licenca propria (ver LICENSE) | 3 | 2 | 2 | sim | A collection of coastal tools designed to implement skill assessments for coasta |
| NOAA-PSL | [COARE-algorithm](https://github.com/NOAA-PSL/COARE-algorithm) | 54 | Fortran | MIT | 1 | 0 | 0 | nao | Repository of the COARE bulk air-sea flux algorithm in python, matlab, and fortr |
| NOAA-PSL | [graph-ufs](https://github.com/NOAA-PSL/graph-ufs) | 1 | Jupyter Notebook | Apache-2.0 | 13 | 0 | 1 | nao | Repository for training and evaluating GraphCast with UFS data |
| noaa-oar-arl | [monet](https://github.com/noaa-oar-arl/monet) | 48 | Python | licenca propria (ver LICENSE) | 6 | 0 | 0 | nao | The Model and ObservatioN Evaluation Toolkit (MONET) |
| noaa-oar-arl | [monetio](https://github.com/noaa-oar-arl/monetio) | 30 | Python | MIT | 35 | 1 | 0 | nao | The Model and ObservatioN Evaluation Tool I/O package |
| noaa-oar-arl | [canopy-app](https://github.com/noaa-oar-arl/canopy-app) | 8 | Fortran | MIT | 25 | 0 | 14 | nao | Stand-alone/column canopy codes and parameterizations |
| noaa-ocs-modeling | [OCSMesh](https://github.com/noaa-ocs-modeling/OCSMesh) | 32 | Python | CC0-1.0 | 49 | 0 | 0 | nao | OCSMesh is a mesh preparation tool for coastal ocean modeling applications. |
| ufs-community | [ufs-weather-model](https://github.com/ufs-community/ufs-weather-model) | 195 | Fortran | licenca propria (ver LICENSE) | 116 | 0 | 0 | nao | UFS Weather Model |
| ufs-community | [ufs-srweather-app](https://github.com/ufs-community/ufs-srweather-app) | 71 | Python | licenca propria (ver LICENSE) | 61 | 0 | 0 | nao | UFS Short-Range Weather Application |
| ufs-community | [UFS_UTILS](https://github.com/ufs-community/UFS_UTILS) | 31 | Fortran | licenca propria (ver LICENSE) | 40 | 0 | 0 | nao | Utilities for the NCEP models. |
| ufs-community | [CATChem](https://github.com/ufs-community/CATChem) | 3 | Fortran | Apache-2.0 | 56 | 2 | 1 | nao |  |
| ufs-community | [ufs-da-workflow](https://github.com/ufs-community/ufs-da-workflow) | 0 | Jinja | CC0-1.0 | 25 | 0 | 1 | nao | UFS DA (Data Assimilation) Workflow |

### Labs/agencias (351 de 1139 candidatos)

| Org | Repo | ★ | Lang | Licenca | Issues | GFI | HW | CONTRIB. | Descricao |
|---|---|---|---|---|---|---|---|---|---|
| ACCESS-NRI | [access-nri-intake-catalog](https://github.com/ACCESS-NRI/access-nri-intake-catalog) | 14 | Python | Apache-2.0 | 59 | 1 | 1 | nao | Tools and configuration info used to manage ACCESS-NRI's intake catalogue. Admin |
| ACCESS-NRI | [interactive-data-catalogue](https://github.com/ACCESS-NRI/interactive-data-catalogue) | 3 | TypeScript | Apache-2.0 | 40 | 1 | 0 | nao | ACCESS-NRI Interactive Catalog Viewer, built with Vue. Admins: @charles-turner-1 |
| ACCESS-NRI | [meorg_client](https://github.com/ACCESS-NRI/meorg_client) | 1 | Python | Apache-2.0 | 9 | 1 | 0 | nao | API Client for ModelEvaluation.org. |
| ACCESS-NRI | [access-experiment-generator](https://github.com/ACCESS-NRI/access-experiment-generator) | 1 | Python | Apache-2.0 | 16 | 1 | 1 | nao | Automating the setup and management of ACCESS models. Admins: @minghangli-uni, @ |
| ACCESS-NRI | [containerised-environments-infra](https://github.com/ACCESS-NRI/containerised-environments-infra) | 0 | Shell | Apache-2.0 | 16 | 0 | 1 | nao | Repository to manage the infrastructure for the deployment of containerised Pyth |
| E3SM-Project | [E3SM](https://github.com/E3SM-Project/E3SM) | 444 | Fortran | licenca propria (ver LICENSE) | 666 | 0 | 29 | sim | Energy Exascale Earth System Model source code.  NOTE:  use "maint" branches for |
| E3SM-Project | [e3sm_diags](https://github.com/E3SM-Project/e3sm_diags) | 50 | Jupyter Notebook | BSD-3-Clause | 76 | 0 | 0 | nao | E3SM Diagnostics package  |
| E3SM-Project | [EKAT](https://github.com/E3SM-Project/EKAT) | 22 | C++ | licenca propria (ver LICENSE) | 19 | 0 | 1 | sim | Tools and libraries for writing Kokkos-enabled HPC C++ in E3SM ecosystem |
| E3SM-Project | [containers](https://github.com/E3SM-Project/containers) | 4 | Python | BSD-3-Clause | 17 | 0 | 1 | sim | Container recipes for the E3SM project |
| ESA-PhiLab | [OpenSarToolkit](https://github.com/ESA-PhiLab/OpenSarToolkit) | 248 | Python | MIT | 16 | 0 | 0 | sim | High-level functionality for the inventory, download and pre-processing of Senti |
| ESA-PhiLab | [Major-TOM](https://github.com/ESA-PhiLab/Major-TOM) | 234 | Jupyter Notebook | Apache-2.0 | 7 | 0 | 0 | nao | Expandable Datasets for Earth Observation |
| ESA-PhiLab | [iris](https://github.com/ESA-PhiLab/iris) | 170 | JavaScript | GPL-3.0 | 17 | 0 | 0 | nao | Semi-automatic tool for manual segmentation of multi-spectral and geo-spatial im |
| ESA-PhiLab | [phidown](https://github.com/ESA-PhiLab/phidown) | 105 | Jupyter Notebook | Apache-2.0 | 1 | 0 | 0 | nao | Manage, search, and download Earth Observation data with Φ-down from Copernicus  |
| ESA-PhiLab | [PyRawS](https://github.com/ESA-PhiLab/PyRawS) | 93 | Jupyter Notebook | Apache-2.0 | 5 | 0 | 0 | nao | Python for Raw Sentinel-2 data (PyRawS) is an open-source software providing uti |
| ESA-PhiLab | [sarpyx](https://github.com/ESA-PhiLab/sarpyx) | 67 | Jupyter Notebook | Apache-2.0 | 4 | 0 | 0 | sim | sarpyx is a specialized Python package for advanced Synthetic Aperture Radar (SA |
| NCAR | [wrf-python](https://github.com/NCAR/wrf-python) | 501 | Python | Apache-2.0 | 72 | 0 | 0 | sim | A collection of diagnostic and interpolation routines for use with output from t |
| NCAR | [ncl](https://github.com/NCAR/ncl) | 277 | C | licenca propria (ver LICENSE) | 140 | 0 | 0 | sim | The NCAR Command Language (NCL) is a scripting language for the analysis and vis |
| NCAR | [DART](https://github.com/NCAR/DART) | 266 | Fortran | Apache-2.0 | 176 | 1 | 1 | nao | Data Assimilation Research Testbed |
| NCAR | [wrf_hydro_nwm_public](https://github.com/NCAR/wrf_hydro_nwm_public) | 248 | Fortran | licenca propria (ver LICENSE) | 141 | 0 | 0 | sim | WRF-Hydro model code |
| NCAR | [VAPOR](https://github.com/NCAR/VAPOR) | 210 | C++ | BSD-3-Clause | 348 | 0 | 0 | nao | VAPOR is the Visualization and Analysis Platform for Ocean, Atmosphere, and Sola |
| NCAR | [ParallelIO](https://github.com/NCAR/ParallelIO) | 161 | C | Apache-2.0 | 90 | 0 | 0 | nao | A high-level Parallel I/O Library for structured grid applications |
| NCAR | [geocat-comp](https://github.com/NCAR/geocat-comp) | 147 | Python | Apache-2.0 | 39 | 0 | 0 | sim | GeoCAT-comp provides implementations of computational functions for operating on |
| NCAR | [FastEddy-model](https://github.com/NCAR/FastEddy-model) | 130 | C | Apache-2.0 | 5 | 0 | 0 | nao | An NSF NCAR developed, parallelized and GPU-resident, large-eddy simulation code |
| NCAR | [lrose-core](https://github.com/NCAR/lrose-core) | 117 | C++ | licenca propria (ver LICENSE) | 25 | 0 | 0 | nao | Core C/C++ code for LROSE.  |
| NCAR | [noahmp](https://github.com/NCAR/noahmp) | 110 | Fortran | licenca propria (ver LICENSE) | 13 | 0 | 0 | nao | Noah-MP Community Repository |
| NCAR | [miles-credit](https://github.com/NCAR/miles-credit) | 105 | Python | Apache-2.0 | 37 | 0 | 1 | nao | NCAR MILES Community Research Earth Digital Intelligence Twin (CREDIT): research |
| NCAR | [ccpp-physics](https://github.com/NCAR/ccpp-physics) | 79 | Fortran | licenca propria (ver LICENSE) | 47 | 0 | 0 | nao | Collection of physics parameterizations compliant with the Common Community Phys |
| NCAR | [lrose-titan](https://github.com/NCAR/lrose-titan) | 75 | Shell | BSD-2-Clause | 4 | 0 | 0 | nao | TITAN system within LROSE |
| NCAR | [geocat-examples](https://github.com/NCAR/geocat-examples) | 74 | Python | Apache-2.0 | 17 | 0 | 0 | sim | GeoCAT-examples provides a gallery of visualization examples demonstrating how t |
| NCAR | [geocat-viz](https://github.com/NCAR/geocat-viz) | 66 | Python | licenca propria (ver LICENSE) | 20 | 0 | 0 | sim | GeoCAT-viz contains tools to help plot geoscience data, including convenience an |
| NCAR | [hrldas](https://github.com/NCAR/hrldas) | 66 | Jupyter Notebook | licenca propria (ver LICENSE) | 9 | 0 | 0 | nao | HRLDAS (High Resolution Land Data Assimilation System) |
| NCAR | [CM1](https://github.com/NCAR/CM1) | 58 | Fortran | MIT | 5 | 0 | 0 | nao | Cloud Model 1 (CM1), a numerical model for idealized studies of the atmosphere w |
| NCAR | [ADF](https://github.com/NCAR/ADF) | 51 | Python | CC-BY-4.0 | 73 | 3 | 3 | nao | A unified collection of python scripts used to generate standard plots from CAM  |
| NCAR | [gdex-api-client](https://github.com/NCAR/gdex-api-client) | 45 | Python | MIT | 7 | 0 | 0 | nao | GDEX apps clients.  Subdirectories will be organized by language, e.g. python, p |
| NCAR | [wrf_hydro_arcgis_preprocessor](https://github.com/NCAR/wrf_hydro_arcgis_preprocessor) | 43 | Python | licenca propria (ver LICENSE) | 5 | 0 | 0 | nao |  |
| NCAR | [music-box](https://github.com/NCAR/music-box) | 35 | Jupyter Notebook | Apache-2.0 | 17 | 0 | 0 | nao | A box/column model using MICM chemistry |
| NCAR | [CUPiD](https://github.com/NCAR/CUPiD) | 34 | Jupyter Notebook | Apache-2.0 | 107 | 1 | 0 | nao | CUPiD is a “one stop shop” that enables and integrates timeseries file generatio |
| NCAR | [MELODIES-MONET](https://github.com/NCAR/MELODIES-MONET) | 33 | Python | Apache-2.0 | 110 | 0 | 1 | nao | MELODIES MONET - diagnostic tool for evaluating models against a variety of obse |
| NCAR | [wrf_hydro_gis_preprocessor](https://github.com/NCAR/wrf_hydro_gis_preprocessor) | 31 | Python | MIT | 9 | 0 | 0 | nao |  |
| NCAR | [pop-tools](https://github.com/NCAR/pop-tools) | 28 | Python | Apache-2.0 | 35 | 0 | 1 | sim | Tools to support analysis of POP2-CESM model solutions |
| NCAR | [HPC-Docs](https://github.com/NCAR/HPC-Docs) | 13 | HTML | CC-BY-SA-4.0 | 19 | 0 | 3 | nao | NCAR HPC Docs Repository |
| NCAR | [aircraft_projects](https://github.com/NCAR/aircraft_projects) | 4 | Shell | Apache-2.0 | 12 | 0 | 1 | nao | EOL / RAF Field Project Aircraft data acquisition and processing configurations. |
| NatLabRockies | [api-umbrella](https://github.com/NatLabRockies/api-umbrella) | 2201 | Ruby | MIT | 256 | 0 | 0 | nao | Open source API management platform |
| NatLabRockies | [EnergyPlus](https://github.com/NatLabRockies/EnergyPlus) | 1587 | C++ | licenca propria (ver LICENSE) | 882 | 0 | 0 | nao | EnergyPlus™ is a whole building energy simulation program that engineers, archit |
| NatLabRockies | [OpenStudio](https://github.com/NatLabRockies/OpenStudio) | 649 | C++ | licenca propria (ver LICENSE) | 254 | 0 | 0 | sim | OpenStudio is a cross-platform collection of software tools to support whole bui |
| NatLabRockies | [SAM](https://github.com/NatLabRockies/SAM) | 490 | C++ | BSD-3-Clause | 98 | 0 | 0 | sim | System Advisor Model (SAM) |
| NatLabRockies | [floris](https://github.com/NatLabRockies/floris) | 302 | Python | BSD-3-Clause | 48 | 0 | 0 | sim | A controls-oriented engineering wake model. |
| NatLabRockies | [OpenOA](https://github.com/NatLabRockies/OpenOA) | 254 | Jupyter Notebook | BSD-3-Clause | 12 | 0 | 0 | nao | This library provides a framework for assessing wind plant performance using ope |
| NatLabRockies | [rdtools](https://github.com/NatLabRockies/rdtools) | 189 | Python | MIT | 37 | 0 | 0 | sim | PV Analysis Tools in Python |
| NatLabRockies | [ROSCO](https://github.com/NatLabRockies/ROSCO) | 178 | Jupyter Notebook | Apache-2.0 | 11 | 0 | 1 | nao | A Reference Open Source Controller for Wind Turbines |
| NatLabRockies | [HPC](https://github.com/NatLabRockies/HPC) | 157 | Jupyter Notebook | licenca propria (ver LICENSE) | 65 | 0 | 0 | sim | A collection of various resources, examples, and executables for the general NLR |
| NatLabRockies | [pysam](https://github.com/NatLabRockies/pysam) | 152 | C | BSD-3-Clause | 16 | 0 | 0 | sim | Python Wrapper for the System Advisor Model |
| NatLabRockies | [resstock](https://github.com/NatLabRockies/resstock) | 147 | Ruby | licenca propria (ver LICENSE) | 101 | 0 | 1 | nao | Highly granular modeling of residential building stocks at national, regional, a |
| NatLabRockies | [reV](https://github.com/NatLabRockies/reV) | 143 | Python | BSD-3-Clause | 10 | 2 | 0 | nao | reV is an open-source geospatial techno-economic tool that estimates energy tech |
| NatLabRockies | [sup3r](https://github.com/NatLabRockies/sup3r) | 140 | Python | BSD-3-Clause | 5 | 0 | 0 | nao | The Super-Resolution for Renewable Resource Data (sup3r) software uses generativ |
| NatLabRockies | [mappymatch](https://github.com/NatLabRockies/mappymatch) | 128 | Python | BSD-3-Clause | 17 | 0 | 0 | sim | Pure-python package for map matching |
| NatLabRockies | [REopt_API](https://github.com/NatLabRockies/REopt_API) | 127 | Python | licenca propria (ver LICENSE) | 53 | 0 | 0 | sim | The model for the REopt API, which is used as the back-end for the REopt Webtool |
| NatLabRockies | [hsds-examples](https://github.com/NatLabRockies/hsds-examples) | 126 | Jupyter Notebook | licenca propria (ver LICENSE) | 6 | 0 | 0 | nao | Examples of using the HSDS Service to Access NREL WIND Toolkit data |
| NatLabRockies | [bifacial_radiance](https://github.com/NatLabRockies/bifacial_radiance) | 109 | HTML | BSD-3-Clause | 85 | 0 | 0 | nao | Toolkit for working with RADIANCE for the ray-trace modeling of Bifacial Photovo |
| NatLabRockies | [openstudio-standards](https://github.com/NatLabRockies/openstudio-standards) | 105 | Ruby | licenca propria (ver LICENSE) | 73 | 0 | 0 | nao |  |
| NatLabRockies | [phygnn](https://github.com/NatLabRockies/phygnn) | 104 | Python | BSD-3-Clause | 3 | 0 | 0 | nao | physics-guided neural networks (phygnn) |
| NatLabRockies | [ParaEMT_public](https://github.com/NatLabRockies/ParaEMT_public) | 104 | Python | BSD-3-Clause | 5 | 0 | 0 | nao |  |
| NatLabRockies | [solar-data-tools](https://github.com/NatLabRockies/solar-data-tools) | 102 | Jupyter Notebook | licenca propria (ver LICENSE) | 7 | 1 | 0 | sim | Some data analysis tools for working with historical PV solar time-series data s |
| NatLabRockies | [EvoProtGrad](https://github.com/NatLabRockies/EvoProtGrad) | 97 | Jupyter Notebook | BSD-3-Clause | 5 | 0 | 0 | sim | Directed evolution of proteins in sequence space with gradients |
| NatLabRockies | [ssc](https://github.com/NatLabRockies/ssc) | 96 | C++ | BSD-3-Clause | 34 | 0 | 0 | sim | SAM Simulation Core (SSC) contains the underlying performance and financial mode |
| NatLabRockies | [turbine-models](https://github.com/NatLabRockies/turbine-models) | 94 | Python | BSD-3-Clause | 16 | 0 | 0 | sim | Documentation for the turbine models in this repository is available below. |
| NatLabRockies | [floorspace.js](https://github.com/NatLabRockies/floorspace.js) | 85 | JavaScript | licenca propria (ver LICENSE) | 59 | 0 | 0 | nao |  |
| NatLabRockies | [dgen](https://github.com/NatLabRockies/dgen) | 79 | Jupyter Notebook | BSD-3-Clause | 31 | 0 | 0 | nao | The Distributed Generation Market Demand (dGen) model simulates customer adoptio |
| NatLabRockies | [MATBOX_Microstructure_analysis_toolbox](https://github.com/NatLabRockies/MATBOX_Microstructure_analysis_toolbox) | 76 | MATLAB | licenca propria (ver LICENSE) | 18 | 0 | 3 | nao | MATBOX is an open-source MATLAB toolbox dedicated to microstructure analsyis of  |
| NatLabRockies | [OCHRE](https://github.com/NatLabRockies/OCHRE) | 75 | Python | BSD-3-Clause | 110 | 1 | 0 | nao | A Python-based building energy modeling (BEM) tool designed to model flexible lo |
| NatLabRockies | [BuildingMOTIF](https://github.com/NatLabRockies/BuildingMOTIF) | 72 | Jupyter Notebook | licenca propria (ver LICENSE) | 85 | 0 | 0 | nao | Building Metadata OnTology Interoperability Framework (BuildingMOTIF). For model |
| NatLabRockies | [OpenStudio-HPXML](https://github.com/NatLabRockies/OpenStudio-HPXML) | 70 | Ruby | licenca propria (ver LICENSE) | 151 | 0 | 1 | nao | Modeling of residential buildings in EnergyPlus using OpenStudio/HPXML. |
| NatLabRockies | [elm](https://github.com/NatLabRockies/elm) | 69 | Python | BSD-3-Clause | 2 | 0 | 0 | nao | ELM is a collection of utilities to apply Large Language Models (LLMs) to energy |
| NatLabRockies | [wex](https://github.com/NatLabRockies/wex) | 66 | C | BSD-3-Clause | 30 | 0 | 0 | nao | WEX, which is short for WxWidgets Extensions, is a cross-platform library of gra |
| NatLabRockies | [GEOPHIRES-X](https://github.com/NatLabRockies/GEOPHIRES-X) | 66 | Python | MIT | 83 | 5 | 0 | sim | GEOPHIRES is NREL's free and open-source geothermal techno-economic simulator.  |
| NatLabRockies | [alfalfa](https://github.com/NatLabRockies/alfalfa) | 62 | Python | licenca propria (ver LICENSE) | 75 | 0 | 0 | nao | Alfalfa is a web service that enables runtime interaction with building energy m |
| NatLabRockies | [nfp](https://github.com/NatLabRockies/nfp) | 62 | Python | licenca propria (ver LICENSE) | 3 | 0 | 0 | nao | Keras layers for end-to-end learning with rdkit and pymatgen |
| NatLabRockies | [fastsim](https://github.com/NatLabRockies/fastsim) | 62 | Rust | Apache-2.0 | 33 | 0 | 0 | nao | FASTSim (Future Automotive Systems Technology Simulator) is a vehicle simulation |
| NatLabRockies | [OpenStudio-server](https://github.com/NatLabRockies/OpenStudio-server) | 59 | Ruby | licenca propria (ver LICENSE) | 18 | 0 | 0 | sim | The OpenStudio Server is a docker or Helm deployable instance which allows for l |
| NatLabRockies | [PyPSSE](https://github.com/NatLabRockies/PyPSSE) | 56 | Python | BSD-3-Clause | 3 | 0 | 0 | nao |  |
| NatLabRockies | [MoorPy](https://github.com/NatLabRockies/MoorPy) | 56 | Python | BSD-3-Clause | 9 | 0 | 0 | nao |  |
| NatLabRockies | [ComStock](https://github.com/NatLabRockies/ComStock) | 54 | Ruby | licenca propria (ver LICENSE) | 74 | 0 | 0 | nao | National scale modeling of the U.S. commercial building stock supported by U.S.  |
| NatLabRockies | [PVDegradationTools](https://github.com/NatLabRockies/PVDegradationTools) | 49 | Jupyter Notebook | licenca propria (ver LICENSE) | 57 | 1 | 3 | sim | Set of tools to calculate degradation responses and degradation related paramete |
| NatLabRockies | [REopt.jl](https://github.com/NatLabRockies/REopt.jl) | 47 | Julia | Apache-2.0 | 116 | 0 | 0 | nao |  |
| NatLabRockies | [Panel-Segmentation](https://github.com/NatLabRockies/Panel-Segmentation) | 46 | Python | MIT | 6 | 0 | 0 | nao | This open-source package provides a framework for automatically detecting and ex |
| NatLabRockies | [BioReactorDesign](https://github.com/NatLabRockies/BioReactorDesign) | 46 | Liquid | BSD-3-Clause | 9 | 0 | 1 | nao | Bio Reactor Design (BiRD): a toolbox to simulate and analyze different designs o |
| NatLabRockies | [ATB-calc](https://github.com/NatLabRockies/ATB-calc) | 46 | Python | BSD-3-Clause | 4 | 0 | 0 | nao | Python files and Jupyter notebooks for processing the Annual Technology Baseline |
| NatLabRockies | [REopt-Analysis-Scripts](https://github.com/NatLabRockies/REopt-Analysis-Scripts) | 45 | Jupyter Notebook | BSD-3-Clause | 13 | 0 | 0 | nao |  |
| NatLabRockies | [SOWFA-6](https://github.com/NatLabRockies/SOWFA-6) | 45 | C++ | licenca propria (ver LICENSE) | 21 | 0 | 0 | nao |  |
| NatLabRockies | [PV_ICE](https://github.com/NatLabRockies/PV_ICE) | 45 | HTML | licenca propria (ver LICENSE) | 15 | 0 | 0 | nao | An open-source tool to quantify Solar Photovoltaics (PV) Energy and Mass Flows i |
| NatLabRockies | [flasc](https://github.com/NatLabRockies/flasc) | 45 | Jupyter Notebook | BSD-3-Clause | 17 | 0 | 0 | nao | A rich floris-driven suite for SCADA analysis |
| NatLabRockies | [openstudio-mcp](https://github.com/NatLabRockies/openstudio-mcp) | 45 | Python | licenca propria (ver LICENSE) | 33 | 0 | 0 | nao |  |
| NatLabRockies | [windtools](https://github.com/NatLabRockies/windtools) | 44 | Python | Apache-2.0 | 5 | 0 | 0 | nao | Python tools for wind simulation setup, data processing, and analysis |
| NatLabRockies | [gdx-pandas](https://github.com/NatLabRockies/gdx-pandas) | 43 | Python | BSD-3-Clause | 3 | 0 | 0 | nao | Python interface to read and write GAMS GDX files using pandas.DataFrames as the |
| NatLabRockies | [PyDSS](https://github.com/NatLabRockies/PyDSS) | 42 | Python | licenca propria (ver LICENSE) | 17 | 0 | 0 | nao |  |
| NatLabRockies | [OpenStudio-PAT](https://github.com/NatLabRockies/OpenStudio-PAT) | 40 | Ruby | licenca propria (ver LICENSE) | 104 | 0 | 0 | sim | The Parametric Analysis Tool (PAT) is part of the OpenStudio collection of softw |
| NatLabRockies | [HOPP](https://github.com/NatLabRockies/HOPP) | 40 | Python | BSD-3-Clause | 68 | 0 | 1 | sim |  |
| NatLabRockies | [plexosdb](https://github.com/NatLabRockies/plexosdb) | 39 | Python | BSD-3-Clause | 17 | 0 | 0 | nao | Database Manager for use with PLEXOS XML files |
| NatLabRockies | [bifacialvf](https://github.com/NatLabRockies/bifacialvf) | 35 | Python | licenca propria (ver LICENSE) | 12 | 0 | 0 | nao | Bifacial PV View Factor model for system performance calculation |
| NatLabRockies | [lk](https://github.com/NatLabRockies/lk) | 34 | C | BSD-3-Clause | 1 | 0 | 0 | sim | LK (Language Kit) is a simple but powerful scripting language that is designed t |
| NatLabRockies | [electrolyzer](https://github.com/NatLabRockies/electrolyzer) | 34 | Python | licenca propria (ver LICENSE) | 31 | 0 | 0 | sim |  |
| NatLabRockies | [rex](https://github.com/NatLabRockies/rex) | 33 | Python | BSD-3-Clause | 5 | 0 | 1 | nao | REsource eXtraction Tool (rex) |
| NatLabRockies | [reVeal](https://github.com/NatLabRockies/reVeal) | 32 | Python | BSD-3-Clause | 12 | 0 | 0 | nao | The reV Extension for Analyzing Large Loads (reVeal) is an open-source geospatia |
| NatLabRockies | [altrios](https://github.com/NatLabRockies/altrios) | 30 | Rust | licenca propria (ver LICENSE) | 24 | 0 | 0 | nao |  |
| NatLabRockies | [routee-compass](https://github.com/NatLabRockies/routee-compass) | 29 | Rust | BSD-3-Clause | 93 | 4 | 0 | sim | An energy-aware vehicle routing engine |
| NatLabRockies | [buildstockbatch](https://github.com/NatLabRockies/buildstockbatch) | 26 | Python | licenca propria (ver LICENSE) | 67 | 3 | 0 | nao |  |
| NatLabRockies | [OpenStudio-workflow-gem](https://github.com/NatLabRockies/OpenStudio-workflow-gem) | 16 | Ruby | licenca propria (ver LICENSE) | 30 | 0 | 1 | nao |  |
| NatLabRockies | [buildstock-query](https://github.com/NatLabRockies/buildstock-query) | 16 | Jupyter Notebook | BSD-3-Clause | 24 | 0 | 1 | nao | BuildStockQuery is a python library for querying datasets generated by ResStock™ |
| NatLabRockies | [infrasys](https://github.com/NatLabRockies/infrasys) | 13 | Python | BSD-3-Clause | 30 | 0 | 1 | nao | Data store for components and time series in support of Python-based modeling pa |
| NatLabRockies | [OpenStudio-HPXML-Calibration](https://github.com/NatLabRockies/OpenStudio-HPXML-Calibration) | 9 | Ruby | licenca propria (ver LICENSE) | 27 | 0 | 1 | nao | A package to automatically calibrate an HPXML model to utility bills |
| NatLabRockies | [Marmot](https://github.com/NatLabRockies/Marmot) | 8 | Python | licenca propria (ver LICENSE) | 18 | 2 | 0 | nao | Marmot is a data formatting and visualization tool for production cost and capac |
| NatLabRockies | [arco](https://github.com/NatLabRockies/arco) | 6 | Rust | BSD-3-Clause | 48 | 1 | 0 | sim |  |
| NatLabRockies | [chronify](https://github.com/NatLabRockies/chronify) | 5 | Python | BSD-3-Clause | 28 | 0 | 1 | nao |  |
| NatLabRockies | [LabProcessTracker](https://github.com/NatLabRockies/LabProcessTracker) | 3 | Python | licenca propria (ver LICENSE) | 5 | 0 | 1 | nao | QR/barcode scanning infrastructure for tracking samples and processes |
| NatLabRockies | [bambam](https://github.com/NatLabRockies/bambam) | 2 | Rust | BSD-3-Clause | 19 | 1 | 0 | sim | The Behavior and Advanced Mobility Big Access Model |
| NatLabRockies | [SUNI](https://github.com/NatLabRockies/SUNI) | 1 | C++ | BSD-3-Clause | 15 | 1 | 0 | nao | Solar Uncertainty Integrator (SUNI) estimates uncertainty for high-resolution, s |
| ORNL | [HeCBench](https://github.com/ORNL/HeCBench) | 308 | C++ | BSD-3-Clause | 8 | 0 | 0 | nao |  |
| ORNL | [HydraGNN](https://github.com/ORNL/HydraGNN) | 127 | Python | BSD-3-Clause | 21 | 0 | 0 | sim | Distributed PyTorch implementation of multi-headed graph convolutional neural ne |
| ORNL | [ReSolve](https://github.com/ORNL/ReSolve) | 85 | C++ | licenca propria (ver LICENSE) | 32 | 1 | 0 | sim | Library of GPU-resident linear solvers |
| ORNL | [TASMANIAN](https://github.com/ORNL/TASMANIAN) | 78 | C++ | licenca propria (ver LICENSE) | 6 | 0 | 0 | nao | The Toolkit for Adaptive Stochastic Modeling and Non-Intrusive ApproximatioN |
| ORNL | [AdditiveFOAM](https://github.com/ORNL/AdditiveFOAM) | 78 | C++ | licenca propria (ver LICENSE) | 5 | 0 | 0 | sim | A continuum physics code for additive manufacturing built on OpenFOAM |
| ORNL | [CabanaPD](https://github.com/ORNL/CabanaPD) | 39 | C++ | BSD-3-Clause | 50 | 0 | 0 | sim | Peridynamics with the Cabana library |
| ORNL | [flowcept](https://github.com/ORNL/flowcept) | 38 | Python | MIT | 24 | 0 | 0 | sim | Runtime provenance for AI and scientific workflows—capture, enrich, and query wo |
| ORNL | [Equilipy](https://github.com/ORNL/Equilipy) | 34 | Python | BSD-3-Clause | 1 | 0 | 0 | sim | Open-source python package for multicomponent multiphase equilibrium CALPHAD cal |
| ORNL | [GridKit](https://github.com/ORNL/GridKit) | 27 | C++ | licenca propria (ver LICENSE) | 83 | 3 | 4 | sim | Modeling framework for power systems simulations and analysis. |
| Unidata | [MetPy](https://github.com/Unidata/MetPy) | 1447 | Python | BSD-3-Clause | 393 | 11 | 0 | sim | MetPy is a collection of tools in Python for reading, visualizing and performing |
| Unidata | [netcdf4-python](https://github.com/Unidata/netcdf4-python) | 841 | Python | MIT | 154 | 0 | 0 | nao | netcdf4-python: python/numpy interface to the netCDF C library |
| Unidata | [netcdf-c](https://github.com/Unidata/netcdf-c) | 607 | C | BSD-3-Clause | 291 | 0 | 0 | sim | Official GitHub repository for netCDF-C libraries and utilities. |
| Unidata | [netcdf-fortran](https://github.com/Unidata/netcdf-fortran) | 271 | Fortran | licenca propria (ver LICENSE) | 124 | 0 | 0 | sim | Official GitHub repository for netCDF-Fortran libraries, which depend on the net |
| Unidata | [siphon](https://github.com/Unidata/siphon) | 245 | Python | BSD-3-Clause | 84 | 2 | 0 | sim | Siphon - A collection of Python utilities for retrieving atmospheric and oceanic |
| Unidata | [awips2](https://github.com/Unidata/awips2) | 227 | Java | MIT | 86 | 0 | 0 | nao | Weather forecasting display and analysis package developed by NWS/Raytheon, rele |
| Unidata | [netcdf-java](https://github.com/Unidata/netcdf-java) | 200 | Java | BSD-3-Clause | 73 | 0 | 2 | sim | The Unidata netcdf-java library |
| Unidata | [netcdf-cxx4](https://github.com/Unidata/netcdf-cxx4) | 152 | C++ | licenca propria (ver LICENSE) | 60 | 0 | 0 | nao | Official GitHub repository for netCDF-C++ libraries and utilities. |
| Unidata | [cftime](https://github.com/Unidata/cftime) | 96 | Python | MIT | 19 | 0 | 1 | nao | Time-handling functionality from netcdf4-python. |
| Unidata | [IDV](https://github.com/Unidata/IDV) | 92 | Java | licenca propria (ver LICENSE) | 17 | 0 | 0 | nao | The Integrated Data Viewer (IDV) from Unidata is a framework for analyzing and d |
| Unidata | [gempak](https://github.com/Unidata/gempak) | 84 | C | BSD-3-Clause | 50 | 0 | 0 | nao | Analysis and product generation for meteorological data.  |
| Unidata | [tds](https://github.com/Unidata/tds) | 83 | Java | BSD-3-Clause | 28 | 1 | 1 | sim | THREDDS Data Server |
| Unidata | [UDUNITS-2](https://github.com/Unidata/UDUNITS-2) | 71 | C | licenca propria (ver LICENSE) | 61 | 0 | 0 | nao | API and utility for arithmetic manipulation of units of physical quantities |
| Unidata | [tomcat-docker](https://github.com/Unidata/tomcat-docker) | 67 | Shell | BSD-3-Clause | 1 | 0 | 0 | nao | Security-hardened Tomcat container for thredds-docker. |
| Unidata | [LDM](https://github.com/Unidata/LDM) | 50 | C | licenca propria (ver LICENSE) | 49 | 0 | 0 | sim | The Unidata Local Data Manager (LDM) system includes network client and server p |
| Unidata | [python-awips](https://github.com/Unidata/python-awips) | 48 | Python | licenca propria (ver LICENSE) | 6 | 0 | 0 | nao | A framework for querying AWIPS meteorological datasets from an EDEX Data Server. |
| Unidata | [thredds-docker](https://github.com/Unidata/thredds-docker) | 40 | Dockerfile | BSD-3-Clause | 1 | 0 | 0 | nao | Dockerized THREDDS |
| ecmwf | [cfgrib](https://github.com/ecmwf/cfgrib) | 464 | Python | Apache-2.0 | 113 | 0 | 0 | sim | A Python interface to map GRIB files to the NetCDF Common Data Model following t |
| ecmwf | [ecmwf-opendata](https://github.com/ecmwf/ecmwf-opendata) | 339 | Python | Apache-2.0 | 38 | 0 | 0 | nao | A package to download ECMWF open data |
| ecmwf | [cdsapi](https://github.com/ecmwf/cdsapi) | 322 | Python | Apache-2.0 | 43 | 0 | 0 | sim | Python API to access the Copernicus Climate Data Store (CDS)  |
| ecmwf | [earthkit](https://github.com/ecmwf/earthkit) | 315 | Python | Apache-2.0 | 10 | 0 | 0 | nao | Python tools to work with weather and climate data |
| ecmwf | [notebook-examples](https://github.com/ecmwf/notebook-examples) | 274 | Jupyter Notebook | Apache-2.0 | 1 | 0 | 0 | nao | Example notebooks showing how to work with ECMWF services and data |
| ecmwf | [eccodes](https://github.com/ecmwf/eccodes) | 272 | C++ | Apache-2.0 | 26 | 0 | 0 | nao | ECMWF's GRIB and BUFR decoding/encoding library |
| ecmwf | [anemoi-core](https://github.com/ecmwf/anemoi-core) | 163 | Python | Apache-2.0 | 135 | 4 | 7 | nao | Core packages for Anemoi. |
| ecmwf | [atlas](https://github.com/ecmwf/atlas) | 153 | C++ | Apache-2.0 | 36 | 0 | 0 | nao | A library for numerical weather prediction and climate modelling |
| ecmwf | [metview-python](https://github.com/ecmwf/metview-python) | 147 | Python | Apache-2.0 | 23 | 1 | 0 | sim | Python interface to Metview meteorological workstation and batch system |
| ecmwf | [anemoi](https://github.com/ecmwf/anemoi) | 145 | Python | Apache-2.0 | 17 | 0 | 0 | nao |  |
| ecmwf | [WeatherGenerator](https://github.com/ecmwf/WeatherGenerator) | 145 | Python | Apache-2.0 | 626 | 10 | 0 | sim | The repository of the WeatherGenerator project |
| ecmwf | [eccodes-python](https://github.com/ecmwf/eccodes-python) | 137 | Python | Apache-2.0 | 13 | 0 | 0 | sim | Python interface to the ecCodes GRIB/BUFR decoder/encoder |
| ecmwf | [earthkit-data](https://github.com/ecmwf/earthkit-data) | 122 | Python | Apache-2.0 | 73 | 0 | 0 | nao | A format-agnostic Python interface for geospatial data |
| ecmwf | [thermofeel](https://github.com/ecmwf/thermofeel) | 99 | Python | Apache-2.0 | 1 | 0 | 0 | sim | thermofeel is a library to calculate human thermal comfort indexes |
| ecmwf | [ecmwf-api-client](https://github.com/ecmwf/ecmwf-api-client) | 96 | Python | Apache-2.0 | 3 | 0 | 0 | nao | Python API to access ECMWF archive |
| ecmwf | [anemoi-datasets](https://github.com/ecmwf/anemoi-datasets) | 88 | Python | Apache-2.0 | 55 | 4 | 1 | nao | Datasets for Machine Learning weather forecasting models |
| ecmwf | [ecflow](https://github.com/ecmwf/ecflow) | 65 | C++ | Apache-2.0 | 8 | 0 | 0 | nao | ECMWF's workflow manager  |
| ecmwf | [magics](https://github.com/ecmwf/magics) | 64 | Jupyter Notebook | Apache-2.0 | 13 | 0 | 0 | nao | Plotting package to visualise meteorological data in GRIB, NetCDF, BUFR and ODB  |
| ecmwf | [skinnywms](https://github.com/ecmwf/skinnywms) | 53 | Python | Apache-2.0 | 7 | 0 | 0 | nao | Lightweight WMS server for serving maps of netCDF and GRIB data |
| ecmwf | [magics-python](https://github.com/ecmwf/magics-python) | 52 | Python | Apache-2.0 | 12 | 0 | 0 | sim | Python interface to Magics meteorological plotting package |
| ecmwf | [polytope](https://github.com/ecmwf/polytope) | 49 | Python | Apache-2.0 | 29 | 0 | 0 | sim | A library for extracting polytope "features" from datacubes |
| ecmwf | [fckit](https://github.com/ecmwf/fckit) | 48 | Fortran | Apache-2.0 | 15 | 0 | 0 | nao | A Fortran toolkit for interoperating Fortran with C/C++ |
| ecmwf | [anemoi-inference](https://github.com/ecmwf/anemoi-inference) | 44 | Python | Apache-2.0 | 33 | 1 | 0 | nao | Inference of Machine Learning weather forecasting models |
| ecmwf | [fdb](https://github.com/ecmwf/fdb) | 43 | C++ | Apache-2.0 | 27 | 0 | 0 | nao | Fdb is a domain-specific object store for meteorological objects |
| ecmwf | [ecbuild](https://github.com/ecmwf/ecbuild) | 39 | CMake | Apache-2.0 | 19 | 0 | 0 | nao | A CMake-based build system, consisting of a collection of CMake macros and funct |
| ecmwf | [infero](https://github.com/ecmwf/infero) | 37 | C++ | Apache-2.0 | 4 | 0 | 0 | nao | A lower-level API for Machine Learning inference in operations |
| ecmwf | [earthkit-hydro](https://github.com/ecmwf/earthkit-hydro) | 34 | Python | Apache-2.0 | 18 | 0 | 0 | nao | A Python library for common hydrological functions |
| ecmwf | [eckit](https://github.com/ecmwf/eckit) | 30 | C++ | Apache-2.0 | 23 | 0 | 0 | nao | A C++ toolkit that supports development of tools and applications at ECMWF. |
| ecmwf | [anemoi-transform](https://github.com/ecmwf/anemoi-transform) | 9 | Python | Apache-2.0 | 24 | 2 | 5 | nao |  |
| ecmwf | [anemoi-plugins](https://github.com/ecmwf/anemoi-plugins) | 2 | Python | Apache-2.0 | 3 | 2 | 0 | nao | Plugins for anemoi |
| ecmwf-lab | [ai-models](https://github.com/ecmwf-lab/ai-models) | 596 | Python | Apache-2.0 | 7 | 0 | 0 | nao | Run AI-based weather forecasting models with ECMWF data |
| esa | [pagmo2](https://github.com/esa/pagmo2) | 937 | C++ | GPL-3.0 | 34 | 0 | 3 | nao | A C++ platform to perform parallel computations of optimisation tasks (global an |
| esa | [pygmo2](https://github.com/esa/pygmo2) | 542 | C++ | MPL-2.0 | 34 | 1 | 1 | nao | A Python platform to perform parallel computations of optimisation tasks (global |
| esa | [pykep](https://github.com/esa/pykep) | 426 | C++ | MPL-2.0 | 2 | 0 | 0 | nao | PyKEP is a scientific library providing basic tools for research in interplaneta |
| esa | [asn1scc](https://github.com/esa/asn1scc) | 341 | F# | licenca propria (ver LICENSE) | 20 | 0 | 0 | nao | ASN1SCC: An open source ASN.1 compiler for embedded systems |
| esa | [torchquad](https://github.com/esa/torchquad) | 229 | Python | GPL-3.0 | 15 | 2 | 4 | sim | Numerical integration in arbitrary dimensions on the GPU using PyTorch / TF / JA |
| esa | [dSGP4](https://github.com/esa/dSGP4) | 98 | Python | GPL-3.0 | 2 | 0 | 0 | nao | dSGP4: differentiable SGP4. Supports differentiability, ML integration & embaras |
| esa | [opengeode](https://github.com/esa/opengeode) | 77 | Python | LGPL-3.0 | 3 | 0 | 0 | nao | OpenGEODE - a free SDL State Machine editor for space applications...and more |
| esa | [polyhedral-gravity-model](https://github.com/esa/polyhedral-gravity-model) | 39 | Jupyter Notebook | GPL-3.0 | 2 | 0 | 0 | sim | Implementation of a polyhedral gravity model in C++17 with a Python Binding |
| idaholab | [moose](https://github.com/idaholab/moose) | 2374 | C++ | LGPL-2.1 | 2853 | 51 | 0 | sim | Multiphysics Object Oriented Simulation Environment |
| idaholab | [raven](https://github.com/idaholab/raven) | 268 | Python | Apache-2.0 | 181 | 0 | 0 | sim | RAVEN is a flexible and multi-purpose probabilistic risk analysis, validation an |
| idaholab | [STIG](https://github.com/idaholab/STIG) | 101 | TypeScript | BSD-3-Clause | 42 | 4 | 11 | nao | Structured Threat Intelligence Graph |
| idaholab | [DeepLynx](https://github.com/idaholab/DeepLynx) | 100 | C# | licenca propria (ver LICENSE) | 1 | 0 | 0 | sim | DeepLynx Nexus is version 2 of the DeepLynx data warehouse, and acts as the cent |
| idaholab | [virtual_test_bed](https://github.com/idaholab/virtual_test_bed) | 78 | SWIG | CC-BY-4.0 | 136 | 0 | 0 | nao | The National Reactor Innovation Center's (NRIC) Virtual Test Bed Repository |
| idaholab | [LIGGGHTS-INL](https://github.com/idaholab/LIGGGHTS-INL) | 68 | C++ | GPL-2.0 | 2 | 0 | 0 | nao | LIGGGHTS-INL is a capability-extended adaptation of the LIGGGHTS Open Source Dis |
| idaholab | [MontePy](https://github.com/idaholab/MontePy) | 66 | Python | MIT | 107 | 18 | 0 | sim | MontePy is the most user friendly Python library (API) to read, edit, and write  |
| idaholab | [mastodon](https://github.com/idaholab/mastodon) | 51 | Assembly | LGPL-2.1 | 42 | 0 | 0 | nao | A MOOSE app for structural dynamics, seismic analysis, and risk assessment.  |
| idaholab | [civet](https://github.com/idaholab/civet) | 38 | Python | Apache-2.0 | 28 | 0 | 0 | nao | Continuous Integration, Verification, Enhancement, and Testing |
| idaholab | [tmap8](https://github.com/idaholab/tmap8) | 38 | Python | LGPL-2.1 | 59 | 0 | 0 | nao | Tritium Migration Analysis Program, Version 8 |
| idaholab | [HYBRID](https://github.com/idaholab/HYBRID) | 37 | C | Apache-2.0 | 7 | 0 | 0 | nao | HYBRID is a modeling toolset to assess the integration and economic viability of |
| idaholab | [falcon](https://github.com/idaholab/falcon) | 35 | C++ | LGPL-2.1 | 7 | 0 | 0 | nao | Fracturing And Liquid CONservation  |
| idaholab | [EMRALD](https://github.com/idaholab/EMRALD) | 35 | TypeScript | MIT | 28 | 0 | 0 | nao | Event Modeling Risk Assessment using Linked Diagrams (EMRALD) is a software tool |
| idaholab | [HERON](https://github.com/idaholab/HERON) | 33 | Python | Apache-2.0 | 53 | 3 | 0 | sim | Holistic Energy Resource Optimization Network (HERON) is a modeling toolset and  |
| lanl | [pyxDamerauLevenshtein](https://github.com/lanl/pyxDamerauLevenshtein) | 258 | Python | BSD-3-Clause | 1 | 0 | 0 | nao | pyxDamerauLevenshtein implements the Damerau-Levenshtein (DL) edit distance algo |
| lanl | [scico](https://github.com/lanl/scico) | 173 | Python | BSD-3-Clause | 19 | 0 | 0 | nao | Scientific Computational Imaging COde |
| lanl | [LaGriT](https://github.com/lanl/LaGriT) | 134 | Fortran | licenca propria (ver LICENSE) | 131 | 2 | 1 | sim | Los Alamos Grid Toolbox (LaGriT) is a library of user callable tools that provid |
| lanl | [Architector](https://github.com/lanl/Architector) | 101 | Python | licenca propria (ver LICENSE) | 5 | 0 | 0 | nao | The architector python package - for 3D metal complex design. C22085 |
| lanl | [hippynn](https://github.com/lanl/hippynn) | 98 | Python | licenca propria (ver LICENSE) | 20 | 0 | 0 | nao | python library for atomistic machine learning |
| lanl | [dfnWorks](https://github.com/lanl/dfnWorks) | 97 | Python | licenca propria (ver LICENSE) | 15 | 0 | 0 | nao | dfnWorks is a parallelized computational suite to generate three-dimensional dis |
| lanl | [pyHarmonySearch](https://github.com/lanl/pyHarmonySearch) | 90 | Python | BSD-3-Clause | 1 | 0 | 0 | nao | pyHarmonySearch is a pure Python implementation of the harmony search (HS) globa |
| lanl | [PYSEQM](https://github.com/lanl/PYSEQM) | 87 | Python | licenca propria (ver LICENSE) | 3 | 0 | 0 | nao | an interface to semi-empirical quantum chemistry methods implemented with pytorc |
| lanl | [ursa](https://github.com/lanl/ursa) | 86 | Python | licenca propria (ver LICENSE) | 27 | 0 | 0 | sim | Universal Research and Scientific Agent |
| lanl | [Fierro](https://github.com/lanl/Fierro) | 77 | C++ | BSD-3-Clause | 22 | 3 | 1 | nao | Fierro is a C++ code designed to aid the research and development of numerical m |
| lanl | [FEHM](https://github.com/lanl/FEHM) | 70 | GLSL | licenca propria (ver LICENSE) | 40 | 0 | 0 | sim | Finite Element Heat and Mass Transfer Code |
| lanl | [Draco](https://github.com/lanl/Draco) | 59 | C++ | licenca propria (ver LICENSE) | 2 | 0 | 0 | nao | An object-oriented component library supporting radiation transport applications |
| lanl | [LATTE](https://github.com/lanl/LATTE) | 49 | Fortran | licenca propria (ver LICENSE) | 20 | 0 | 0 | nao | Developer repository for the LATTE code |
| lanl | [ALF](https://github.com/lanl/ALF) | 47 | Python | licenca propria (ver LICENSE) | 1 | 0 | 0 | nao | An open-source active learning framework for training machine-learned interatomi |
| lanl | [artemis](https://github.com/lanl/artemis) | 42 | C++ | licenca propria (ver LICENSE) | 26 | 1 | 0 | nao | Astrophysical multifluid radiation hydrodynamics code |
| lanl | [singularity-eos](https://github.com/lanl/singularity-eos) | 40 | C++ | BSD-3-Clause | 107 | 21 | 16 | nao | Performance portable equations of state and mixed cell closures |
| lanl | [nubhlight](https://github.com/lanl/nubhlight) | 39 | C | licenca propria (ver LICENSE) | 17 | 2 | 0 | nao | General Relativistic Neutrino Radiation Magnetohydrodynamics for Neutron Star Me |
| lanl | [CGMF](https://github.com/lanl/CGMF) | 31 | Jupyter Notebook | BSD-3-Clause | 3 | 0 | 0 | nao | CGMF nuclear fission fragment de-excitation statistical code |
| lanl | [BEE](https://github.com/lanl/BEE) | 22 | Python | licenca propria (ver LICENSE) | 175 | 1 | 0 | nao |  |
| lanl | [spiner](https://github.com/lanl/spiner) | 9 | C++ | BSD-3-Clause | 12 | 1 | 3 | nao | Performance portable routines for generic, tabulated, multi-dimensional data |
| lanl | [ports-of-call](https://github.com/lanl/ports-of-call) | 8 | C++ | BSD-3-Clause | 17 | 2 | 3 | nao | Performance Portability Utilities |
| lanl | [Transform-Your-World](https://github.com/lanl/Transform-Your-World) | 3 | Python | MIT | 2 | 1 | 0 | nao | Machine learning library for developing scientific transformer models, particula |
| lanl | [brag](https://github.com/lanl/brag) | 2 | Python | licenca propria (ver LICENSE) | 4 | 1 | 0 | sim | Basic RAG |
| llnl | [zfp](https://github.com/llnl/zfp) | 887 | C++ | BSD-3-Clause | 43 | 0 | 2 | sim | Compressed numerical arrays that support high-speed random access |
| llnl | [sundials](https://github.com/llnl/sundials) | 696 | C | BSD-3-Clause | 60 | 0 | 0 | sim | Official development repository for SUNDIALS - a SUite of Nonlinear and DIfferen |
| llnl | [rose](https://github.com/llnl/rose) | 693 | C | BSD-3-Clause | 36 | 0 | 0 | nao | ROSE is an open-source compiler framework engineered by LLNL supporting program  |
| llnl | [RAJA](https://github.com/llnl/RAJA) | 602 | C++ | BSD-3-Clause | 197 | 0 | 6 | sim | RAJA Performance Portability Layer (C++) |
| llnl | [OGhidra](https://github.com/llnl/OGhidra) | 453 | Python | licenca propria (ver LICENSE) | 24 | 0 | 0 | nao | OGhidra bridges Large Language Models (LLMs) via Ollama with the Ghidra reverse  |
| llnl | [Umpire](https://github.com/llnl/Umpire) | 422 | C++ | MIT | 37 | 0 | 0 | sim | An application-focused API for memory management on NUMA & GPU architectures |
| llnl | [Caliper](https://github.com/llnl/Caliper) | 421 | C++ | BSD-3-Clause | 38 | 0 | 0 | nao | Caliper is an instrumentation and performance profiling library |
| llnl | [blt](https://github.com/llnl/blt) | 297 | C++ | BSD-3-Clause | 142 | 0 | 0 | sim | A streamlined CMake build system foundation for developing HPC software |
| llnl | [HPC-Tutorials](https://github.com/llnl/HPC-Tutorials) | 258 | C | MIT | 4 | 0 | 0 | nao | Future home of hpc-tutorials.llnl.gov |
| llnl | [LEAP](https://github.com/llnl/LEAP) | 254 | Cuda | MIT | 48 | 0 | 0 | nao | comprehensive library of 3D transmission Computed Tomography (CT) algorithms wit |
| llnl | [SAMRAI](https://github.com/llnl/SAMRAI) | 250 | C++ | licenca propria (ver LICENSE) | 59 | 0 | 0 | sim | Structured Adaptive Mesh Refinement Application Infrastructure - a scalable C++  |
| llnl | [conduit](https://github.com/llnl/conduit) | 247 | C++ | licenca propria (ver LICENSE) | 223 | 0 | 0 | nao | Simplified Data Exchange for HPC Simulations |
| llnl | [smith](https://github.com/llnl/smith) | 245 | C++ | BSD-3-Clause | 178 | 0 | 1 | sim | Smith is a high order nonlinear thermomechanical simulation code |
| llnl | [libROM](https://github.com/llnl/libROM) | 238 | C++ | licenca propria (ver LICENSE) | 44 | 0 | 3 | nao | Data-driven model reduction library with an emphasis on large scale parallelism  |
| llnl | [hiop](https://github.com/llnl/hiop) | 231 | C++ | licenca propria (ver LICENSE) | 70 | 0 | 0 | nao | HPC solver for nonlinear optimization problems |
| llnl | [axom](https://github.com/llnl/axom) | 197 | C++ | BSD-3-Clause | 251 | 2 | 3 | sim | CS infrastructure components for HPC applications |
| llnl | [units](https://github.com/llnl/units) | 173 | C++ | BSD-3-Clause | 6 | 0 | 1 | sim | A run-time C++ library for working with units of measurement and conversions bet |
| llnl | [maestrowf](https://github.com/llnl/maestrowf) | 161 | Python | MIT | 91 | 9 | 0 | sim | A tool to easily orchestrate general computational workflows both locally and on |
| llnl | [merlin](https://github.com/llnl/merlin) | 151 | Python | licenca propria (ver LICENSE) | 42 | 0 | 0 | sim | Machine Learning for HPC Workflows |
| llnl | [RAJAPerf](https://github.com/llnl/RAJAPerf) | 135 | Jupyter Notebook | BSD-3-Clause | 79 | 0 | 0 | nao | RAJA Performance Suite |
| llnl | [fpzip](https://github.com/llnl/fpzip) | 126 | C++ | BSD-3-Clause | 2 | 0 | 0 | nao | Lossless compressor of multidimensional floating-point arrays |
| llnl | [umap](https://github.com/llnl/umap) | 115 | C++ | LGPL-2.1 | 5 | 0 | 0 | nao | User-space Page Management |
| llnl | [CHAI](https://github.com/llnl/CHAI) | 111 | C++ | BSD-3-Clause | 30 | 0 | 0 | sim | Copy-hiding array abstraction to automatically migrate data between memory space |
| llnl | [Spindle](https://github.com/llnl/Spindle) | 110 | C | licenca propria (ver LICENSE) | 45 | 0 | 0 | nao | Scalable dynamic library and python loading in HPC environments |
| llnl | [scr](https://github.com/llnl/scr) | 108 | C | licenca propria (ver LICENSE) | 110 | 0 | 0 | sim | SCR caches checkpoint data in storage on the compute nodes of a Linux cluster to |
| llnl | [camp](https://github.com/llnl/camp) | 107 | C++ | BSD-3-Clause | 37 | 0 | 1 | nao | Compiler agnostic metaprogramming library providing concepts, type operations an |
| llnl | [msr-safe](https://github.com/llnl/msr-safe) | 97 | C | GPL-2.0 | 25 | 0 | 0 | nao | Allows safer access to model specific registers (MSRs) |
| llnl | [shroud](https://github.com/llnl/shroud) | 97 | Fortran | BSD-3-Clause | 13 | 0 | 0 | sim | Shroud: generate Fortran and Python wrappers for C and C++ libraries |
| llnl | [ExaCA](https://github.com/llnl/ExaCA) | 96 | C++ | MIT | 11 | 0 | 0 | sim | Cellular automata code for alloy nucleation and solidification written with Kokk |
| llnl | [Aluminum](https://github.com/llnl/Aluminum) | 91 | C++ | licenca propria (ver LICENSE) | 3 | 0 | 0 | nao | High-performance, GPU-aware communication library |
| llnl | [GOTCHA](https://github.com/llnl/GOTCHA) | 88 | C | licenca propria (ver LICENSE) | 17 | 0 | 1 | nao | GOTCHA is a library for wrapping function calls in shared libraries |
| llnl | [Abmarl](https://github.com/llnl/Abmarl) | 85 | Python | licenca propria (ver LICENSE) | 67 | 2 | 0 | nao | Agent Based Modeling and Reinforcement Learning |
| llnl | [spheral](https://github.com/llnl/spheral) | 84 | C++ | BSD-3-Clause | 40 | 1 | 0 | nao |  |
| llnl | [variorum](https://github.com/llnl/variorum) | 83 | C++ | MIT | 81 | 0 | 0 | sim | Vendor-neutral library for exposing power and performance features across divers |
| llnl | [benchpark](https://github.com/llnl/benchpark) | 79 | Python | Apache-2.0 | 149 | 0 | 0 | nao | An open collaborative repository for reproducible specifications of HPC benchmar |
| llnl | [SSAPy](https://github.com/llnl/SSAPy) | 78 | Python | MIT | 1 | 0 | 0 | sim | A Python package allowing for fast and precise orbital modeling. |
| llnl | [lmt](https://github.com/llnl/lmt) | 77 | C | GPL-2.0 | 15 | 0 | 0 | nao | Lustre Monitoring Tools |
| llnl | [mttime](https://github.com/llnl/mttime) | 73 | Python | licenca propria (ver LICENSE) | 6 | 0 | 0 | nao | Time Domain Moment Tensor Inversion in Python |
| llnl | [open-ai-co-scientist](https://github.com/llnl/open-ai-co-scientist) | 72 | Python | MIT | 20 | 0 | 0 | sim | Open-source implementation of Google's AI co-scientist for genenerating research |
| llnl | [ExaConstit](https://github.com/llnl/ExaConstit) | 71 | C++ | BSD-3-Clause | 10 | 0 | 1 | nao | A crystal plasticity FEM code that runs on the GPU |
| llnl | [STAT](https://github.com/llnl/STAT) | 70 | C | licenca propria (ver LICENSE) | 10 | 0 | 0 | nao | STAT - the Stack Trace Analysis Tool |
| llnl | [pyranda](https://github.com/llnl/pyranda) | 69 | Fortran | licenca propria (ver LICENSE) | 2 | 0 | 0 | nao | A Python driven, Fortran powered Finite Difference solver for arbitrary hyperbol |
| llnl | [paraview_mcp](https://github.com/llnl/paraview_mcp) | 69 | Python | BSD-3-Clause | 1 | 0 | 0 | sim | ParaView-MCP integrates multimodal LLMs with ParaView via Model Context Protocol |
| llnl | [metall](https://github.com/llnl/metall) | 68 | C++ | licenca propria (ver LICENSE) | 6 | 0 | 0 | nao | Persistent memory allocator for data-centric analytics |
| llnl | [llnl.github.io](https://github.com/llnl/llnl.github.io) | 67 | JavaScript | MIT | 13 | 0 | 0 | nao | Public home for LLNL software catalog |
| llnl | [GridDyn](https://github.com/llnl/GridDyn) | 63 | C++ | BSD-3-Clause | 4 | 0 | 0 | sim | GridDyn is an open-source power transmission simulation software package |
| llnl | [quandary](https://github.com/llnl/quandary) | 60 | C++ | MIT | 11 | 0 | 0 | nao | Optimal control for open quantum systems |
| llnl | [scraper](https://github.com/llnl/scraper) | 59 | Python | MIT | 15 | 0 | 4 | nao | Python library for getting metadata from source code hosting tools |
| llnl | [UEDGE](https://github.com/llnl/UEDGE) | 56 | Fortran | LGPL-2.1 | 20 | 0 | 0 | nao | 2D fluid simulation of plasma and neutrals in magnetic fusion devices |
| llnl | [H5Z-ZFP](https://github.com/llnl/H5Z-ZFP) | 55 | C | licenca propria (ver LICENSE) | 27 | 0 | 0 | nao | A registered ZFP compression plugin for HDF5 |
| llnl | [proteus](https://github.com/llnl/proteus) | 54 | C++ | Apache-2.0 | 65 | 0 | 1 | sim | Programmable JIT Compilation and Optimization for C/C++ using LLVM |
| llnl | [mpibind](https://github.com/llnl/mpibind) | 53 | C | MIT | 5 | 0 | 0 | nao | Pragmatic, Productive, and Portable Affinity for HPC |
| llnl | [zero-rk](https://github.com/llnl/zero-rk) | 48 | C++ | BSD-3-Clause | 1 | 0 | 0 | nao | Zero-order Reaction Kinetics (Zero-RK) is a software package that simulates chem |
| llnl | [mgmol](https://github.com/llnl/mgmol) | 47 | C++ | licenca propria (ver LICENSE) | 20 | 0 | 0 | nao | MGmol is a scalable O(N) First-Principles Molecular Dynamics code that is capabl |
| llnl | [Silo](https://github.com/llnl/Silo) | 46 | C | licenca propria (ver LICENSE) | 95 | 0 | 0 | nao | Mesh and Field I/O Library and Scientific Database |
| llnl | [AMPE](https://github.com/llnl/AMPE) | 45 | C++ | licenca propria (ver LICENSE) | 9 | 0 | 0 | nao | Adaptive Mesh Phase-field Evolution |
| llnl | [FPChecker](https://github.com/llnl/FPChecker) | 44 | Python | Apache-2.0 | 15 | 0 | 0 | nao | A dynamic analysis tool to detect floating-point errors in HPC applications. |
| llnl | [Surfactant](https://github.com/llnl/Surfactant) | 43 | Python | MIT | 75 | 2 | 6 | sim | Modular framework for file information extraction and dependency analysis to gen |
| llnl | [Kripke](https://github.com/llnl/Kripke) | 42 | C++ | BSD-3-Clause | 9 | 0 | 0 | nao | Kripke is a simple, scalable, 3D Sn deterministic particle transport code |
| llnl | [ygm](https://github.com/llnl/ygm) | 40 | C++ | licenca propria (ver LICENSE) | 14 | 0 | 0 | sim |  |
| llnl | [Tribol](https://github.com/llnl/Tribol) | 40 | C++ | MIT | 52 | 0 | 0 | sim | Modular interface physics library featuring state-of-the-art contact physics met |
| llnl | [hatchet](https://github.com/llnl/hatchet) | 37 | JavaScript | MIT | 37 | 0 | 0 | nao | Graph-indexed Pandas DataFrames for analyzing hierarchical performance data |
| llnl | [GPLaSDI](https://github.com/llnl/GPLaSDI) | 37 | Python | MIT | 12 | 0 | 0 | nao | Gaussian process-based interpretable latent space dynamics identification throug |
| llnl | [fudge](https://github.com/llnl/fudge) | 36 | Python | licenca propria (ver LICENSE) | 10 | 0 | 0 | nao | For Updating Data and Generating Evaluations (FUDGE): LLNL code for managing nuc |
| llnl | [inq](https://github.com/llnl/inq) | 36 | C++ | MPL-2.0 | 2 | 0 | 0 | nao | This is a mirror. Please check our main website on gitlab. |
| llnl | [CARE](https://github.com/llnl/CARE) | 33 | C++ | BSD-3-Clause | 37 | 0 | 0 | nao | CHAI and RAJA provide an excellent base on which to build portable codes. CARE e |
| llnl | [coda-calibration-tool](https://github.com/llnl/coda-calibration-tool) | 31 | Java | Apache-2.0 | 1 | 0 | 0 | nao | Tool for calibrating seismic coda source models |
| llnl | [csld](https://github.com/llnl/csld) | 31 | Python | MIT | 5 | 0 | 0 | nao | Compressive sensing lattice dynamics |
| llnl | [uberenv](https://github.com/llnl/uberenv) | 30 | Shell | licenca propria (ver LICENSE) | 37 | 0 | 1 | nao | Automates using spack to build and deploy software |
| llnl | [gLaSDI](https://github.com/llnl/gLaSDI) | 30 | Python | MIT | 1 | 0 | 0 | nao |  |
| llnl | [hubcast](https://github.com/llnl/hubcast) | 28 | Python | Apache-2.0 | 13 | 1 | 0 | sim | An event driven synchronization application for bridging GitHub and GitLab |
| llnl | [PyDV](https://github.com/llnl/PyDV) | 15 | Python | licenca propria (ver LICENSE) | 30 | 1 | 0 | nao | PyDV: Python Data Visualizer |
| llnl | [INGRID](https://github.com/llnl/INGRID) | 11 | Python | MIT | 20 | 1 | 0 | nao | Interactive Grid Generator for Tokamak Boundary Region |
| pnnl | [neuromancer](https://github.com/pnnl/neuromancer) | 1378 | Python | licenca propria (ver LICENSE) | 17 | 0 | 0 | sim | Pytorch-based framework for solving parametric constrained optimization problems |
| pnnl | [HyperNetX](https://github.com/pnnl/HyperNetX) | 718 | Python | licenca propria (ver LICENSE) | 19 | 0 | 0 | sim | Python package for hypergraph analysis and visualization. |
| pnnl | [OpenCGRA](https://github.com/pnnl/OpenCGRA) | 183 | Verilog | licenca propria (ver LICENSE) | 4 | 0 | 0 | nao | OpenCGRA is an open-source framework for modeling, testing, and evaluating CGRAs |
| pnnl | [SHAD](https://github.com/pnnl/SHAD) | 137 | C++ | Apache-2.0 | 66 | 1 | 1 | nao | Scalable High-performance Algorithms and Data-structures |
| pnnl | [NWGraph](https://github.com/pnnl/NWGraph) | 101 | C++ | licenca propria (ver LICENSE) | 5 | 0 | 0 | nao | Complete Project Documentation |
| pnnl | [L2O-MINLP](https://github.com/pnnl/L2O-MINLP) | 72 | Jupyter Notebook | licenca propria (ver LICENSE) | 1 | 0 | 0 | nao | Learning-to-Optimize for Mixed-Integer Non-Linear Programming |
| pnnl | [lamellar-runtime](https://github.com/pnnl/lamellar-runtime) | 68 | Rust | licenca propria (ver LICENSE) | 26 | 0 | 0 | nao | Lamellar is an asynchronous tasking runtime for HPC systems developed in RUST |
| pnnl | [chemreasoner](https://github.com/pnnl/chemreasoner) | 66 | Python | licenca propria (ver LICENSE) | 8 | 0 | 0 | nao | ChemReasoner - Catalyst Discovery via Large Language Model-driven Reasoning |
| pnnl | [soda-opt](https://github.com/pnnl/soda-opt) | 64 | C++ | licenca propria (ver LICENSE) | 3 | 0 | 0 | sim |  |
| pnnl | [agent-cage](https://github.com/pnnl/agent-cage) | 64 | Python | licenca propria (ver LICENSE) | 1 | 0 | 0 | sim | Agent Cage is a Docker-based sandbox prototype that gives AI agents complete acc |
| pnnl | [cactus](https://github.com/pnnl/cactus) | 52 | Jupyter Notebook | BSD-2-Clause | 3 | 0 | 0 | sim | LLM Agent that leverages cheminformatics tools to provide informed responses. |
| pnnl | [tesp](https://github.com/pnnl/tesp) | 49 | Python | licenca propria (ver LICENSE) | 36 | 0 | 0 | nao |  |
| pnnl | [COMET](https://github.com/pnnl/COMET) | 45 | C++ | licenca propria (ver LICENSE) | 28 | 0 | 0 | nao |  |
| pnnl | [FragNet](https://github.com/pnnl/FragNet) | 42 | Python | licenca propria (ver LICENSE) | 2 | 0 | 0 | nao | A Graph Neural Network for molecular property prediction with four levels of Int |
| pnnl | [deimos](https://github.com/pnnl/deimos) | 41 | Python | BSD-3-Clause | 4 | 0 | 0 | nao |  |
| pnnl | [NWQ-Sim](https://github.com/pnnl/NWQ-Sim) | 37 | OpenQASM | MIT | 5 | 0 | 0 | nao |  |
| pnnl | [blueprint-styler](https://github.com/pnnl/blueprint-styler) | 35 | TypeScript | licenca propria (ver LICENSE) | 5 | 0 | 0 | nao | Custom themes and styles for Palantir's Blueprint js React component library |
| pnnl | [qasmtrans](https://github.com/pnnl/qasmtrans) | 31 | OpenQASM | licenca propria (ver LICENSE) | 5 | 0 | 0 | nao | A C++ based quantum transpiler for NISQ devices |
| pnnl | [building-energy-standards-data](https://github.com/pnnl/building-energy-standards-data) | 31 | Python | licenca propria (ver LICENSE) | 11 | 0 | 0 | nao | Database of building energy standards data for building energy simulation |
| pnnl | [UQ-MLIP](https://github.com/pnnl/UQ-MLIP) | 31 | Python | licenca propria (ver LICENSE) | 1 | 0 | 0 | nao | Uncertainty quantification (UQ) for Machine-Learning Interatomic Potentials (MLI |
| pnnl | [BEM-AI](https://github.com/pnnl/BEM-AI) | 31 | Python | licenca propria (ver LICENSE) | 17 | 0 | 0 | nao |  |
| pnnl | [ieee-std-1815-2-test-tool](https://github.com/pnnl/ieee-std-1815-2-test-tool) | 1 | Rust | BSD-3-Clause | 39 | 2 | 0 | sim | Profile editor, reference control station, reference outstation, and conformance |
| sandialabs | [wiretap](https://github.com/sandialabs/wiretap) | 1118 | Go | licenca propria (ver LICENSE) | 12 | 0 | 0 | nao | Wiretap is a transparent, VPN-like proxy server that tunnels traffic via WireGua |
| sandialabs | [toyplot](https://github.com/sandialabs/toyplot) | 449 | Jupyter Notebook | licenca propria (ver LICENSE) | 16 | 0 | 0 | nao | Interactive plotting for Python. |
| sandialabs | [Albany](https://github.com/sandialabs/Albany) | 333 | C++ | licenca propria (ver LICENSE) | 74 | 0 | 4 | nao | Sandia National Laboratories' Albany multiphysics code |
| sandialabs | [InterSpec](https://github.com/sandialabs/InterSpec) | 232 | C++ | LGPL-2.1 | 5 | 0 | 0 | nao | spectral radiation analysis software |
| sandialabs | [cross-sim](https://github.com/sandialabs/cross-sim) | 226 | Jupyter Notebook | licenca propria (ver LICENSE) | 13 | 0 | 0 | nao | CrossSim: accuracy simulation of analog in-memory computing |
| sandialabs | [qthreads](https://github.com/sandialabs/qthreads) | 200 | C | licenca propria (ver LICENSE) | 41 | 0 | 0 | nao | Lightweight locality-aware user-level threading runtime. |
| sandialabs | [pyGSTi](https://github.com/sandialabs/pyGSTi) | 191 | Python | Apache-2.0 | 132 | 0 | 0 | sim | A python implementation of Gate Set Tomography |
| sandialabs | [seacas](https://github.com/sandialabs/seacas) | 190 | C | licenca propria (ver LICENSE) | 37 | 0 | 2 | sim | The Sandia Engineering Analysis Code Access System (SEACAS) is a suite of prepro |
| sandialabs | [snl-quest](https://github.com/sandialabs/snl-quest) | 160 | Python | licenca propria (ver LICENSE) | 45 | 0 | 0 | nao | An open source, Python-based software platform for energy storage simulation and |
| sandialabs | [gr-fhss_utils](https://github.com/sandialabs/gr-fhss_utils) | 83 | C++ | GPL-3.0 | 9 | 0 | 0 | nao | Bursty modem utilities |
| sandialabs | [gr-pdu_utils](https://github.com/sandialabs/gr-pdu_utils) | 77 | C++ | GPL-3.0 | 3 | 0 | 0 | nao | GNU Radio PDU Utilities |
| sandialabs | [slycat](https://github.com/sandialabs/slycat) | 76 | JavaScript | licenca propria (ver LICENSE) | 138 | 0 | 0 | nao | A web-based analysis and visualization platform for HPC ensembles and high dimen |
| sandialabs | [pecos](https://github.com/sandialabs/pecos) | 76 | Python | licenca propria (ver LICENSE) | 2 | 0 | 0 | nao | Python package for performance monitoring of time series data |
| sandialabs | [pyapprox](https://github.com/sandialabs/pyapprox) | 76 | Python | MIT | 2 | 0 | 0 | nao | Flexible and efficient tools for high-dimensional approximation, scientific mach |
| sandialabs | [ctadl](https://github.com/sandialabs/ctadl) | 66 | Python | licenca propria (ver LICENSE) | 3 | 0 | 0 | sim | CTADL is a static taint analysis tool |
| sandialabs | [optimism](https://github.com/sandialabs/optimism) | 54 | Python | licenca propria (ver LICENSE) | 16 | 0 | 0 | nao | Computational solid mechanics made easy with Jax |
| sandialabs | [hyram](https://github.com/sandialabs/hyram) | 51 | Python | GPL-3.0 | 4 | 0 | 0 | sim |  |
| sandialabs | [pyttb](https://github.com/sandialabs/pyttb) | 51 | Python | licenca propria (ver LICENSE) | 34 | 0 | 0 | sim | Python Tensor Toolbox |
| sandialabs | [sdynpy](https://github.com/sandialabs/sdynpy) | 50 | Python | licenca propria (ver LICENSE) | 13 | 0 | 0 | nao | A Structural Dynamics Python Library |
| sandialabs | [SpecUtils](https://github.com/sandialabs/SpecUtils) | 48 | C++ | LGPL-2.1 | 5 | 0 | 0 | nao | A library for opening, manipulating, and exporting gamma spectral files |
| sandialabs | [reverse_argparse](https://github.com/sandialabs/reverse_argparse) | 48 | Python | BSD-3-Clause | 30 | 0 | 0 | sim | A Python library to determine what exactly the user ran at the command line, alo |
| sandialabs | [Spitfire](https://github.com/sandialabs/Spitfire) | 44 | Python | licenca propria (ver LICENSE) | 14 | 0 | 1 | nao | Spitfire is a Python/C++ library for constructing tabulated chemistry models and |
| sandialabs | [Fugu](https://github.com/sandialabs/Fugu) | 42 | Python | BSD-3-Clause | 3 | 0 | 0 | nao |  |
| sandialabs | [MEWS](https://github.com/sandialabs/MEWS) | 40 | Python | licenca propria (ver LICENSE) | 4 | 0 | 0 | nao | Multi-scenario Extreme Weather Simulator (MEWS) |
| sandialabs | [Prove-It](https://github.com/sandialabs/Prove-It) | 39 | Jupyter Notebook | licenca propria (ver LICENSE) | 136 | 0 | 0 | sim | A tool for proving and organizing general theorems using Python.  |
| sandialabs | [verdict](https://github.com/sandialabs/verdict) | 39 | C++ | licenca propria (ver LICENSE) | 3 | 0 | 0 | nao | verdict |
| sandialabs | [OWENS.jl](https://github.com/sandialabs/OWENS.jl) | 36 | Julia | LGPL-3.0 | 3 | 0 | 0 | nao |  |
| sandialabs | [spack-manager](https://github.com/sandialabs/spack-manager) | 31 | Python | licenca propria (ver LICENSE) | 15 | 0 | 1 | sim | A project and machine deployment model using Spack |
| sandialabs | [PEAT](https://github.com/sandialabs/PEAT) | 31 | Python | GPL-3.0 | 41 | 4 | 5 | sim | The Process Extraction and Analysis Tool (PEAT), a Operational Technology (OT) d |
| sandialabs | [lim1tr](https://github.com/sandialabs/lim1tr) | 30 | Python | AGPL-3.0 | 1 | 0 | 0 | nao | Lithium-Ion Modeling with 1-D Thermal Runaway (LIM1TR) |
| sandialabs | [sceptre-phenix](https://github.com/sandialabs/sceptre-phenix) | 28 | JavaScript | GPL-3.0 | 51 | 2 | 1 | sim | phenix is an orchestration tool and GUI for Sandia's minimega platform |
| sandialabs | [WecOptTool](https://github.com/sandialabs/WecOptTool) | 21 | Python | GPL-3.0 | 33 | 1 | 1 | sim | WEC Design Optimization Toolbox |
| sandialabs | [OpenCSP](https://github.com/sandialabs/OpenCSP) | 13 | Python | licenca propria (ver LICENSE) | 91 | 7 | 0 | sim | Code for Concentrating Solar Power |
| sandialabs | [sceptre-phenix-apps](https://github.com/sandialabs/sceptre-phenix-apps) | 11 | Python | GPL-3.0 | 36 | 2 | 0 | sim | Apps written to work with the latest version of phenix |
| sandialabs | [firewheel](https://github.com/sandialabs/firewheel) | 10 | Python | licenca propria (ver LICENSE) | 36 | 2 | 1 | sim | FIREWHEEL is an experiment orchestration tool that assists a user in building an |
| sandialabs | [pytribeam](https://github.com/sandialabs/pytribeam) | 8 | Python | licenca propria (ver LICENSE) | 24 | 0 | 4 | sim | Automated data collection for the TriBeam microscope |
| sandialabs | [sansmic](https://github.com/sandialabs/sansmic) | 4 | C++ | BSD-3-Clause | 12 | 2 | 0 | sim | Sandia solution mining code |
| sandialabs | [forcefinder](https://github.com/sandialabs/forcefinder) | 4 | Jupyter Notebook | licenca propria (ver LICENSE) | 7 | 1 | 0 | sim | Advanced inverse source estimation tools for force reconstruction, TPA, and MIMO |

### Comunidade cientifica (192 de 321 candidatos)

| Org | Repo | ★ | Lang | Licenca | Issues | GFI | HW | CONTRIB. | Descricao |
|---|---|---|---|---|---|---|---|---|---|
| FEniCS | [dolfinx](https://github.com/FEniCS/dolfinx) | 1219 | C++ | LGPL-3.0 | 123 | 4 | 0 | sim | Next generation FEniCS problem solving environment for solving finite element pr |
| FEniCS | [ffcx](https://github.com/FEniCS/ffcx) | 195 | Python | licenca propria (ver LICENSE) | 49 | 2 | 0 | nao | Next generation FEniCS Form Compiler for finite element forms |
| FEniCS | [ufl](https://github.com/FEniCS/ufl) | 152 | Python | LGPL-3.0 | 60 | 0 | 0 | nao | UFL - Unified Form Language |
| FEniCS | [basix](https://github.com/FEniCS/basix) | 148 | C++ | MIT | 49 | 1 | 0 | sim | FEniCSx finite element basis evaluation library |
| KratosMultiphysics | [Kratos](https://github.com/KratosMultiphysics/Kratos) | 1373 | C++ | licenca propria (ver LICENSE) | 752 | 0 | 31 | sim | Kratos Multiphysics (A.K.A Kratos) is a framework for building parallel multi-di |
| KratosMultiphysics | [GiDInterface](https://github.com/KratosMultiphysics/GiDInterface) | 39 | Python | licenca propria (ver LICENSE) | 37 | 0 | 3 | nao | The graphical user interface of Kratos for GiD. Featuring CFD, CSM, DEM, PFEM, e |
| NGSolve | [ngsolve](https://github.com/NGSolve/ngsolve) | 586 | C++ | LGPL-2.1 | 22 | 0 | 0 | sim |   Netgen/NGSolve is a high performance multiphysics finite element software. It  |
| NGSolve | [netgen](https://github.com/NGSolve/netgen) | 396 | C++ | LGPL-2.1 | 115 | 0 | 0 | sim |  |
| OPM | [ResInsight](https://github.com/OPM/ResInsight) | 217 | C++ | GPL-3.0 | 839 | 0 | 0 | sim | 3D viewer and post processing of reservoir models |
| OPM | [opm-simulators](https://github.com/OPM/opm-simulators) | 170 | C++ | GPL-3.0 | 498 | 0 | 0 | sim | OPM Flow and experimental simulators, including components such as well models e |
| OPM | [LBPM](https://github.com/OPM/LBPM) | 93 | C++ | GPL-3.0 | 47 | 0 | 0 | nao | Pore scale modelling |
| OPM | [IFEM](https://github.com/OPM/IFEM) | 47 | C | licenca propria (ver LICENSE) | 26 | 0 | 0 | nao | IFEM - Isogeometric Toolbox for the solution of PDEs |
| OPM | [opm-common](https://github.com/OPM/opm-common) | 40 | C++ | GPL-3.0 | 216 | 0 | 0 | nao | Common components for OPM, in particular build system (cmake). |
| PolymathicAI | [the_well](https://github.com/PolymathicAI/the_well) | 4469 | Jupyter Notebook | BSD-3-Clause | 23 | 0 | 0 | nao | A 15TB Collection of Physics Simulation Datasets |
| PolymathicAI | [walrus](https://github.com/PolymathicAI/walrus) | 312 | Python | MIT | 8 | 0 | 0 | nao |  |
| PolymathicAI | [AstroCLIP](https://github.com/PolymathicAI/AstroCLIP) | 182 | Python | MIT | 7 | 0 | 0 | nao | Multimodal contrastive pretraining for astronomical data |
| PolymathicAI | [AION](https://github.com/PolymathicAI/AION) | 152 | Jupyter Notebook | MIT | 4 | 0 | 0 | nao | Polymathic's  Large Omnimodal Model for Astronomy |
| PolymathicAI | [MIMIC](https://github.com/PolymathicAI/MIMIC) | 46 | Python | MIT | 1 | 0 | 0 | nao | A Generative Multimodal Model for Biomolecules |
| SciML | [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl) | 3166 | Julia | licenca propria (ver LICENSE) | 115 | 0 | 1 | nao | Multi-language suite for high-performance solvers of differential equations and  |
| SciML | [ModelingToolkit.jl](https://github.com/SciML/ModelingToolkit.jl) | 1693 | Julia | licenca propria (ver LICENSE) | 712 | 2 | 0 | sim | An acausal modeling framework for automatically parallelized scientific machine  |
| SciML | [NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl) | 1235 | Julia | licenca propria (ver LICENSE) | 165 | 3 | 0 | nao | Physics-Informed Neural Networks (PINN) Solvers of (Partial) Differential Equati |
| SciML | [DiffEqFlux.jl](https://github.com/SciML/DiffEqFlux.jl) | 929 | Julia | MIT | 55 | 3 | 1 | nao | Pre-built implicit layer architectures with O(1) backprop, GPUs, and stiff+non-s |
| SciML | [Optimization.jl](https://github.com/SciML/Optimization.jl) | 845 | Julia | MIT | 139 | 1 | 0 | nao | Mathematical Optimization in Julia. Local, global, gradient-based and derivative |
| SciML | [OrdinaryDiffEq.jl](https://github.com/SciML/OrdinaryDiffEq.jl) | 692 | Julia | licenca propria (ver LICENSE) | 592 | 10 | 11 | sim | High performance ordinary differential equation (ODE) and differential-algebraic |
| SciML | [diffeqpy](https://github.com/SciML/diffeqpy) | 612 | Python | MIT | 11 | 0 | 0 | nao | Solving differential equations in Python using DifferentialEquations.jl and the  |
| SciML | [Catalyst.jl](https://github.com/SciML/Catalyst.jl) | 530 | Julia | licenca propria (ver LICENSE) | 111 | 4 | 1 | nao | Chemical reaction network and systems biology interface for scientific machine l |
| SciML | [BlackBoxOptim.jl](https://github.com/SciML/BlackBoxOptim.jl) | 466 | Julia | licenca propria (ver LICENSE) | 47 | 0 | 0 | nao | Black-box optimization for Julia |
| SciML | [DataDrivenDiffEq.jl](https://github.com/SciML/DataDrivenDiffEq.jl) | 431 | Julia | MIT | 90 | 2 | 5 | nao | Data driven modeling and automated discovery of dynamical systems for the SciML  |
| SciML | [SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl) | 397 | Julia | licenca propria (ver LICENSE) | 125 | 4 | 0 | nao | A component of the DiffEq ecosystem for enabling sensitivity analysis for scient |
| SciML | [Surrogates.jl](https://github.com/SciML/Surrogates.jl) | 384 | Julia | licenca propria (ver LICENSE) | 32 | 0 | 0 | sim | Surrogate modeling and optimization for scientific machine learning (SciML) |
| SciML | [ComponentArrays.jl](https://github.com/SciML/ComponentArrays.jl) | 378 | Julia | MIT | 81 | 0 | 1 | nao | Arrays with arbitrarily nested named components. |
| SciML | [Evolutionary.jl](https://github.com/SciML/Evolutionary.jl) | 353 | Julia | licenca propria (ver LICENSE) | 42 | 0 | 0 | nao | Evolutionary & genetic algorithms for Julia  |
| SciML | [SciMLBenchmarks.jl](https://github.com/SciML/SciMLBenchmarks.jl) | 346 | Julia | MIT | 37 | 0 | 0 | nao | Scientific machine learning (SciML) benchmarks, AI for science, and (differentia |
| SciML | [DiffEqDocs.jl](https://github.com/SciML/DiffEqDocs.jl) | 330 | Julia | licenca propria (ver LICENSE) | 24 | 0 | 0 | nao | Documentation for the DiffEq differential equations and scientific machine learn |
| SciML | [DiffEqGPU.jl](https://github.com/SciML/DiffEqGPU.jl) | 328 | Julia | MIT | 48 | 1 | 0 | nao | GPU-acceleration routines for DifferentialEquations.jl and the broader SciML sci |
| SciML | [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl) | 313 | Julia | MIT | 116 | 4 | 0 | nao | High-performance and differentiation-enabled nonlinear solvers (Newton methods), |
| SciML | [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl) | 295 | Julia | licenca propria (ver LICENSE) | 51 | 0 | 0 | nao | LinearSolve.jl: High-Performance Unified Interface for Linear Solvers in Julia.  |
| SciML | [DataInterpolations.jl](https://github.com/SciML/DataInterpolations.jl) | 269 | Julia | MIT | 48 | 1 | 0 | nao | A library of data interpolation and smoothing functions |
| SciML | [Integrals.jl](https://github.com/SciML/Integrals.jl) | 246 | Julia | MIT | 29 | 1 | 0 | nao | A common interface for quadrature and numerical integration for the SciML scient |
| SciML | [SciMLStyle](https://github.com/SciML/SciMLStyle) | 236 | Julia | MIT | 5 | 0 | 0 | nao | A style guide for stylish Julia developers |
| SciML | [RecursiveArrayTools.jl](https://github.com/SciML/RecursiveArrayTools.jl) | 234 | Julia | licenca propria (ver LICENSE) | 54 | 0 | 0 | nao | Tools for easily handling objects like arrays of arrays and deeper nestings in s |
| SciML | [ReservoirComputing.jl](https://github.com/SciML/ReservoirComputing.jl) | 233 | Julia | MIT | 41 | 2 | 0 | sim | Reservoir computing utilities for scientific machine learning (SciML) |
| SciML | [Sundials.jl](https://github.com/SciML/Sundials.jl) | 216 | Julia | BSD-2-Clause | 26 | 0 | 0 | nao | Julia interface to Sundials, including a nonlinear solver (KINSOL), ODEs (CVODE  |
| SciML | [MethodOfLines.jl](https://github.com/SciML/MethodOfLines.jl) | 212 | Julia | MIT | 110 | 0 | 0 | nao | Automatic Finite Difference PDE solving with Julia SciML |
| SciML | [SciMLBase.jl](https://github.com/SciML/SciMLBase.jl) | 179 | Julia | MIT | 140 | 2 | 0 | nao | The Base interface of the SciML ecosystem |
| SciML | [ModelingToolkitStandardLibrary.jl](https://github.com/SciML/ModelingToolkitStandardLibrary.jl) | 171 | Julia | MIT | 75 | 0 | 0 | nao | A standard library of components to model the world and beyond |
| SciML | [Julia_Modeling_Workshop](https://github.com/SciML/Julia_Modeling_Workshop) | 159 | HTML | MIT | 1 | 0 | 0 | nao | High-Performance Scientific Modeling with Julia and SciML |
| SciML | [JumpProcesses.jl](https://github.com/SciML/JumpProcesses.jl) | 151 | Julia | licenca propria (ver LICENSE) | 94 | 0 | 3 | nao | Build and simulate jump equations like Gillespie simulations and jump diffusions |
| SciML | [diffeqr](https://github.com/SciML/diffeqr) | 148 | R | licenca propria (ver LICENSE) | 7 | 0 | 0 | nao | Solving differential equations in R using DifferentialEquations.jl and the SciML |
| SciML | [NBodySimulator.jl](https://github.com/SciML/NBodySimulator.jl) | 138 | Julia | licenca propria (ver LICENSE) | 16 | 0 | 0 | nao | A differentiable simulator for scientific machine learning (SciML) with N-body p |
| SciML | [StructuralIdentifiability.jl](https://github.com/SciML/StructuralIdentifiability.jl) | 130 | Julia | MIT | 18 | 1 | 0 | nao | Fast and automatic structural identifiability software for ODE systems |
| SciML | [SymbolicNumericIntegration.jl](https://github.com/SciML/SymbolicNumericIntegration.jl) | 130 | Julia | MIT | 7 | 0 | 0 | nao | SymbolicNumericIntegration.jl: Symbolic-Numerics for Solving Integrals |
| SciML | [PreallocationTools.jl](https://github.com/SciML/PreallocationTools.jl) | 127 | Julia | licenca propria (ver LICENSE) | 1 | 0 | 0 | nao | Tools for building non-allocating pre-cached functions in Julia, allowing for GC |
| SciML | [DiffEqBayes.jl](https://github.com/SciML/DiffEqBayes.jl) | 125 | Julia | licenca propria (ver LICENSE) | 16 | 0 | 0 | nao | Extension functionality which uses Stan.jl, DynamicHMC.jl, and Turing.jl to esti |
| SciML | [LabelledArrays.jl](https://github.com/SciML/LabelledArrays.jl) | 125 | Julia | licenca propria (ver LICENSE) | 20 | 0 | 0 | nao | Arrays which also have a label for each element for easy scientific machine lear |
| SciML | [QuasiMonteCarlo.jl](https://github.com/SciML/QuasiMonteCarlo.jl) | 124 | Julia | MIT | 22 | 0 | 0 | nao | Lightweight and easy generation of quasi-Monte Carlo sequences with a ton of dif |
| SciML | [DiffEqProblemLibrary.jl](https://github.com/SciML/DiffEqProblemLibrary.jl) | 123 | Julia | licenca propria (ver LICENSE) | 11 | 0 | 3 | nao | A library of premade problems for examples and testing differential equation sol |
| SciML | [PolyChaos.jl](https://github.com/SciML/PolyChaos.jl) | 122 | Julia | MIT | 16 | 2 | 1 | nao | A Julia package to construct orthogonal polynomials, their quadrature rules, and |
| SciML | [RuntimeGeneratedFunctions.jl](https://github.com/SciML/RuntimeGeneratedFunctions.jl) | 114 | Julia | MIT | 15 | 1 | 0 | nao | Functions generated at runtime without world-age issues or overhead |
| SciML | [ExponentialUtilities.jl](https://github.com/SciML/ExponentialUtilities.jl) | 109 | Julia | licenca propria (ver LICENSE) | 34 | 1 | 0 | nao | Fast and differentiable implementations of matrix exponentials, Krylov exponenti |
| SciML | [FEniCS.jl](https://github.com/SciML/FEniCS.jl) | 106 | Julia | licenca propria (ver LICENSE) | 10 | 0 | 0 | nao | A scientific machine learning (SciML) wrapper for the FEniCS Finite Element libr |
| SciML | [ConcreteStructs.jl](https://github.com/SciML/ConcreteStructs.jl) | 105 | Julia | MIT | 12 | 0 | 0 | nao |   🏩🏠🌆🏨🌇🏦 |
| SciML | [DiffEqCallbacks.jl](https://github.com/SciML/DiffEqCallbacks.jl) | 100 | Julia | licenca propria (ver LICENSE) | 31 | 0 | 0 | nao | A library of useful callbacks for hybrid scientific machine learning (SciML) wit |
| SciML | [EllipsisNotation.jl](https://github.com/SciML/EllipsisNotation.jl) | 99 | Julia | licenca propria (ver LICENSE) | 3 | 0 | 0 | nao | Julia-based implementation of ellipsis array indexing notation `..` |
| SciML | [SciMLDocs](https://github.com/SciML/SciMLDocs) | 99 | Julia | MIT | 10 | 0 | 0 | nao | Global documentation for the Julia SciML Scientific Machine Learning Organizatio |
| SciML | [EasyModelAnalysis.jl](https://github.com/SciML/EasyModelAnalysis.jl) | 88 | Julia | MIT | 29 | 0 | 0 | nao | High level functions for analyzing the output of simulations |
| SciML | [HighDimPDE.jl](https://github.com/SciML/HighDimPDE.jl) | 87 | Julia | licenca propria (ver LICENSE) | 15 | 0 | 0 | nao | A Julia package for Deep Backwards Stochastic Differential Equation (Deep BSDE)  |
| SciML | [FastBroadcast.jl](https://github.com/SciML/FastBroadcast.jl) | 84 | Julia | MIT | 9 | 0 | 0 | nao |  |
| SciML | [ParameterizedFunctions.jl](https://github.com/SciML/ParameterizedFunctions.jl) | 78 | Julia | licenca propria (ver LICENSE) | 1 | 0 | 0 | nao | A simple domain-specific language (DSL) for defining differential equations for  |
| SciML | [MultiScaleArrays.jl](https://github.com/SciML/MultiScaleArrays.jl) | 77 | Julia | licenca propria (ver LICENSE) | 7 | 0 | 0 | nao | A framework for developing multi-scale arrays for use in scientific machine lear |
| SciML | [sciml.ai](https://github.com/SciML/sciml.ai) | 76 | CSS | MIT | 8 | 0 | 0 | nao | The SciML Scientific Machine Learning Software Organization Website |
| SciML | [ADTypes.jl](https://github.com/SciML/ADTypes.jl) | 73 | Julia | MIT | 15 | 0 | 0 | nao | Repository for automatic differentiation backend types |
| SciML | [SciMLExpectations.jl](https://github.com/SciML/SciMLExpectations.jl) | 71 | Julia | licenca propria (ver LICENSE) | 37 | 0 | 0 | nao | Fast uncertainty quantification for scientific machine learning (SciML) and diff |
| SciML | [CellMLToolkit.jl](https://github.com/SciML/CellMLToolkit.jl) | 68 | Julia | licenca propria (ver LICENSE) | 12 | 0 | 0 | nao | CellMLToolkit.jl is a Julia library that connects CellML models to the Scientifi |
| SciML | [GlobalSensitivity.jl](https://github.com/SciML/GlobalSensitivity.jl) | 67 | Julia | MIT | 33 | 0 | 0 | nao | Robust, Fast, and Parallel Global Sensitivity Analysis (GSA) in Julia |
| SciML | [Static.jl](https://github.com/SciML/Static.jl) | 64 | Julia | MIT | 23 | 0 | 0 | nao | Static types useful for dispatch and generated functions. |
| SciML | [DiffEqNoiseProcess.jl](https://github.com/SciML/DiffEqNoiseProcess.jl) | 63 | Julia | licenca propria (ver LICENSE) | 28 | 0 | 0 | nao | A library of noise processes for stochastic systems like stochastic differential |
| SciML | [DiffEqParamEstim.jl](https://github.com/SciML/DiffEqParamEstim.jl) | 62 | Julia | licenca propria (ver LICENSE) | 23 | 0 | 0 | nao | Easy scientific machine learning (SciML) parameter estimation with pre-built los |
| SciML | [BoundaryValueDiffEq.jl](https://github.com/SciML/BoundaryValueDiffEq.jl) | 61 | Julia | licenca propria (ver LICENSE) | 59 | 1 | 0 | nao | Boundary value problem (BVP) solvers for scientific machine learning (SciML) |
| SciML | [DeepEquilibriumNetworks.jl](https://github.com/SciML/DeepEquilibriumNetworks.jl) | 59 | Julia | MIT | 1 | 0 | 0 | nao | Implicit Layer Machine Learning via Deep Equilibrium Networks, O(1) backpropagat |
| SciML | [SciMLOperators.jl](https://github.com/SciML/SciMLOperators.jl) | 56 | Julia | MIT | 29 | 0 | 0 | nao | SciMLOperators.jl: Matrix-Free Operators for the SciML Scientific Machine Learni |
| SciML | [MuladdMacro.jl](https://github.com/SciML/MuladdMacro.jl) | 52 | Julia | licenca propria (ver LICENSE) | 1 | 0 | 0 | nao | This package contains a macro for converting expressions to use muladd calls and |
| SciML | [MinimallyDisruptiveCurves.jl](https://github.com/SciML/MinimallyDisruptiveCurves.jl) | 52 | Julia | MIT | 2 | 0 | 0 | nao | Finds relationships between the parameters of a mathematical model |
| SciML | [FiniteVolumeMethod.jl](https://github.com/SciML/FiniteVolumeMethod.jl) | 52 | Julia | MIT | 7 | 0 | 0 | nao | Solver for two-dimensional conservation equations using the finite volume method |
| SciML | [DiffEqPhysics.jl](https://github.com/SciML/DiffEqPhysics.jl) | 51 | Julia | licenca propria (ver LICENSE) | 8 | 0 | 0 | nao | A library for building differential equations arising from physical problems for |
| SciML | [MomentClosure.jl](https://github.com/SciML/MomentClosure.jl) | 51 | Julia | MIT | 2 | 0 | 0 | nao | Tools to generate and study moment equations for any chemical reaction network u |
| SciML | [ProcessSimulator.jl](https://github.com/SciML/ProcessSimulator.jl) | 51 | Julia | MIT | 8 | 0 | 0 | nao |  |
| SciML | [SBMLToolkit.jl](https://github.com/SciML/SBMLToolkit.jl) | 44 | Julia | MIT | 20 | 0 | 0 | nao | SBML differential equation and chemical reaction model (Gillespie simulations) f |
| SciML | [ModelOrderReduction.jl](https://github.com/SciML/ModelOrderReduction.jl) | 43 | Julia | MIT | 39 | 0 | 0 | nao | High-level model-order reduction to automate the acceleration of large-scale sim |
| SciML | [NeuralOperators.jl](https://github.com/SciML/NeuralOperators.jl) | 43 | Julia | MIT | 48 | 0 | 0 | nao | DeepONets, (Fourier) Neural Operators, Physics-Informed Neural Operators, and mo |
| SciML | [MonteCarloIntegration.jl](https://github.com/SciML/MonteCarloIntegration.jl) | 41 | Julia | MIT | 12 | 0 | 0 | nao | A package for multi-dimensional integration using monte carlo methods |
| SciML | [ModelingToolkitNeuralNets.jl](https://github.com/SciML/ModelingToolkitNeuralNets.jl) | 41 | Julia | MIT | 10 | 0 | 0 | nao | Symbolic-Numeric Neural DAEs and Universal Differential Equations for Automating |
| SciML | [SciMLWorkshop.jl](https://github.com/SciML/SciMLWorkshop.jl) | 40 | Julia | MIT | 2 | 0 | 0 | nao | Workshop materials for training in scientific computing and scientific machine l |
| SciML | [DASSL.jl](https://github.com/SciML/DASSL.jl) | 36 | Julia | licenca propria (ver LICENSE) | 2 | 0 | 0 | nao | Solves stiff differential algebraic equations (DAE) using variable stepsize back |
| SciML | [RootedTrees.jl](https://github.com/SciML/RootedTrees.jl) | 36 | Julia | MIT | 14 | 0 | 0 | sim | A collection of functionality around rooted trees to generate order conditions f |
| SciML | [DifferenceEquations.jl](https://github.com/SciML/DifferenceEquations.jl) | 36 | Julia | MIT | 5 | 0 | 0 | nao | Solving difference equations with DifferenceEquations.jl and the SciML ecosystem |
| SciML | [DiffEqFinancial.jl](https://github.com/SciML/DiffEqFinancial.jl) | 34 | Julia | licenca propria (ver LICENSE) | 1 | 0 | 0 | nao | Differential equation problem specifications and scientific machine learning for |
| SciML | [SteadyStateDiffEq.jl](https://github.com/SciML/SteadyStateDiffEq.jl) | 31 | Julia | licenca propria (ver LICENSE) | 14 | 0 | 0 | nao | Solvers for steady states in scientific machine learning (SciML) |
| SciML | [ParallelParticleSwarms.jl](https://github.com/SciML/ParallelParticleSwarms.jl) | 31 | Julia | MIT | 15 | 0 | 0 | nao | GPU accelerated Particle Swarm Optimization |
| SciML | [DataInterpolationsND.jl](https://github.com/SciML/DataInterpolationsND.jl) | 30 | Julia | MIT | 32 | 2 | 0 | nao | Interpolation of arbitrarily high dimensional array data |
| SciML | [DASKR.jl](https://github.com/SciML/DASKR.jl) | 11 | Julia | licenca propria (ver LICENSE) | 5 | 0 | 1 | nao | Interface to DASKR, a differential algebraic system solver for the SciML scienti |
| dealii | [dealii](https://github.com/dealii/dealii) | 1734 | C++ | licenca propria (ver LICENSE) | 636 | 0 | 0 | sim | The development repository for the deal.II finite element library |
| dealii | [candi](https://github.com/dealii/candi) | 80 | Shell | LGPL-3.0 | 41 | 0 | 0 | nao | candi - (Compile & Install) - Downloads, configures, builds and installs deal.II |
| dealii | [code-gallery](https://github.com/dealii/code-gallery) | 49 | C++ | licenca propria (ver LICENSE) | 14 | 0 | 1 | nao | A collection of codes based on deal.II contributed by deal.II users |
| deepmodeling | [deepmd-kit](https://github.com/deepmodeling/deepmd-kit) | 2054 | Python | LGPL-3.0 | 236 | 1 | 0 | sim | A deep learning package for many-body potential energy representation and molecu |
| deepmodeling | [jax-fem](https://github.com/deepmodeling/jax-fem) | 772 | Python | GPL-3.0 | 24 | 0 | 0 | nao | Differentiable Finite Element Method with JAX |
| deepmodeling | [dpgen](https://github.com/deepmodeling/dpgen) | 400 | Python | LGPL-3.0 | 15 | 0 | 0 | nao | The deep potential generator to generate a deep-learning based model of interato |
| deepmodeling | [dpdata](https://github.com/deepmodeling/dpdata) | 254 | Python | LGPL-3.0 | 19 | 0 | 0 | nao | A Python package for manipulating atomistic data of software in computational sc |
| deepmodeling | [DMFF](https://github.com/deepmodeling/DMFF) | 199 | Python | LGPL-3.0 | 12 | 0 | 0 | nao | DMFF (Differentiable Molecular Force Field) is a Jax-based python package that p |
| deepmodeling | [Uni-Lab-OS](https://github.com/deepmodeling/Uni-Lab-OS) | 178 | Python | GPL-3.0 | 50 | 0 | 0 | nao | A Platform for Laboratory Automation. |
| deepmodeling | [CrystalFormer](https://github.com/deepmodeling/CrystalFormer) | 153 | Jupyter Notebook | Apache-2.0 | 5 | 0 | 0 | nao | A Foundation Model for Crystal Structure Generation and Prediction |
| deepmodeling | [DeePTB](https://github.com/deepmodeling/DeePTB) | 123 | Python | LGPL-3.0 | 45 | 0 | 0 | sim | DeePTB: A deep learning package for tight-binding Hamiltonian with ab initio acc |
| deepmodeling | [reacnetgenerator](https://github.com/deepmodeling/reacnetgenerator) | 103 | Python | LGPL-3.0 | 14 | 0 | 1 | nao | an automatic reaction network generator for reactive molecular dynamics simulati |
| deepmodeling | [AI4S-agent-tools](https://github.com/deepmodeling/AI4S-agent-tools) | 90 | Python | MIT | 2 | 0 | 0 | sim | Collecting a variety of Agent-Ready tool modules |
| deepmodeling | [dpdispatcher](https://github.com/deepmodeling/dpdispatcher) | 62 | Python | LGPL-3.0 | 5 | 0 | 0 | sim | generate HPC scheduler systems jobs input scripts and submit these scripts to HP |
| deepmodeling | [deepmd-gnn](https://github.com/deepmodeling/deepmd-gnn) | 56 | Python | LGPL-3.0 | 13 | 0 | 0 | nao | DeePMD-kit plugin for various graph neural network models |
| deepmodeling | [APEX](https://github.com/deepmodeling/APEX) | 48 | Python | LGPL-3.0 | 2 | 0 | 0 | nao | APEX: Alloy Properties EXplorer using simulations |
| deepmodeling | [dpti](https://github.com/deepmodeling/dpti) | 44 | Python | LGPL-3.0 | 63 | 0 | 0 | nao | A Python Package to Automate Thermodynamic Integration Calculations for Free Ene |
| deepmodeling | [unimol_tools](https://github.com/deepmodeling/unimol_tools) | 35 | Python | MIT | 18 | 0 | 0 | nao | unimol_tools: a easy-use & auto-ml molecule property prediction tool |
| firedrakeproject | [firedrake](https://github.com/firedrakeproject/firedrake) | 679 | Python | licenca propria (ver LICENSE) | 466 | 9 | 0 | nao | Firedrake is an automated system for the portable solution of partial differenti |
| firedrakeproject | [Irksome](https://github.com/firedrakeproject/Irksome) | 35 | Jupyter Notebook | licenca propria (ver LICENSE) | 31 | 0 | 0 | nao | Solvers for Implicit Runge Kutta methods |
| firedrakeproject | [petsctools](https://github.com/firedrakeproject/petsctools) | 4 | Python | LGPL-3.0 | 13 | 1 | 0 | nao | Pythonic extensions for petsc4py and slepc4py. |
| jax-ml | [jax](https://github.com/jax-ml/jax) | 36395 | Python | Apache-2.0 | 2623 | 2 | 35 | sim | Composable transformations of Python+NumPy programs: differentiate, vectorize, J |
| jax-ml | [scaling-book](https://github.com/jax-ml/scaling-book) | 1469 | SCSS | MIT | 14 | 0 | 0 | nao | Home for "How To Scale Your Model", a short blog-style textbook about scaling LL |
| jax-ml | [jax-triton](https://github.com/jax-ml/jax-triton) | 472 | Python | Apache-2.0 | 11 | 0 | 0 | sim | jax-triton contains integrations between JAX and OpenAI Triton |
| jax-ml | [ml_dtypes](https://github.com/jax-ml/ml_dtypes) | 363 | C++ | Apache-2.0 | 43 | 0 | 2 | sim | A stand-alone implementation of several NumPy dtype extensions used in machine l |
| jax-ml | [oryx](https://github.com/jax-ml/oryx) | 326 | Python | Apache-2.0 | 20 | 1 | 0 | sim | Oryx is a library for probabilistic programming and deep learning built on top o |
| jax-ml | [jax-ai-stack](https://github.com/jax-ml/jax-ai-stack) | 313 | Python | Apache-2.0 | 36 | 0 | 0 | sim |  |
| jax-ml | [jax-llm-examples](https://github.com/jax-ml/jax-llm-examples) | 286 | Python | Apache-2.0 | 9 | 0 | 0 | sim | Minimal yet performant LLM examples in pure JAX |
| jax-ml | [bayeux](https://github.com/jax-ml/bayeux) | 249 | Python | Apache-2.0 | 11 | 0 | 0 | sim | State of the art inference for your bayesian models.  |
| jax-ml | [bonsai](https://github.com/jax-ml/bonsai) | 246 | Python | Apache-2.0 | 38 | 3 | 2 | sim | Minimal, lightweight JAX implementations of popular models. |
| jax-ml | [jax-tpu-embedding](https://github.com/jax-ml/jax-tpu-embedding) | 37 | Python | Apache-2.0 | 11 | 0 | 0 | sim | High-performance large-scale embedding acceleration for JAX on Google TPU Sparse |
| materialsproject | [pymatgen](https://github.com/materialsproject/pymatgen) | 1988 | Python | licenca propria (ver LICENSE) | 132 | 0 | 0 | nao | Python Materials Genomics (pymatgen) is a robust materials analysis code that de |
| materialsproject | [fireworks](https://github.com/materialsproject/fireworks) | 427 | Python | licenca propria (ver LICENSE) | 75 | 0 | 0 | sim | The Fireworks Workflow Management Repo. |
| materialsproject | [atomate2](https://github.com/materialsproject/atomate2) | 352 | Python | licenca propria (ver LICENSE) | 64 | 0 | 0 | sim | atomate2 is a library of computational materials science workflows  |
| materialsproject | [crystaltoolkit](https://github.com/materialsproject/crystaltoolkit) | 204 | Python | licenca propria (ver LICENSE) | 76 | 0 | 3 | nao | Crystal Toolkit is a framework for building web apps for materials science and i |
| materialsproject | [custodian](https://github.com/materialsproject/custodian) | 188 | Python | MIT | 34 | 0 | 0 | nao | A simple, robust and flexible just-in-time job management framework in Python. |
| materialsproject | [api](https://github.com/materialsproject/api) | 179 | Python | licenca propria (ver LICENSE) | 8 | 0 | 0 | nao | New API client for the Materials Project  |
| materialsproject | [reaction-network](https://github.com/materialsproject/reaction-network) | 133 | Python | licenca propria (ver LICENSE) | 6 | 0 | 0 | sim | Reaction Network is a Python package for predicting likely inorganic chemical re |
| materialsproject | [jobflow](https://github.com/materialsproject/jobflow) | 128 | Python | licenca propria (ver LICENSE) | 44 | 0 | 0 | sim | jobflow is a library for writing computational workflows. |
| materialsproject | [emmet](https://github.com/materialsproject/emmet) | 71 | Python | licenca propria (ver LICENSE) | 18 | 0 | 0 | nao | Be a master builder of databases of material properties. Avoid the Kragle. |
| materialsproject | [pymatgen-analysis-defects](https://github.com/materialsproject/pymatgen-analysis-defects) | 67 | Python | licenca propria (ver LICENSE) | 6 | 0 | 0 | nao | Defect analysis modules for pymatgen |
| materialsproject | [pymatgen-db](https://github.com/materialsproject/pymatgen-db) | 52 | Python | MIT | 3 | 0 | 0 | nao | Pymatgen-db provides an addon to the Python Materials Genomics (pymatgen) librar |
| materialsproject | [maggma](https://github.com/materialsproject/maggma) | 44 | Python | licenca propria (ver LICENSE) | 38 | 0 | 0 | nao | Building blocks for scientific data pipelines |
| materialsproject | [pyrho](https://github.com/materialsproject/pyrho) | 43 | Python | licenca propria (ver LICENSE) | 7 | 0 | 0 | sim |  |
| materialsproject | [MPContribs](https://github.com/materialsproject/MPContribs) | 42 | Python | MIT | 27 | 0 | 0 | nao | Platform for materials scientists to contribute and disseminate their materials  |
| materialsproject | [dash-mp-components](https://github.com/materialsproject/dash-mp-components) | 32 | Python | licenca propria (ver LICENSE) | 6 | 0 | 0 | sim | Plotly Dash components developed by the Materials Project |
| mfem | [mfem](https://github.com/mfem/mfem) | 2259 | C++ | BSD-3-Clause | 299 | 0 | 3 | sim | Lightweight, general, scalable C++ library for finite element methods |
| mfem | [PyMFEM](https://github.com/mfem/PyMFEM) | 299 | SWIG | BSD-3-Clause | 34 | 0 | 0 | nao | Python wrapper for MFEM |
| openmm | [pdbfixer](https://github.com/openmm/pdbfixer) | 677 | Python | licenca propria (ver LICENSE) | 61 | 0 | 0 | nao | PDBFixer fixes problems in PDB files |
| openmm | [openmmforcefields](https://github.com/openmm/openmmforcefields) | 382 | Rich Text Format | licenca propria (ver LICENSE) | 87 | 0 | 0 | nao | CHARMM and AMBER forcefields for OpenMM (with small molecule support) |
| openmm | [spice-dataset](https://github.com/openmm/spice-dataset) | 205 | Python | MIT | 21 | 0 | 0 | nao | A collection of QM data for training potential functions |
| openmm | [openmm-ml](https://github.com/openmm/openmm-ml) | 184 | Python | licenca propria (ver LICENSE) | 31 | 0 | 0 | sim | High level API for using machine learning models in OpenMM simulations |
| openmm | [NNPOps](https://github.com/openmm/NNPOps) | 102 | C++ | licenca propria (ver LICENSE) | 27 | 0 | 6 | nao | High-performance operations for neural network potentials |
| openmm | [openmm-setup](https://github.com/openmm/openmm-setup) | 84 | HTML | licenca propria (ver LICENSE) | 5 | 0 | 0 | nao | An application for configuring and running simulations with OpenMM |
| pangeo-data | [awesome-open-climate-science](https://github.com/pangeo-data/awesome-open-climate-science) | 603 |  | CC0-1.0 | 17 | 0 | 0 | nao | Awesome Open Atmospheric, Ocean, and Climate Science  |
| pangeo-data | [climpred](https://github.com/pangeo-data/climpred) | 258 | Python | MIT | 12 | 0 | 0 | sim | :earth_americas: Verification of weather and climate forecasts :earth_africa: |
| pangeo-data | [xESMF](https://github.com/pangeo-data/xESMF) | 252 | Python | MIT | 59 | 0 | 1 | nao | Universal Regridder for Geospatial Data |
| pangeo-data | [scikit-downscale](https://github.com/pangeo-data/scikit-downscale) | 196 | Python | Apache-2.0 | 13 | 0 | 0 | nao | Statistical climate downscaling in Python |
| pangeo-data | [rechunker](https://github.com/pangeo-data/rechunker) | 181 | Jupyter Notebook | MIT | 50 | 0 | 0 | nao | Disk-to-disk chunk transformation for chunked arrays. |
| pangeo-data | [pangeo-docker-images](https://github.com/pangeo-data/pangeo-docker-images) | 140 | Dockerfile | MIT | 34 | 1 | 3 | nao | Docker Images For Pangeo Jupyter Environment |
| pangeo-data | [pangeo.io](https://github.com/pangeo-data/pangeo.io) | 18 | JavaScript | Apache-2.0 | 18 | 2 | 9 | nao | Pangeo Website |
| precice | [precice](https://github.com/precice/precice) | 979 | C++ | LGPL-3.0 | 241 | 15 | 5 | sim | A coupling library and ecosystem for partitioned multi-physics and multi-scale s |
| precice | [openfoam-adapter](https://github.com/precice/openfoam-adapter) | 173 | C++ | GPL-3.0 | 50 | 4 | 1 | sim | OpenFOAM-preCICE adapter |
| precice | [tutorials](https://github.com/precice/tutorials) | 143 | C | LGPL-3.0 | 69 | 6 | 4 | sim | Various tutorial cases for the coupling library preCICE with real solvers. These |
| precice | [calculix-adapter](https://github.com/precice/calculix-adapter) | 60 | C | GPL-3.0 | 11 | 3 | 0 | nao | preCICE-adapter for the CSM code CalculiX |
| precice | [fenics-adapter](https://github.com/precice/fenics-adapter) | 36 | Python | LGPL-3.0 | 20 | 0 | 4 | nao | preCICE-adapter for the open source computing platform FEniCS |
| precice | [precice.github.io](https://github.com/precice/precice.github.io) | 31 | HTML | MIT | 45 | 2 | 0 | nao | The website of preCICE |
| precice | [python-bindings](https://github.com/precice/python-bindings) | 30 | Cython | LGPL-3.0 | 29 | 2 | 1 | nao | Python language bindings for preCICE |
| precice | [micro-manager](https://github.com/precice/micro-manager) | 24 | Python | LGPL-3.0 | 32 | 5 | 2 | sim | A manager tool to facilitate two-scale coupling in multi-physics simulations usi |
| precice | [dealii-adapter](https://github.com/precice/dealii-adapter) | 21 | C++ | LGPL-3.0 | 8 | 1 | 0 | nao | A coupled structural solver written with the C++ finite element library deal.II |
| precice | [su2-adapter](https://github.com/precice/su2-adapter) | 19 | C++ | LGPL-3.0 | 5 | 0 | 2 | nao | preCICE-adapter for the CFD code SU2 - :heart: Maintainer needed :heart: https:/ |
| precice | [fenicsx-adapter](https://github.com/precice/fenicsx-adapter) | 18 | Python | LGPL-3.0 | 13 | 2 | 4 | nao |  preCICE adapter for the open source computing platform FEniCSx |
| precice | [code_aster-adapter](https://github.com/precice/code_aster-adapter) | 16 | Python | GPL-2.0 | 5 | 0 | 1 | nao | preCICE-adapter for the FEM code code_aster  |
| precice | [PreCICE.jl](https://github.com/precice/PreCICE.jl) | 13 | Julia | LGPL-3.0 | 8 | 1 | 1 | nao | Julia language bindings for preCICE |
| precice | [aste](https://github.com/precice/aste) | 11 | Python | GPL-3.0 | 31 | 6 | 1 | sim | Artificial Solver Testing Environment |
| precice | [vm](https://github.com/precice/vm) | 8 | Shell | MIT | 18 | 0 | 1 | nao | Vagrant box with preCICE and examples preinstalled |
| precice | [dumux-adapter](https://github.com/precice/dumux-adapter) | 7 | C++ | licenca propria (ver LICENSE) | 7 | 1 | 2 | nao |  |
| precice | [matlab-bindings](https://github.com/precice/matlab-bindings) | 5 | MATLAB | LGPL-3.0 | 17 | 0 | 1 | nao | MATLAB language bindings for preCICE |
| su2code | [SU2](https://github.com/su2code/SU2) | 1815 | C++ | licenca propria (ver LICENSE) | 111 | 2 | 0 | nao | SU2: An Open-Source Suite for Multiphysics Simulation and Design |
| su2code | [TestCases](https://github.com/su2code/TestCases) | 73 | GLSL | LGPL-2.1 | 9 | 0 | 0 | nao | Test cases for SU2 |
| su2code | [Tutorials](https://github.com/su2code/Tutorials) | 58 | Python | LGPL-2.1 | 12 | 0 | 0 | nao | Collection of tutorials for SU2 |
| su2code | [su2code.github.io](https://github.com/su2code/su2code.github.io) | 30 | SCSS | LGPL-2.1 | 10 | 0 | 0 | nao | SU2 Project Website |
| tum-pbs | [PhiFlow](https://github.com/tum-pbs/PhiFlow) | 1945 | Python | MIT | 30 | 0 | 0 | sim | A differentiable PDE solving framework for machine learning |
| tum-pbs | [ConFIG](https://github.com/tum-pbs/ConFIG) | 123 | Python | MIT | 2 | 1 | 0 | nao | [ICLR2025 Spotlight] Official implementation of Conflict-Free Inverse Gradients  |
| tum-pbs | [PhiML](https://github.com/tum-pbs/PhiML) | 112 | Python | MIT | 4 | 0 | 0 | sim | Intuitive scientific computing with dimension types for Jax, PyTorch, TensorFlow |
| tum-pbs | [apebench](https://github.com/tum-pbs/apebench) | 109 | Python | MIT | 15 | 0 | 0 | nao | [Neurips 2024] A benchmark suite for autoregressive neural emulation of PDEs. (≥ |
| tum-pbs | [pde-transformer](https://github.com/tum-pbs/pde-transformer) | 90 | Jupyter Notebook | Apache-2.0 | 1 | 0 | 0 | nao | PDE-Transformer is a neural network architecture designed to efficiently process |
| tum-pbs | [PBFM](https://github.com/tum-pbs/PBFM) | 83 | Python | Apache-2.0 | 1 | 0 | 0 | nao | [ICLR 2026] Official implementation of PBFM - Physics-Based Flow Matching |
| tum-pbs | [diffSPH](https://github.com/tum-pbs/diffSPH) | 61 | Jupyter Notebook | Apache-2.0 | 2 | 0 | 0 | nao |  |
| tum-pbs | [PICT](https://github.com/tum-pbs/PICT) | 57 | Python | Apache-2.0 | 6 | 0 | 0 | nao | Official repository of the PICT Solver  |
| tum-pbs | [Tadpole](https://github.com/tum-pbs/Tadpole) | 23 | Python | Apache-2.0 | 1 | 1 | 0 | nao | [NeurIPS2026] Flexible scientific foundation models trained with synthetic onlin |

## Ja contribuidos antes (clones em `forkes/`)

NVIDIA/physicsnemo, PolymathicAI/the_well, jax-md/jax-md, Ceyron/exponax. Ver tambem a anotacao de PRs abertos na memoria do projeto.