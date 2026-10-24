# 06 Zoo models as NIM-style services: PINNeAPPle + PINNeAPPle Model Services

**What.** `pinneapple_tools.model_services` serves the PINNeAPPle zoo (airfoil, heat sinks, rover wheel contact),
a physics-check service (`pp.trust_card`) and the intent extractor on NVIDIA NIM's routes: `GET /v1/health/ready`,
`GET /v1/models` (OpenAI list format with the model card), `POST /v1/infer`, `POST /v1/chat/completions` (with
`nvext.guided_json`). Every answer carries PINNeAPPle's trust report for *that* query. A client sweeps the airfoil
surrogate (NACA 2412, Re 4e6) from -6 to 20 degrees, beyond its validity range on purpose.

**Why.** A model endpoint that only returns numbers hides when it is extrapolating. These services answer with the
checks of their own answer (card metrics reproduced, physics checks, validity range), standard library only, CPU,
no key. The same `ModelServiceClient` talks to NVIDIA NIM on build.nvidia.com with `NVIDIA_API_KEY`.

| Step | Who | Ran here |
|---|---|---|
| zoo models, `pp.zoo.validate` per query | PINNeAPPle | yes |
| server (NIM routes), client | PINNeAPPle Model Services | yes |
| the same client against NVIDIA NIM (`ModelServiceClient(NIM_URL, api_key=...)`) | NVIDIA NIM | **no** (needs `NVIDIA_API_KEY`) |

```bash
python examples/nvidia/06_model_services_zoo/run.py                                  # ~7 s
python -m pinneapple_tools.model_services serve --port 8000                           # the server alone
docker compose -f docker/model_services/compose.yaml up --build                       # as a container
```

## Results (this run: 4-core Xeon 2.1 GHz)

- 6 services listed; NACA 2412 at 4 deg: Cl **0.6615** against OpenFOAM's 0.6625 (0.15 %); answer APPROVED.
- 27 angles: all 17 inside the card's validity range (-2 to 14 deg) APPROVED; the 10 outside (-6 to -3, 15 to 20)
  answered **REVIEW** with the extrapolation warning, none APPROVED.
- Heat-sink service (12 fins, 30 x 1.5 mm, 3 m/s, 60 W): R_base 0.582 K/W, dp 8.0 Pa, APPROVED.
- Unknown model -> 404, missing inputs -> 400. Round trip per request (prediction + trust report): median **35 ms**,
  p95 50 ms. **Decision: APPROVED.**

Figures in `results/`: `architecture.png`, `sweep.png` (answers coloured by decision, latency), `checks.png`.
