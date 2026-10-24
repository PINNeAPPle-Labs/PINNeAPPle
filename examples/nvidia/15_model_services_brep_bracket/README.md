# 15 From a sentence to an exact, checked part: PINNeAPPle + Model Services + NVIDIA USD BRep

**What.** Two plain-language requests go to `pinneapple-intent` on PINNeAPPle Model Services, through the
OpenAI-format `/v1/chat/completions` with `nvext.guided_json`. This is the same call NVIDIA NIM takes. The service is
the deterministic extractor of `pinneapple_design.intent`: every value carries the quote it was read from, and its
trust report says when a number could not be placed. The client builds only what was APPROVED.

For the accepted request, PINNeAPPle sizes the plate from stock thicknesses (cantilever theory, yield / safety factor).
USD BRep builds the exact L bracket (inner fillet r = t, two M6 clearance holes). PINNeAPPle then verifies the
geometry and checks the beam against its hexahedral finite elements.

**Why.** Text-to-design fails quietly when a number is misread. Here the extractor's own trust report stops the
ambiguous request, and the accepted one ends in an exact solid with its checks.

| Step | Who | Ran here |
|---|---|---|
| extraction with quotes, trust of the extraction | Model Services (`pinneapple-intent`, no network) | yes |
| sizing, finite elements (C3D8I), geometry checks | PINNeAPPle | yes |
| exact bracket, mass, tessellation | USD BRep 1.0.1 | yes |
| an NVIDIA NIM LLM behind the same client | NVIDIA NIM | **no** (needs `NVIDIA_API_KEY`) |

```bash
PINNEAPPLE_BREP_PYTHON=/opt/brep/bin/python python examples/nvidia/15_model_services_brep_bracket/run.py   # ~8 s
```

## Results (this run: 4-core Xeon 2.1 GHz)

- Request 1, "holds 1.5 kN at 120 mm from the wall, safety factor 2":
  - Read as load 1500 N ('1.5 kN'), arm 0.12 m ('120 mm'), SF 2 ('2').
  - **APPROVED**; the schema is satisfied; round trip 35 ms.
- Request 2, "holding 1.5 kN on a 120 mm arm, 50 mm wide":
  - The extractor took '50 mm' as the arm and could not place '120 mm'.
  - **REVIEW**, so nothing is built from it. This is a real limit of the deterministic parser, caught by its own
    trust report.
- Sizing for request 1, 80 mm wide: 10 mm plate (beam root stress 123.7 MPa <= 276 / 2).
  - Exact bracket 370.0 g; volume equals the closed form to 2e-9.
  - Mesh chord error <= 2e-5 m; 1.2 mm elements resolve the 3.3 mm hole radius.
- Finite elements (1 040 nodes) give a root stress of 125.1 MPa, 1.1 % from beam theory. Tip deflection is 1.55 mm
  against 1.45 mm: the FE includes the mounting leg's rotation.
- The peak stress in the fillet is not resolved by this model; the bracket template's CalculiX path covers it.
- **Decision: APPROVED.**

Figures in `results/`: `architecture.png`, `request_to_part.png` (the part and both service answers), `checks.png`.
