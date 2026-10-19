# Physics AI Model Gateway (day 31)

**Live:** https://models.pinneapple.org · **Port:** 8114 · **Prefix:** `MDL`

Publish physical models and run them through one interface: versioned endpoints that check every query against the
model's validity range, log it, and meter it.

## Pain

Trained physical models end up as notebooks, pickles and one-off Flask scripts. Each has its own call format. None
says when a query is outside what the model was trained on, and nobody knows who calls what, how fast it answers, or
how often it extrapolates.

## Input

- A trained model, given in one of these forms:
  - a pre-trained zoo model (three are published at start-up);
  - a `ModelRef` (`pinneapple_data.contracts`) whose URI is `zoo:<name>`, or the http(s) URI of a model behind the same
    API — a Surrogate Studio endpoint or another gateway;
  - an ONNX graph with declared inputs, outputs and validity bounds;
  - a Surrogate Studio `model.zip`.
- The run configuration: OOD policy (warn or refuse), stage (development, staging, production) and an optional
  override reason for a model whose trust decision is REJECT.
- Input data: `{input: value or list}`, a list of row objects, or rows in input order. Up to `MDL_MAX_ROWS` per call.
- The desired version: latest, a stage name, or a version id.

## Output

- **A versioned endpoint**: `POST /v1/models/<name>/predict`, or `/versions/<v>/predict` for one version.
- **Predictions** with a per-row OOD assessment and the reasons. Under `refuse`, out-of-range rows get no prediction;
  under `warn`, they are flagged. A caller may tighten the policy but never relax it.
- **Metadata**: kind, method, stage, trust decision, source, validity range, and the version as a contracts `ModelRef`
  (`/api/models/<name>/model_ref.json`).
- **Operational metrics** per version (`/v1/metrics`): calls, rows, errors, OOD rows, refused rows, latency p50 / p95.
- **Run log** (`/v1/models/<name>/runs`): time, version, caller, rows, OOD rows, refusals, latency and error for every
  call, including rejected ones.
- **Exports**: ONNX / TorchScript for PyTorch surrogates; the published ONNX file for ONNX models.

## Integrations

- REST (`/docs`).
- A Python client with no dependencies beyond the standard library: `pinneapple_registry.gateway_client.GatewayClient`.
- PyTorch (surrogates), ONNX (onnxruntime), TorchScript export.
- The zoo.
- `ModelRef` from the Surrogate Studio, the Benchmark Arena and the other apps.
- API keys and usage metering from the shared runtime.
- JAX models are not supported natively; export them to ONNX first.

## Methods (PINNeAPPle)

- `pinneapple_registry.ModelGateway`, on `ModelStore`: timestamped versions, stages, a REJECT trust report that blocks
  staging and production without an override, kinds `pinneapple.surrogate` / `pinneapple.zoo` / `onnx` / `remote`, the
  run log and metrics.
- `gateway_api.gateway_router`: the REST shape, shared with the Surrogate Studio.
- OOD gates: the surrogate's own gate (training range + nearest-neighbour distance); the validity box of zoo, ONNX and
  remote models.

## Monetization

Metered inferences (free tier: 50 000 rows per month), hosted models (5 free), private deployments with API keys, and
enterprise plans: SSO, a dedicated instance, an SLA, and audit export of the run log.

## Limits

- Publishing a remote model makes this server call that URL, and a `model.zip` is unpickled when loaded (checked
  against the sha256 in its card). Both are admin-only. Only ONNX may be published with an API key.
- The OOD gate sees inputs only. For zoo, ONNX and remote models it is the box of declared ranges, so a query inside
  the box but in a gap of the training data is not flagged. Surrogates from the studio also check the
  nearest-neighbour distance.
- Latency is measured server side. A remote model's latency includes the call to its server; that server's own
  limits and logs also apply.
- One worker per container: the model cache and the run log live in one process. The log is a JSON-lines file on the
  data volume; it is not a time-series database.
- Metering counts rows; it does not bill.
- JAX models must be exported to ONNX first.
