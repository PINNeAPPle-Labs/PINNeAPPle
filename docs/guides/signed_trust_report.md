# A trust report that can be re-verified later

A `TrustReport` decides whether a result may be deployed (`APPROVED` / `REVIEW` / `REJECT`). Once it leaves the
process that produced it — attached to a model card, sent to a client, archived with a simulation — the question
changes: *is this still the report of this model, and is the number in it still what the model does?*

A signature alone answers neither. It says who signed; it does not say that the signer's number was right, that the
model is the one signed, or that the inputs were the ones used. `SignedTrustReport`
(`pinneapple_analysis.trust.signed_report`) signs the report **together with the model and the inputs** and stores the
physics residual, and verification recomputes that residual from the loaded files.

| layer | what it checks | what it catches on its own |
|---|---|---|
| DSSE signature (`pinneapple_security.signing`) | the payload: the report, the SHA-256 of the artifacts, the residual spec and its recorded value | the report or its record was edited |
| SHA-256 of the model and of the inputs | the bytes are the signed ones, and their base64 is the canonical encoding that was written | the model was swapped or the inputs changed |
| **recomputed physics residual** | the model's own prediction on the signed grid, by the field card's grid check | a signed payload whose number the model does not have — a stale tool, a different run, a hand-typed table |

The third layer is the point: the first two are what the library already did for files. A payload that is internally
consistent and correctly signed still fails if the recorded residual is not the one the model and inputs produce.

## Usage

```python
import pinneapple_security.signing as signing
from pinneapple_analysis.trust import ResidualSpec, SignedTrustReport, TrustReport

bundle = SignedTrustReport.create(
    report,                                     # the TrustReport: checks, weights, decision inputs
    model=model,                                # a torch.nn.Module: points (N, ndim) -> values (N,)
    inputs={"x": grid_x, "y": grid_y},          # the tensor-product grid the residual is taken on
    residual=ResidualSpec(pde="laplace"),       # what to recompute, as data (never code from the bundle)
    key=signing.generate_key(),                 # Ed25519; signing.hmac_key() needs no dependency
)
bundle.save("run-0417.pptrust")

checked = SignedTrustReport.load("run-0417.pptrust").verify(public_key)
checked.ok, checked.residual_recorded, checked.residual_recomputed
checked.report.decisao
```

Verification returns the numbers it checked (`residual_recorded`, `residual_recomputed`, `rel_error`) and the report as
signed. A failure raises a `SignedReportError` subclass that names the layer: `SignatureFailure`, `ArtifactMismatch`,
`ResidualMismatch`, `ResidualNotComputable`.

* **The model** is serialised as a state dict. Verification rebuilds it from the class path recorded in the payload,
  or from `model_factory=lambda: MyNet(...)` when the class needs constructor arguments — pass the factory for a model
  that cannot be built with no arguments.
* **The residual** is the one `pp.trust_card` uses: `laplace(u) = f` / `poisson`, or `u_t = alpha laplace(u) + f`
  with `pde="heat"`, by three-point finite differences on the model's prediction, rms over the interior divided by the
  rms of the sum of the terms that must balance. The spec is data (`pde`, `alpha`, `time_coord`, an optional `source`
  array, `rel_tol`), so verifying a bundle never runs code the bundle names beyond building the model.
* **`rel_tol`** (1e-6 by default) is tolerance for round-off of the same arithmetic in another process, not a
  modelling tolerance. A model that drifted by 0.04 % fails it.

## What it does not do

Verification imports the class the payload names (or calls your factory), so **a bundle is exactly as safe as the key
that signed it**: verify with the public key of a signer you trust, and treat an unverified bundle as a program rather
than as data. It also cannot tell you *why* a residual is what it is: a `ResidualMismatch` says the number and the
model disagree, and the report's own checks say what was measured.

## Measured

`tests/pinneapple_analysis/test_signed_trust_report.py` holds the acceptance test of #644: a bundle whose recorded
residual is recomputed on load, and a property test over 100 random single-bit flips stratified over the three parts
of the file (the signed payload, the model, the inputs) — every flip fails verification. Measured once over the whole
file, **39184 single-bit flips at every byte position of a 4898-byte bundle all raised a `SignedReportError`**.

Two tests cover what a hash cannot: a payload whose residual was changed and then **re-signed with the right key**
(`ResidualMismatch`, the numbers in the message), and a **model whose weights drifted with the SHA-256 and the
signature updated** (`ResidualMismatch`, 0.04 % apart).

```bash
pytest tests/pinneapple_analysis/test_signed_trust_report.py -q
```
