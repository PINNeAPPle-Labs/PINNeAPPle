# Fail-closed trust policy

The opt-in `pinneapple_veriphysics.trust_policy` module separates scientific
measurements from operational authorization. Declare mandatory check IDs in a
versioned policy, bind each measured check to the SHA-256 of the artifact, and
evaluate it for a specific scope and expiry. Only an approved, unexpired decision
matching the artifact, scope and policy version permits use. Missing, ambiguous,
inconclusive or mismatched evidence blocks use. A failed mandatory check rejects
the artifact regardless of aggregate confidence scores.

The caller must authenticate evidence, define the scope and required checks,
persist the decision, and enforce it at each workflow transition. This module
does not sign evidence, perform independent review, or automatically protect
existing workflow entry points. Those integrations require separate changes.
