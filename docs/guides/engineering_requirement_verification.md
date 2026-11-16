# Engineering requirement verification — initial contract

This is an initial, dependency-free bridge for engineering requirements and
external solver/VeriPhysics check results. It is **not** a solver integration,
a standards compliance certification, or a guarantee of physical correctness.

## Example

```python
from pinneapple_physics.engineering_trace import EngineeringTrace, Requirement
from pinneapple_physics.engineering_verification import record_check_result, release_gate

trace = EngineeringTrace()
trace.add_requirement(
    Requirement("P-001", "Maximum pressure", "design-spec#section-2", "Pa",
                upper_bound=1_000_000)
)
assert release_gate(trace)["approved"] is False

# In a real workflow, this result must come from a trusted, persisted check.
record_check_result(
    trace,
    requirement_id="P-001",
    evidence_id="run-001-pressure",
    artifact_uri="runs/run-001/result.json",
    result={"check": "max_pressure", "passed": True,
            "value": 950_000.0, "unit": "Pa"},
)
assert release_gate(trace)["approved"] is True
print(trace.report())
```

## Contract and limitations

- Requirements must carry a source; numeric limits require units.
- A check result must explicitly supply a boolean `passed` and a check name.
- Numeric results must be finite and use the requirement's exact unit string.
  **No unit conversion** is performed; convert with a vetted unit library before recording.
- An unverified or failed requirement blocks release.
- A recorded `passed=True` is a claim by the check producer, **not** proof of
  solver convergence, source authenticity, or engineering certification.
- Multiple checks are currently combined conservatively: any failed check
  blocks approval. Reverification/supersession semantics remain to be designed.
- This adapter does not execute VeriPhysics or validate artifact contents.
  Evidence provenance, signed artifact hashes, check producer identity, and
  stale-evidence invalidation are required before production use.

Follow-up: #1842, #1844, and Engineering API #361.
