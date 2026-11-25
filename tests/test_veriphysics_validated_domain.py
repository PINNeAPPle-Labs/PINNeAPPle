from pinneapple_veriphysics.validated_domain import (
    DomainPolicy,
    GeometryEvidence,
    ValidatedDomain,
)


def test_geometry_domain():
    policy = DomainPolicy(.05, (.01,), .1, .4)
    def evidence(x, error=.01):
        return GeometryEvidence((x,), "v1", str(x), True, error, (.001,), .02)
    domain = ValidatedDomain({"span": (0, 1)}, "v1", policy,
                             [(0.5,)], [evidence(.3), evidence(.7)])
    assert domain.assess({"span": .5})["status"] == "EMPIRICALLY_SUPPORTED"
    assert domain.assess({"span": .5, "camber": .1})["reason"] == "parameter_schema_changed"
    assert domain.assess({"span": 2})["reason"] == "outside_training_bounds"
    assert domain.extend([evidence(.5, .2)]).assess({"span": .5})["status"] == "UNSUPPORTED"
    assert domain.assess({"span": .5})["status"] == "EMPIRICALLY_SUPPORTED"


def test_expanded_domain_requires_new_evidence():
    domain = ValidatedDomain({"span": (0, 1), "camber": (0, 1)}, "v2",
                             DomainPolicy(.05, (.01,), .1, .4), [(.5, .5)])
    assert domain.assess({"span": .5, "camber": .5})["status"] == "UNSUPPORTED"
