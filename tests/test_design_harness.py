from pinneapple_veriphysics.design_harness import DesignVerificationHarness

def make():
    return DesignVerificationHarness([[0,1],[0,1]],[[.5,.5],[.7,.6]],1.0,threshold=.1)

def test_provisional_is_not_verified():
    h=make()
    d=h.check([.5,.5],1.5,.01,check_id="pre")
    assert d.status=="provisional"
    assert h.ledger.status("pre")=="failed"
    assert h.ledger.latest("pre").valid

def test_hard_constraint():
    assert make().check([2,.5],1.9,0).status=="reject"

def test_ood_and_uncertainty():
    assert make().check([0,0],1.8,0).status=="escalate"
    assert make().check([.5,.5],1.15,1.0).status=="escalate"

def test_reference_controls_final_decision():
    h=make()
    bad=h.check([.5,.5],1.5,0,reference=lambda x:1.01,reference_baseline=1.,check_id="bad")
    good=h.check([.5,.5],1.5,0,reference=lambda x:1.2,reference_baseline=1.,check_id="good")
    assert bad.status=="reject"
    assert good.status=="verified"
    assert h.ledger.status("bad")=="failed"
    assert h.ledger.status("good")=="passed"
