"""One registry protocol for every extensible object type (#200): the same suite runs for all ten kinds."""
import numpy as np
import pytest

import pinneapple as pp
import pinneapple_core.registry as registry
from pinneapple_core.registry import AlreadyRegistered, AmbiguousKind, UnknownKind, UnknownName, WrongType

PROBLEM = lambda n=8: pp.get_preset("poisson_2d", **{})  # noqa: E731 - a preset factory stand-in, replaced per test


def _preset(**kwargs):
    """A preset factory: a callable that builds a ProblemSpec."""
    from pinneapple_physics.pde_environment.builder import ProblemBuilder

    return (ProblemBuilder("registry_coupon").domain(x=(0, 1)).fields("u").pde("poisson")
            .bc("dirichlet", field="u", value=0.0, on="x_min")
            .bc("dirichlet", field="u", value=1.0, on="x_max").build())


def _solver(problem, **options):
    """A solve method: a callable ``(problem, **options) -> Solution``."""
    raise pp.MethodNotAvailable("this is the registry coupon, it never solves anything")


def _operator(values, mesh=None):
    return np.asarray(values) * 2.0


def _benchmark(**options):
    return {"ran": True}


def _model_class():
    import torch

    class RegistryCoupon(torch.nn.Module):
        def __init__(self, width: int = 4):
            super().__init__()
            self.net = torch.nn.Linear(2, width)

        def forward(self, x):
            return self.net(x)

    return RegistryCoupon


def _geometry():
    from pinneapple_core.geometry import Geometry

    return Geometry.box([0.0, 0.0], [1.0, 1.0])


def _material():
    from pinneapple_design.intent.materials import MATERIALS

    return next(iter(MATERIALS.values()))


def _dataset_info():
    from pinneapple_data.datasets.registry import DatasetInfo

    return DatasetInfo(id="registry_coupon", name="Registry coupon", category="physics",
                       description="a dataset that exists only for this test", fields=["u"])


def _dataset_loader(**kwargs):
    return {"u": np.zeros((4, 2))}


def _optimizer():
    from pinneapple_core.optim import PhysicsOptimizer

    class RegistryOptimizer(PhysicsOptimizer):
        """A minimiser coupon: same interface, no cleverness of its own."""

    return RegistryOptimizer


def _analysis():
    """An analysis coupon: a declared calculation, the object ``kind="analysis"`` takes."""
    from pinneapple_core.analysis import Analysis, InputSpec, OutputSpec

    def coupon(x):
        return {"y": 2.0 * x}

    return Analysis(id="registry_coupon_analysis", function=coupon, label="registry coupon",
                    inputs={"x": InputSpec(label="the only input")},
                    outputs={"y": OutputSpec(unit="-", label="the double")}, example={"x": 1.0})


SAMPLES = {                       # kind -> (object, extra keyword arguments for register, a wrong object for the kind)
    "physics": (_preset, {}, 42),
    "model": (_model_class(), {}, "not a model"),
    "solver": (_solver, {}, object()),
    "operator": (_operator, {}, "not an operator"),
    "dataset": (_dataset_info(), {"loader": _dataset_loader}, object()),
    "benchmark": (_benchmark, {}, 3.5),
    "geometry": (_geometry(), {}, "not a geometry"),
    "material": (_material(), {}, "not a material"),
    "optimizer": (_optimizer(), {}, 7),
    "analysis": (_analysis(), {}, 11),
}


def _cleanup(kind: str, name: str) -> None:
    """Remove a coupon from the unified registry *and* from the library home it reached."""
    try:
        pp.unregister(kind, name)
    except UnknownName:
        pass


@pytest.fixture(params=sorted(SAMPLES))
def kind(request):
    k = request.param
    name = f"registry_coupon_{k}"
    _cleanup(k, name)
    yield k, name, *SAMPLES[k]
    _cleanup(k, name)


def test_every_kind_is_declared_once_with_what_it_accepts():
    assert registry.kinds() == ["analysis", "benchmark", "dataset", "geometry", "material", "model", "operator",
                                "optimizer", "physics", "solver"]
    for name, reg in registry.KINDS.items():
        assert reg.kind == name and reg.accepts_doc and reg.home
    assert pp.registry is registry


def test_register_list_resolve_and_unregister_are_the_same_for_every_kind(kind):
    k, name, obj, extra, _wrong = kind
    assert name not in pp.registered(k)

    returned = pp.register(obj, kind=k, name=name, **extra)
    assert returned is obj, "register returns the object, so it can also be used as a decorator"
    assert name in pp.registered(k) and isinstance(pp.registered(k), list)
    assert pp.resolve(k, name) is obj

    pp.unregister(k, name)
    assert name not in pp.registered(k)
    with pytest.raises(UnknownName):
        pp.resolve(k, name)


def test_the_object_of_the_wrong_type_is_refused_naming_what_the_kind_takes(kind):
    k, name, _obj, _extra, wrong = kind
    with pytest.raises(WrongType) as exc:
        pp.register(wrong, kind=k, name=name)
    assert k in str(exc.value) and registry.KINDS[k].accepts_doc in str(exc.value)
    assert name not in pp.registered(k)


def test_a_duplicate_is_refused_and_replace_overwrites(kind):
    k, name, obj, extra, _wrong = kind
    pp.register(obj, kind=k, name=name, **extra)
    with pytest.raises(AlreadyRegistered):
        pp.register(obj, kind=k, name=name, **extra)
    def marker_fn(**options):
        return "the replacement coupon"

    marker = marker_fn if k in ("physics", "solver", "operator", "benchmark") else obj
    if k == "optimizer":
        from pinneapple_core.optim import PhysicsOptimizer

        class ReplacementOptimizer(PhysicsOptimizer):
            """The object the replace=True half of this test swaps in."""

        marker = ReplacementOptimizer
    if k == "analysis":
        from pinneapple_core.analysis import Analysis, InputSpec, OutputSpec

        def replacement(x):
            return {"y": 3.0 * x}

        marker = Analysis(id=name, function=replacement, label="the replacement coupon",
                          inputs={"x": InputSpec()}, outputs={"y": OutputSpec()}, example={"x": 1.0})
    pp.register(marker, kind=k, name=name, replace=True, **extra)
    assert pp.resolve(k, name) is marker
    pp.unregister(k, name)
    assert name not in pp.registered(k)


def test_a_registered_solver_is_reachable_from_solve_and_a_preset_from_get_preset():
    name = "registry_coupon_solver"
    pp.register(_solver, kind="solver", name=name)
    try:
        assert name in pp.list_methods(), "registering a solver reaches pp.solve's own list"
        with pytest.raises(pp.MethodNotAvailable, match="registry coupon"):
            pp.solve(pp.get_preset("laplace_2d"), name)
    finally:
        pp.unregister("solver", name)
    assert name not in pp.list_methods()


def test_the_bridge_is_recorded_and_a_missing_one_is_reported_not_raised():
    name = "registry_coupon_operator"
    pp.register(_operator, kind="operator", name=name)
    try:
        assert registry.KINDS["operator"].describe(name)["home"] == "this module"
    finally:
        pp.unregister("operator", name)
    # a dataset without its loader cannot reach DatasetRegistry: the object is still registered, and says why not
    info = _dataset_info()
    pp.register(info, kind="dataset")
    try:
        described = registry.KINDS["dataset"].describe("registry_coupon")
        assert described["bridge"].startswith("unavailable: ValueError") and "loader" in described["bridge"]
        assert pp.resolve("dataset", "registry_coupon") is info
    finally:
        pp.unregister("dataset", "registry_coupon")


def test_inference_only_when_the_type_settles_it():
    cls = _model_class()
    pp.register(cls, name="registry_coupon_inferred_model")          # a torch Module subclass is a model
    try:
        assert "registry_coupon_inferred_model" in pp.registered("model")
        assert "registry_coupon_inferred_model" in pp.list_models(), "the model registry is the same list"
    finally:
        pp.unregister("model", "registry_coupon_inferred_model")

    with pytest.raises(AmbiguousKind, match="pass kind"):
        pp.register(_operator, name="registry_coupon_ambiguous")     # a plain callable could be three kinds


def test_the_decorator_form_registers_and_returns_the_function():
    @pp.register(kind="operator", name="registry_coupon_decorated")
    def coupon(values, mesh=None):
        return np.asarray(values)

    try:
        assert pp.resolve("operator", "registry_coupon_decorated") is coupon
        assert np.allclose(coupon(np.ones(3)), np.ones(3))
    finally:
        pp.unregister("operator", "registry_coupon_decorated")


def test_unknown_kind_and_unknown_name_say_what_is_available():
    with pytest.raises(UnknownKind, match="known kinds"):
        pp.registered("nonsense")
    with pytest.raises(UnknownName, match="is not registered"):
        pp.resolve("operator", "nothing_here")


def test_the_library_ships_names_in_every_bridged_kind():
    for k in ("physics", "model", "solver", "dataset", "material"):
        assert pp.registered(k), f"{k} should list what the library already has"
    assert "fem" in pp.registered("solver") and pp.resolve("solver", "fem") is not None
    assert "poisson_2d" in pp.registered("physics")


def test_a_registered_material_and_dataset_reach_their_homes():
    material = _material()
    pp.register(material, kind="material", name="registry_coupon_material")
    try:
        from pinneapple_design.intent.materials import MATERIALS

        assert MATERIALS["registry_coupon_material"] is material
    finally:
        pp.unregister("material", "registry_coupon_material")
        assert "registry_coupon_material" not in MATERIALS

    info = _dataset_info()
    pp.register(info, kind="dataset", loader=_dataset_loader)
    try:
        from pinneapple_data.datasets.registry import DatasetRegistry

        assert info.id in DatasetRegistry.ids()
        assert DatasetRegistry.load(info.id)["u"].shape == (4, 2)
    finally:
        pp.unregister("dataset", info.id)
        assert info.id not in DatasetRegistry.ids()
