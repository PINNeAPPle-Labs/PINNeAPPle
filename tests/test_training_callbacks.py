"""Physics checks during training (#704). The same PhysicsNeMo FNO on Darcy is trained in plain Lightning and in a
PhysicsNeMo training loop: both log identical physics metrics (the checks of pp.validate against reference solutions
of `operator_benchmarks.darcy_2d`). A physics loss with the wrong sign makes the residual grow while the loss falls:
the early warning fires long before the end of training; the correct weight raises none."""
from __future__ import annotations

import copy
import json
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("physicsnemo")

from pinneapple_analysis.uncertainty.residual_bound import ResidualLoss, darcy_problem  # noqa: E402
from pinneapple_neural.callbacks import (  # noqa: E402
    PhysicsMonitor,
    PhysicsNeMoHook,
    PhysicsWarning,
)
from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob  # noqa: E402

STEPS, EVERY = 400, 40


@pytest.fixture(scope="module")
def darcy():
    a = ob.threshold_coefficient(ob.periodic_grf(72, (16, 16), alpha=2.0, tau=3.0, seed=1))
    u = np.stack([ob.darcy_2d(x, 1.0) for x in a])
    return a, u, ResidualLoss([darcy_problem(x) for x in a[:64]])


class Scaled(torch.nn.Module):
    def __init__(self, net, c):
        super().__init__()
        self.net, self.c = net, c

    def forward(self, x):
        return self.net(x) * self.c


def fno(u):
    from physicsnemo.models.fno import FNO
    torch.manual_seed(0)
    return Scaled(FNO(1, 1, decoder_layer_size=16, dimension=2, latent_channels=16, num_fno_layers=2,
                      num_fno_modes=6, padding=4), float(np.abs(u).max()))


def loader(a, u):
    ds = torch.utils.data.TensorDataset(torch.tensor(a[:64, None], dtype=torch.float32),
                                        torch.tensor(u[:64, None], dtype=torch.float32), torch.arange(64))
    return torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False)


def loss_fn(net, batch, RL, w, u2):
    x, y, idx = batch
    p = net(x)
    return ((p - y) ** 2).mean() / u2 + w * RL(p, idx)


def monitor(a, u, **kw):
    return PhysicsMonitor("darcy_2d", reference=(a[64:], u[64:]), every=EVERY, patience=3, **kw)


def physicsnemo_loop(darcy, w, mon, steps=STEPS):
    a, u, RL = darcy
    net = fno(u)
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    hook, u2, step = PhysicsNeMoHook(mon), float((u[:64] ** 2).mean()), 0
    while step < steps:
        for batch in loader(a, u):
            loss = loss_fn(net, batch, RL, w, u2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            if hook(net, step, loss) or step == steps:
                return net, step
    return net, step


def test_lightning_and_physicsnemo_log_identical_physics_metrics(darcy, tmp_path):
    L = pytest.importorskip("lightning.pytorch")
    from lightning.pytorch.loggers import CSVLogger

    from pinneapple_neural.callbacks import PhysicsValidationCallback
    a, u, RL = darcy
    u2 = float((u[:64] ** 2).mean())

    class Module(L.LightningModule):
        def __init__(self):
            super().__init__()
            self.net = fno(u)

        def forward(self, x):
            return self.net(x)

        def training_step(self, batch, i):
            return loss_fn(self.net, batch, RL, 0.01, u2)

        def configure_optimizers(self):
            return torch.optim.Adam(self.net.parameters(), 3e-3)

    m_pl = monitor(a, u, report_path=tmp_path / "pl" / "report.json")
    trainer = L.Trainer(max_steps=STEPS, accelerator="cpu", callbacks=[PhysicsValidationCallback(m_pl)],
                        logger=CSVLogger(tmp_path, name="csv"), enable_checkpointing=False,
                        enable_progress_bar=False, enable_model_summary=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.fit(Module(), loader(a, u))
        m_nemo = monitor(a, u)
        physicsnemo_loop(darcy, 0.01, m_nemo)
    assert [h["step"] for h in m_pl.history] == [h["step"] for h in m_nemo.history] == list(range(EVERY, STEPS + 1,
                                                                                                    EVERY))
    for hp, hn in zip(m_pl.history, m_nemo.history, strict=True):
        assert hp["metrics"].keys() == hn["metrics"].keys()
        for k in hp["metrics"]:
            assert hp["metrics"][k] == pytest.approx(hn["metrics"][k], rel=1e-5, abs=1e-9), (hp["step"], k)
    assert "physics/residual.energy_bound" in m_pl.history[0]["metrics"]
    assert not m_pl.alerts and not m_nemo.alerts                        # the correct weight raises no warning
    last = m_pl.history[-1]["metrics"]
    assert last["physics/reference.rel_l2"] < 0.15 and last["physics/residual.energy_bound"] < 0.6
    # logged to the Trainer's logger, and the report and trust card written at the end
    import csv
    rows = list(csv.DictReader(open(next((tmp_path / "csv").rglob("metrics.csv")))))
    logged = [r for r in rows if r.get("physics/residual.energy_bound")]
    assert len(logged) == STEPS // EVERY
    assert json.loads((tmp_path / "pl" / "report.json").read_text())["schema"].startswith("validation-report")
    card = json.loads((tmp_path / "pl" / "report.trust.json").read_text())
    assert card["decisao"] in ("APPROVED", "REVIEW", "REJECT") and card["training_alerts"] == []


def test_wrong_residual_weight_triggers_the_early_warning(darcy):
    a, u, _ = darcy
    mon = monitor(a, u)
    with pytest.warns(PhysicsWarning, match="worsened for 3 consecutive checks"):
        physicsnemo_loop(darcy, -0.01, mon)                             # sign error: the residual is maximised
    first = min(x["step"] for x in mon.alerts)
    assert first == 4 * EVERY < STEPS                                   # the earliest step patience 3 allows
    assert {x["check"] for x in mon.alerts} >= {"residual.energy_bound", "reference.rel_l2"}
    stop = monitor(a, u, action="stop")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, steps = physicsnemo_loop(darcy, -0.01, stop)
    assert stop.should_stop and steps == first                          # action="stop" ends training there


def test_hugging_face_trainer_callback(darcy, tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("accelerate")                                   # the Trainer needs it with PyTorch
    from transformers import Trainer, TrainingArguments

    from pinneapple_neural.callbacks import PhysicsTrainerCallback
    a, u, RL = darcy
    u2 = float((u[:64] ** 2).mean())

    class HFModel(torch.nn.Module):
        def __init__(self, w):
            super().__init__()
            self.net, self.w = fno(u), w

        def forward(self, x, y, idx):
            return {"loss": loss_fn(self.net, (x, y, idx), RL, self.w, u2)}

    ds = [{"x": torch.tensor(a[i, None], dtype=torch.float32), "y": torch.tensor(u[i, None], dtype=torch.float32),
           "idx": torch.tensor(i)} for i in range(64)]
    results = {}
    for w in (0.01, -0.01):
        mon = monitor(a, u, action="stop", model_fn=lambda m: m.net)
        args = TrainingArguments(output_dir=str(tmp_path / str(w)), max_steps=STEPS, per_device_train_batch_size=8,
                                 learning_rate=3e-3, lr_scheduler_type="constant", logging_steps=1,
                                 save_strategy="no", report_to=[], use_cpu=True, disable_tqdm=True)
        tr = Trainer(model=HFModel(w), args=args, train_dataset=ds)
        PhysicsTrainerCallback(mon).attach(tr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tr.train()
        logged = [h for h in tr.state.log_history if "physics/residual.energy_bound" in h]
        results[w] = (mon, tr.state.global_step, logged)
    mon, steps, logged = results[0.01]
    assert steps == STEPS and not mon.alerts and len(logged) == STEPS // EVERY
    mon, steps, logged = results[-0.01]
    assert mon.should_stop and steps < STEPS and mon.alerts[0]["step"] == steps
    assert mon.report is not None                                       # the final report ran at train end
    # the metrics are those of pp.validate on the same weights
    net = copy.deepcopy(results[0.01][0].model_fn(tr.model))
    assert mon.evaluate(tr.model) == PhysicsMonitor("darcy_2d", reference=(a[64:], u[64:])).evaluate(net)


class _Scripted:
    """A monitor whose metrics follow a script (no model): to test the early-warning rule by itself."""

    def __init__(self, values, **kw):
        self.mon = PhysicsMonitor("darcy_2d", every=1, patience=2, **kw)
        it = iter(values)
        self.mon.evaluate = lambda module: next(it)


def test_early_warning_follows_each_checks_direction_and_checks_a_step_once():
    with warnings.catch_warnings():
        warnings.simplefilter("error", PhysicsWarning)
        # pinn.trivial rises as a PINN learns (higher is better): no alarm while the loss falls
        good = _Scripted([{"physics/pinn.trivial": v} for v in (0.0, 0.3, 0.6, 0.9)]).mon
        for k, loss in enumerate((4.0, 3.0, 2.0, 1.0), 1):
            good.step(None, k, loss)
        assert not good.alerts
        # the same rise of an error metric does alarm
        bad = _Scripted([{"physics/residual.energy_bound": v} for v in (0.0, 0.3, 0.6, 0.9)]).mon
        with pytest.raises(PhysicsWarning):
            for k, loss in enumerate((4.0, 3.0, 2.0, 1.0), 1):
                bad.step(None, k, loss)
    # gradient accumulation: the same global step reported k times is checked once
    acc = _Scripted([{"physics/residual.energy_bound": 1.0}] * 2).mon
    assert acc.step(None, 5, 1.0) is not None and acc.step(None, 5, 0.9) is None and len(acc.history) == 1
