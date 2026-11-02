"""Core coverage (#1129, toward 1.0) for the trainer's HPC helpers, distributed queries, callbacks, checkpoints,
inference, metric configuration and history plot.

The distributed helpers run for real in a single-process ``gloo`` group on CPU (world size 1). There, a ZeRO
optimizer step must equal a plain optimizer step; PowerSGD must leave the gradients unchanged during its warm-up;
and the Top-K hook must keep exactly k gradient entries. FSDP and CUDA graphs need an accelerator, so on CPU only
their documented refusals are checked.
"""
import math
import os

import pytest

torch = pytest.importorskip("torch")
dist = pytest.importorskip("torch.distributed")

from pinneapple_neural.trainer import callbacks as CB  # noqa: E402
from pinneapple_neural.trainer import checkpoint as CK  # noqa: E402
from pinneapple_neural.trainer import distributed as D  # noqa: E402
from pinneapple_neural.trainer import hpc as H  # noqa: E402
from pinneapple_neural.trainer import inference as INF  # noqa: E402
from pinneapple_neural.trainer import metrics_cfg as MC  # noqa: E402
from pinneapple_neural.trainer import viz as V  # noqa: E402


@pytest.fixture()
def gloo(tmp_path):
    if not dist.is_available() or dist.is_initialized():
        pytest.skip("needs a fresh torch.distributed")
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


# ------------------------------------------------------------------------------------------------ distributed
def test_rank_and_world_size_outside_and_inside_a_process_group(gloo):
    assert D.is_distributed() and D.get_rank() == 0 and D.get_world_size() == 1
    cfg = D.DDPTrainerConfig()
    assert cfg.backend in ("nccl", "gloo") and cfg.grad_clip >= 0


def test_not_distributed_by_default():
    if dist.is_initialized():
        pytest.skip("a process group is active")
    assert not D.is_distributed() and D.get_rank() == 0 and D.get_world_size() == 1


# ------------------------------------------------------------------------------------------------ HPC
def _pair():
    torch.manual_seed(0)
    a = torch.nn.Linear(6, 1)
    b = torch.nn.Linear(6, 1)
    b.load_state_dict(a.state_dict())
    return a, b


def test_zero_optimizer_step_equals_a_plain_step(gloo):
    a, b = _pair()
    za = H.wrap_zero_optimizer(torch.optim.Adam, a.parameters(), lr=0.1)
    pb = torch.optim.Adam(b.parameters(), lr=0.1)
    assert type(za).__name__ == "ZeroRedundancyOptimizer"
    x, y = torch.randn(8, 6), torch.randn(8, 1)
    for m, o in ((a, za), (b, pb)):
        o.zero_grad()
        ((m(x) - y) ** 2).mean().backward()
        o.step()
    assert all(torch.allclose(p, q) for p, q in zip(a.parameters(), b.parameters(), strict=True))


def test_zero_optimizer_without_a_process_group_is_the_plain_optimizer():
    if dist.is_initialized():
        pytest.skip("a process group is active")
    assert type(H.wrap_zero_optimizer(torch.optim.SGD, torch.nn.Linear(2, 1).parameters(), lr=0.1)) is torch.optim.SGD


def test_topk_hook_keeps_exactly_k_gradient_entries(gloo):
    torch.manual_seed(0)
    ddp = torch.nn.parallel.DistributedDataParallel(torch.nn.Linear(10, 1))
    H.register_topk_hook(ddp, compression_ratio=0.3)
    ddp(torch.randn(4, 10)).sum().backward()
    g = torch.cat([p.grad.view(-1) for p in ddp.parameters()])
    assert int((g != 0).sum()) == int(11 * 0.3)                                 # 11 parameters, one bucket


def test_powersgd_hook_is_exact_during_its_warm_up(gloo):
    a, b = _pair()
    ddp = torch.nn.parallel.DistributedDataParallel(a)
    H.register_powersgd_hook(ddp, matrix_approximation_rank=1, start_powersgd_iter=10)
    x = torch.randn(5, 6)
    ddp(x).sum().backward()
    b(x).sum().backward()
    assert all(torch.allclose(p.grad, q.grad) for p, q in zip(a.parameters(), b.parameters(), strict=True))


def test_compression_hooks_warn_on_a_model_without_ddp():
    for fn in (H.register_topk_hook, H.register_powersgd_hook):
        with pytest.warns(UserWarning):
            fn(torch.nn.Linear(2, 1))


def test_fsdp_and_cuda_graphs_need_an_accelerator():
    with pytest.raises(ValueError, match="sharding_strategy"):
        H.wrap_fsdp(torch.nn.Linear(2, 1), H.FSDPConfig(sharding_strategy="NOPE"))
    assert H.FSDPConfig().sharding_strategy == "FULL_SHARD"
    if torch.cuda.is_available():
        pytest.skip("this test describes the CPU-only case")
    with pytest.raises(RuntimeError, match="CUDA"):
        H.CUDAGraphModule(torch.nn.Linear(2, 1))


def test_launch_scripts():
    cfg = H.TorchRunConfig(n_nodes=2, n_procs_per_node=4, master_addr="10.0.0.1", max_restarts=3)
    assert cfg.world_size == 8
    cmd = H.build_torchrun_cmd("train.py", cfg, script_args="--epochs 5")
    for part in ("torchrun", "--nnodes=2", "--nproc_per_node=4", "--master_addr=10.0.0.1", "--max_restarts=3",
                 "train.py --epochs 5"):
        assert part in cmd
    s = H.SLURMConfig(job_name="pinn", n_nodes=2, n_gpus_per_node=4, modules=["cuda/12"], conda_env="pp")
    script = H.build_slurm_script(cmd, s, output_dir="logs")
    lines = script.splitlines()
    assert lines[0] == "#!/bin/bash" and "#SBATCH --job-name=pinn" in lines and "#SBATCH --nodes=2" in lines
    assert any("module load cuda/12" in ln for ln in lines) and cmd in script


def test_profiler_config_and_batch_size_search(tmp_path):
    pc = H.ProfilerConfig(trace_dir=str(tmp_path))
    assert pc.trace_dir == str(tmp_path) and pc.record_shapes
    calls = []

    def fits(bs):
        calls.append(bs)
        return bs <= 300                                                         # "memory" holds 300 samples
    assert H.AutoBatchSizeFinder(fits, lo=1, hi=4096, verbose=False).find() == 300
    assert len(calls) <= 2 + math.ceil(math.log2(4096))                          # binary search
    with pytest.warns(UserWarning):
        assert H.AutoBatchSizeFinder(lambda bs: False, lo=8, hi=64, verbose=False).find() == 0


# ------------------------------------------------------------------------------------------------ callbacks
def test_early_stopping_counts_epochs_without_improvement():
    es = CB.EarlyStopping(monitor="val", patience=2, min_delta=0.1)
    for v in (1.0, 0.95, 0.5, 0.45, 0.44):                                       # improvements need 0.1
        es.update({"val": v})
    assert es.best == 0.5 and es.bad_epochs == 2 and es.stop
    mx = CB.EarlyStopping(monitor="acc", mode="max", patience=5)
    mx.update({"acc": 0.7})
    mx.update({"acc": 0.8})
    mx.update({"other": 1.0})                                                    # missing key: ignored
    assert mx.best == 0.8 and mx.bad_epochs == 0


def test_model_checkpoint_saves_only_improvements(tmp_path):
    m = torch.nn.Linear(2, 1)
    ck = CB.ModelCheckpoint(path=str(tmp_path / "ck" / "best.pt"), monitor="val")
    assert ck.maybe_save(m, {"epoch": 0}, {"val": 1.0})
    assert not ck.maybe_save(m, {"epoch": 1}, {"val": 2.0}) and not ck.maybe_save(m, {}, {})
    assert ck.maybe_save(m, {"epoch": 2}, {"val": 0.5})
    saved = torch.load(ck.path)
    assert saved["best"] == 0.5 and saved["extra"] == {"epoch": 2}


def test_checkpoint_round_trip(tmp_path):
    m = torch.nn.Linear(3, 2)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    ck = CK.Checkpoint(model_state=m.state_dict(), optim_state=opt.state_dict(), cfg={"lr": 0.1},
                       meta={"epoch": 4}, normalizers={"x": [0.0, 1.0]})
    path = os.path.join(str(tmp_path), "sub", "run.pt")
    CK.save_checkpoint(path, ck)
    back = CK.load_checkpoint(path)
    assert back.cfg == {"lr": 0.1} and back.meta == {"epoch": 4} and back.normalizers == {"x": [0.0, 1.0]}
    m2 = torch.nn.Linear(3, 2)
    m2.load_state_dict(back.model_state)
    assert all(torch.equal(p, q) for p, q in zip(m.parameters(), m2.parameters(), strict=True))


# ------------------------------------------------------------------------------------------------ utilities
def test_predict_switches_to_eval_mode():
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.9))
    x = torch.randn(3, 4)
    y1, y2 = INF.predict(m, x, device="cpu"), INF.predict(m, x)
    assert not m.training and torch.equal(y1, y2)                                # no dropout at inference


def test_metrics_from_configuration():
    b = MC.build_metrics_from_cfg(["mse", {"name": "MAE"}])
    out = b.compute(torch.tensor([[1.0], [3.0]]), torch.tensor([[0.0], [0.0]]))
    assert out["mse"] == pytest.approx(5.0) and out["mae"] == pytest.approx(2.0)
    assert set(MC.build_metrics_from_cfg([]).compute(torch.ones(2, 1), torch.ones(2, 1))) >= {"mse", "rel_l2"}
    with pytest.raises(KeyError):
        MC.build_metrics_from_cfg(["nope"])


def test_plot_history_draws_each_key():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hist = [{"epoch": i, "train_total": 1.0 / (i + 1), "val_total": 2.0 / (i + 1)} for i in range(5)]
    fig = V.plot_history(hist, show=False)
    ax = (fig or plt.gcf()).axes[0]
    assert len(ax.get_lines()) == 2 and ax.get_yscale() == "log"
    plt.close("all")
