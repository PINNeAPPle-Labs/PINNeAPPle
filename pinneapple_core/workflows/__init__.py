"""Simulation workflows: a DAG of steps, executors and adapters as plugins, traceable runs, datasets from runs.

It orchestrates tools that already exist (solvers, Docker, Slurm); it does not replace them.

==========================  ========================================================================================
``Workflow``, ``Step``      the DAG: dependencies, order, cycle check, JSON; ``workflow_from_cases`` per case table
``WorkflowRunner``          runs, resume, per-step state and log, retries, skip-done by content hash, cancel
``Executor`` + registry     where commands run: ``local``, ``docker``, ``slurm`` (sbatch scripts, dry run by
                            default); ``register_executor(name, cls)`` adds one
``Adapter`` + registry      what a step kind does: ``command``, ``python``, ``openfoam``, ``calculix``;
                            ``register_adapter(adapter)`` adds one (a GUI-driven step, another solver)
``dataset_from_run``        finished runs -> ``DatasetSpec`` + table (CSV / Parquet / HDF5) for training
``TEMPLATES``               whitelisted workflows for public servers (Python slab heating, CalculiX beam,
                            OpenFOAM cavity)
==========================  ========================================================================================

    from pinneapple_core.workflows import TEMPLATES, WorkflowRunner, dataset_from_run
    t = TEMPLATES["slab_heat"]
    cases = t.example_cases(6)
    runner = WorkflowRunner("work/runner")
    run = runner.run(t.workflow(cases))
    spec, path = dataset_from_run(runner, run["id"], cases, output_step="solve", path="work/slab.parquet")

**Plugging in** (the Engineering GUI Agent does this for GUI-driven steps)::

    from pinneapple_core.workflows import Adapter, Executor, register_adapter, register_executor

    class GuiAdapter(Adapter):
        kind = "gui"
        def available(self, step=None): return True, ""
        def run(self, ctx):                     # ctx: StepContext (workdir, params, upstream, executor, log)
            ...                                 # drive the GUI, write files into ctx.workdir
            return {"screenshot": "after.png"}  # JSON-able outputs, stored and hashed for skip-done

    register_adapter(GuiAdapter())              # steps with kind="gui" now run
"""
from .adapters import (
    ADAPTERS,
    Adapter,
    CalculiXAdapter,
    CommandAdapter,
    OpenFOAMAdapter,
    PythonAdapter,
    StepCancelled,
    StepContext,
    StepFailed,
    get_adapter,
    register_adapter,
)
from .dag import Step, Workflow, WorkflowError, fill, workflow_from_cases
from .datasets import dataset_from_run, id_column_of, read_cases
from .engine import STATES, WorkflowRunner, step_hash
from .executors import (
    EXECUTORS,
    DockerExecutor,
    ExecResult,
    Executor,
    LocalExecutor,
    SlurmExecutor,
    get_executor,
    register_executor,
)

__all__ = [
    "Workflow", "Step", "WorkflowError", "fill", "workflow_from_cases", "WorkflowRunner", "STATES", "step_hash",
    "Executor", "ExecResult", "LocalExecutor", "DockerExecutor", "SlurmExecutor", "EXECUTORS", "register_executor",
    "get_executor", "Adapter", "StepContext", "StepFailed", "StepCancelled", "CommandAdapter", "PythonAdapter",
    "OpenFOAMAdapter", "CalculiXAdapter", "ADAPTERS", "register_adapter", "get_adapter", "dataset_from_run",
    "read_cases", "id_column_of", "TEMPLATES", "Template",
]


def __getattr__(name):
    # the templates build ParameterSpaces (pinneapple_data); load them on first use to keep this import light
    if name in ("TEMPLATES", "Template"):
        from . import templates
        return getattr(templates, name)
    raise AttributeError(name)
