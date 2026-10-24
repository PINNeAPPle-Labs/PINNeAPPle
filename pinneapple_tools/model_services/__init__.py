"""PINNeAPPle Model Services: ready-made PINNeAPPle models as local microservices with the shape of NVIDIA NIM.

Each service is a model with a card (task, inputs, outputs, validity range, measured metrics, references) and a
trust function; every answer carries PINNeAPPle's trust report (decision APPROVED / REVIEW / REJECT with its checks),
and a query outside the validity range says so. The HTTP routes follow NIM's: ``GET /v1/health/ready``,
``GET /v1/health/live``, ``GET /v1/models`` (OpenAI list format), ``POST /v1/infer`` (``{"model", "inputs"}``) and,
for language services, ``POST /v1/chat/completions`` (OpenAI format, with ``nvext.guided_json``).

    from pinneapple_tools.model_services import default_registry, serve_in_thread, ModelServiceClient
    server, url = serve_in_thread(default_registry())
    c = ModelServiceClient(url)
    c.models(); c.infer("airfoil_cst_rans_mlp", {...})        # outputs + trust
    ModelServiceClient("https://integrate.api.nvidia.com/v1", api_key=os.environ["NVIDIA_API_KEY"]).chat(...)

``python -m pinneapple_tools.model_services serve --port 8000`` runs the server; ``docker/model_services`` has a
Dockerfile and a compose file. Built-in services: every model of the PINNeAPPle zoo (``pp.zoo``, trust from
``pp.zoo.validate``), physics checks of any field (``pp.trust_card``), and ``pinneapple-intent``, the deterministic
specification extractor of ``pinneapple_design.intent`` behind the chat route (no network, no key: every value it
returns comes with the quote it was read from).
"""
from .client import ModelServiceClient, ServiceError
from .registry import ModelService, Registry, default_registry
from .server import make_server, serve_in_thread

__all__ = ["ModelService", "Registry", "default_registry", "make_server", "serve_in_thread", "ModelServiceClient",
           "ServiceError"]
