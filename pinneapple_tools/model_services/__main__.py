"""python -m pinneapple_tools.model_services serve [--host 0.0.0.0] [--port 8000] [--no-zoo] [--api-key KEY]"""
import argparse

from .registry import default_registry
from .server import make_server


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pinneapple_tools.model_services")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve", help="serve the PINNeAPPle model services (NIM-compatible routes)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--no-zoo", action="store_true", help="only the physics-check and intent services")
    s.add_argument("--api-key", default=None)
    a = ap.parse_args(argv)
    reg = default_registry(zoo=not a.no_zoo)
    srv = make_server(reg, a.host, a.port, a.api_key)
    print(f"PINNeAPPle Model Services on http://{a.host}:{srv.server_address[1]}/v1 "
          f"({', '.join(s.id for s in reg.list())})", flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
