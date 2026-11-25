"""Command-line entry point for Atlas metadata synchronization and inspection."""
import argparse
import json
from .store import AtlasStore
from .sync import sync_public_query
from .reconcile import reconcile_dois


def main(argv=None):
    parser = argparse.ArgumentParser(description="PINNeAPPle Global Physics AI Atlas")
    parser.add_argument("--db", default="physics_ai_atlas.sqlite3")
    sub = parser.add_subparsers(dest="command", required=True)
    sync = sub.add_parser("sync", help="Query public metadata sources")
    sync.add_argument("query")
    sync.add_argument("--limit", type=int, default=10)
    sub.add_parser("reconcile", help="Link identical DOIs across sources")
    graph = sub.add_parser("graph", help="Print graph as JSON")
    graph.add_argument("--limit", type=int, default=100)
    args = parser.parse_args(argv)
    store = AtlasStore(args.db)
    try:
        if args.command == "sync":
            result = sync_public_query(store, args.query, limit=args.limit)
        elif args.command == "reconcile":
            result = {"doi_relations": reconcile_dois(store)}
        else:
            result = store.graph(limit=args.limit)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        store.close()


if __name__ == "__main__":
    main()
