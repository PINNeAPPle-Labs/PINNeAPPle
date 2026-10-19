"""Publish every zoo model and dataset in ``pinneapple_hub/zoo/manifest.json`` to the Hugging Face Hub, with the demo
Space, and check what the Hub serves (the logic is :mod:`pinneapple_hub.zoo.publish`).

    HF_TOKEN=... python scripts/zoo/publish_to_hub.py              # create/update repos, upload, tag <revision>
    python scripts/zoo/publish_to_hub.py --dry-run                 # build the derived files and cards, list uploads
    python scripts/zoo/publish_to_hub.py --staging DIR             # also Hub-only items, files in DIR/<name>/<file>
    python scripts/zoo/publish_to_hub.py --hub-fallback            # Hub-only datasets converted from the Hub copy
    HF_TOKEN=... python scripts/zoo/publish_to_hub.py --space      # the demo Space (static, from the Hub models)
    python scripts/zoo/publish_to_hub.py --verify                  # anonymous download + SHA-256 of every file
    python scripts/zoo/publish_to_hub.py --verify-viewer           # datasets-server is-valid / splits / first-rows
    python scripts/zoo/publish_to_hub.py --verify-models           # load from the Hub, reproduce card and example
    python scripts/zoo/publish_to_hub.py --verify-space            # the Space runs and serves its page

One repo per item under the PINNeAPPle-Labs organisation: the manifest's files (SHA-256 checked before upload), the
files derived from them (Parquet copies of datasets, safetensors and ONNX copies of models) and the README card; the
commit is tagged with the item's ``revision``. Without a token the upload steps print a warning and exit 0 (the CI
workflow relies on that to skip cleanly when the HF_TOKEN secret is not set).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pinneapple_hub.zoo import publish as pub  # noqa: E402

plan = pub.plan                                          # kept importable: tests and tools call plan()

NO_TOKEN = ("::warning::HF_TOKEN is not set: nothing was uploaded. Add the HF_TOKEN repository secret (a token "
            "with write access to the PINNeAPPle-Labs organisation) and re-run this workflow.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--staging", type=Path, help="directory with the files of Hub-only items (<name>/<file>)")
    ap.add_argument("--hub-fallback", action="store_true",
                    help="convert Hub-only datasets that are not staged from their copy on the Hub")
    ap.add_argument("--only", nargs="*", default=[], help="only these items (names)")
    ap.add_argument("--no-tags", action="store_true", help="upload to main without creating the revision tags")
    ap.add_argument("--work", type=Path, help="directory for the derived files (default: a temporary one)")
    ap.add_argument("--space", action="store_true", help="build and publish the static demo Space")
    ap.add_argument("--verify", action="store_true", help="only verify the published files (anonymous download)")
    ap.add_argument("--allow-unpublished", action="store_true",
                    help="--verify: a Hub-only item missing from the Hub is a note, not a failure")
    ap.add_argument("--verify-viewer", action="store_true", help="check the dataset viewer of every dataset")
    ap.add_argument("--verify-models", action="store_true", help="check every model as downloaded from the Hub")
    ap.add_argument("--verify-space", action="store_true", help="wait for the demo Space to run")
    ap.add_argument("--timeout", type=float, default=2400, help="seconds to wait for the viewer / the Space")
    a = ap.parse_args(argv)
    token = os.environ.get("HF_TOKEN", "").strip()
    if a.verify:
        return 1 if pub.verify_files(plan(need_files=False, only=a.only), a.allow_unpublished) else 0
    if a.verify_viewer:
        return 1 if pub.verify_viewer(a.only or None, timeout=a.timeout) else 0
    if a.verify_models:
        return 1 if pub.verify_models(a.only or None) else 0
    if a.verify_space:
        return pub.verify_space(timeout=a.timeout)
    if a.space:
        if not token:
            print(NO_TOKEN)
            return 0
        pub.publish_space(token, ROOT / "apps" / "zoo_space", ROOT / "apps" / "playground" / "site", work=a.work)
        return 0
    items = plan(a.staging, hub_fallback=a.hub_fallback, work=a.work, only=a.only)
    size = sum(Path(p).stat().st_size for it in items for p in [*it["files"].values(), *it["derived"].values()])
    print(f"{len(items)} items, {sum(len(it['files']) + len(it['derived']) for it in items)} files ready, "
          f"{size / 1e6:.2f} MB (Hub-only datasets from the Hub copy are prepared while publishing)")
    for it in items:
        files = [*it["files"], *it["derived"]] if it["readme"] is not None else ["(prepared from the Hub copy)"]
        print(f"  {it['repo_type']:7s} {it['repo']} @ {it['revision']}: {', '.join(files)}")
    if a.dry_run:
        return 0
    if not token:
        print(NO_TOKEN)
        return 0
    failed = pub.publish(items, token, work=a.work, tags=not a.no_tags)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
