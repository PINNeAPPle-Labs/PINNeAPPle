"""``python -m pinneapple_security <command>``: manifests, signatures, scans and log sanitising from the shell.

    manifest DIR [-o manifest.json]          write the Merkle manifest of a folder
    verify-manifest DIR manifest.json        compare a folder with its manifest (exit 1 if anything changed)
    keygen KEYFILE [--hmac]                  new Ed25519 (or HMAC) key; prints the public key JSON
    sign FILE KEYFILE                        detached signature FILE.sig
    verify FILE PUBKEY.json|KEYFILE          check FILE.sig (exit 1 if invalid)
    scan-model FILE...                       scan checkpoints / pickles for code execution (exit 1 if unsafe)
    scan-secrets PATH                        credentials in code and configs (exit 1 if any)
    sanitize-log LOG [-o OUT] [--hide WORD]  strip user names, hosts, e-mails and IPs from a log
    sbom [-o sbom.json]                      CycloneDX SBOM of this Python environment
"""
from __future__ import annotations

import argparse
import json
import sys


def _load_verify_key(path: str):
    from . import signing

    with open(path) as f:
        d = json.load(f)
    return signing.VerifyKey.from_dict(d) if "public_key" in d else signing.SigningKey.load(path).public()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m pinneapple_security", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("manifest")
    a.add_argument("dir")
    a.add_argument("-o", "--out")
    a = sub.add_parser("verify-manifest")
    a.add_argument("dir")
    a.add_argument("manifest")
    a = sub.add_parser("keygen")
    a.add_argument("keyfile")
    a.add_argument("--hmac", action="store_true")
    a = sub.add_parser("sign")
    a.add_argument("file")
    a.add_argument("keyfile")
    a = sub.add_parser("verify")
    a.add_argument("file")
    a.add_argument("key")
    a = sub.add_parser("scan-model")
    a.add_argument("files", nargs="+")
    a = sub.add_parser("scan-secrets")
    a.add_argument("path")
    a = sub.add_parser("sanitize-log")
    a.add_argument("log")
    a.add_argument("-o", "--out")
    a.add_argument("--hide", action="append", default=[])
    a = sub.add_parser("sbom")
    a.add_argument("-o", "--out")
    args = p.parse_args(argv)

    if args.cmd == "manifest":
        from .integrity import build_manifest
        m = build_manifest(args.dir)
        if args.out:
            m.save(args.out)
        print(json.dumps({"merkle_root": m.root, "files": len(m.files)}))
        return 0
    if args.cmd == "verify-manifest":
        from .integrity import Manifest
        r = Manifest.load(args.manifest).verify(args.dir)
        print(json.dumps(r, indent=1))
        return 0 if r["ok"] else 1
    if args.cmd == "keygen":
        from . import signing
        k = signing.hmac_key() if args.hmac else signing.generate_key()
        k.save(args.keyfile)
        print(json.dumps({"keyid": k.keyid} if args.hmac else k.public().to_dict()))
        return 0
    if args.cmd == "sign":
        from . import signing
        print(signing.sign_file(args.file, signing.SigningKey.load(args.keyfile)))
        return 0
    if args.cmd == "verify":
        from . import signing
        try:
            print(json.dumps(signing.verify_file(args.file, _load_verify_key(args.key))))
            return 0
        except (signing.VerificationError, OSError) as e:
            print(f"INVALID: {e}", file=sys.stderr)
            return 1
    if args.cmd == "scan-model":
        from .model_security import scan_checkpoint
        bad = 0
        for f in args.files:
            r = scan_checkpoint(f)
            bad += not r.safe
            print(json.dumps(r.to_dict()))
        return 1 if bad else 0
    if args.cmd == "scan-secrets":
        from .secrets_scan import scan_path
        found = scan_path(args.path)
        for s in found:
            print(f"{s.path}:{s.line}: {s.rule} {s.excerpt}")
        return 1 if found else 0
    if args.cmd == "sanitize-log":
        from .privacy import sanitize_log
        with open(args.log, encoding="utf-8", errors="replace") as f:
            clean, counts = sanitize_log(f.read(), extra=args.hide)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(clean)
        else:
            sys.stdout.write(clean)
        print(json.dumps(counts), file=sys.stderr)
        return 0
    if args.cmd == "sbom":
        from .attestation import sbom
        doc = json.dumps(sbom(), indent=1)
        if args.out:
            with open(args.out, "w") as f:
                f.write(doc)
        else:
            print(doc)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
