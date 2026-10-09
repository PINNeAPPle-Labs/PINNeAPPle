"""Personal and sensitive data in text and tables: detection, redaction, pseudonymisation, k-anonymity, log sanitising.

* :func:`find_pii` finds e-mails, phone numbers, IP addresses, payment card numbers (Luhn-checked), IBANs (mod-97
  checked) and the Brazilian CPF and CNPJ (check digits verified, so random 11/14-digit numbers are not flagged).
* :func:`redact` replaces each finding with a label; :func:`pseudonymize` replaces a value with a keyed token (HMAC):
  the same input always gives the same token, so joins still work, and without the key the token can not be reversed
  or recomputed by a dictionary attack.
* :func:`k_anonymity` measures the smallest group sharing the same quasi-identifiers (a table is k-anonymous when every
  row is indistinguishable from at least k-1 others on them) and lists the groups below ``k``.
* :func:`sanitize_log` removes from solver/training logs what identifies people and machines (user names in home
  paths, host names, e-mails, IPs) before a log is shared with a vendor, a client or a public issue.

Regex detection has false negatives (free-text names, addresses): it is a safety net, not a guarantee of anonymity.
"""
from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["Finding", "find_pii", "redact", "pseudonymize", "pseudonymize_column", "k_anonymity", "sanitize_log",
           "valid_cpf", "valid_cnpj", "luhn_ok", "iban_ok", "PII_KINDS"]


def valid_cpf(s: str) -> bool:
    d = [int(c) for c in re.sub(r"\D", "", s)]
    if len(d) != 11 or len(set(d)) == 1:
        return False
    for n in (9, 10):
        r = sum(d[i] * (n + 1 - i) for i in range(n)) * 10 % 11
        if (0 if r == 10 else r) != d[n]:
            return False
    return True


def valid_cnpj(s: str) -> bool:
    d = [int(c) for c in re.sub(r"\D", "", s)]
    if len(d) != 14 or len(set(d)) == 1:
        return False
    w1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    for n, w in ((12, w1), (13, [6] + w1)):
        r = sum(a * b for a, b in zip(d[:n], w, strict=False)) % 11
        if (0 if r < 2 else 11 - r) != d[n]:
            return False
    return True


def luhn_ok(s: str) -> bool:
    d = [int(c) for c in re.sub(r"\D", "", s)]
    if not 13 <= len(d) <= 19:
        return False
    tot = 0
    for i, x in enumerate(reversed(d)):
        if i % 2:
            x *= 2
            x -= 9 if x > 9 else 0
        tot += x
    return tot % 10 == 0


def iban_ok(s: str) -> bool:
    s = re.sub(r"\s", "", s).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", s):
        return False
    n = "".join(str(int(c, 36)) for c in s[4:] + s[:4])
    return int(n) % 97 == 1


def _ipv4_ok(s: str) -> bool:
    return all(0 <= int(p) <= 255 for p in s.split("."))


# kind -> (pattern, validator)
_PATTERNS: dict[str, tuple[re.Pattern, Callable[[str], bool] | None]] = {
    "email": (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), None),
    "cnpj": (re.compile(r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b"), valid_cnpj),
    "cpf": (re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"), valid_cpf),
    "card": (re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), luhn_ok),
    "iban": (re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b"), iban_ok),
    "ipv4": (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), _ipv4_ok),
    "ipv6": (re.compile(r"\b(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}\b"), None),
    "phone": (re.compile(r"(?<![\w.])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,3}\)[ .-]?|\d{2,3}[ .-])\d{4,5}[ .-]?\d{4}(?![\w.])"), None),
}
PII_KINDS = tuple(_PATTERNS)


@dataclass(frozen=True)
class Finding:
    kind: str
    start: int
    end: int
    text: str


def find_pii(text: str, kinds: Iterable[str] = PII_KINDS) -> list[Finding]:
    """Non-overlapping findings, earlier kinds in ``PII_KINDS`` winning ties (a CNPJ is not also a phone)."""
    taken: list[tuple[int, int]] = []
    out: list[Finding] = []
    for kind in kinds:
        pat, ok = _PATTERNS[kind]
        for m in pat.finditer(text):
            s, e = m.span()
            if any(s < b and a < e for a, b in taken):
                continue
            if ok is not None and not ok(m.group()):
                continue
            taken.append((s, e))
            out.append(Finding(kind, s, e, m.group()))
    return sorted(out, key=lambda f: f.start)


def redact(text: str, kinds: Iterable[str] = PII_KINDS, label: Callable[[Finding], str] = lambda f: f"[{f.kind.upper()}]") -> str:
    out, last = [], 0
    for f in find_pii(text, kinds):
        out += [text[last:f.start], label(f)]
        last = f.end
    return "".join(out + [text[last:]])


def pseudonymize(value: Any, key: bytes, prefix: str = "id_", length: int = 16) -> str:
    """Keyed, deterministic token for ``value`` (HMAC-SHA256, truncated to ``length`` hex characters)."""
    if len(key) < 16:
        raise ValueError("pseudonymisation key must be at least 16 bytes")
    return prefix + hmac.new(key, str(value).encode("utf-8"), hashlib.sha256).hexdigest()[:length]


def pseudonymize_column(values: Sequence[Any], key: bytes, prefix: str = "id_") -> list[str]:
    return [pseudonymize(v, key, prefix) for v in values]


def k_anonymity(rows: Any, quasi_identifiers: Sequence[str], k: int = 5) -> dict[str, Any]:
    """``rows``: a pandas DataFrame or a list of dicts. Returns the table's k, the groups smaller than ``k`` and the
    share of rows in them."""
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")
    groups: dict[tuple, int] = {}
    for r in rows:
        key = tuple(r.get(q) for q in quasi_identifiers)
        groups[key] = groups.get(key, 0) + 1
    if not groups:
        return {"k": 0, "ok": False, "small_groups": [], "rows_at_risk_pct": 0.0}
    small = sorted(((dict(zip(quasi_identifiers, g, strict=True)), n) for g, n in groups.items() if n < k), key=lambda x: x[1])
    total = sum(groups.values())
    return {"k": min(groups.values()), "ok": not small, "n_groups": len(groups),
            "small_groups": [{"values": v, "size": n} for v, n in small],
            "rows_at_risk_pct": 100.0 * sum(n for _, n in small) / total}


_HOME = re.compile(r"(?P<pre>/home/|/Users/|[A-Za-z]:\\Users\\)(?P<user>[^/\\\s:]+)")
_HOSTLINE = re.compile(r"(?im)^(?P<key>\s*(?:Host|Hostname|host|hostname)\s*[:=]\s*)(?P<val>\S+)")
_FQDN = re.compile(r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:local|lan|internal|corp|intranet)\b", re.I)


def sanitize_log(text: str, extra: Sequence[str] = (), keep_paths: bool = True) -> tuple[str, dict[str, int]]:
    """Strip personal and infrastructure identifiers from a log. ``extra``: literal strings to hide too (project or
    client names). Returns the clean text and how many of each kind were replaced."""
    counts: dict[str, int] = {}

    def sub(pat, repl, s, kind):
        s2, n = pat.subn(repl, s)
        if n:
            counts[kind] = counts.get(kind, 0) + n
        return s2

    text = sub(_HOME, lambda m: m.group("pre") + "<user>", text, "user_path")
    text = sub(_HOSTLINE, lambda m: m.group("key") + "<host>", text, "host")
    text = sub(_FQDN, "<host>", text, "internal_host")
    for f in reversed(find_pii(text, ("email", "ipv4", "ipv6"))):     # before the custom words, which may sit inside them
        text = text[:f.start] + f"<{f.kind}>" + text[f.end:]
        counts[f.kind] = counts.get(f.kind, 0) + 1
    for e in extra:
        if e:
            text = sub(re.compile(re.escape(e), re.I), "<redacted>", text, "custom")
    if not keep_paths:
        text = sub(re.compile(r"(?:/[\w.<>-]+){2,}"), "<path>", text, "path")
    return text, counts
