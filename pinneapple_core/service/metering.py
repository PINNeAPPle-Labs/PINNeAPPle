"""Usage metering per app, principal and unit, with free limits per calendar month (UTC).

    meter = UsageMeter("/data/service.sqlite", app="campaign",
                       limits={"cases_planned": 500, "campaigns": 20})
    meter.consume("ip:203.0.113.7", "cases_planned", 40)    # records it, or raises UsageLimitExceeded
    meter.usage("ip:203.0.113.7")                            # {"period": "2026-10", "units": {...}}

A principal is whoever the use is counted against: ``key:<id>`` (API key), ``user:<name>`` (login) or ``ip:<addr>``
(anonymous). Limits come from code defaults and the environment: ``<PREFIX>_FREE_<UNIT>`` (``CAMPAIGN_FREE_
CASES_PLANNED=1000``; ``unlimited`` or ``-1`` removes the limit). This is metering only; billing is not here.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .db import LocalDB, transaction

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    seq INTEGER PRIMARY KEY AUTOINCREMENT, app TEXT, principal TEXT, unit TEXT, amount REAL, period TEXT,
    ts REAL, ref TEXT
);
CREATE INDEX IF NOT EXISTS usage_lookup ON usage(app, principal, period, unit);
"""


class UsageLimitExceeded(Exception):
    def __init__(self, unit: str, used: float, amount: float, limit: float, period: str):
        self.unit, self.used, self.amount, self.limit, self.period = unit, used, amount, limit, period
        super().__init__(f"free limit for '{unit}' reached: {used:g} used + {amount:g} requested > {limit:g} "
                         f"per month ({period})")

    def to_dict(self) -> dict[str, Any]:
        return {"unit": self.unit, "used": self.used, "requested": self.amount, "limit": self.limit,
                "period": self.period, "detail": str(self)}


def current_period(ts: float | None = None) -> str:
    return time.strftime("%Y-%m", time.gmtime(ts if ts is not None else time.time()))


def limits_from_env(prefix: str, defaults: dict[str, float | None], environ=None) -> dict[str, float | None]:
    """Free limits per unit: ``defaults`` overridden by ``<PREFIX>_FREE_<UNIT>`` environment variables."""
    env = os.environ if environ is None else environ
    out: dict[str, float | None] = {}
    for unit, default in defaults.items():
        raw = env.get(f"{prefix}_FREE_{unit.upper()}")
        if raw is None or raw.strip() == "":
            out[unit] = default
        elif raw.strip().lower() in ("unlimited", "none", "-1"):
            out[unit] = None
        else:
            out[unit] = float(raw)
    return out


class UsageMeter:
    def __init__(self, db_path: str | Path, app: str, limits: dict[str, float | None] | None = None):
        self.app = app
        self.limits = dict(limits or {})
        self._db = LocalDB(db_path, _SCHEMA)

    def used(self, principal: str, unit: str, period: str | None = None) -> float:
        row = self._db.conn.execute("SELECT COALESCE(SUM(amount), 0) FROM usage WHERE app=? AND principal=? "
                                 "AND unit=? AND period=?", (self.app, principal, unit, period or current_period()))
        return float(row.fetchone()[0])

    def check(self, principal: str, unit: str, amount: float = 1) -> None:
        """Raise :class:`UsageLimitExceeded` if ``amount`` more would pass the free limit (records nothing)."""
        limit = self.limits.get(unit)
        if limit is not None:
            used = self.used(principal, unit)
            if used + amount > limit:
                raise UsageLimitExceeded(unit, used, amount, limit, current_period())

    def record(self, principal: str, unit: str, amount: float = 1, ref: str = "") -> None:
        """Count use without checking the limit (e.g. compute seconds measured after a job)."""
        self._db.conn.execute("INSERT INTO usage(app, principal, unit, amount, period, ts, ref) VALUES (?,?,?,?,?,?,?)",
                           (self.app, principal, unit, float(amount), current_period(), time.time(), ref))

    def consume(self, principal: str, unit: str, amount: float = 1, ref: str = "") -> None:
        """Check the limit and record, atomically (two server workers cannot both pass the last free unit)."""
        with transaction(self._db.conn):
            self.check(principal, unit, amount)
            self._db.conn.execute("INSERT INTO usage(app, principal, unit, amount, period, ts, ref) "
                               "VALUES (?,?,?,?,?,?,?)",
                               (self.app, principal, unit, float(amount), current_period(), time.time(), ref))

    def usage(self, principal: str, period: str | None = None) -> dict[str, Any]:
        period = period or current_period()
        rows = self._db.conn.execute("SELECT unit, SUM(amount) used FROM usage WHERE app=? AND principal=? AND period=? "
                                  "GROUP BY unit", (self.app, principal, period))
        used = {r["unit"]: float(r["used"]) for r in rows}
        units = {}
        for unit in sorted(set(used) | set(self.limits)):
            u, lim = used.get(unit, 0.0), self.limits.get(unit)
            units[unit] = {"used": u, "limit": lim, "remaining": None if lim is None else max(lim - u, 0.0)}
        return {"app": self.app, "principal": principal, "period": period, "units": units}

    def totals(self, period: str | None = None) -> dict[str, Any]:
        """Use per unit over every principal (for the operator)."""
        period = period or current_period()
        rows = self._db.conn.execute("SELECT unit, SUM(amount) used, COUNT(DISTINCT principal) principals FROM usage "
                                  "WHERE app=? AND period=? GROUP BY unit", (self.app, period))
        return {"app": self.app, "period": period,
                "units": {r["unit"]: {"used": float(r["used"]), "principals": r["principals"]} for r in rows}}
