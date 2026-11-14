"""Text in two languages, and numbers written the way each language writes them.

A text in this library is a ``{"pt": ..., "en": ...}`` mapping built by :func:`tr`; the language in force
(:func:`language`, set by :func:`use_language` or by the ``PINNEAPPLE_LANG`` environment variable) decides which
one a person reads. English is the source and Portuguese the other language, the same convention
:mod:`pinneapple_reason` uses for its chain of whys.

Numbers follow the language, because a report a client reads is not only words: ``1234.5`` is "1,234.5" in
English and "1.234,5" in Portuguese, and a percent is written the same way.

    >>> from pinneapple_core.analysis.text import tr, text_of, fmt_number, use_language
    >>> text_of(tr("válvula", "valve"), "pt")
    'válvula'
    >>> with use_language("pt"):
    ...     fmt_number(1234.5)
    '1.234,5'
    >>> fmt_number(1234.5, lang="en")
    '1,234.5'
"""
from __future__ import annotations

import contextlib
import contextvars
import math
import numbers
import os
import string
from collections.abc import Iterator, Mapping
from typing import Any

__all__ = ["SUPPORTED_LANGUAGES", "tr", "text_of", "say", "language", "use_language", "fmt_number", "fmt_percent",
           "plural", "counted", "format_template"]

#: the languages a text can be written in, English first (it is the source every message is written in)
SUPPORTED_LANGUAGES = ("en", "pt")

_FORCED: contextvars.ContextVar[str | None] = contextvars.ContextVar("pinneapple_language", default=None)


def tr(pt: str, en: str) -> dict[str, str]:
    """A text in Portuguese and English, as every user-facing string in this module is."""
    return {"pt": pt, "en": en}


def text_of(text: Any, lang: str | None = None) -> str:
    """The text in ``lang`` (the language in force when ``lang`` is None), falling back to the other language."""
    if isinstance(text, Mapping):
        wanted = _norm(lang) if lang else language()
        other = "en" if wanted == "pt" else "pt"
        return str(text.get(wanted) or text.get(other) or text.get("en") or "")
    return "" if text is None else str(text)


def _norm(lang: str | None) -> str:
    key = str(lang or "").strip().lower()
    if not key:
        return language()
    if key.startswith("pt"):
        return "pt"
    if key.startswith("en"):
        return "en"
    raise ValueError(f"unknown language {lang!r}; this library writes text in {SUPPORTED_LANGUAGES}")


def language() -> str:
    """The language in force: :func:`use_language`, else ``PINNEAPPLE_LANG``, else English."""
    forced = _FORCED.get()
    if forced:
        return forced
    env = os.environ.get("PINNEAPPLE_LANG", "").strip().lower()
    return env[:2] if env[:2] in SUPPORTED_LANGUAGES else "en"


@contextlib.contextmanager
def use_language(lang: str) -> Iterator[str]:
    """Read and write every text in ``lang`` for the duration of the block (``"en"`` or ``"pt"``)."""
    token = _FORCED.set(_norm(lang))
    try:
        yield _FORCED.get() or "en"
    finally:
        _FORCED.reset(token)


def fmt_number(value: Any, digits: int = 1, lang: str | None = None) -> str:
    """A number in the language's format, without trailing zeros: ``1234.5`` -> "1,234.5" / "1.234,5".

    A missing or non-finite value is an em dash, never "nan": a report that prints "nan" is a report that
    someone reads as a measurement.
    """
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if not isinstance(value, numbers.Real):
        return str(value)
    number = float(value)
    if not math.isfinite(number):
        return "—"
    text = f"{number:,.{max(int(digits), 0)}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in {"-0", "", "-"}:
        text = "0"
    if _norm(lang) == "en":
        return text
    return text.replace(",", "\u0000").replace(".", ",").replace("\u0000", ".")


def fmt_percent(fraction: Any, digits: int = 0, lang: str | None = None) -> str:
    """A 0–1 fraction as a percentage in the language's format: ``0.734`` -> "73%" / "73 %"."""
    if fraction is None or not isinstance(fraction, numbers.Real) or not math.isfinite(float(fraction)):
        return "—"
    return f"{fmt_number(float(fraction) * 100.0, digits, lang)}%"


def plural(count: Any, one: str, many: str, lang: str | None = None) -> str:
    """ "1 change", "3 changes": the noun already in the language of the sentence."""
    number = float(count or 0)
    return f"{fmt_number(number, lang=lang)} {one if abs(number) == 1 else many}"


def counted(count: Any, one: Any, many: Any, lang: str | None = None) -> str:
    """A count with its noun, where the noun is a :func:`tr` text with a ``{count}`` field.

    ``counted(3, tr("{count} falha", "{count} failure"), tr("{count} falhas", "{count} failures"))``
    -> "3 failures" / "3 falhas". ``one`` is used when the count is exactly ±1.
    """
    number = float(count or 0)
    template = text_of(one if abs(number) == 1 else many, lang)
    return format_template(template, {"count": fmt_number(number, 0, lang)}, lang=lang)


class _Formatter(string.Formatter):
    """``{name}`` formatted in the language, ``{name:.2}`` with two decimals, ``{name:%}`` as a percentage."""

    def __init__(self, lang: str | None = None) -> None:
        super().__init__()
        self.lang = lang

    def format_field(self, value: Any, format_spec: str) -> str:  # noqa: A003 - the stdlib name
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            if format_spec == "%":
                return fmt_percent(value, lang=self.lang)
            if format_spec.startswith(".") and format_spec[1:].isdigit():
                return fmt_number(value, int(format_spec[1:]), lang=self.lang)
        return super().format_field("—" if value is None else value, "")


class _Metrics(dict):
    """A metric a sentence names but the analysis did not report reads as an em dash, not as a KeyError."""

    def __missing__(self, key: str) -> None:
        return None


def format_template(template: str, values: Mapping[str, Any], lang: str | None = None) -> str:
    """One sentence from a template and the numbers it names, in the language in force.

    ``format_template("f = {f:.4} at Re = {Re:.0}", {"f": 0.0317, "Re": 45000})``
    -> "f = 0.0317 at Re = 45,000".

    A template naming something the result does not carry prints an em dash; a template the numbers cannot
    fill (a bad field, an unclosed brace) is returned as written rather than raising: a finding is read by
    a person who is waiting for it.
    """
    formatter = _Formatter(lang)
    try:
        return formatter.vformat(template, (), _Metrics(values))
    except (ValueError, IndexError, KeyError, AttributeError):
        return template


def say(template: Any, lang: str | None = None, **values: Any) -> str:
    """A :func:`tr` text (or a plain string) worded in ``lang`` and filled with ``values``."""
    return format_template(text_of(template, lang), values, lang)
