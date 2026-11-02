"""Deprecation helpers (issue #1128): how a core name changes without breaking its users.

The policy (``docs/api_stability.md``): a core name that is renamed, moved or removed keeps working for at least one
minor release, with a :class:`PinneappleDeprecationWarning` that names the replacement and the release it goes away
in, and a ``changed`` changelog fragment starting with ``**Deprecated:**``. Experimental names do not need this, but
may use it.

    from pinneapple_stability.deprecation import deprecated, deprecated_alias, renamed_kwargs

    @deprecated(since="0.9", removed_in="1.1", use="pinneapple_physics.solving.solve")
    def old_solve(...): ...

    old_name = deprecated_alias(new_function, "old_name", since="0.9", removed_in="1.1")

    @renamed_kwargs(since="0.9", removed_in="1.1", hidden="hidden_dim")
    def build(hidden_dim=64): ...
"""
from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from typing import Any

__all__ = ["PinneappleDeprecationWarning", "deprecated", "deprecated_alias", "renamed_kwargs", "message"]


class PinneappleDeprecationWarning(FutureWarning):
    """A deprecated PINNeAPPle name was used. A FutureWarning, which Python shows by default (a plain
    DeprecationWarning is hidden outside ``__main__``): the people who must act on it are the library's users."""


def message(what: str, since: str, removed_in: str, use: str | None = None) -> str:
    """The one wording every deprecation uses."""
    tail = f"; use {use} instead" if use else ""
    return f"{what} is deprecated since {since} and will be removed in {removed_in}{tail}"


def _name(obj: Any) -> str:
    return f"{getattr(obj, '__module__', '?')}.{getattr(obj, '__qualname__', getattr(obj, '__name__', '?'))}"


def deprecated(*, since: str, removed_in: str, use: str | None = None) -> Callable:
    """Decorator for a function or class: a warning on every call (or instantiation), behaviour unchanged."""
    def wrap(obj):
        msg = message(_name(obj), since, removed_in, use)
        if isinstance(obj, type):
            init = obj.__init__

            @functools.wraps(init)
            def __init__(self, *a, **k):
                warnings.warn(msg, PinneappleDeprecationWarning, stacklevel=2)
                init(self, *a, **k)
            obj.__init__ = __init__
            obj.__deprecated__ = msg
            return obj

        @functools.wraps(obj)
        def inner(*a, **k):
            warnings.warn(msg, PinneappleDeprecationWarning, stacklevel=2)
            return obj(*a, **k)
        inner.__deprecated__ = msg
        return inner
    return wrap


def deprecated_alias(target: Callable, old_name: str, *, since: str, removed_in: str) -> Callable:
    """The old name of something renamed: calls ``target`` and warns, naming the new one."""
    msg = message(old_name, since, removed_in, _name(target))

    @functools.wraps(target)
    def alias(*a, **k):
        warnings.warn(msg, PinneappleDeprecationWarning, stacklevel=2)
        return target(*a, **k)
    alias.__deprecated__ = msg
    alias.__name__ = alias.__qualname__ = old_name.rsplit(".", 1)[-1]
    return alias


def renamed_kwargs(*, since: str, removed_in: str, **old_to_new: str) -> Callable:
    """Decorator: keyword arguments renamed (``old="new"``) still work, with a warning; both at once is an error."""
    def wrap(fn):
        @functools.wraps(fn)
        def inner(*a, **k):
            for old, new in old_to_new.items():
                if old in k:
                    if new in k:
                        raise TypeError(f"{fn.__name__}() got both {old!r} (deprecated) and {new!r}")
                    warnings.warn(message(f"argument {old!r} of {_name(fn)}", since, removed_in, repr(new)),
                                  PinneappleDeprecationWarning, stacklevel=2)
                    k[new] = k.pop(old)
            return fn(*a, **k)
        return inner
    return wrap
