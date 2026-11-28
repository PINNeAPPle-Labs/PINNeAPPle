"""Uma thread só para o gmsh — sempre a mesma, porque ele não sobrevive a ser chamado de threads diferentes.

O gmsh traz o próprio runtime OpenMP, e a sessão dele fica presa à thread que a inicializou: a segunda chamada feita de
outra thread derruba o processo inteiro, com SIGSEGV dentro de ``__kmpc_critical_with_hint``, sem traceback e sem
exceção que dê para tratar. Medido nesta máquina com a esfera:

* duas ``mesh_sphere`` na thread principal: as duas passam;
* duas ``mesh_sphere``, cada uma numa thread própria: a primeira passa e a segunda mata o processo;
* três ``mesh_sphere`` chamadas de três threads diferentes, todas executadas nesta thread única: as três passam.

Isso importa porque um servidor roda cada caso numa thread do seu pool, e a segunda esfera do dia derrubava o processo
inteiro. :func:`run` é o único caminho: ele mantém um ``ThreadPoolExecutor`` de um trabalhador só, com a sessão do gmsh
viva nele, e devolve o resultado (ou a exceção) para quem chamou.
"""
from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

__all__ = ["run"]

T = TypeVar("T")

_EXEC: ThreadPoolExecutor | None = None
_TRAVA = threading.Lock()


def _executor() -> ThreadPoolExecutor:
    global _EXEC
    if _EXEC is None:
        with _TRAVA:
            if _EXEC is None:
                _EXEC = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gmsh")
    return _EXEC


def run(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Roda ``fn`` na thread do gmsh e devolve o que ela devolveu; a exceção dela sobe aqui, com o traceback dela.

    Uma exceção de Python atravessa a thread sem problema — o que não atravessa é o gmsh ser tocado de fora dela.
    """
    return _executor().submit(fn, *args, **kwargs).result()
