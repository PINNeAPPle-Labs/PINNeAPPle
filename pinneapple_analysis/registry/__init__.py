"""O registro dos algoritmos que este deployment executa, com fonte e estado de verificação."""
from .algorithms import (NOT_EVALUATED, REGISTRY, VERIFIED, Algorithm, Reference, categorias, lacunas, listar,
                         obter)

__all__ = ["Algorithm", "Reference", "REGISTRY", "VERIFIED", "NOT_EVALUATED", "listar", "obter", "categorias",
           "lacunas"]
