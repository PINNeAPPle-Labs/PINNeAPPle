"""Optional SciREX/JAX neural-operator backend; no JAX import at module load."""
from .adapter import ScirexAdapter, ModelSpec, FieldSpec, available_models, validate_field
__all__ = ["ScirexAdapter", "ModelSpec", "FieldSpec", "available_models", "validate_field"]
