"""ASME Form U-DR-1 shortcuts over the generic ``pinneapple_data.formfill`` engine (template ``asme_u-dr-1``).

>>> from pinneapple_data import udr1
>>> docs = [udr1.read_document(p, name=n) for n, p in sources]            # in priority order
>>> comp = udr1.compile_requirements(docs, [c for d in docs for c in udr1.extract_rules(d)])
>>> comp.gaps(), comp.conflicts()
>>> pdf = udr1.fill_form(open("u-dr-1-blank.pdf", "rb").read(), comp)
"""
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..formfill import (Candidate, Compilation, Decision, Document, compile, extract_rules as _extract_rules,
                        fill_pdf, parse_value, read_document, read_filled)
from ..formfill.specs.asme_udr1 import FIELD_BY_KEY, FIELDS, SECTIONS, SPEC

__all__ = ["SPEC", "FIELDS", "FIELD_BY_KEY", "SECTIONS", "Candidate", "Compilation", "Decision", "Document",
           "compile_requirements", "extract_rules", "parse_value", "read_document", "fill_form", "read_filled"]


def extract_rules(doc: Document) -> List[Candidate]:
    return _extract_rules(doc, SPEC)


def compile_requirements(docs: Sequence[Document], candidates: Iterable[Candidate],
                         overrides: Optional[Dict[str, Any]] = None) -> Compilation:
    return compile(docs, candidates, SPEC, overrides)


def fill_form(blank_pdf: bytes, comp: Compilation) -> bytes:
    return fill_pdf(blank_pdf, comp, SPEC)
