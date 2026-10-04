"""ASME Form U-DR-1 compiler: read process datasheets, mechanical datasheets and client specifications, find every
item the form asks for with its source, report conflicts between documents and missing data, and fill the official
fillable PDF.

>>> from pinneapple_data import udr1
>>> docs = [udr1.read_document(p, name=n) for n, p in sources]            # in priority order
>>> comp = udr1.compile_requirements(docs, [c for d in docs for c in udr1.extract_rules(d)])
>>> comp.gaps(), comp.conflicts()
>>> pdf = udr1.fill_form(open("u-dr-1-blank.pdf", "rb").read(), comp)
"""
from .catalog import FIELD_BY_KEY, FIELDS, SECTIONS, Field
from .extract import (Candidate, Compilation, Decision, Document, compile_requirements, extract_rules, parse_value,
                      read_document)
from .fill import fill_form, form_values, read_filled

__all__ = ["FIELDS", "FIELD_BY_KEY", "SECTIONS", "Field", "Candidate", "Compilation", "Decision", "Document",
           "compile_requirements", "extract_rules", "parse_value", "read_document", "fill_form", "form_values",
           "read_filled"]
