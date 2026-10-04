# U-DR-1 Compiler

**Drop the process datasheet, the mechanical datasheet and the client specification, get ASME Form U-DR-1 filled in,
with the source of every value and a list of what is missing or contradictory.**

Day 6 of the PINNeAPPle 30-app program. Before a manufacturer designs a pressure vessel under ASME BPVC Section VIII
Division 1, the user's design requirements go on Form U-DR-1: service, operating and design conditions, MDMT, corrosion
allowances, loadings, materials, nozzles, joint types, examination and testing. Those answers are spread over three or four
documents, and compiling them by hand takes an engineer hours and misses things. This service:

1. **Reads** every document: PDF tables and "label: value" lines (plain text too), with their units.
2. **Compiles** the 129 items of the form. Each value keeps its document, page and the exact text it was read from.
   Quantities are compared in SI, with gauge and absolute pressure told apart (`barg`, `bar(a)`, `psig`, `MPa(g)`, ...).
3. **Flags** conflicts between documents (the document higher in the list wins, the others are shown) and the required
   items no document states, so you know whom to ask. Your answers in the Review tab override everything.
4. **Fills** the official fillable PDF: text fields, check boxes, radio groups, the nozzle schedule and the joint table;
   text too long for its box goes to General Notes. Date, user and registration ID are left for the engineer.

Optionally, Claude also reads each PDF (for free-form specifications the label rules miss). A value from Claude is kept only
if the text it quotes as its source is found in the document; everything else is listed as rejected, never used silently.

Library: `pinneapple_data.udr1` (`read_document`, `extract_rules`, `compile_requirements`, `fill_form`,
`extract_with_claude`). The item catalog (`udr1.FIELDS`) is the reusable part: the same compiled data can feed other forms
and calculations.

## The form is not bundled

Form U-DR-1 is ASME's document. Download the fillable PDF (07/25 revision) from asme.org and either upload it in the Form
tab or point `UDR_FORM_PDF` to it. `apps/deploy/forms/*.pdf` is git-ignored.

## Validation (`tests/test_udr1.py`)

The three documents in [`examples/`](examples) describe a fictitious separator V-101
(`python examples/make_samples.py` regenerates them). From them the compiler finds:

- 72 of 129 items, 27 of the 28 required ones, and the 6 nozzles, including a row the table continues on the next page.
- Exactly the planted conflict: the internal shell corrosion allowance is 6 mm in the owner specification and 3 mm in the
  mechanical datasheet.
- Exactly the planted gap: no document says whether the service is cyclic.
- Reversing the document order flips the conflict's winner. Engineer overrides close the gap and the conflict.
- Filled values read back from the PDF; a PDF that is not the form is refused.
- The Claude path is tested with a stub client: a fabricated quote and an invalid answer are rejected.

The fill test runs when `UDR_FORM_PDF` is set and is skipped otherwise.

## Run it

```bash
pip install -e . -r apps/design_requirements/requirements.txt
cd apps/design_requirements && UDR_FORM_PDF=/path/to/u-dr-1.pdf uvicorn design_requirements.api:app --port 8085
curl -F files=@process_datasheet.pdf -F files=@mechanical_datasheet.pdf localhost:8085/api/compile
curl -F files=@process_datasheet.pdf -F form=@u-dr-1.pdf localhost:8085/api/fill -o U-DR-1_filled.pdf
```

Environment: `UDR_USER` / `UDR_PASSWORD`, `UDR_FORM_PDF`, `UDR_MAX_MB` (40), `UDR_MAX_FILES` (8), `UDR_MAX_HEAVY` (2),
and for the Claude extraction `ANTHROPIC_API_KEY` with `UDR_CLAUDE_MODEL` (the model ID to use).
Deploy with the other apps: [`apps/deploy`](../deploy/README.md).

The result is a draft for the engineer to check and sign. ASME and BPVC are trademarks of ASME; this tool is not
affiliated with or endorsed by ASME.
