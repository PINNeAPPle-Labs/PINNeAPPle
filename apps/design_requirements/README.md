# Form Compiler

**Pick the document format, drop the datasheets and specifications, get the form or datasheet compiled, with the source
of every value and a list of what is missing or contradictory.**

Day 6 of the PINNeAPPle 30-app program. Engineering forms and datasheets collect data that lives in other documents
(process datasheet, mechanical datasheet, owner's specifications, calculations). Compiling them by hand takes hours,
and conflicts between documents or missing data surface late, as questions from the vendor. This service:

1. **Reads** every document: PDF tables and "label: value" lines (plain text too), with their units; tables with one
   column per side (shell side | tube side) are read cell by cell.
2. **Compiles** every item of the chosen format. Each value keeps its document, page and the exact text it was read
   from. Quantities are compared in SI, with gauge and absolute pressure told apart (`barg`, `bar(a)`, `mbarg`, `psig`).
3. **Flags** conflicts between documents (the document higher in the list wins, the others are shown) and the required
   items no document states, so you know whom to ask. Your answers in the Review tab override everything.
4. **Outputs** the filled official form (U-DR-1) or a compiled datasheet PDF (open items first, every value with its
   source), a CSV checklist, and a JSON data record (value, unit, SI value, source) for other forms and calculations.

## Supported formats

| Format (template id) | Data of | Items (required) | Output | Example set |
|---|---|---|---|---|
| Pressure vessel (`asme_u-dr-1`) | ASME BPVC VIII-1 Form U-DR-1 | 129 (28) + nozzle schedule | fills the official fillable form | separator V-101 |
| Relief valve (`psv`) | API 520 Part I / API 526 | 42 (13) | datasheet PDF | PSV-101 |
| Shell-and-tube exchanger (`shell_tube`) | TEMA / API 660, per side | 62 (29) | datasheet PDF | gas cooler E-101 |
| Storage tank (`tank`) | API 650 (Annex L) | 41 (21) + nozzle schedule | datasheet PDF | diesel tank T-201 |
| Centrifugal pump (`pump`) | API 610 | 42 (17) | datasheet PDF | condensate pumps P-101 A/B |

The datasheets are the tool's own layout with the data those standards ask for; the standards' forms are copyrighted
and are not reproduced. Each format is a `FormSpec` in `pinneapple_data/formfill/specs/` (items, the phrasings documents
use for them, units, options, required items, tables); adding one is a module and a line in the registry. The library
also reads any fillable PDF (`FormSpec.from_pdf`) or a JSON spec, for scripted use.

## Local LLM (optional, Ollama)

The label rules read datasheets well, but miss values written only in sentences or under unusual labels. With
`UDR_OLLAMA_MODEL` set, a model served by your own [Ollama](https://ollama.com) server also reads each document. The
documents never leave your network: their text (with tables and page numbers) goes to Ollama, which answers in a JSON
schema built from the form spec. An answer is kept only if the text it quotes as its source is in the document **and**
the value is in that quote; everything else is listed as rejected, never used silently.

## The U-DR-1 form is not bundled

Form U-DR-1 is ASME's document. Download the fillable PDF (07/25 revision) from asme.org and either upload it in the Form
tab or point `UDR_FORM_PDF` to it. `apps/deploy/forms/*.pdf` is git-ignored.

## Validation (`tests/test_formfill.py`)

Every format has a set of fictitious documents in [`examples/`](examples) (`python examples/make_samples.py`
regenerates them), with one conflict and one missing required item planted on purpose. The tests check the values found
and that exactly the planted conflict and gap are reported:

| Format | Found | Planted conflict | Planted gap |
|---|---|---|---|
| U-DR-1, separator V-101 | 72 of 129 items, 27 of 28 required, 6 nozzles (one row continued on the next page) | shell corrosion allowance 6 mm (owner) vs 3 mm (mechanical DS) | cyclic service |
| PSV-101 | 32 of 42, 12 of 13 | set pressure 15 barg (relief load summary) vs 14.5 barg (sizing calc) | valve type |
| Gas cooler E-101 | 56 of 62, 28 of 29 | tube-side design pressure 15 vs 16 barg | TEMA class |
| Diesel tank T-201 | 37 of 41, 20 of 21, 5 nozzles | shell corrosion allowance 3 mm (tank spec) vs 2 mm (mechanical DS) | maximum emptying rate |
| Pumps P-101 A/B | 27 of 42, 16 of 17 | NPSH available 4.2 m (process DS) vs 3.8 m (hydraulic calc) | viscosity |

Also tested: reversing the document order flips a conflict's winner; engineer overrides close gaps and conflicts; the
filled U-DR-1 is read back field by field (when `UDR_FORM_PDF` is set); numbers with thousands separators and decimal
commas; `mPa·s` vs `MPa`; the local-LLM path against a fake Ollama server (a fabricated quote, a value missing from its
quote, an invalid answer and an invented table row are rejected).

## Run it

```bash
pip install -e . -r apps/design_requirements/requirements.txt
cd apps/design_requirements && UDR_FORM_PDF=/path/to/u-dr-1.pdf uvicorn design_requirements.api:app --port 8085
curl -F template=psv -F files=@relief_load_summary.pdf -F files=@psv_sizing.pdf localhost:8085/api/compile
curl -F template=shell_tube -F files=@process_ds.pdf -F files=@mech_ds.pdf localhost:8085/api/datasheet -o E-101.pdf
curl -F files=@process_datasheet.pdf -F form=@u-dr-1.pdf localhost:8085/api/fill -o U-DR-1_filled.pdf
```

Local LLM: `ollama serve`, `ollama pull <model>`, then start the app with `UDR_OLLAMA_MODEL=<model>` (and
`UDR_OLLAMA_URL` if Ollama is not on `localhost:11434`).

Environment: `UDR_USER` / `UDR_PASSWORD`, `UDR_FORM_PDF`, `UDR_MAX_MB` (40), `UDR_MAX_FILES` (8), `UDR_MAX_HEAVY` (2),
`UDR_OLLAMA_URL`, `UDR_OLLAMA_MODEL`, `UDR_OLLAMA_TIMEOUT` (600 s).
Deploy with the other apps: [`apps/deploy`](../deploy/README.md) (an optional `ollama` service is included).

The result is a draft for the engineer to check and sign. ASME and BPVC are trademarks of ASME; this tool is not
affiliated with or endorsed by ASME.
