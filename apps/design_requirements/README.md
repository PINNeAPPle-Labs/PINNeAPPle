# Form Compiler

**Drop the datasheets and specifications, get the engineering form filled in, with the source of every value and a list
of what is missing or contradictory.** ASME Form U-DR-1 is built in; any other fillable PDF works too.

Day 6 of the PINNeAPPle 30-app program. Engineering forms collect data that lives in other documents. Before a
manufacturer designs a pressure vessel under ASME BPVC Section VIII Division 1, for example, the user's design
requirements go on Form U-DR-1: service, operating and design conditions, MDMT, corrosion allowances, loadings,
materials, nozzles, joint types, examination and testing. Those answers are spread over three or four documents (process
datasheet, mechanical datasheet, owner's specifications), and compiling them by hand takes an engineer hours and misses
things. This service:

1. **Reads** every document: PDF tables and "label: value" lines (plain text too), with their units.
2. **Compiles** every item of the form. Each value keeps its document, page and the exact text it was read from.
   Quantities are compared in SI, with gauge and absolute pressure told apart (`barg`, `bar(a)`, `psig`, `MPa(g)`, ...).
3. **Flags** conflicts between documents (the document higher in the list wins, the others are shown) and the required
   items no document states, so you know whom to ask. Your answers in the Review tab override everything.
4. **Fills** the fillable PDF: text fields, check boxes, radio groups and table rows (the U-DR-1 nozzle schedule);
   text too long for its box goes to the notes field. Date, user and registration ID are left for the engineer.
5. **Exports** the compiled data as a JSON record (value, unit, SI value, source per item), for other forms and
   calculations.

## Any form, not only U-DR-1

The engine (`pinneapple_data.formfill`) is generic; a `FormSpec` describes what one form asks for: items (label, kind,
units, options, required), how documents phrase them (label patterns), table rows, and the PDF field each answer goes
to. Three ways to get one:

| Template | What you provide | Items |
|---|---|---|
| `asme_u-dr-1` (built in) | the documents (and your copy of the form to fill) | 129 items with their usual phrasings, required items, nozzle schedule, check boxes |
| `auto` | the documents and **any fillable PDF** | one item per form field, labelled from its tooltip or name; fields with a readable label are found by the rules, the local LLM can find the rest |
| `custom` | the documents and a **JSON spec** | whatever you write: start from `/api/spec` (U-DR-1) or `/api/spec/from-pdf` (any PDF), add labels, patterns, options and required items |

Adding a built-in template is a module that builds a `FormSpec` (see `pinneapple_data/formfill/specs/asme_udr1.py`).

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

The three documents in [`examples/`](examples) describe a fictitious separator V-101
(`python examples/make_samples.py` regenerates them). With the U-DR-1 template the compiler finds:

- 72 of 129 items, 27 of the 28 required ones, and the 6 nozzles, including a row the table continues on the next page.
- Exactly the planted conflict: the internal shell corrosion allowance is 6 mm in the owner specification and 3 mm in the
  mechanical datasheet.
- Exactly the planted gap: no document says whether the service is cyclic.
- Reversing the document order flips the conflict's winner. Engineer overrides close the gap and the conflict.
- Filled values read back from the PDF; a PDF that is not the form is refused.

Generic engine and LLM:

- A fillable form the tool has never seen (built in the test with reportlab) becomes a spec, is filled from a datasheet
  and read back; its JSON spec round-trips, and edited patterns, kinds and required flags take effect.
- The local-LLM path runs against a fake Ollama HTTP server: a fabricated quote, a value missing from its quote, an
  invalid answer and an invented table row are rejected; long documents are sent in page chunks; an unreachable server
  or missing model is reported.

The U-DR-1 fill test runs when `UDR_FORM_PDF` is set and is skipped otherwise.

## Run it

```bash
pip install -e . -r apps/design_requirements/requirements.txt
cd apps/design_requirements && UDR_FORM_PDF=/path/to/u-dr-1.pdf uvicorn design_requirements.api:app --port 8085
curl -F files=@process_datasheet.pdf -F files=@mechanical_datasheet.pdf localhost:8085/api/compile
curl -F files=@process_datasheet.pdf -F form=@u-dr-1.pdf localhost:8085/api/fill -o U-DR-1_filled.pdf
curl -F template=auto -F form=@any_fillable_form.pdf -F files=@datasheet.pdf localhost:8085/api/fill -o filled.pdf
```

Local LLM: `ollama serve`, `ollama pull <model>`, then start the app with `UDR_OLLAMA_MODEL=<model>` (and
`UDR_OLLAMA_URL` if Ollama is not on `localhost:11434`).

Environment: `UDR_USER` / `UDR_PASSWORD`, `UDR_FORM_PDF`, `UDR_MAX_MB` (40), `UDR_MAX_FILES` (8), `UDR_MAX_HEAVY` (2),
`UDR_OLLAMA_URL`, `UDR_OLLAMA_MODEL`, `UDR_OLLAMA_TIMEOUT` (600 s).
Deploy with the other apps: [`apps/deploy`](../deploy/README.md) (an optional `ollama` service is included).

The result is a draft for the engineer to check and sign. ASME and BPVC are trademarks of ASME; this tool is not
affiliated with or endorsed by ASME.
