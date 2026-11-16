# Publication roadmap — PINNeAPPle arXiv preprint

**Working title:** Reproducible cross-framework evaluation of physics-informed learning: matched
numerics, divergent defaults, and the limits of physics-based error detection
**Authors (requested order):** Yan Barros; Enrico Schiassi
**Branch:** `research/arxiv-reproducible-physics-ai-paper`

Legend: `[x]` done and verified · `[~]` done but needs a human decision or signature · `[ ]` open.

---

## Done

- [x] **Preregistered protocol** for the cross-framework comparison, committed before any result
      (`benchmarks/vs_frameworks/PROTOCOL.md`), with every deviation listed and justified
      (`benchmarks/vs_frameworks/REPORT.md`, Appendix A of the paper).
- [x] **Independent reference solutions** for every benchmark: Cole–Hopf by Gauss–Hermite quadrature
      for Burgers, manufactured solutions for Poisson and Helmholtz, ETDRK4 with a resolution study
      for Allen–Cahn (2048 vs 4096 modes: 1.7e-4 in the max norm).
- [x] **One neutral evaluator** for all frameworks, from exported `.npz` predictions; no framework's
      own reported metric is used.
- [x] **Tracks A1 (matched), A2 (native), B (operators), C (inverse), D (ergonomics)** executed, with
      3–5 seeds, all runs recorded with configuration, versions, thread count and load average.
- [x] **A defect found, diagnosed and fixed** in PINNeAPPle's `InversePINN` (adaptive loss weighting
      inflated ν by 12.5×), plus four further bugs found by the harness. All listed in Appendix B.
- [x] **A new experiment for the paper**: a 72-case controlled corpus measuring what
      `pinneapple check` catches and misses, with reference-free and reference-based detectors
      reported apart (`experiments/verification/`).
- [x] **Full manuscript** (`paper.tex`, 19 pages) with related work, methods, results, threats to
      validity, reproducibility statement, two appendices and 20 verified references.
- [x] **Every number generated from a committed record** by
      `scripts/make_tables.py` and `experiments/verification/analyze.py`; no table edited by hand.
- [x] **PDF builds clean** with tectonic (no overfull boxes, bibliography resolved).
- [x] **Bibliography verified**: every DOI resolved against Crossref (journal entries) or
      DataCite/arXiv (preprints) when written. Two candidate references were dropped because the DOIs
      did not resolve to the papers claimed.
- [x] **Reproduction instructions** (`REPRODUCE.md`) and a `Makefile` with `results`, `verification`,
      `check` and `bundle` targets.
- [x] **arXiv source bundle** (`arxiv/pinneapple-preprint.tar.gz`) built from the sources only.

## Needs a human decision — do these before submitting

- [ ] **Author consent and order.** Both authors must approve authorship, order and scope, and confirm
      the manuscript is what they agreed to. The paper states where PINNeAPPle is worse than the
      alternatives; that is deliberate and both authors should be comfortable with it.
- [ ] **Affiliations and identifiers.** The paper uses the affiliations and ORCIDs of the JOSS
      submission (`paper/paper.md`): PINNeAPPle Labs (Brazil); University of Arizona and University of
      Bologna. Confirm these are current, and confirm who is the corresponding author.
- [ ] **Funding, competing interests, acknowledgements.** None are declared in the manuscript. Add any
      that apply, or state explicitly that there are none.
- [ ] **Software licence and preprint policy.** Confirm the repository licence covers the code the
      artifacts point to, and that no employer or institutional policy restricts posting.
- [ ] **Independent review.** At least one reader expert in scientific ML who is not an author should
      read the draft, with the two specific questions: are any claims stronger than the data, and is
      the CPU-only framing clear enough that no reader transfers the cost ratios to a GPU.

## Open at submission time

- [ ] **arXiv category.** Proposed: `cs.LG` primary, `physics.comp-ph` cross-list. Confirm against the
      current category descriptions; the paper is a systems-and-evaluation study, so `cs.LG` is the
      primary claim.
- [ ] **Endorsement.** A first submission to `cs.LG` normally needs an endorsement for at least one
      author. Check the current rules before the submission window, and obtain it early if needed.
- [ ] **Source-package limits and metadata.** Check the current arXiv size limit and metadata fields;
      the source bundle is well under any historical limit. Title, abstract and comments are ready in
      `arxiv/METADATA.md`.
- [ ] **Upload and inspect the compiled preview.** arXiv compiles from source; confirm the preview has
      the tables, the figures and the bibliography as they appear locally, and respond to moderation.
- [ ] **After announcement:** add the arXiv ID and the archival DOI to `CITATION.cff`, the README and
      the repository website; tag the release the artifacts point to, so the claim "this commit
      reproduces these numbers" resolves.

## Deliberately not done

- **No GPU runs.** The protocol allowed them (`--device cuda` on every runner) and none were made. The
  paper says so in the abstract, in §4 and in the threats to validity. A reviewer may ask for them;
  the honest answer is that they are future work and that no cost ratio here transfers to a GPU.
- **No increase in seed count.** Five seeds is the protocol's minimum and the largest the CPU budget
  allowed. The paper states plainly that no difference reaches significance and reports ranges
  instead.
- **No Allen–Cahn in A1.** Cut in advance, first item of the protocol's cut order; documented in the
  figure caption, the table caption and Appendix A.
- **No claim of superiority for PINNeAPPle as a whole.** The paper reports four places where it loses
  and says which measurements they come from.

## Non-negotiable scientific safeguards

No invented measurements, no fabricated citations, no implied physical guarantees without a proof, no
selective reporting of successful seeds, and no claim that arXiv acceptance is certain.
