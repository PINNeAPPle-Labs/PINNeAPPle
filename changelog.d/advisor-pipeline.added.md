- **Added:** `pinneapple_advisor` answers the ten questions a simulation study raises before anything runs -- is
  the physical model right, is it 3-D, a mesh, what the user already has (geometry, mesh, dataset), which solver and
  what one run costs, sweep the solver or train a surrogate, which architecture among PINN / XPINN / DeepONet / FNO /
  MeshGraphNet / PINO / inverse PINN, how to generate the dataset, and what to check and look at -- and returns the
  pipeline as JSON (`pinneapple.advisor/1`) plus a Mermaid, SVG, HTML and Markdown diagram: `pp.advise(text)` /
  `pinneapple_advisor.advise`, the node `advisor.pipeline`, `python -m pinneapple_advisor`. A free-text description is
  read by keyword rules in English and Portuguese and, for the fields still open, by a local Ollama model whose
  answers are validated field by field; the architecture, the solver family and the direct-versus-surrogate decision
  come from the library's own recommendation APIs (`architecture_recommendation`, `solver_orchestration`) and a cost
  model that prints its formula and assumptions (`pinneapple_advisor.estimate`, "order of magnitude" until the user
  passes a measured solve). Nothing in a plan is executed. Guide: `docs/guides/pipeline_advisor.md`.
