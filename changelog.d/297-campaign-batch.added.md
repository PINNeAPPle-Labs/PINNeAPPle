- `pinneapple_analysis.surrogate_loop.campaign_solver`: adapts a per-case evaluator into the ``solver(X)`` that
  :func:`~pinneapple_analysis.surrogate_loop.loop` calls, so each round's batch is one submission to the campaign
  runner (#295) rather than a Python loop over designs — the queue, the retries, the concurrency and the
  content-addressed cache are the runner's, not a second copy of them. A repeated identical loop costs no solver time
  (every design is answered from the cache), and a design that failed raises with the runner's own classification
  instead of entering the loop as a zero, which the acquisition would read as an excellent optimum.
