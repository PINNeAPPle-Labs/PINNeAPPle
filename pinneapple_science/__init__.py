"""PINNeAPPle science (``pp.science``): experiments by area of science, at academic and research level, theory
and practice (a virtual bench), in 0-D (lumped), 1-D, 2-D, 3-D and 3-D + time; and the atlas of every area.

    import pinneapple as pp
    pp.science.atlas()                                  # every area: available / partial / planned, with pointers
    pp.science.experiments(area="electromagnetism", dim="3D+t")
    r = pp.science.run("em.faraday_induction")          # a magnet falling through a loop, 3-D in time
    print(r)                                            # computed vs reference, physics checks, trust decision
    r.save("out/faraday")                               # result.json, figures, the animation

Every experiment declares its area, level (``academic`` / ``research``), kind (``theory`` / ``practice``) and
dimensions, and every result carries comparisons with cited references, physics checks and a trust report
(:class:`pinneapple_analysis.trust.TrustReport`). Committed results, Lab manifests and seals:
``python -m pinneapple_science report all`` (:mod:`.report`). The atlas is data
(``pinneapple_science/data/atlas.json``) and points to the code that already covers each area elsewhere in the
library; electromagnetism is built on ``pp.em`` and ``pp.maxwell``. See docs/guides/science.md.
"""
from .atlas import Atlas, validate
from .atlas import load as atlas
from .core import DIMS, KINDS, LEVELS, Comparison, Experiment, Reference, Result
from .registry import experiments, get, names, register, run

__all__ = ["atlas", "validate", "Atlas", "experiments", "get", "names", "register", "run", "Experiment", "Result",
           "Reference", "Comparison", "LEVELS", "KINDS", "DIMS"]
