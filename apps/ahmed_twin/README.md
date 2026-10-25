# Ahmed Twin

Surrogate digital twin of the Ahmed body from the OpenFOAM study in `examples/end_to_end/ahmed_body`: sliders for
slant angle, ground clearance and speed; the predicted surface pressure on the body in 3-D (three.js) with a mean /
90 % half-width toggle; CD and CL with their calibrated 90 % band and a sweep chart; a warning outside the region the
surrogate was validated in; and a button that runs a verifying OpenFOAM case at the slider point and reports it
against the prediction.

```bash
cd apps/ahmed_twin
uvicorn ahmed_twin.api:app --port 8171        # then open http://localhost:8171
```

The page is written at start-up by `pinneapple_autolab.domains.ahmed_study.twin` (library); this app only serves
it and queues the verifying runs (`pinneapple_autolab.study.campaign.run_campaign`, one at a time). OpenFOAM is needed
only for the button. A static copy of the page (button disabled) is in `examples/end_to_end/ahmed_body/results/twin`.
