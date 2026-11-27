- **A CalculiX step runs the backend its own verdict reports**: `CalculiXAdapter` asked the solver contract whether
  CalculiX is usable -- which is yes when either the `ccx` binary is on `PATH` or Docker can run the bundled image --
  and then always spelled `ccx -i <job>`, so on a host with Docker and without CalculiX the template was offered,
  accepted and died with exit 127. The command line now comes from `calculix.runner.ccx_command`, which is the one
  place that knows both (`backend` is also a step parameter now); a step on a remote executor keeps the plain
  `ccx -i <job>`, because an sbatch script or a Kubernetes Job runs it on a machine without this one's Docker. The
  CI job that runs the simulation runner installs `calculix-ccx`, as the CalculiX bridge tier already did, so the
  beam template is exercised against the real solver instead of the image.
