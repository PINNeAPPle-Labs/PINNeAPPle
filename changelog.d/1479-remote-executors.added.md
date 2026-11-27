- **Kubernetes and cloud batch executors** (E8/#1479): `executor="kubernetes"` runs one Job per step through
  `kubectl` with the step's directory mounted at `/work`, streams the pod's lines while it runs, reads the container
  exit code and deletes the Job on cancel or timeout; `executor="batch"` does the same through a `BatchProvider`
  (no vendor is wired until a real case picks one). A step that cannot run remotely — a `local_only` adapter, no
  volume, no upload — fails naming the step and the reason instead of surfacing as a timeout. Local remains the
  default and stays dependency-free.
