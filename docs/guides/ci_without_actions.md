# Running the gate without GitHub Actions

GitHub Actions has been unavailable on this repository since 2026-11-05 (billing), so a pull request arrives here with
an empty checks tab. Three ways to give a reviewer the same evidence, cheapest first.

## 1. The local gate, pasted into the pull request

`scripts/local_gate.sh` runs what the workflow would have run for the files the change touches: `ruff` over them,
`pytest` over the test files among them, and the repository's own structural checks. One line per stage, so its output
is exactly the evidence a reviewer reads from a green check.

```bash
./scripts/local_gate.sh                    # what the diff touches, against origin/main
./scripts/local_gate.sh tests/test_x.py    # an explicit list
STRICT=1 ./scripts/local_gate.sh           # repository-wide checks block too
```

`.githooks/pre-push` calls it, and `git config core.hooksPath .githooks` wires it, so a push that would have failed CI
fails before it leaves the machine. `git push --no-verify` skips it when a push is genuinely urgent.

The gate's own rule: **lint and the touched tests decide**. The repository-wide checks (`check_example_imports.py`,
`check_experiment_manifests.py`) are advisory because they read the whole tree and report a state the diff does not
decide — at the time of writing, 118 template imports are broken on `main` and `check_experiment_manifests.py` needs a
package this environment does not install.

## 2. A self-hosted runner on the server (free, and restores the real checks tab)

Self-hosted runners do not consume the hosted minutes that the billing block is about: a private repository can run as
many jobs as it likes on its own hardware. The server at `159.69.246.142` (4 vCPU, 7 GB, no GPU) is enough for lint and
the CPU test suite.

```bash
# on the server, as the user that will run the jobs
mkdir -p ~/actions-runner && cd ~/actions-runner
curl -O -L https://github.com/actions/runner/releases/latest/download/actions-runner-linux-x64-2.331.0.tar.gz
tar xzf actions-runner-linux-x64-2.331.0.tar.gz
./config.sh --url https://github.com/PINNeAPPle-Labs/PINNeAPPle \
            --token <REGISTRATION_TOKEN> \
            --labels self-hosted,linux,x64,pinneapple \
            --unattended --replace
sudo ./svc.sh install && sudo ./svc.sh start
```

`<REGISTRATION_TOKEN>` comes from the repository's *Settings → Actions → Runners → New self-hosted runner*, or from
`gh api -X POST repos/PINNeAPPle-Labs/PINNeAPPle/actions/runners/registration-token --jq .token` with an admin session.

Then the jobs need to run there. The least invasive change is to keep `ubuntu-latest` for anything that genuinely needs
GitHub's image and add a self-hosted job for the CPU suite:

```yaml
jobs:
  lint-and-core:
    runs-on: [self-hosted, linux, pinneapple]
    steps:
      - uses: actions/checkout@v7
      - run: ./scripts/local_gate.sh --all
```

The runner needs `python3.11`, `ruff` and the package's test dependencies installed once (`pip install -e ".[pyvista]"`,
`pip install pytest`). Anything that needs a GPU stays manual — this machine has no accelerator.

## 3. Rehearse the workflow locally with `act`

`act` runs a workflow file in Docker on the workstation, using the same YAML.

```bash
brew install act
act pull_request -W .github/workflows/tests.yml -j test
```

It needs Docker (the local image was cleared on 2026-11-20 to free 46 GB) and a runner image of several GB, so it is the
last option here — it is slower than the gate and adds nothing it does not already check for a change of this size.
