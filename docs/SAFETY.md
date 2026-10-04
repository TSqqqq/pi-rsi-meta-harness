# Safety and Research Integrity

This harness intentionally uses **bounded RSI**. The system may improve how it searches, but it must not autonomously alter the definition of success or weaken its own control boundary.

## Immutable scientific boundary

Configure evaluator/test paths under `[files].immutable`. During initialization the harness records SHA-256 hashes. Experiment worktrees are checked before expensive execution.

Do not include mutable training code in the immutable list, and do include final evaluation logic/data split manifests when practical.

## The meta-RSI agent may change/suggest

- exploration vs exploitation allocation
- which historical nodes to revive
- scout roles
- hypothesis-generation policy
- candidate batch size and fidelity allocation
- model-routing thresholds
- numeric search spaces
- which mechanism families deserve more compute

## It must not change

- primary metric definition
- locked final held-out test split
- evaluator semantics
- authentication credentials
- system permissions / sudo policy
- maximum safety boundary
- historical experiment results

## Dynamic Python dependencies

The harness does not preinstall Optuna/Ray/SQLAlchemy. Agents can request packages through `rsi_harness.safe_pip`.

The helper rejects URLs, VCS installs, editable installs, and arbitrary shell strings. It only installs into a project-local virtual environment. For stronger isolation, run the entire target repository and Pi process inside a container or restricted user account.

Important: Pi tools/extensions execute with the OS permissions of the Pi process. The harness is not an OS sandbox. For unattended execution, isolate the process at the operating-system/container level and do not expose host Docker sockets or unrelated credentials.

## Publication integrity

A high automated score is not sufficient evidence for a paper. Publication-grade claims should use full-fidelity multi-seed validation, fair baselines, ablations, compute accounting, and a final held-out evaluation that was not repeatedly optimized against.
