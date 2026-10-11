# Safety and Research Integrity

This harness intentionally uses **bounded RSI**. The system may improve how it searches, but it must not autonomously alter the definition of success or weaken its own control boundary.

## Immutable scientific boundary

Configure evaluator/test paths under `[files].immutable`. During initialization the harness records SHA-256 hashes. Experiment worktrees are checked before expensive execution.

Do not include mutable training code in the immutable list, and do include final evaluation logic/data split manifests when practical.

## The scout may change

- which branches to extend, revive or combine
- the split of a batch between new mechanisms and knob tuning
- which experiments to run first

## Benchmark integrity

Experiments must improve the model, not the score. The evaluator therefore runs the method only through a fixed
driver (template `examples/fixed_driver.py`, kept in an immutable path): the method runs in a sandbox that sees
its code, the model and the raw inputs; it gets one test item at a time, shuffled; grouped items go to separate
processes. The scout prompt also forbids benchmark exploits explicitly. Keep a regression test with a fake
method that tries every known leak (reading labels, earlier predictions, scores; writing files; relating grouped
items through shared state) and assert that all of them fail.

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
