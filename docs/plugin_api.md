# Workflow Plugin API

AeroGNC-Lab supports trusted local workflow extensions through the
`aerognc.workflows` Python entry-point group. Plugins are optional and are not needed
by any built-in simulation.

Each entry point must load a callable that accepts a
`aerognc.project.registry.WorkflowContext` and returns a
`aerognc.project.registry.WorkflowResult`. Register a stable workflow name, semantic
plugin version, and compatible workflow API version through `WorkflowDescriptor`.
The current API version is available as `WORKFLOW_API_VERSION`.

Workflow code must:

- validate all plugin-specific parameters;
- use `context.configuration_path` rather than the process working directory;
- consume `context.seed` for every stochastic operation;
- call `context.cancellation.raise_if_cancelled()` at bounded intervals;
- send monotonic progress through `context.report_progress()`;
- return unit-labelled `ResultDataset` channels and structured requirement outcomes;
- remain inside the public-safe civilian research scope.

Entry-point discovery failures, duplicate names, and incompatible API versions are
reported as isolated plugin issues. They do not prevent built-in workflows from being
registered. Plugins are ordinary trusted Python code: the entry-point mechanism is an
extension boundary, not a security sandbox.

