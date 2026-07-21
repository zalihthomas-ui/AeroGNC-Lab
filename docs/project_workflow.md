# Engineering Project Workflow

An AeroGNC project is a portable YAML description of named scenarios. It keeps
configuration, deterministic inputs, completed evidence, and reports together without
making the numerical models depend on the desktop interface. The command line and the
workbench use the same service layer.

## Create and validate a project

From the repository root:

```powershell
python -m aerognc.cli project init projects\my_study --name "My study"
python -m aerognc.cli project validate projects\my_study\project.aerognc.yaml
```

The generated project is intentionally empty. Copy or create a supported configuration
inside the project root, then add a scenario to `project.aerognc.yaml`. The checked-in
[`portfolio_demo.aerognc.yaml`](../projects/portfolio_demo.aerognc.yaml) is a complete
example with 3-DOF, 6-DOF, and orbit-tour scenarios.

Project paths are resolved from the project file, never from the process working
directory. Referenced configuration files and the result directory must remain inside
the declared workspace root. Unknown keys, duplicate scenarios, unsafe paths, and
missing configurations fail before propagation.

## Run and inspect evidence

```powershell
python -m aerognc.cli project run projects\portfolio_demo.aerognc.yaml nominal-3dof
python -m aerognc.cli project list projects\portfolio_demo.aerognc.yaml
python -m aerognc.cli project report projects\portfolio_demo.aerognc.yaml RUN_ID
```

Every request records a completed, failed, or cancelled immutable manifest. A completed
run stores unit-labelled channels in compressed NumPy and readable CSV form, plus a
JSON description and SHA-256 hashes. The local SQLite file is only a rebuildable index;
the run directories are the authoritative evidence.

The report is a self-contained, offline HTML file. It reloads committed evidence and
shows provenance, warnings, requirement outcomes, events, extrema, and selected SVG
histories. This prevents an interface from reporting transient objects that were never
saved.

## Compare runs

```powershell
python -m aerognc.cli project compare projects\portfolio_demo.aerognc.yaml `
  BASELINE_RUN_ID CANDIDATE_RUN_ID --channels altitude_m,mach
```

Comparison uses the intersection of the two time ranges, interpolates onto a declared
uniform grid, and rejects missing or differently dimensioned channels. It reports bias,
RMS difference, maximum absolute difference, final difference, and correlation. The
JSON comparison and HTML report are written alongside project results.

## Reproducibility contract

- A run input fingerprint includes project/scenario identity, configuration hash,
  random seed, solver settings, scalar parameters, and safety scope.
- Measured runtime and generated file locations are excluded from that fingerprint.
- Completed run directories cannot be overwritten.
- Writes use a temporary sibling directory and atomic rename.
- Artifact hashes are checked when data are reloaded.
- Seeded workflows receive the seed explicitly; hidden global random state is not part
  of the project API.
- Generated reports are derived views and do not alter the immutable run manifest.

## Recovery

If the process stops before the atomic rename, the incomplete `.partial-*` directory is
not considered a run and may be reviewed or removed manually. If the SQLite index is
lost or damaged, reconstruct it from manifests using the `ResultStore.rebuild_index()`
Python API. Failed and cancelled manifests remain inspectable but never masquerade as
completed datasets.

