# FMI interoperability preparation

`attitude_controller_interface/modelDescription.xml` is a deterministic FMI 3.0
Co-Simulation *interface contract* for the fictional quaternion attitude controller.
It fixes scalar-first body-to-NED quaternion inputs, FRD body-rate inputs, synthetic
controller parameters, body-moment outputs, SI units, value references, dependencies,
and a 10 ms intended communication step.

It is not an FMU. AeroGNC-Lab does not yet include the mandatory FMI C API source or
platform binary, and no `.fmu` archive has been built, schema-certified, imported, or
executed. `STATUS.json` records those facts explicitly. Regenerate the files with:

```powershell
python -m aerognc.cli fmi-interface
```

The next valid step would be to create a minimal C wrapper around the already
separated controller law, build platform binaries in CI, validate the complete
archive against the official FMI schema/checker, and execute a numerical
Python-versus-FMU equivalence case. Until those steps exist, this directory is useful
for interface review only and must not be advertised as FMI runtime validation.
