# Optional GMAT validation interface

This directory contains an independent two-body comparison interface for NASA's
General Mission Analysis Tool (GMAT). GMAT is not a dependency of AeroGNC-Lab and
was not detected or executed for the current release. Consequently, the repository
does not contain or claim a GMAT numerical pass result.

`two_body_validation.script` propagates a wholly synthetic civilian verification
spacecraft around Earth with an Earth point-mass force model and reports elapsed
seconds plus Cartesian `EarthMJ2000Eq` state in GMAT's km/km-s units. It deliberately
uses a simple standard body/frame so a future external run can isolate propagation
and exchange-format differences from the fictional Helios catalog.

To create evidence in an environment with GMAT:

1. Record the exact GMAT release, platform, and executable hash if practical.
2. Open and execute `two_body_validation.script` explicitly; do not assume that
   executable discovery means it ran.
3. Retain the GMAT log and `gmat_two_body_report.txt`.
4. Compare the report through
   `aerognc.interoperability.compare_gmat_report` against the Python universal-conic
   states at the same 60 s epochs.
5. Record tolerances, force-model settings, frame, time scale, and any discrepancy
   before changing the validation status in project documentation.

The generated orbit-tour outputs also include a CCSDS OEM/KVN 3.0 engineering file.
The fictional Helios center and spacecraft identifiers are not operational registry
entries, so the OEM demonstrates syntax/unit/frame interoperability rather than an
agency-approved mission product.
