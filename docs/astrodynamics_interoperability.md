# Astrodynamics time, frames, and interoperability

## Internal frame and time definitions

Atmospheric flight remains in the NED/FRD conventions defined in
`coordinate_systems.md`. Interplanetary states use `HELIOS_ECLIPJ2000`: a
primary-centred inertial frame whose x axis is the J2000 mean equinox, z axis is the
J2000 mean ecliptic north pole, and y completes the right-handed triad. Exchange
states may use `J2000`, the mean equator/equinox frame. The current implementation
uses one fixed mean-obliquity rotation. Position and velocity rotate with the same
orthonormal matrix and round-trip numerically; no precession, nutation, Earth
orientation, or topocentric transformation is implied.

UTC conversion uses a bundled, auditable leap-second table. For the release epoch,
TAI minus UTC is 37 s and TT minus TAI is 32.184 s. Julian dates are generated for
UTC, TAI, and TT. A documented low-order periodic approximation supplies TDB minus
TT for preliminary interplanetary exchange. It is not a replacement for a SPICE
time kernel in precision navigation. The table is declared valid only through
2026-12-31, following [IERS Bulletin C 72](https://datacenter.iers.org/data/html/bulletinc-072.html),
which states that no leap second is introduced at the end of December 2026. The
TDB convention and approximation follow the public
[NAIF SPICE time documentation](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/time.html).

## CCSDS orbit exchange

The orbit-tour command writes a CCSDS Orbit Ephemeris Message using OEM 3.0 keyword
value notation, following [CCSDS 502.0-B-3 Orbit Data Messages](https://ccsds.org/publications/allpubs/).
The internal platform remains SI. The writer performs one explicit exchange-boundary
conversion from metres and metres per second to the OEM-required kilometres and
kilometres per second. The parser reverses that conversion. Tests round-trip headers,
epochs, states, identifiers, units, and malformed input.

The example identifiers (`HELIOS`, `FICTIONAL-SELENE-001`) are synthetic and exist to
exercise interoperability. They are not registered operational object identifiers.
The export is suitable for schema/tool experiments, not conjunction assessment or
navigation delivery.

## GMAT and SPICE interfaces

`gmat_validation/two_body_validation.script` and the generated
`gmat_two_body_validation.script` define a standalone fictional Earth point-mass
case, a Prince-Dormand propagator, and a numerical report. The comparison reader can
ingest a user-executed report, checks exact sample epochs, converts GMAT kilometres
to SI, and reports maximum/RMS position and velocity differences. The script syntax
uses the public GMAT
[Propagate](https://documentation.help/GMAT/Propagate.html) and
[ReportFile](https://documentation.help/GMAT/ReportFile.html) interfaces.

GMAT was not detected in the development environment and the script was not
executed. `spiceypy` was also absent and no kernels were supplied, so no SPICE
comparison was executed. The machine-readable status file records `executed=false`
for both. AeroGNC-Lab never treats tool detection, script generation, or an OEM file
write as external numerical validation.

To add valid evidence later:

1. Record the external tool release and exact public kernels/configuration.
2. Execute the generated case outside Python and retain its unedited report/log.
3. Run the provided comparison reader against a matching independently generated SI
   reference.
4. Declare tolerances before examining differences and investigate frame/time/model
   mismatches rather than tuning the tolerance afterward.
5. Update the validation report only with the actual command, hashes, metrics, and
   execution date.

## Known limitations

The leap-second table requires maintenance after its declared validity date. The
fixed ecliptic/equator rotation is adequate only for the synthetic J2000 examples.
The low-order TDB approximation, analytical fictional ephemerides, and point-mass
GMAT template omit effects needed for real navigation. No EOP, high-order gravity,
relativistic time transfer, clock covariance, kernel pedigree system, or operational
OD interface is claimed.
