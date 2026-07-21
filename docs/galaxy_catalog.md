# Milky Way and planetary catalog layer

## What is included

The repository bundles three deliberately separate data products:

1. Approximate Milky Way context (barred-spiral morphology, roughly 100,000 light-year
   disk, an estimated 100–400 billion stars, the Sun in the Orion Spur, and an
   approximately 230 million-year Galactic orbit).
2. The eight IAU Solar System planets with selected mean descriptive properties.
3. A point-in-time snapshot of every confirmed planet returned by the NASA Exoplanet
   Archive `pscomppars` table at acquisition: 6,324 planets in 4,738 host systems.

This is not “all planets in the Milky Way.” Most Galactic planets are unobserved, the
true total is unknown, and discovery catalogs are strongly selection-biased. The
context values are approximate, while blank exoplanet fields mean the selected
composite parameter was not reported. NASA describes the Milky Way as a barred
spiral more than 100,000 light-years across and explains that observations imply
planetary systems are extremely common; see [NASA Galaxy Basics](https://science.nasa.gov/universe/galaxies/),
[NASA's scale overview](https://science.nasa.gov/universe/exoplanets/our-milky-way-galaxy-how-big-is-space/),
and the [Milky Way structure illustration](https://science.nasa.gov/resource/the-milky-way-galaxy/).
The eight-planet classification follows [NASA's planet overview](https://science.nasa.gov/solar-system/planets/)
and [IAU Resolution B5](https://www.iau.org/static/resolutions/Resolution_GA26-5-6.pdf).

## Provenance and reproducibility

`data/catalogs/nasa_confirmed_exoplanets.csv` was retrieved at
2026-07-19T22:05:28Z using the official
[NASA Exoplanet Archive TAP service](https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html).
The adjacent metadata file records the exact ADQL query, table, selected fields,
retrieval time, row count, and SHA-256. The loader refuses a modified CSV, duplicate
planet names, unsorted rows, non-finite reported numbers, or metadata/count mismatch.
The refresh utility queries the same official endpoint and atomically replaces the
snapshot only after a schema, uniqueness, and minimum-size check.

```powershell
python scripts/update_exoplanet_catalog.py
python -m aerognc.cli catalog --query TRAPPIST-1 --output results/catalog_query
python -m aerognc.cli catalog --max-distance-pc 50
python -m aerognc.cli workbench
```

The snapshot fields cover names, host-system confirmed-planet count, discovery
method/year, orbital period, semimajor axis, radius, mass, system distance, ICRS
right ascension/declination, and selected stellar properties. These are composite
published values, not complete uncertainty distributions and not instantaneous
Cartesian ephemerides.

## Coordinate conversion

For rows with ICRS direction and distance, AeroGNC-Lab directly converts the ICRS
unit vector into Galactic axes with a fixed orthonormal matrix. Heliocentric
Cartesian coordinates use parsecs: +x points toward Galactic longitude zero (the
Galactic-centre direction), +y toward longitude 90 degrees, and +z toward the north
Galactic pole. The transform and its transpose round-trip away from angular wrapping;
the known Galactic-centre direction is an automated check.

The resulting plot is a map of *confirmed detections around the Sun*. It is not an
image of the Galaxy's intrinsic planet distribution. Transit survey footprints,
microlensing fields, telescope sensitivity, follow-up practice, distance errors, and
missing values create the visible structure. Discovery-year colours and method
counts help expose those biases rather than hiding them behind a decorative spiral.
The unified workbench also provides a rotatable, zoomable and pickable 3D host-system
view for the current filters. Multiple planets at the same host are collapsed into
one point, and the inspector lists the selected planets rather than implying that
overlapping marks are separate stars.

## Simulation boundary

Real catalog rows are read-only astronomy context. They are not automatically passed
to the interplanetary solver because most lack orbital phase, orientation, complete
mass, covariance, host-star state, and a common high-precision epoch. Transfers
between different host stars are interstellar problems and cannot be represented
honestly by the patched-conic planetary model. Executable portfolio missions continue
to use the fully specified fictional Helios system. A future real Solar System mode
would require explicitly selected public SPICE kernels, frame/time pedigree, and
independent validation before any numerical claim.

## Limitations

- The confirmed catalog changes continually; the bundled data are a dated snapshot.
- Composite values can be revised and missingness is not random.
- Only selected scalar fields are retained to keep the public repository compact.
- No Gaia covariance, proper motion, radial velocity, stellar multiplicity model,
  habitability score, or occurrence-rate correction is implemented.
- The visualization uses heliocentric positions and does not model the Galactic
  potential, spiral-arm dynamics, dust extinction, or stellar evolution.

These limits are intentional: the feature demonstrates provenance, validation,
coordinate mathematics, data quality, and understandable UI browsing without
fabricating a complete Milky Way database.
