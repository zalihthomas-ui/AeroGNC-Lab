# Rotating-frame strapdown navigation and delayed error-state filtering

## Scope

This workflow is a public-safe navigation verification case for a fictional civilian
research rocket. It uses synthetic motion and synthetic measurements. It is not a
production navigation unit, and its numerical results are not evidence for a real
vehicle. The configured command is:

```bash
python -m aerognc.cli advanced-navigation --config configs/advanced_navigation.yaml
```

The workflow extends the simpler vertical filter and the baseline 15-state ESKF with
rotating-oblate-planet mechanisation, coning/sculling compensation, delayed updates,
fixed-lag replay, innovation gating, sensor-health state, covariance-consistency
statistics, and an observability check.

## State and conventions

The nominal state is

\[
  \mathbf{x} = (\varphi,\lambda,h,\mathbf{v}^n,
  \mathbf{q}_{nb},\mathbf{b}_g,\mathbf{b}_a),
\]

where position is geodetic latitude, longitude, and ellipsoidal altitude; velocity is
resolved in local North-East-Down; and the Hamilton scalar-first quaternion rotates
body-FRD components into navigation-NED components. The 15-component error state is

\[
  \delta\mathbf{x} =
  [\delta\mathbf{p}^{n},\delta\mathbf{v}^{n},
  \delta\boldsymbol{\theta}^{n},\delta\mathbf{b}_g,
  \delta\mathbf{b}_a]^T.
\]

All internal angles are radians, angular rates are rad/s, distances are metres,
velocities are m/s, accelerations are m/s2, and covariance entries use the associated
squared SI units. Plot-only degree conversions are explicit.

## Strapdown mechanisation

Two adjacent IMU increments receive directly implemented two-sample compensation:

\[
\Delta\boldsymbol{\theta}_c = \Delta\boldsymbol{\theta}_1
+\Delta\boldsymbol{\theta}_2
+\frac{2}{3}\Delta\boldsymbol{\theta}_1\times\Delta\boldsymbol{\theta}_2,
\]

\[
\Delta\mathbf{v}_c = \Delta\mathbf{v}_1+\Delta\mathbf{v}_2
+\frac{1}{2}(\Delta\boldsymbol{\theta}_1+\Delta\boldsymbol{\theta}_2)
\times(\Delta\mathbf{v}_1+\Delta\mathbf{v}_2)
+\frac{2}{3}(\Delta\boldsymbol{\theta}_1\times\Delta\mathbf{v}_2
+\Delta\mathbf{v}_1\times\Delta\boldsymbol{\theta}_2).
\]

Quaternion attitude integration removes gyro bias and navigation-frame rotation. The
velocity update includes specific force, central-plus-J2 gravity, Coriolis, and
transport-rate terms. Geodetic position is advanced using meridian and prime-vertical
curvature radii. Quaternion normalization is an explicit numerical safeguard.

## Delayed error-state EKF

The filter linearizes the inertial error dynamics to propagate the error covariance
and process noise. Position/velocity and barometric innovations use the Joseph form

\[
\mathbf{P}^{+}=(\mathbf{I}-\mathbf{K}\mathbf{H})\mathbf{P}^{-}
(\mathbf{I}-\mathbf{K}\mathbf{H})^T
+\mathbf{K}\mathbf{R}\mathbf{K}^T,
\]

followed by nominal-state injection and the small-angle attitude reset. A delayed
measurement is applied at its timestamped fixed-lag snapshot; every later accepted
measurement and propagation record is then replayed in chronological order. Records
older than the configured lag are rejected explicitly instead of being applied at the
wrong epoch.

Normalized innovation squared (NIS) gates protect each measurement channel. Repeated
rejections move a sensor from healthy to degraded and then failed; repeated accepted
innovations allow recovery. The demo injects deterministic GNSS and barometer
dropout, bias-step, stuck, and spike conditions and reports accepted/rejected counts.

## Verification evidence

The configured 22 s case verifies:

- quaternion norm and covariance symmetry/positive semidefiniteness;
- full 15-state local observability-Gramian rank for the defined maneuver;
- fixed-lag replay of up to 18 propagation steps;
- measurable coning/sculling improvement over uncompensated increments;
- bounded RMS position, velocity, and attitude errors;
- NIS and normalized-estimation-error-squared ensemble consistency fractions;
- deterministic fault injection, rejection, health degradation, and recovery.

The consistency ensemble uses the configured fixed master seed and chi-square bounds.
It is a statistical software check for the synthetic assumptions, not certification
that the covariance models represent a physical navigation system. Earth-orientation
parameters, lever arms, clock states, scale-factor/misalignment calibration, terrain,
ionosphere, multipath, relativistic corrections, and operational GNSS processing are
outside this version's scope.
