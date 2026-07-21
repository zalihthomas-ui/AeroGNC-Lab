# Geodesy and Rotating-Planet Flight Mechanics

This optional model removes the flat, nonrotating-frame assumption from a
representative ascent without changing the original local-NED regression model. The
configured body, Orbis-A, is fictional; its dimensions, rotation, gravity, (J_2),
launch site, and flight results are synthetic.

## Frames and coordinates

The planet-fixed Cartesian frame (e) is Earth-centred/Earth-fixed-like (ECEF):

- (+z_e) is the positive rotation pole;
- (+x_e) intersects the equator at zero east longitude;
- (+y_e) completes a right-handed frame at (90^\circ) east longitude.

Geodetic coordinates are latitude \(\varphi\), east longitude \(\lambda\), and
ellipsoidal altitude \(h\). The axisymmetric reference ellipsoid uses equatorial
semi-axis \(a\), flattening \(f=(a-b)/a\), and first eccentricity squared
\(e^2=f(2-f)\). With prime-vertical radius

\[
N(\varphi)=\frac{a}{\sqrt{1-e^2\sin^2\varphi}},
\]

the geodetic-to-ECEF map is

\[
\mathbf r_e=
\begin{bmatrix}
(N+h)\cos\varphi\cos\lambda\\
(N+h)\cos\varphi\sin\lambda\\
\left(N(1-e^2)+h\right)\sin\varphi
\end{bmatrix}.
\]

ECEF-to-geodetic conversion uses a bounded fixed-point iteration and fails
explicitly if it does not converge. A dedicated polar branch avoids the longitude
and altitude singularity. Round trips are tested at the equator, mid-latitudes, and
near the pole.

The ECEF-to-local-NED direction-cosine matrix is

\[
C_{ne}=\begin{bmatrix}
-\sin\varphi\cos\lambda&-\sin\varphi\sin\lambda&\cos\varphi\\
-\sin\lambda&\cos\lambda&0\\
-\cos\varphi\cos\lambda&-\cos\varphi\sin\lambda&-\sin\varphi
\end{bmatrix}.
\]

It maps ECEF-resolved vectors to North-East-Down components. Local position is the
ECEF displacement from the geodetic launch site, resolved by this matrix. It is a
tangent-plane diagnostic, while integration remains in ECEF.

## Inertial/fixed state transformation

For positive eastward rotation \(\boldsymbol\Omega=[0,0,\Omega]^T\), the
inertial-to-ECEF state transformation includes transport velocity:

\[
\mathbf r_e=C_{ei}\mathbf r_i,\qquad
\mathbf v_e=C_{ei}\left(\mathbf v_i-\boldsymbol\Omega\times\mathbf r_i\right).
\]

The inverse restores \(\boldsymbol\Omega\times\mathbf r\). Tests verify state
round-trip consistency and the expected velocity of a point stationary in ECEF.
The library also exposes planet rotation and local transport rates in NED for the
later inertial-navigation layer.

## Rotating-frame dynamics

The six-state order is

\[
\mathbf x_e=[\mathbf r_e^T,\mathbf v_e^T]^T.
\]

Specific force \(\mathbf f_e\) contains thrust and aerodynamics but excludes gravity
and frame terms. The propagated equation is

\[
\dot{\mathbf r}_e=\mathbf v_e,
\qquad
\dot{\mathbf v}_e=\mathbf f_e+\mathbf g_{J_2,e}
-2\boldsymbol\Omega\times\mathbf v_e
-\boldsymbol\Omega\times(\boldsymbol\Omega\times\mathbf r_e).
\]

Central plus first-order (J_2) gravity is evaluated directly in ECEF. Coriolis and
centrifugal terms are separate public functions so their signs can be tested
independently. Surface gravity is reported along geodetic down, including the
centrifugal term.

## Configured demonstration

Run:

```bash
python -m aerognc.cli rotating-ascent \
  --config configs/rotating_planet_ascent.yaml
```

The deterministic Orbis-A case detects burnout at 3.350 s, apogee near 1091.8 m at
15.46 s, and impact near 31.54 s. The command writes ECEF, geodetic, local-NED,
atmospheric, load, event, and summary data plus a four-panel figure.

## Inertial quaternion 6-DOF composition

The optional rigid-body composition propagates

\[
\mathbf x_i=[\mathbf r_i^T,\mathbf v_i^T,q_{ib}^T,
(\boldsymbol\omega_{b/i}^b)^T]^T.
\]

Body-axis thrust, aerodynamics, and control are rotated into the inertial frame by
\(C_{ib}\); only planet gravity is added outside the body-force vector:

\[
\dot{\mathbf r}_i=\mathbf v_i,\qquad
\dot{\mathbf v}_i=C_{ib}\frac{\mathbf F_b}{m}+C_{ie}\mathbf g_e.
\]

The Euler equation retains prescribed variable inertia,

\[
\dot{\boldsymbol\omega}_{b/i}^b=I^{-1}
\left(\mathbf M_b-\boldsymbol\omega\times I\boldsymbol\omega
-\dot I\boldsymbol\omega\right),
\qquad
\dot q_{ib}=\tfrac12q_{ib}\otimes[0,\boldsymbol\omega_{b/i}^b].
\]

For aerodynamic loads, the inertial state is transformed to ECEF, planet-fixed wind
is subtracted there, and the resulting relative-air vector is rotated into body axes.
The controller reference begins as a local \(q_{nb}\) schedule and is explicitly
composed into \(q_{ib}\) at the current geodetic location. Planet and transport rates
are removed when reporting body rate relative to local NED.

Run the configured case with:

```bash
python -m aerognc.cli rotating-six-dof --config configs/rotating_six_dof.yaml
```

The synthetic eight-second run stores inertial, ECEF, geodetic, NED, attitude,
aerodynamic, control, mass, and load channels. Quaternion normalization error is
bounded below \(10^{-9}\) by the verified scenario.

## Limitations

- The ellipsoid is axisymmetric and rotates uniformly.
- Only central and first-order (J_2) gravity are included.
- Atmosphere and wind are inherited from the lower-atmosphere launch-site model; no
  global weather field rotates with longitude.
- The original ECEF ascent remains translational; the separately configured inertial
  composition adds rigid-body attitude without changing the flat-NED regression case.
- The local attitude reference neglects a commanded transport-rate feed-forward
  term; measured local-frame rate is still removed in feedback.
- Orbis-A results are software-verification evidence, not predictions for Earth or a
  real vehicle.
