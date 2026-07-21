# Coordinate Systems and Engineering Conventions

This document is normative. Source code, configuration, plots, and tests use SI
units and these conventions unless a field explicitly carries another unit in its
name (for example, a human-facing configuration angle ending in `_deg`). Conversion
occurs once at the configuration boundary.

## Frames

### Local navigation frame, \(n\)

The navigation frame is a flat-Earth, local North-East-Down (NED) tangent frame
whose origin is the launch point:

- \(+x_n\): north, m
- \(+y_n\): east, m
- \(+z_n\): down, m

The baseline local 3-DOF/6-DOF models assume this tangent frame does not rotate
during a flight. Position is \(\mathbf r_n=[r_N,r_E,r_D]^T\) in metres and velocity is
\(\mathbf v_n=[v_N,v_E,v_D]^T\) in metres per second. Geometric altitude above the
launch datum is

\[
h=-r_D.
\]

This is not mean-sea-level altitude unless the launch datum is configured as such.

### Planet-fixed Cartesian frame, \(e\)

The optional rotating-planet translation uses a right-handed ECEF-like frame. Its
\(+z_e\) axis is the positive rotation pole, \(+x_e\) crosses the equator at zero
east longitude, and \(+y_e\) crosses the equator at \(90^\circ\) east. The planet
rotates eastward with \(\boldsymbol\Omega=[0,0,\Omega]^T\) expressed in ECEF.

Geodetic latitude is positive north, longitude is positive east, and altitude is
ellipsoidal height along the surface normal. The ECEF-to-NED matrix is named
\(C_{ne}\) and maps ECEF-resolved components to the local tangent frame. Integration
uses ECEF position/velocity; launch-relative NED and geodetic coordinates are logged
diagnostics. See [geodesy and rotating-planet mechanics](geodesy_rotating_planet.md).

The primary-centred interplanetary inertial frame remains separate: it is nonrotating
and ecliptic-aligned as documented in the interplanetary model. No function silently
mixes local NED, ECEF, and primary-centred inertial coordinates.

### Planet-centred inertial frame, \(i\)

The optional rotating-planet 6-DOF model propagates Cartesian position and velocity
in a nonrotating planet-centred inertial frame. At \(t=0\), its axes coincide with
the synthetic ECEF axes. The matrix \(C_{ei}(t)\) maps inertial components into ECEF;
the state transformation includes the transport term
\(\boldsymbol\Omega\times\mathbf r\). Gravity is evaluated in ECEF and rotated into
inertial components, while wind and atmosphere remain planet-fixed.

The quaternion \(q_{ib}\) maps FRD body components into the planet-centred inertial
frame. Its angular rate is \(\boldsymbol\omega_{b/i}^b\). Local \(q_{nb}\),
\(\boldsymbol\omega_{b/n}^b\), geodetic coordinates, and NED velocity are derived
diagnostics; they are not substituted into the inertial state without an explicit
transformation.

### Body frame, \(b\)

The right-handed body frame is fixed to the vehicle at its instantaneous centre of
mass and uses Forward-Right-Down (FRD):

- \(+x_b\): forward through the nose
- \(+y_b\): right
- \(+z_b\): down

Forces \([X,Y,Z]^T\) and moments \([L,M,N]^T\) are resolved in body axes. Positive
moments follow the right-hand rule: roll about \(+x_b\), pitch about \(+y_b\), and
yaw about \(+z_b\). Angular velocity is
\(\boldsymbol\omega_{b/n}^b=[p,q,r]^T\) in rad/s.

### Wind frame, \(w\)

The wind-frame \(+x_w\) axis is aligned with the vehicle velocity relative to the
air. \(+z_w\) lies in the body \(x_b-z_b\) plane and points generally down; \(+y_w\)
completes a right-handed frame. Angle of attack and sideslip are

\[
\alpha=\operatorname{atan2}(V_{a,z}^b,V_{a,x}^b),\qquad
\beta=\arcsin\left(V_{a,y}^b/\|\mathbf V_a^b\|\right).
\]

They are defined as zero below the configured minimum airspeed. Wind velocity is
the motion of the air mass relative to navigation coordinates, so
\(\mathbf V_a^n=\mathbf v_n-\mathbf v_{wind}^n\).

## Rotations and quaternions

The direction-cosine matrix \(C_{nb}\) maps body-resolved components into navigation
components:

\[
\mathbf a_n=C_{nb}\mathbf a_b,\qquad C_{bn}=C_{nb}^T.
\]

Quaternions are Hamilton, scalar-first arrays
\(q_{nb}=[q_0,q_1,q_2,q_3]^T\). They encode the active body-to-navigation rotation
and use Hamilton multiplication \(\otimes\). Composition is
\(q_{ac}=q_{ab}\otimes q_{bc}\): first rotate from \(c\) to \(b\), then from \(b\)
to \(a\). A vector is rotated by \(q\otimes[0,\mathbf v]\otimes q^*\).

For body angular rate resolved in body axes,

\[
\dot q_{nb}=\tfrac12 q_{nb}\otimes[0,\boldsymbol\omega_{b/n}^b].
\]

Quaternion sign is not physically unique; comparisons account for \(q\) and
\(-q\). Numerical propagation renormalises at accepted integration steps.

Euler angles are aerospace intrinsic yaw-pitch-roll (3-2-1): yaw \(\psi\) about
navigation down, pitch \(\theta\) about the intermediate right axis, then roll
\(\phi\) about body forward. The corresponding mapping is
\(C_{nb}=R_z(\psi)R_y(\theta)R_x(\phi)\). Euler angles are reporting and command
interfaces only; dynamics use quaternions.

## State ordering

The point-mass state is

\[
\mathbf x_{3D}=[r_N,r_E,r_D,v_N,v_E,v_D]^T.
\]

The rigid-body state is

\[
\mathbf x_{6D}=[r_N,r_E,r_D,v_N,v_E,v_D,
q_0,q_1,q_2,q_3,p,q,r]^T.
\]

The rotating-planet translational state is

\[
\mathbf x_e=[x_e,y_e,z_e,v_{x,e},v_{y,e},v_{z,e}]^T.
\]

The rotating-planet rigid-body state is

\[
\mathbf x_{6D,i}=[\mathbf r_i^T,\mathbf v_i^T,
q_{ib,0},q_{ib,1},q_{ib,2},q_{ib,3},
(\boldsymbol\omega_{b/i}^b)^T]^T.
\]

The constrained pitch-plane ascent state is

\[
\mathbf x_g=[r_N,r_D,v_N,v_D,m_p,\theta]^T,
\]

where \(m_p\) is remaining propellant and \(\theta\) is thrust-axis elevation above
the local horizontal.

Time is seconds, mass kilograms, inertia kg m\(^2\), force newtons, moment N m,
pressure pascals, density kg/m\(^3\), temperature kelvin, and angles/rates inside
the numerical core radians and radians per second.

## Orbit-sandbox planet-centred inertial frame

The near-planet orbit sandbox uses a spherical planet-centred inertial frame (P).
Its (+z_P) axis is the primary rotation axis. The relative satellite state is

\[
\mathbf x_o=[x_P,y_P,z_P,v_{x,P},v_{y,P},v_{z,P}]^T.
\]

For full N-body propagation, this six-state ordering is concatenated for the primary,
each configured moon, and the finite-mass satellite. Logged satellite coordinates are
formed relative to the propagated primary. The atmosphere velocity in the perturbed
model is (oldsymbol\omega_P\times\mathbf r_P). The frame and all body definitions
are fictional; real observational catalog coordinates are not silently substituted.

## Aircraft planet-centred inertial and live-display frames

The aircraft plant also propagates planet-centred inertial position/velocity but uses
(q_{IB}), the Hamilton scalar-first active rotation from FRD body components into
inertial components. Its state is

\[
\mathbf x_a=[\mathbf r_I^T,\mathbf v_I^T,q_{IB}^T,
\boldsymbol\omega_B^T,m,\delta_a,\delta_e,\delta_r,\tau]^T.
\]

At each position, a spherical local NED triad is formed from the radial up direction
and the planet spin axis. Local roll, pitch and heading are reports derived from
(C_{NB}=C_{NI}C_{IB}); the propagated attitude remains quaternion-based.

The live plot is a display-only east-north-altitude frame fixed at the initial
location. This mapping does not feed back into gravity, aerodynamics, or the state.
Imported meshes are converted once to (+x_B) forward, (+y_B) right, (+z_B) down.
The supported source conventions are declared in the UI; no axis orientation is
guessed from triangle geometry.
