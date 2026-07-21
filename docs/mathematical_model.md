# Mathematical Model

The equations in this document use the conventions defined in
[`coordinate_systems.md`](coordinate_systems.md). Model-specific assumptions and
validation evidence are expanded as implementation phases are completed.

## Three-degree-of-freedom point mass

\[
\dot{\mathbf r}_n=\mathbf v_n,
\qquad
\dot{\mathbf v}_n=\frac{\mathbf F_T^n+\mathbf F_A^n}{m}+\mathbf g_n.
\]

The air-relative velocity and scalar dynamic pressure are

\[
\mathbf V_a^n=\mathbf v_n-\mathbf v_{wind}^n,
\qquad
\bar q=\tfrac12\rho\|\mathbf V_a^n\|^2.
\]

Drag opposes \(\mathbf V_a^n\):

\[
\mathbf F_A^n=-\bar q S C_D(M)\frac{\mathbf V_a^n}{\|\mathbf V_a^n\|}.
\]

Mass is computed from the integrated/normalised synthetic thrust curve and is
bounded below by dry mass. The first implementation treats expelled propellant
momentum through specified thrust; no additional \(\dot m\mathbf v\) term is added.

## Rotating oblate-planet translation

The optional ECEF state includes central-plus-\(J_2\) gravity and the rotating-frame
terms explicitly:

\[
\dot{\mathbf r}_e=\mathbf v_e,
\quad
\dot{\mathbf v}_e=\mathbf f_e+\mathbf g_{J_2,e}
-2\boldsymbol\Omega\times\mathbf v_e
-\boldsymbol\Omega\times(\boldsymbol\Omega\times\mathbf r_e).
\]

Specific force \(\mathbf f_e\) contains thrust and drag but excludes gravity and
apparent terms. Geodetic/ECEF transformations and local-NED diagnostics are defined
in [geodesy and rotating-planet mechanics](geodesy_rotating_planet.md).

## Multidimensional aerodynamic coefficients

Regular-grid coefficient databases use direct multilinear interpolation over named
axes and return analytic in-cell gradients. All six coefficients share one complete
tensor grid and one explicit error/clamp/extrapolation policy. The file contract,
coefficient ordering, provenance hash, and gradient equation are documented in the
[aerodynamic database specification](aerodynamic_database.md).

## Six-degree-of-freedom rigid body

Applied body force excludes gravity. With \(C_{nb}\) mapping body to navigation,

\[
\dot{\mathbf r}_n=\mathbf v_n,
\qquad
\dot{\mathbf v}_n=\frac{C_{nb}\mathbf F_b}{m}+\mathbf g_n.
\]

Quaternion kinematics are

\[
\dot q_{nb}=\tfrac12q_{nb}\otimes[0,\boldsymbol\omega]^T.
\]

Euler's equation about the instantaneous centre of mass, including prescribed
inertia variation, is

\[
I\dot{\boldsymbol\omega}=\mathbf M_b-
\boldsymbol\omega\times(I\boldsymbol\omega)-\dot I\boldsymbol\omega.
\]

The model assumes the configured mass-property schedule already captures internal
mass redistribution and that off-axis momentum flux is negligible. Quaternion norm
is restored after accepted fixed steps; a zero-norm quaternion is a hard error.

## Interplanetary restricted N-body motion

The interplanetary state uses a nonrotating primary-centred ecliptic frame:

\[
\mathbf x=[x,y,z,v_x,v_y,v_z]^T,
\qquad \dot{\mathbf r}=\mathbf v.
\]

Orbiting bodies follow configured elliptical Keplerian ephemerides, including
inclination, ascending node and argument-of-periapsis rotations. Spacecraft acceleration includes central
gravity plus each body's direct and primary-frame indirect term:

\[
\dot{\mathbf v}=-\mu_0\frac{\mathbf r}{r^3}
+\sum_i\mu_i\left[
\frac{\mathbf r_i-\mathbf r}{\|\mathbf r_i-\mathbf r\|^3}
-\frac{\mathbf r_i}{\|\mathbf r_i\|^3}
\right].
\]

This is a restricted model: the spacecraft has no gravitational influence and the
prescribed planets do not mutually perturb one another. Encounter diagnostics compare
primary-relative energy and speed before/after a boundary while checking approximate
conservation of planet-relative excess-speed magnitude. See the full
[interplanetary mission model](interplanetary_mission.md) for RTN initialization,
transfer/flyby equations, scenario evidence, UI controls, and limitations.

## Universal conics, Lambert arcs, and maneuvers

Universal-variable propagation defines

\[
\alpha=\frac{2}{r_0}-\frac{v_0^2}{\mu},\qquad z=\alpha\chi^2,
\]

and solves the universal time equation with Stumpff \(C(z),S(z)\). Lagrange
coefficients \(f,g,\dot f,\dot g\) recover the endpoint state. The Lambert solver
uses the same functions to find departure/arrival velocities for a requested
zero-revolution time of flight. B-plane and flyby equations are given in
[advanced astrodynamics](advanced_astrodynamics.md).

An impulse is an exact velocity discontinuity. Its mass change obeys the ideal rocket
equation. During a finite burn,

\[
\mathbf a_T=\frac{T}{m}\hat{\mathbf d},\qquad
\dot m=-\frac{T}{I_{sp}g_0}.
\]

RTN directions are recomputed from instantaneous position and angular momentum.
Dry mass is enforced after every accepted step.

## Optional perturbations and full N-body motion

The J2 implementation uses the standard equatorial Cartesian acceleration. Solar
radiation pressure follows a cannonball area/mass model with inverse-square distance;
the relativistic option applies the first Schwarzschild post-Newtonian correction.
These terms are opt-in. The full N-body verification model propagates every massive
body with pairwise Newtonian forces and exposes total momentum, barycentre, and
kinetic-plus-potential energy.

## Linear flight dynamics and navigation error state

For \(\dot x=f(x,u)\), trim solves selected steady residuals and central differences
form \(A=\partial f/\partial x\) and \(B=\partial f/\partial u\). Continuous LQR
solves

\[
A^TP+PA-PBR^{-1}B^TP+Q=0,
\qquad K=R^{-1}B^TP.
\]

The 15-state navigation error ordering is
\([\delta p_n,\delta v_n,\delta\theta_b,\delta b_g,\delta b_a]^T\). Its nominal
quaternion propagation and covariance Jacobian follow the same NED/FRD convention as
the 6-DOF plant. See [flight-control analysis](flight_control_analysis.md) and
[error-state navigation](error_state_navigation.md).

## Envelope model and constrained ascent

The envelope workflow trims and linearises a nonlinear two-state pitch surrogate
\(x=[\alpha,q]^T\) using the same atmosphere, mass/inertia schedule, actuator limit,
and aerodynamic database as the ascent plant. A Hamiltonian LQR solution is designed
at every Mach-altitude-mass point and gain components are interpolated trilinearly.
Equations, between-grid verification, and uncertainty assumptions are in
[flight-envelope trim and scheduled control](flight_envelope.md).

The constrained pitch-plane ascent propagates
\([r_N,r_D,v_N,v_D,m_p,\theta]^T\). Throttle scales thrust and nominal propellant
flow together, while a bounded first-order pitch response follows a time schedule.
The offline search and online max-Q/load/angle governance are documented in
[constrained ascent guidance](constrained_ascent_guidance.md).

## Numerical integration

For \(\dot x=f(t,x)\), one classical fixed RK4 step of size \(h\) is

\[
\begin{aligned}
k_1&=f(t,x),\\
k_2&=f(t+h/2,x+hk_1/2),\\
k_3&=f(t+h/2,x+hk_2/2),\\
k_4&=f(t+h,x+hk_3),\\
x_{n+1}&=x_n+\frac{h}{6}(k_1+2k_2+2k_3+k_4).
\end{aligned}
\]

Scalar event functions are checked for directed sign crossings between accepted
steps. Event time/state is estimated by bracketed interpolation; terminal events
truncate the result. Convergence order and an independent SciPy comparison are
required evidence, not assumptions.

For variable-step work, the directly implemented Dormand--Prince 5(4) pair evaluates
seven stage derivatives. Its fifth-order update and embedded fourth-order estimate
use the published Butcher weights (b_i^{(5)}) and (b_i^{(4)}):

\[
x_{n+1}^{(5)}=x_n+h\sum_{i=1}^{7}b_i^{(5)}k_i,\qquad
e=x_{n+1}^{(5)}-x_{n+1}^{(4)}.
\]

The dimensionless root-mean-square error norm is

\[
E=\sqrt{\frac{1}{n}\sum_j
\left(\frac{e_j}{a_{\mathrm{tol}}+r_{\mathrm{tol}}
\max(|x_{n,j}|,|x_{n+1,j}|)}\right)^2}.
\]

A step is accepted when (E\leq1). The next step uses the bounded controller
(h_{n+1}=h_n\,\mathrm{clip}(0.9E^{-1/5},0.2,5)). State-dependent event roots are
refined on a cubic Hermite interpolant using endpoint states and derivatives;
safeguarded bisection stops at the configured time tolerance. Accepted/rejected step
counts and every derivative evaluation are retained as evidence.

For first-order uncertainty and targeting studies, finite central differences form
(A=\partial f/\partial x) and (B=\partial f/\partial p). The augmented equations
are

\[
\dot\Phi=A\Phi,\quad \Phi(t_0)=I,\qquad
\dot S=AS+B,\quad S(t_0)=0.
\]

The resulting state-transition matrix (Phi) and selected-parameter sensitivity
(S) are verified against a linear-system matrix exponential. Versioned JSON/NPZ
checkpoints preserve epoch, state, recommended next step, and checksummed metadata.
A separate logical-time scheduler computes each dispatch epoch from integer tick
number and task period, avoiding accumulated time drift and wall-clock-dependent
ordering.

## Near-planet orbit sandbox

The orbit sandbox reuses the same custom RK4 boundary for five deliberately distinct
plants: analytical force-free translation, central two-body gravity, prescribed
restricted-three-body differential acceleration, full pairwise finite-mass N-body
gravity, and a low-orbit composition with central gravity, (J_2), rotating-reference-
atmosphere drag, mass/area/(C_D), and optional dry-mass-bounded ideal impulses.
Governing equations, state/frame definitions, event semantics and finite-horizon
lifetime interpretation are specified in the [Satellite Orbit Sandbox](orbit_sandbox.md).

## Coefficient-driven aircraft plant

The fictional aircraft extends the state to planet-centred inertial translation,
body-to-inertial Hamilton quaternion, body rates, mass, three physical control-surface
states, and throttle. At each derivative stage, rotating-atmosphere-relative velocity
forms (alpha,eta,M,\bar q); synthetic (C_L,C_D,C_Y,C_l,C_m,C_n) then generate
FRD forces and moments. Lift loses slope and drag rises beyond the configured stall
boundary. Central plus (J_2) gravity, fuel flow, mass-scaled inertia and bounded
actuator dynamics remain explicit. The complete equations, signs, live input mapping,
stall-speed relation and 100 km research-ascent scope are specified in
[Fictional Aircraft Simulation and Live Flight](aircraft_simulation.md).
