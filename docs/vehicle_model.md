# Fictional Vehicle Model

The baseline **Asteria-SR1** is a wholly fictional civilian research/sounding rocket.
Every geometry, thrust, mass-property, aerodynamic, and actuator value is synthetic;
the configuration does not correspond to a real vehicle or operational system.

The separate fictional **Aquila-X1** civilian research-aircraft plant has its own
coefficient, propulsion, fuel, actuator, stall, live-control, and high-altitude
assumptions. They are documented in
[Fictional Aircraft Simulation and Live Flight](aircraft_simulation.md); its imported
OBJ/STL geometry is deliberately visual only.

## Configuration and propulsion

[`configs/vehicle_asteria_sr1.yaml`](../configs/vehicle_asteria_sr1.yaml) is the
single readable source for baseline vehicle values. The piecewise-linear thrust
curve is integrated exactly segment-by-segment. The deliberately transparent
solid-motor approximation makes propellant consumption proportional to delivered
impulse:

\[
m_p(t)=m_{p,0}\left(1-\frac{I(t)}{I_{total}}\right),\qquad
\dot m_p=-m_{p,0}\frac{T(t)}{I_{total}}.
\]

Thrust is zero outside the tabulated interval, and remaining propellant is bounded
to \([0,m_{p,0}]\). This is an engineering simulation assumption, not a motor-design
or internal-ballistics model.

## Mass properties

Mass, centre-of-gravity location measured aft from the fictional nose datum, and
the full body inertia tensor interpolate linearly with remaining propellant fraction.
Both wet and dry tensors must be symmetric positive definite. The 6-DOF equations
receive the corresponding prescribed \(\dot I\). Slosh, flexible modes, propellant
geometry, and momentum flux not represented by specified thrust are outside the
first release.

## Aerodynamics

Zero-angle drag is a synthetic Mach table with explicit clamp/error/extrapolation
policy. Angle, body-rate, and control derivatives form a transparent baseline:

\[
C_D=C_{D0}(M)+C_{D\alpha^2}\alpha^2,
\quad C_Z=-C_{N\alpha}\alpha,
\quad C_m=C_{m\alpha}\alpha+C_{mq}\hat q+C_{m\delta}\delta.
\]

Equivalent documented lateral-directional expressions are implemented for
\(C_Y,C_l,C_n\). Drag is constructed opposite air-relative velocity, while force
and moment coefficient signs follow FRD/right-hand conventions. The interface is
designed so a future public, non-proprietary CFD table depending on Mach, angle of
attack, and other variables can replace the baseline provider.

The v0.3 tabulated provider implements that interface directly. A long-form CSV must
contain a complete Cartesian tensor of named independent axes plus all six body-axis
coefficients. The loader rejects duplicate or missing nodes, non-finite values, and
unsupported columns. Multilinear interpolation and local coefficient gradients use
an explicit clamp, error, or linear-extrapolation policy. The synthetic demonstration
database depends on Mach, angle of attack, and pitch-control deflection; the interface
supports additional axes without changing the force/moment consumer. Source hashes,
axis ranges, and boundary-use counts are written to the analysis report. See the
[aerodynamic database specification](aerodynamic_database.md).

## Actuators

Each channel applies command delay, first-order lag, rate limiting, and position
saturation. A diagonal synthetic effectiveness map converts requested roll, pitch,
and yaw moments into three bounded abstract actuator channels. It does not describe
a real steering mechanism.

## Generic staging

The optional fictional multistage model gives every stage a unique name, dry mass,
local-time thrust curve, global ignition time, and optional separation time. Ignition
must follow prior-stage burnout and separation. For attached stage \(k\),

\[
m_k(t)=m_{d,k}+m_{p,k}(t),\qquad
m(t)=m_{payload}+\sum_{k\in attached}m_k(t).
\]

Separation is an explicit dry-mass discontinuity after propellant depletion. The
model checks zero thrust at motor endpoints, exact jettison accounting, event order,
and the time-varying retained dry-mass floor. It does not model internal ballistics,
plume interaction, hot staging, or separation-body trajectories.

## Deployable recovery

The recovery area remains zero through a configured trigger and deployment delay,
ramps to a reefed area, holds, then ramps continuously to full area. The signed
one-axis drag force is

\[
F_D=-\tfrac12\rho v_D|v_D|C_D A(t).
\]

The vertical benchmark reports deployment transitions, opening load, apogee, and
descending ground contact. It is deliberately a design-study model, not parachute
certification or a coupled canopy/vehicle dynamics solver. Full assumptions and the
YAML workflow are in [multistage and recovery analysis](multistage_recovery.md).

## Imported engineering data

`engineering_data.py` provides strict CSV boundaries for thrust, time-varying mass
properties, and regular-grid aerodynamics. Headers carry SI units or canonical
dimensionless/radian names. All values must be finite, independent axes monotonic,
and inertia tensors positive definite. Each import stores the resolved source path,
SHA-256, interpolation method, and explicit exterior policy. Ambiguous headers such
as `time`, `thrust`, or `alpha_deg` are rejected instead of silently converted.
