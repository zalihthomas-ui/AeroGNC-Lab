# Fictional Aircraft Simulation and Live Flight

## Scope

The Aircraft Flight workflow models the fictional civilian `Aquila-X1` research
aircraft. It is intended for flight-mechanics education, controller experiments,
software verification, and portfolio demonstration. Every coefficient, dimension,
mass property, engine value and stall boundary is synthetic. Nothing in the model is
type-certified data for a real aircraft.

Two workflows use the same physical plant:

1. a deterministic hands-off batch run that writes numerical evidence and plots; and
2. a live 3D player that converts keyboard or optional XInput commands into actuator
   requests and propagates the resulting equations in real time.

The live motion is not scripted. The imported 3D object is transformed by the
calculated position and quaternion attitude.

## Frames, signs and state ordering

The planet-centred inertial frame (I) is right-handed. Its (+z_I) axis is the
fictional planet spin axis. The local navigation basis is north-east-down (NED). The
body frame (B) is forward-right-down (FRD):

- (+x_B): forward;
- (+y_B): right wing;
- (+z_B): down;
- positive body rotations follow the right-hand rule; and
- the Hamilton scalar-first quaternion (q_{IB}) actively maps body components into
  inertial components.

The 18-state ordering is

\[
\mathbf x=
\begin{bmatrix}
\mathbf r_I(3) &
\mathbf v_I(3) &
q_{IB}(4) &
\boldsymbol\omega_B(3) &
m &
\delta_a & \delta_e & \delta_r & \tau
\end{bmatrix}^{\mathsf T}.
\]

Position is in metres, velocity in metres per second, body rate in radians per second,
mass in kilograms, control deflections in radians, and throttle (	au) is a fraction
from zero to one. The quaternion occupies indices 6 through 9 and is normalized after
every accepted RK4 step.

Initial latitude, longitude, heading, flight-path angle, bank and angle of attack are
entered in degrees and converted once at the validated configuration boundary. The
local body attitude uses the documented aerospace 3-2-1 yaw-pitch-roll sequence.

## Atmosphere-relative velocity and angles

Atmospheric velocity includes planet rotation and configured local wind:

\[
\mathbf v_{\mathrm{atm}}^I=
\boldsymbol\omega_P\times\mathbf r^I+C_{IN}\mathbf v_{\mathrm{wind}}^N,
\]

\[
\mathbf v_{\mathrm{air}}^B=C_{BI}
\left(\mathbf v^I-\mathbf v_{\mathrm{atm}}^I\right).
\]

For $\mathbf v_{\mathrm{air}}^B=[u,v,w]^{\mathsf T}$,

\[
V=\sqrt{u^2+v^2+w^2},\qquad
\alpha=\operatorname{atan2}(w,u),\qquad
\beta=\sin^{-1}\left(\frac{v}{V}\right).
\]

With FRD axes, positive $\alpha$ gives positive body-down velocity relative to a
nose-up body orientation. Dynamic pressure and Mach number are

\[
\bar q=\frac{1}{2}\rho V^2,
\qquad
M=\frac{V}{a}.
\]

The ISA/reference atmosphere continues to 1500 km so aerodynamic forces naturally
vanish during a high-altitude attempt instead of switching the rigid-body plant to a
different animation.

## Synthetic aerodynamic model

Before stall, the longitudinal coefficients include

\[
C_L=C_{L0}+C_{L\alpha}\alpha+C_{L\delta_e}\delta_e
+C_{Lq}\frac{q\bar c}{2V},
\]

\[
C_D=C_{D0}+kC_L^2+C_{D,\mathrm{stall}}f_s^2+C_{D,\mathrm{control}},
\]

\[
C_m=C_{m0}+C_{m\alpha}\alpha+C_{mq}\frac{q\bar c}{2V}
+C_{m\delta_e}\delta_e.
\]

Lateral coefficients $C_Y,C_l,C_n$ use sideslip, nondimensional roll/yaw rates,
aileron and rudder. Every displayed `CL`, `CD`, `Cm`, wing-area and mass input enters
the force or equation-of-motion calculation; changing one changes the propagated
state in the automated sensitivity test.

### Stall representation

The prepared stall angle and $C_{L\max}$ are synthetic. Up to the configured stall
angle, linear lift is bounded by $C_{L\max}$. Beyond it, lift magnitude decays toward
the configured post-stall fraction while a stall factor $f_s\in[0,1]$ increases the
drag coefficient. The transition completes over the configured ten-degree model
band. This gives visible lift loss and drag rise without pretending to reproduce
unsteady separated flow.

The live stall-speed reference is calculated, not hard-coded:

\[
V_s=\sqrt{\frac{2m g_0 n}{\rho S C_{L\max}}}.
\]

It therefore changes with mass, density, wing area, maximum lift and selected load
factor. It is an onset estimate for this synthetic coefficient model, not a flight
manual value. Spin entry/recovery, hysteresis, buffet, ground effect and dynamic stall
are omitted.

### Forces and moments

Wind-axis aerodynamic force is

\[
\mathbf F_W=\bar qS
\begin{bmatrix}-C_D&C_Y&-C_L\end{bmatrix}^{\mathsf T}.
\]

The implemented $C_{BW}(\alpha,\beta)$ maps it into FRD body components. Moments are

\[
\mathbf M_B=\bar qS
\begin{bmatrix}bC_l&\bar cC_m&bC_n\end{bmatrix}^{\mathsf T}.
\]

The diagnostic dashboard plots the exact coefficients used at each output sample.

## Propulsion, fuel and actuators

Air-breathing thrust is aligned with (+x_B) and depends on throttle, density ratio,
Mach operating margin and altitude operating margin. It falls to zero above its
configured envelope. The optional fictional rocket assist is also aligned with
(+x_B) and consumes mass according to

\[
\dot m_{\mathrm{rocket}}=-\frac{T_R}{I_{sp}g_0}.
\]

Air-breathing fuel flow is proportional to the actual throttle state. Total mass is
never projected below dry mass. Reference inertia scales with mass and the rotational
equation includes the resulting inertia-rate term.

Normalized pilot commands are converted to physical surface targets. Positive pull-up
command requests negative elevator under the documented coefficient signs. Each
surface follows a first-order response with an explicit position limit and symmetric
rate limit. Throttle also follows a first-order state. The player therefore cannot
instantaneously teleport a control surface.

## Nonlinear equations of motion

With applied body force excluding gravity,

\[
\dot{\mathbf r}^I=\mathbf v^I,
\]

\[
\dot{\mathbf v}^I=C_{IB}\frac{\mathbf F_B}{m}+\mathbf g^I_{2B}+\mathbf g^I_{J_2},
\]

\[
\dot q_{IB}=\frac{1}{2}q_{IB}\otimes
\begin{bmatrix}0&\boldsymbol\omega_B^{\mathsf T}\end{bmatrix}^{\mathsf T},
\]

\[
I\dot{\boldsymbol\omega}_B=
\mathbf M_B-\boldsymbol\omega_B\times(I\boldsymbol\omega_B)
-\dot I\boldsymbol\omega_B.
\]

Central gravity and $J_2$ use planet-centred position. The custom fixed-step RK4
solver evaluates aerodynamics, propulsion, mass, gravity and command-dependent
actuator derivatives at every stage. Ground impact and the 100 km reference crossing
are explicit events.

The reported turn rate is the numerical derivative of unwrapped local heading from
the propagated attitude history. It is not a lookup-table "advertised" turn rate.
Load factor is calculated from modeled lift divided by current weight.

## Live controls

Open `run_solver.bat`, select **Aircraft Flight**, then choose **Fly Live with
Keyboard / Controller**. Click the 3D window so it has keyboard focus.

| Input | Command |
|---|---|
| Arrow left/right | roll/aileron request |
| Arrow up/down | pull-up/push-over request |
| A / D | rudder request |
| W / S | increase/decrease throttle |
| Hold R | rocket assist while held |
| T | toggle the research-ascent attitude aid |
| P or Space | pause |
| C | cycle chase/orbit/top/free camera |
| Home | reset the exact initial condition |
| Escape | close player |

On Windows, an available XInput controller can use left stick for bank/pitch,
right-stick X for rudder, right trigger for throttle and A while held for rocket
assist. The adapter is optional: missing libraries or a disconnected controller
return a neutral snapshot and never prevent keyboard operation.

## Imported 3D models

The player accepts bounded UTF-8 OBJ and ASCII/binary STL files. OBJ polygon faces are
triangulated. The UI offers three declared source-axis conventions and converts them
to FRD. Files are subject to byte, vertex and triangle caps so an accidental huge mesh
fails clearly.

Imported geometry changes **appearance only**. It does not infer wing area, inertia,
mass, reference length or aerodynamic coefficients from triangles. This separation is
intentional: arbitrary visual topology is not trustworthy engineering data. The
repository includes `assets/models/aquila_x1.obj` as a small example.

## 100 km research-ascent aid

The optional `T` aid ramps local pitch toward 78 degrees, levels roll, requests full
throttle and enables the fictional rocket assist. It is included so the same plant can
be observed as density and aerodynamic authority diminish. The configured automated
benchmark crosses the 100 km reference boundary with calculated fuel depletion,
gravity, atmosphere, mass and attitude dynamics.

This aid contains no target, pursuit, interception, terminal homing or engagement
logic. A 100 km crossing is not proof of orbit: the vehicle may remain on a suborbital
trajectory. It is also not structural, thermal, stability-and-control, propulsion, or
flight-safety evidence for a real design.

## Reproducible commands

Hands-off batch run:

```bash
python -m aerognc.cli aircraft --config configs/aircraft_sandbox.yaml
```

Live flight with the bundled visual mesh:

```bash
python -m aerognc.cli fly-aircraft --config configs/aircraft_sandbox.yaml
```

The batch output includes trajectory CSV, summary and limitations JSON, 3D path, and
a coefficient/control dashboard. Tests cover configuration rejection, aerodynamic
coefficient response, stall, mass sensitivity, actuator limits, deterministic trim,
turn response, mesh parsing, controller normalization and the 100 km boundary case.
