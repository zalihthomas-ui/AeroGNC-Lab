# Fifteen-State Error-State Navigation Filter

## Purpose and state

The original three-state vertical EKF remains the default end-to-end sensor demo.
`gnc/error_state_ekf.py` adds a carefully scoped 3-D inertial/GNSS/barometric
baseline. It uses a flat, nonrotating local NED frame and body FRD axes. The nominal
state contains NED position and velocity, Hamilton scalar-first body-to-navigation
quaternion \(q_{nb}\), gyro bias, and accelerometer bias.

The local error vector is

\[
\delta x=[\delta p_n,\ \delta v_n,\ \delta\theta_b,\
\delta b_g,\ \delta b_a]^T\in\mathbb R^{15}.
\]

Position is metres, velocity m/s, attitude error radians, gyro bias rad/s, and
accelerometer bias m/s². Covariance follows this exact ordering.

## Nominal propagation

Bias-corrected IMU values are

\[
\omega=\omega_m-b_g,\qquad f_b=f_m-b_a.
\]

Specific force is rotated with \(C_{nb}\) and local gravity is added. Position uses
the constant-acceleration increment; velocity uses the same acceleration; attitude
is updated with the exponential quaternion of \(\omega\Delta t\) and renormalised.
This multiplication order matches the project’s quaternion derivative convention.

The continuous error Jacobian includes position/velocity coupling,
\(-C_{nb}[f_b]_\times\) attitude-to-velocity coupling,
\(-C_{nb}\) accelerometer-bias coupling, rate cross-coupling, and gyro-bias coupling.
A second-order transition approximation propagates covariance. Continuous white
accelerometer/gyro noise densities and bias random walks are mapped into a discrete
process covariance with the configured time step.

## Measurements and injection

The civilian GNSS-like update observes all three NED position and velocity
components. Barometric altitude observes negative NED down position. Both require an
explicit positive-definite measurement covariance. The Joseph covariance form is
used before the estimated error is injected into position, velocity, quaternion, and
bias nominal states. Covariance is symmetrised and checked for loss of positive
semidefiniteness after every predict/update.

Tests show that an aligned stationary case with body specific force
\([0,0,-g]^T\) remains stationary, quaternion norm stays unity, covariance remains
valid, a known yaw rate integrates to the expected angle, and GNSS/barometer updates
reduce position error and uncertainty.

## Assumptions and limitations

This is not a production strapdown INS. It omits Earth rotation, transport rate,
ellipsoidal/geodetic conversion, coning/sculling compensation, IMU scale-factor and
misalignment states, GNSS clock/atmospheric errors, lever arms, delayed-state replay,
magnetic aiding, observability management, and fault detection. The filter is useful
for architecture and covariance verification on a synthetic short-duration research
flight. It does not claim navigation performance for a real vehicle or planet.
