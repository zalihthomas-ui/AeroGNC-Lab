# Sensor Simulation and Vertical Navigation Filter

This document defines the compact vertical demonstration filter. The separate
[advanced-navigation workflow](advanced_navigation.md) defines the rotating-frame
strapdown mechanisation, delayed 15-state ESKF, replay, integrity, observability, and
NIS/NEES consistency work. The two fidelity levels are intentionally not conflated.

## Sensor models

The reusable sensor layer covers three-axis gyroscope and accelerometer, scalar
barometric altitude, and a six-component local position/velocity measurement
comparable to civilian GNSS. Every sensor supports:

- explicit sample rate and acquisition timestamp;
- Gaussian white noise, constant bias, and seeded random-walk bias drift;
- per-channel quantisation;
- delivery delay;
- random dropout probability and scheduled outage intervals; and
- a deterministic reset to the configured random seed.

Measurements carry both acquisition and availability timestamps. The generic model
has no hidden global random state. Error values in `configs/navigation_demo.yaml` are
synthetic and are not specifications for a real device.

## Scoped filter state and process model

The first release intentionally uses a transparent vertical navigation filter rather
than implying a production inertial navigation system. Its state is

\[
x=[h,\ v_U,\ b_a]^T,
\]

where \(h\) is altitude above the launch datum, \(v_U\) is upward velocity, and
\(b_a\) is vertical accelerometer bias. Given gravity-compensated vertical
acceleration measurement \(a_m\), the discrete process is

\[
\begin{aligned}
h_{k+1}&=h_k+v_k\Delta t+\tfrac12(a_m-b_a)\Delta t^2,\\
v_{k+1}&=v_k+(a_m-b_a)\Delta t,\\
b_{a,k+1}&=b_{a,k}+w_b.
\end{aligned}
\]

The corresponding Jacobian is evaluated explicitly. Acceleration uncertainty maps
through \([\Delta t^2/2,\Delta t,0]^T\); a configurable random walk drives the bias.

## Measurements and covariance

Barometer measurement is \(z_b=h+v_b\). The GNSS-like update uses altitude and
upward velocity with

\[
H_g=\begin{bmatrix}1&0&0\\0&1&0\end{bmatrix}.
\]

Measurement covariance values and the positive-definite initial covariance are
declared in YAML. The Joseph covariance update preserves symmetry and numerical
positive semidefiniteness; both are asserted in validation tests.

## Delay handling and limitations

Sparse plots retain measurements at their acquisition timestamps. At delivery, the
demo propagates delayed barometer/GNSS-like measurements to the current epoch using
the estimated vertical velocity and corrected acceleration. This first-order delay
compensation is explicitly not a fixed-lag smoother or out-of-sequence measurement
filter.

The accelerometer input is assumed to have been rotated into local vertical and
gravity-compensated using known truth attitude. The filter does not estimate full 3-D
attitude, gravity error, scale factor, Earth rotation, geodetic coordinates,
ionospheric effects, or clock states. Those are appropriate future extensions only
after this scoped model remains observable and verified.

In the seeded nominal workflow, raw barometer RMS error is 3.018 m and filtered
altitude RMS error is 0.453 m (85.0% reduction). These values validate the configured
software case; they do not claim real sensor performance.

## Fifteen-state extension

The reusable vertical demo is supplemented by a 15-state quaternion error-state
filter for NED position/velocity, attitude error, gyro bias, and accelerometer bias.
It propagates full 3-D IMU inputs and fuses NED GNSS-like position/velocity plus
barometric altitude with Joseph covariance updates. Its state ordering, Jacobian,
noise mapping, assumptions, and numerical tests are documented in the
[error-state navigation filter](error_state_navigation.md). It is an auditable
baseline, not a claim of production INS fidelity.
