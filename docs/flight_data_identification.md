# Synthetic flight-data alignment and robust identification

## Purpose and reproducible command

This workflow demonstrates how asynchronous flight-test-style records can be planned,
aligned, cleaned, identified, diagnosed, and independently validated without using
proprietary or real vehicle data. The plant, clocks, dropouts, outliers, and sensor
errors are synthetic and describe a fictional civilian single-axis experiment.

```bash
python -m aerognc.cli flight-data-identification \
  --config configs/flight_data_identification.yaml
```

The command writes the original command/reference log, asynchronous sensor log,
aligned/cleaned CSV, automatic JSON report, and a publication-style diagnostic
figure. The generated input logs contain measurements only; truth parameters are used
after fitting solely to assess the synthetic benchmark.

## Asynchronous record and clock model

The command logger and sensor logger run at different configured sample rates. The
sensor clock follows

\[
t_s=(1+d\,10^{-6})t_r+b,
\]

where \(b\) is offset in seconds and \(d\) is drift in parts per million. Six common
Gaussian synchronization markers are detected by amplitude-weighted centres. An
ordinary two-parameter least-squares fit recovers scale and offset; sensor timestamps
are mapped to the reference clock before signal processing. The configured case
recovers the 0.370 s offset within 0.1 ms and the 85 ppm drift within 7 ppm.

Resampling uses linear interpolation only when adjacent finite source observations
are separated by no more than the declared maximum gap. Long dropouts remain NaN and
are never silently bridged. A locally detrended Hampel filter fits a quadratic to
neighbours excluding the candidate sample, uses a median-absolute-deviation scale,
and replaces isolated spikes. Samples next to missing intervals are not classified as
isolated outliers.

## Identified model

The synthetic pitch channel is

\[
I\ddot\theta + c\dot\theta + k\theta = u + d_0,
\]

and is estimated in the linear regression form

\[
\dot q=a_\theta\theta+a_q q+b_u u+d.
\]

Angle and rate are locally smoothed with an explicitly implemented polynomial fit;
the same fit supplies the rate derivative. Huber iteratively reweighted least squares
reduces the influence of residual outliers. The physical mapping is

\[
I=1/b_u,\quad c=-a_q/b_u,\quad k=-a_\theta/b_u,\quad d_0=d/b_u.
\]

The coefficient covariance is transformed with the analytical Jacobian to report
approximate 95% intervals. The final 30% of the record is excluded from fitting. The
identified continuous plant is then forward-integrated from the split state with the
custom RK4 solver and compared with that held-out response.

## Diagnostics and acceptance

Numerical acceptance covers clock offset/drift and marker residual, outlier count,
parameter relative error and confidence-interval coverage, coefficient of
determination, missing-data preservation, held-out angle/rate RMS, residual
autocorrelation, Durbin-Watson, Ljung-Box, and input/residual correlation. Adjacent
local-polynomial derivatives share most samples, so whiteness diagnostics deliberately
take one residual per non-overlapping derivative window; all finite samples remain in
the parameter fit.

For the reference record the fit has \(R^2\approx0.9983\); held-out pitch and rate RMS
are approximately 0.076 deg and 0.072 deg/s. Estimated inertia, damping, stiffness,
and disturbance moment are within 0.4% of their synthetic values. These repeatable
numbers validate the software workflow only. A linear rigid plant, approximate local
covariance, Gaussian background noise, affine clock, known marker correspondence, and
single-record excitation are important limitations; real flight-data work would add
calibration, uncertainty in inputs, nonlinear/parameter-varying models, independent
experiments, and engineering review.
