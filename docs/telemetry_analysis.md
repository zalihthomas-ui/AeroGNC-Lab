# Telemetry Reconstruction and Batch Smoothing

## Purpose and data boundary

This workflow turns independently stored, asynchronous CSV measurements into an
auditable normalized record. Analysis code operates on arrays reloaded from disk; it
does not require a hidden simulation truth object. The checked-in
[`telemetry_mapping_example.yaml`](../configs/telemetry_mapping_example.yaml) shows
the complete schema for a fictional civilian research-flight source.

## Mapping schema

Mapping schema `1.0` declares:

- the timestamp source name, source unit, scale to seconds, and offset;
- a quality-flag source, accepted values, and invalid-row policy;
- source/destination channel names and units;
- an explicit affine conversion `destination = scale * source + offset`;
- one missing-value policy per channel: fail, keep `NaN`, or drop the row.

Unknown or missing YAML keys, duplicate channel names, missing CSV columns,
non-numeric values, nonfinite conversions, and non-monotonic retained timestamps fail
with row/channel context. Import records the source and canonical mapping SHA-256,
row counts, absolute source path, mapping version, and normalized destination units.
Normalized CSV and JSON provenance are separate so column names remain stable.

## Clock alignment and gaps

At least three corresponding synchronization-marker epochs define the affine model

\[
 t_{sensor} = a t_{reference} + b,
\]

where `b` is offset and `(a - 1) * 10^6` is clock drift in ppm. Least squares reports
marker-fit RMS. Corrected samples are linearly resampled to the chosen reference
timeline, but interpolation never bridges a user-declared maximum gap. Timestamp
discontinuities and consecutive missing-value intervals are both retained in the
report.

Residual evidence includes finite-pair count, bias, RMS, sample standard deviation,
maximum magnitude, autocorrelation, Ljung--Box statistic and p-value, and a declared
5% whiteness decision. The p-value is a diagnostic, not proof that a model is valid.
The aligned sample CSV and summary JSON are deterministic for identical inputs.

## RTS smoother

The Rauch--Tung--Striebel smoother consumes a stored forward history:

- filtered state and covariance `(x_k|k, P_k|k)`;
- one-step predicted state and covariance `(x_{k+1|k}, P_{k+1|k})`;
- transition matrix `F_k`.

The backward recursion is

\[
 G_k=P_{k|k}F_k^T P_{k+1|k}^{-1},
\]

\[
 x_{k|N}=x_{k|k}+G_k(x_{k+1|N}-x_{k+1|k}),
\]

\[
 P_{k|N}=P_{k|k}+G_k(P_{k+1|N}-P_{k+1|k})G_k^T.
\]

Input covariance histories must be finite, symmetric, and positive semidefinite;
prediction covariances used in the solve must be nonsingular. Each output covariance
is symmetrized and projected only for negative round-off eigenvalues, with the count
and worst pre-projection eigenvalue reported. The deterministic stored-data benchmark
shows smoothed altitude RMS no worse than the forward estimate while preserving PSD.

## Limits

The importer uses user-declared affine unit conversions rather than a general unit
algebra package. Marker correspondence is assumed known and affine over the record;
piecewise clock jumps require segmentation. Linear resampling is not suitable for
angles across a wrap or categorical channels without preprocessing. RTS smoothing is
offline and cannot be used as a causal onboard estimator.
