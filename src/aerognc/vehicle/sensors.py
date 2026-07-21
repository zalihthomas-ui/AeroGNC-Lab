"""Seeded sampled sensor models with timing, delay, quantisation, and dropout."""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from aerognc.mathematics.vectors import FloatArray


@dataclass(frozen=True, slots=True)
class SensorErrorParameters:
    """Error and timing parameters for one sampled vector sensor."""

    sample_rate_hz: float
    noise_std: FloatArray
    constant_bias: FloatArray
    bias_drift_std_per_sqrt_s: FloatArray
    quantisation: FloatArray
    delay_s: float = 0.0
    dropout_probability: float = 0.0
    dropout_intervals_s: tuple[tuple[float, float], ...] = ()

    def __init__(
        self,
        *,
        sample_rate_hz: float,
        noise_std: npt.ArrayLike,
        constant_bias: npt.ArrayLike,
        bias_drift_std_per_sqrt_s: npt.ArrayLike,
        quantisation: npt.ArrayLike,
        delay_s: float = 0.0,
        dropout_probability: float = 0.0,
        dropout_intervals_s: Sequence[tuple[float, float]] = (),
    ) -> None:
        arrays = [
            np.asarray(value, dtype=np.float64)
            for value in (
                noise_std,
                constant_bias,
                bias_drift_std_per_sqrt_s,
                quantisation,
            )
        ]
        dimension = arrays[0].size
        if dimension == 0 or any(array.shape != (dimension,) for array in arrays):
            raise ValueError("sensor error arrays must be nonempty vectors of equal shape")
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("sensor error arrays must be finite")
        if any(np.any(array < 0.0) for array in (arrays[0], arrays[2], arrays[3])):
            raise ValueError("noise, drift, and quantisation values must be nonnegative")
        scalars = np.array([sample_rate_hz, delay_s, dropout_probability])
        if not np.all(np.isfinite(scalars)) or sample_rate_hz <= 0.0 or delay_s < 0.0:
            raise ValueError("sample rate must be positive and delay nonnegative")
        if not 0.0 <= dropout_probability <= 1.0:
            raise ValueError("dropout_probability must lie in [0, 1]")
        intervals: list[tuple[float, float]] = []
        for start_s, end_s in dropout_intervals_s:
            if not np.isfinite([start_s, end_s]).all() or start_s < 0.0 or end_s <= start_s:
                raise ValueError("dropout intervals must be finite, nonnegative, and increasing")
            intervals.append((float(start_s), float(end_s)))
        object.__setattr__(self, "sample_rate_hz", float(sample_rate_hz))
        object.__setattr__(self, "noise_std", arrays[0].copy())
        object.__setattr__(self, "constant_bias", arrays[1].copy())
        object.__setattr__(self, "bias_drift_std_per_sqrt_s", arrays[2].copy())
        object.__setattr__(self, "quantisation", arrays[3].copy())
        object.__setattr__(self, "delay_s", float(delay_s))
        object.__setattr__(self, "dropout_probability", float(dropout_probability))
        object.__setattr__(self, "dropout_intervals_s", tuple(intervals))

    @property
    def dimension(self) -> int:
        """Measurement vector dimension."""
        return int(self.noise_std.size)


@dataclass(frozen=True, slots=True)
class SensorMeasurement:
    """One delayed measurement with acquisition and availability timestamps."""

    sample_time_s: float
    available_time_s: float
    value: FloatArray


class SampledSensor:
    """Generic reproducible vector sensor evaluated on a supplied truth sequence."""

    def __init__(self, parameters: SensorErrorParameters, *, seed: int) -> None:
        self.parameters = parameters
        self.seed = int(seed)
        self._generator = np.random.default_rng(self.seed)
        self._bias_drift = np.zeros(parameters.dimension)
        self._next_sample_time_s = 0.0
        self._last_sample_time_s: float | None = None
        self._queue: deque[SensorMeasurement] = deque()

    def reset(self) -> None:
        """Restore initial deterministic state and random sequence."""
        self._generator = np.random.default_rng(self.seed)
        self._bias_drift = np.zeros(self.parameters.dimension)
        self._next_sample_time_s = 0.0
        self._last_sample_time_s = None
        self._queue.clear()

    def measure(self, time_s: float, truth_value: npt.ArrayLike) -> SensorMeasurement | None:
        """Sample truth when due and return the oldest measurement whose delay elapsed."""
        truth = np.asarray(truth_value, dtype=np.float64)
        if truth.shape != (self.parameters.dimension,) or not np.all(np.isfinite(truth)):
            raise ValueError(
                f"truth_value must be finite with shape ({self.parameters.dimension},)"
            )
        if not np.isfinite(time_s) or time_s < 0.0:
            raise ValueError("time_s must be finite and nonnegative")
        tolerance_s = 1.0e-10
        if time_s + tolerance_s >= self._next_sample_time_s:
            self._generate_sample(float(time_s), truth)
            period_s = 1.0 / self.parameters.sample_rate_hz
            missed_periods = max(
                1, int(np.floor((time_s - self._next_sample_time_s) / period_s)) + 1
            )
            self._next_sample_time_s += missed_periods * period_s
        if self._queue and self._queue[0].available_time_s <= time_s + tolerance_s:
            return self._queue.popleft()
        return None

    def _generate_sample(self, time_s: float, truth: FloatArray) -> None:
        step_s = (
            1.0 / self.parameters.sample_rate_hz
            if self._last_sample_time_s is None
            else time_s - self._last_sample_time_s
        )
        self._last_sample_time_s = time_s
        self._bias_drift += (
            self.parameters.bias_drift_std_per_sqrt_s
            * np.sqrt(max(step_s, 0.0))
            * self._generator.standard_normal(self.parameters.dimension)
        )
        scheduled_dropout = any(
            start_s <= time_s < end_s for start_s, end_s in self.parameters.dropout_intervals_s
        )
        random_dropout = self._generator.random() < self.parameters.dropout_probability
        if scheduled_dropout or random_dropout:
            return
        value = (
            truth
            + self.parameters.constant_bias
            + self._bias_drift
            + self.parameters.noise_std * self._generator.standard_normal(self.parameters.dimension)
        )
        quantisation = self.parameters.quantisation
        value = np.where(
            quantisation > 0.0,
            np.round(value / np.where(quantisation > 0.0, quantisation, 1.0)) * quantisation,
            value,
        )
        self._queue.append(
            SensorMeasurement(
                sample_time_s=time_s,
                available_time_s=time_s + self.parameters.delay_s,
                value=np.asarray(value, dtype=np.float64),
            )
        )


class GyroscopeSensor(SampledSensor):
    """Three-axis body angular-rate sensor [rad/s]."""

    def __init__(self, parameters: SensorErrorParameters, *, seed: int) -> None:
        if parameters.dimension != 3:
            raise ValueError("gyroscope parameters must have dimension 3")
        super().__init__(parameters, seed=seed)


class AccelerometerSensor(SampledSensor):
    """Three-axis specific-force sensor [m/s²]."""

    def __init__(self, parameters: SensorErrorParameters, *, seed: int) -> None:
        if parameters.dimension != 3:
            raise ValueError("accelerometer parameters must have dimension 3")
        super().__init__(parameters, seed=seed)


class BarometricAltimeter(SampledSensor):
    """Scalar geometric-altitude sensor [m]."""

    def __init__(self, parameters: SensorErrorParameters, *, seed: int) -> None:
        if parameters.dimension != 1:
            raise ValueError("barometric-altimeter parameters must have dimension 1")
        super().__init__(parameters, seed=seed)


class CivilianGnssSensor(SampledSensor):
    """Local NED position [m] and velocity [m/s] measurement vector."""

    def __init__(self, parameters: SensorErrorParameters, *, seed: int) -> None:
        if parameters.dimension != 6:
            raise ValueError("GNSS-like parameters must have dimension 6")
        super().__init__(parameters, seed=seed)
