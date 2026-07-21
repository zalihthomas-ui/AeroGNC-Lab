"""Standard research-ascent event definitions."""

from aerognc.mathematics.integrators import EventSpec
from aerognc.mathematics.vectors import FloatArray


def flight_event_specs(burnout_time_s: float) -> tuple[EventSpec, EventSpec, EventSpec]:
    """Return one-shot burnout, apogee, and descending-ground-impact events."""
    return (
        EventSpec(
            "burnout",
            lambda time_s, _state: time_s - burnout_time_s,
            direction=1,
            terminal=False,
        ),
        EventSpec(
            "apogee",
            lambda _time_s, state: float(state[5]),
            direction=1,
            terminal=False,
        ),
        EventSpec(
            "ground_impact",
            _altitude_event,
            direction=-1,
            terminal=True,
        ),
    )


def _altitude_event(_time_s: float, state: FloatArray) -> float:
    """Geometric altitude above launch datum [m]."""
    return -float(state[2])
