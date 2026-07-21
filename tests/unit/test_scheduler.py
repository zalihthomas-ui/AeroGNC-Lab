from collections.abc import Iterator

import pytest

from aerognc.simulation.scheduler import LogicalTimeScheduler, ScheduledInvocation, ScheduledTask


def test_multirate_scheduler_has_stable_logical_order() -> None:
    observed: list[tuple[str, float, int]] = []

    def record(invocation: ScheduledInvocation) -> None:
        observed.append((invocation.task_name, invocation.logical_time_s, invocation.tick))

    tasks = (
        ScheduledTask("plant", 0.1, record),
        ScheduledTask("controller", 0.2, record),
        ScheduledTask("sensor", 0.15, record, phase_s=0.05),
    )
    result = LogicalTimeScheduler().run(tasks, (2.0, 2.5))

    assert observed[:4] == [
        ("plant", 2.0, 0),
        ("controller", 2.0, 0),
        ("sensor", 2.05, 0),
        ("plant", 2.1, 1),
    ]
    ties_at_end = [name for name, time_s, _tick in observed if time_s == pytest.approx(2.5)]
    assert ties_at_end == ["plant", "sensor"]
    assert result.statistics["plant"].invocations == 6
    assert result.statistics["controller"].invocations == 3
    assert result.statistics["sensor"].invocations == 4


def test_deadline_statistics_do_not_affect_dispatch() -> None:
    values: Iterator[float] = iter((0.0, 0.004, 1.0, 1.001))
    records: list[int] = []
    scheduler = LogicalTimeScheduler(clock=lambda: next(values))
    result = scheduler.run(
        (ScheduledTask("control", 1.0, lambda item: records.append(item.tick), deadline_s=0.002),),
        (0.0, 1.0),
    )

    assert records == [0, 1]
    assert result.statistics["control"].missed_deadlines == 1
    assert result.statistics["control"].maximum_execution_time_s == pytest.approx(0.004)


def test_scheduler_checks_cancellation_between_callbacks() -> None:
    calls = 0

    class StopAfterOne:
        def raise_if_cancelled(self) -> None:
            nonlocal calls
            if calls:
                raise RuntimeError("cancelled")

    def callback(_invocation: ScheduledInvocation) -> None:
        nonlocal calls
        calls += 1

    with pytest.raises(RuntimeError, match="cancelled"):
        LogicalTimeScheduler().run(
            (ScheduledTask("task", 0.1, callback),),
            (0.0, 1.0),
            cancellation=StopAfterOne(),
        )
    assert calls == 1


def test_scheduler_rejects_invalid_or_duplicate_tasks() -> None:
    with pytest.raises(ValueError, match="period"):
        ScheduledTask("bad", 0.0, lambda _item: None)
    duplicate = ScheduledTask("same", 0.1, lambda _item: None)
    with pytest.raises(ValueError, match="unique"):
        LogicalTimeScheduler().run((duplicate, duplicate), (0.0, 1.0))
