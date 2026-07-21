import numpy as np

from aerognc.gnc.guidance import AttitudeReferenceSchedule
from aerognc.mathematics.quaternion import quaternion_to_euler321


def test_attitude_schedule_interpolates_and_clamps() -> None:
    schedule = AttitudeReferenceSchedule(
        [0.0, 2.0, 5.0], [[0.0, 85.0, 10.0], [0.0, 75.0, 10.0], [2.0, 60.0, 15.0]]
    )
    np.testing.assert_allclose(np.rad2deg(schedule.euler_at_time_rad(1.0)), [0.0, 80.0, 10.0])
    np.testing.assert_allclose(np.rad2deg(schedule.euler_at_time_rad(-1.0)), [0.0, 85.0, 10.0])
    np.testing.assert_allclose(np.rad2deg(schedule.euler_at_time_rad(8.0)), [2.0, 60.0, 15.0])
    recovered = quaternion_to_euler321(schedule.quaternion_at_time_nb(1.0))
    np.testing.assert_allclose(np.rad2deg(recovered), [0.0, 80.0, 10.0], atol=1e-12)
