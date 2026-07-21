"""Numerical and coordinate mathematics implemented by AeroGNC-Lab."""

from aerognc.mathematics.adaptive_integrators import (
    AdaptiveIntegrationResult,
    AdaptiveOptions,
    AdaptiveStatistics,
    integrate_adaptive,
)
from aerognc.mathematics.geodesy import (
    GeodeticPosition,
    LaunchSite,
    ReferenceEllipsoid,
    body_rotation_rate_ned,
    dcm_ecef_to_ned,
    dcm_inertial_to_ecef,
    ecef_position_to_ned,
    ecef_to_geodetic,
    ecef_to_inertial_state,
    geodetic_to_ecef,
    inertial_to_ecef_state,
    meridian_radius_m,
    ned_position_to_ecef,
    prime_vertical_radius_m,
    transport_rate_ned,
)
from aerognc.mathematics.variational import (
    VariationalResult,
    central_difference_jacobian,
    dynamics_jacobians,
    propagate_variational,
)
from aerognc.mathematics.vectors import as_matrix3, as_vector, skew_symmetric

__all__ = [
    "AdaptiveIntegrationResult",
    "AdaptiveOptions",
    "AdaptiveStatistics",
    "GeodeticPosition",
    "LaunchSite",
    "ReferenceEllipsoid",
    "VariationalResult",
    "as_matrix3",
    "as_vector",
    "body_rotation_rate_ned",
    "central_difference_jacobian",
    "dcm_ecef_to_ned",
    "dcm_inertial_to_ecef",
    "dynamics_jacobians",
    "ecef_position_to_ned",
    "ecef_to_geodetic",
    "ecef_to_inertial_state",
    "geodetic_to_ecef",
    "inertial_to_ecef_state",
    "integrate_adaptive",
    "meridian_radius_m",
    "ned_position_to_ecef",
    "prime_vertical_radius_m",
    "propagate_variational",
    "skew_symmetric",
    "transport_rate_ned",
]
