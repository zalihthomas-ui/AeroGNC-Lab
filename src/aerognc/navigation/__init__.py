"""Navigation state representation and state providers.

This package defines the vehicle state the GNC chain consumes
(:class:`NavigationState`) and the environment struct guidance/control read
(:class:`FlightEnvironment`). State *providers* (perfect-truth passthrough and
estimated-state filters) build on these types; see ``TODO.md`` Phase 7.
"""

from aerognc.navigation.state import FlightEnvironment, NavigationState

__all__ = ["FlightEnvironment", "NavigationState"]
