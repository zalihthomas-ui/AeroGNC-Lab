"""Mission state machine coordinating waypoint execution.

The :class:`MissionManager` owns the :class:`~aerognc.gnc.path_manager.PathManager`
and layers explicit mission states and operator commands (arm / start / pause /
resume / stop / return-home / abort / emergency) on top of waypoint sequencing.
Mission behaviour lives here — never inside UI callbacks — and every state
transition is logged.

Takeoff, climb, approach, flare, and landed states exist in the enumeration for
completeness and future takeoff/landing modules (TODO Phase 8.3/8.5); the default
simulation air-starts and flies NAVIGATE / LOITER / RETURN_HOME / MISSION_COMPLETE.
"""

from dataclasses import dataclass
from enum import StrEnum

from aerognc.gnc.path_manager import MissionPhase, PathManager, PathManagerStatus
from aerognc.mission.mission import Mission
from aerognc.mission.waypoint import WaypointAction
from aerognc.navigation.state import NavigationState


class MissionState(StrEnum):
    """High-level mission states (superset; not all are used by default)."""

    DISARMED = "disarmed"
    PREFLIGHT = "preflight"
    READY = "ready"
    TAKEOFF = "takeoff"
    CLIMB = "climb"
    NAVIGATE = "navigate"
    LOITER = "loiter"
    RETURN_HOME = "return_home"
    APPROACH = "approach"
    FLARE = "flare"
    LANDED = "landed"
    PAUSED = "paused"
    ABORT = "abort"
    EMERGENCY = "emergency"
    MISSION_COMPLETE = "mission_complete"


class SafetyResponse(StrEnum):
    """Recommendation the safety manager can pass into the mission manager."""

    NONE = "none"
    LIMIT = "limit"  # advisory: command already bounded by controllers
    LOITER = "loiter"
    RETURN_HOME = "return_home"
    ABORT = "abort"
    TERMINATE = "terminate"


_TERMINAL_STATES = frozenset(
    {MissionState.ABORT, MissionState.EMERGENCY, MissionState.LANDED, MissionState.MISSION_COMPLETE}
)
_ACTIVE_STATES = frozenset({MissionState.NAVIGATE, MissionState.LOITER, MissionState.RETURN_HOME})


@dataclass(frozen=True, slots=True)
class StateTransition:
    """A single logged mission-state transition."""

    time_s: float
    from_state: MissionState
    to_state: MissionState
    reason: str


@dataclass(frozen=True, slots=True)
class MissionManagerStatus:
    """Immutable snapshot of the mission manager each step."""

    state: MissionState
    active_waypoint_id: int | None
    path_status: PathManagerStatus | None
    mission_complete: bool


class MissionManager:
    """Explicit mission-state machine over a path manager."""

    def __init__(self, mission: Mission, path_manager: PathManager) -> None:
        self.mission = mission
        self.path_manager = path_manager
        self._state = MissionState.DISARMED
        self._time_s = 0.0
        self._transitions: list[StateTransition] = []
        self._last_path_status: PathManagerStatus | None = None
        self._return_home_id = self._find_return_home_waypoint_id()

    @property
    def state(self) -> MissionState:
        return self._state

    @property
    def transitions(self) -> tuple[StateTransition, ...]:
        return tuple(self._transitions)

    # -- operator commands ----------------------------------------------------
    def arm(self) -> None:
        """Validate the mission and move DISARMED -> READY."""
        self.mission.validate()
        if self._state is MissionState.DISARMED:
            self._transition(MissionState.READY, "armed and mission validated")

    def start(self) -> None:
        """Begin execution (READY -> NAVIGATE)."""
        if self._state is MissionState.READY:
            self.path_manager.reset()
            self._transition(MissionState.NAVIGATE, "mission started")

    def pause(self) -> None:
        if self._state in _ACTIVE_STATES:
            self._transition(MissionState.PAUSED, "operator pause")

    def resume(self) -> None:
        if self._state is MissionState.PAUSED:
            self._transition(MissionState.NAVIGATE, "operator resume")

    def stop(self) -> None:
        self._transition(MissionState.ABORT, "operator stop")

    def abort(self) -> None:
        self._transition(MissionState.ABORT, "operator abort")

    def trigger_emergency(self, reason: str = "emergency") -> None:
        self._transition(MissionState.EMERGENCY, reason)

    def request_return_home(self, reason: str = "return-to-home requested") -> bool:
        """Jump to the return-home leg if the mission defines one; else abort."""
        if self._return_home_id is None:
            self._transition(MissionState.ABORT, f"{reason}: no return-home waypoint")
            return False
        index = self.path_manager.index_of_waypoint(self._return_home_id)
        if index is None:
            self._transition(MissionState.ABORT, f"{reason}: return-home leg missing")
            return False
        self.path_manager.force_active_index(index)
        self._transition(MissionState.RETURN_HOME, reason)
        return True

    # -- step -----------------------------------------------------------------
    def update(
        self,
        vehicle_state: NavigationState,
        dt_s: float,
        safety_response: SafetyResponse = SafetyResponse.NONE,
    ) -> MissionManagerStatus:
        """Advance the mission one step and return the status."""
        self._time_s += dt_s
        self._apply_safety(safety_response)

        if self._state not in _ACTIVE_STATES:
            return MissionManagerStatus(
                state=self._state,
                active_waypoint_id=(
                    self._last_path_status.active_waypoint_id if self._last_path_status else None
                ),
                path_status=self._last_path_status,
                mission_complete=self._state is MissionState.MISSION_COMPLETE,
            )

        path_status = self.path_manager.update(vehicle_state.position_ned_m, dt_s)
        self._last_path_status = path_status
        self._sync_state_from_path(path_status)
        return MissionManagerStatus(
            state=self._state,
            active_waypoint_id=path_status.active_waypoint_id,
            path_status=path_status,
            mission_complete=self._state is MissionState.MISSION_COMPLETE,
        )

    # -- internals ------------------------------------------------------------
    def _apply_safety(self, response: SafetyResponse) -> None:
        if response in (SafetyResponse.NONE, SafetyResponse.LIMIT):
            return  # advisory only; controllers already bound the commands
        if response is SafetyResponse.TERMINATE:
            self.trigger_emergency("safety: terminate")
        elif response is SafetyResponse.ABORT:
            self._transition(MissionState.ABORT, "safety: abort")
        elif response is SafetyResponse.RETURN_HOME and self._state in _ACTIVE_STATES:
            self.request_return_home("safety: return-to-home")
        # LOITER recommendation is advisory; the path/guidance already hold pattern.

    def _sync_state_from_path(self, path_status: PathManagerStatus) -> None:
        if path_status.mission_complete:
            self._transition(MissionState.MISSION_COMPLETE, "final waypoint reached")
            return
        if self._state is MissionState.RETURN_HOME:
            return  # stay in RTH until complete
        active_action = self._action_for_waypoint(path_status.active_waypoint_id)
        if active_action is WaypointAction.RETURN_HOME:
            self._transition(MissionState.RETURN_HOME, "return-home leg active")
        elif path_status.phase is MissionPhase.LOITER:
            self._transition(MissionState.LOITER, "loiter leg active")
        else:
            self._transition(MissionState.NAVIGATE, "navigating to waypoint")

    def _transition(self, to_state: MissionState, reason: str) -> None:
        if to_state is self._state:
            return
        self._transitions.append(StateTransition(self._time_s, self._state, to_state, reason))
        self._state = to_state

    def _action_for_waypoint(self, waypoint_id: int) -> WaypointAction | None:
        for waypoint in self.mission.waypoints:
            if waypoint.id == waypoint_id:
                return waypoint.action
        return None

    def _find_return_home_waypoint_id(self) -> int | None:
        for waypoint in self.mission.waypoints:
            if waypoint.action is WaypointAction.RETURN_HOME:
                return waypoint.id
        return None
