"""Shared project-run service and built-in workflow adapters."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from aerognc.configuration import load_three_dof_configuration
from aerognc.configuration.multistage_recovery_loader import (
    load_multistage_recovery_configuration,
)
from aerognc.configuration.orbit_tour_loader import load_orbit_tour_configuration
from aerognc.configuration.rotating_six_dof_loader import (
    load_rotating_six_dof_configuration,
)
from aerognc.configuration.six_dof_loader import load_six_dof_configuration
from aerognc.project.manifest import (
    RequirementOutcome,
    RunStatus,
    file_sha256,
    new_run_manifest,
)
from aerognc.project.models import EngineeringProject, ProjectScenario
from aerognc.project.registry import (
    CancellationToken,
    ProgressCallback,
    WorkflowCancellationError,
    WorkflowContext,
    WorkflowDescriptor,
    WorkflowRegistry,
    WorkflowResult,
)
from aerognc.project.report import write_engineering_report
from aerognc.project.result_store import ResultDataset, ResultStore, StoredRun
from aerognc.simulation.multistage_recovery import simulate_configured_multistage_recovery
from aerognc.simulation.orbit_assisted_tour import simulate_orbit_assisted_tour
from aerognc.simulation.rotating_six_dof import simulate_rotating_six_dof
from aerognc.simulation.simulator import simulate_three_dof
from aerognc.simulation.six_dof_simulator import simulate_six_dof


class ProjectRunError(RuntimeError):
    """Raised after a failed/cancelled run has been recorded."""

    def __init__(self, run_id: str, status: str, reason: str) -> None:
        self.run_id = run_id
        self.status = status
        self.reason = reason
        super().__init__(f"project run {run_id} {status}: {reason}")


def _reject_parameters(context: WorkflowContext) -> None:
    if context.parameters:
        raise ValueError(
            f"built-in workflow does not accept scenario parameters: {sorted(context.parameters)}"
        )


def _three_dof_workflow(context: WorkflowContext) -> WorkflowResult:
    context.cancellation.raise_if_cancelled()
    _reject_parameters(context)
    context.report_progress(0.05, "Loading 3-DOF configuration")
    configuration = load_three_dof_configuration(context.configuration_path)
    context.cancellation.raise_if_cancelled()
    context.report_progress(0.15, "Propagating point-mass ascent")
    result = simulate_three_dof(configuration)
    context.cancellation.raise_if_cancelled()
    dataset = ResultDataset.from_simulation_result(
        result,
        metadata={"navigation_frame": "NED", "body_model": "point-mass"},
    )
    event_names = tuple(str(event["name"]) for event in dataset.events)
    expected_events = ("burnout", "apogee", "ground_impact")
    minimum_mass = float(np.min(dataset.channels["mass_kg"]))
    dry_mass = configuration.vehicle.mass_properties.dry_mass_kg
    requirements = (
        RequirementOutcome(
            "RUN-3DOF-EVENTS",
            event_names == expected_events,
            detail=f"observed={event_names}; expected={expected_events}",
        ),
        RequirementOutcome(
            "RUN-3DOF-MASS",
            minimum_mass >= dry_mass - 1.0e-10,
            value=minimum_mass,
            limit=dry_mass,
            margin=minimum_mass - dry_mass,
            unit="kg",
        ),
    )
    context.report_progress(1.0, "3-DOF run complete")
    return WorkflowResult(
        dataset,
        {
            "method": "fixed-step classical RK4",
            "step_s": configuration.simulation.step_s,
            "maximum_time_s": configuration.simulation.maximum_time_s,
        },
        requirements,
    )


def _six_dof_workflow(context: WorkflowContext) -> WorkflowResult:
    context.cancellation.raise_if_cancelled()
    _reject_parameters(context)
    context.report_progress(0.05, "Loading quaternion 6-DOF configuration")
    configuration = load_six_dof_configuration(context.configuration_path)
    context.report_progress(0.15, "Propagating closed-loop rigid-body ascent")
    result = simulate_six_dof(configuration)
    context.cancellation.raise_if_cancelled()
    dataset = ResultDataset.from_simulation_result(
        result,
        metadata={"navigation_frame": "NED", "body_frame": "FRD"},
    )
    quaternion_error = float(np.max(np.abs(dataset.channels["quaternion_norm"] - 1.0)))
    attitude_error = float(np.max(dataset.channels["attitude_error_deg"]))
    requirements = (
        RequirementOutcome(
            "RUN-6DOF-QUAT",
            quaternion_error <= 1.0e-9,
            value=quaternion_error,
            limit=1.0e-9,
            margin=1.0e-9 - quaternion_error,
            unit="1",
        ),
        RequirementOutcome(
            "RUN-6DOF-ATT",
            attitude_error <= 10.0,
            value=attitude_error,
            limit=10.0,
            margin=10.0 - attitude_error,
            unit="deg",
        ),
    )
    context.report_progress(1.0, "6-DOF run complete")
    return WorkflowResult(
        dataset,
        {
            "method": "sampled closed-loop fixed-step RK4",
            "step_s": configuration.step_s,
            "duration_s": configuration.duration_s,
        },
        requirements,
    )


def _orbit_tour_workflow(context: WorkflowContext) -> WorkflowResult:
    context.cancellation.raise_if_cancelled()
    _reject_parameters(context)
    context.report_progress(0.05, "Loading fictional orbit-tour configuration")
    configuration = load_orbit_tour_configuration(context.configuration_path)
    context.report_progress(0.15, "Solving Lambert legs and parking orbit")
    simulation = simulate_orbit_assisted_tour(configuration)
    context.cancellation.raise_if_cancelled()
    dataset = ResultDataset.from_simulation_result(
        simulation.result,
        metadata={
            "frame": "HELIOS_ECLIPJ2000",
            "model": "preliminary patched conics with ideal impulsive burns",
        },
    )
    assessment = simulation.assessment
    requirements = tuple(
        RequirementOutcome(identifier, passed)
        for identifier, passed in (
            ("RUN-TOUR-ORDER", assessment.ordered_events_pass),
            ("RUN-TOUR-SOI", assessment.sphere_of_influence_pass),
            ("RUN-TOUR-REV", assessment.parking_revolutions_pass),
            ("RUN-TOUR-DV", assessment.delta_v_pass),
            ("RUN-TOUR-MASS", assessment.final_mass_pass),
            ("RUN-TOUR-DRY", assessment.dry_mass_pass),
            ("RUN-TOUR-END", assessment.lambert_endpoint_pass),
        )
    )
    context.report_progress(1.0, "Orbit-assisted tour complete")
    return WorkflowResult(
        dataset,
        {
            "method": "two zero-revolution Lambert legs and analytical parking orbit",
            "first_leg_samples": configuration.first_leg_samples,
            "parking_orbit_samples": configuration.parking_orbit_samples,
            "second_leg_samples": configuration.second_leg_samples,
        },
        requirements,
        warnings=("Preliminary ideal-impulse patched-conic analysis; not operational navigation.",),
    )


def _rotating_six_dof_workflow(context: WorkflowContext) -> WorkflowResult:
    context.cancellation.raise_if_cancelled()
    _reject_parameters(context)
    context.report_progress(0.05, "Loading rotating-planet 6-DOF configuration")
    configuration = load_rotating_six_dof_configuration(context.configuration_path)
    context.report_progress(0.15, "Propagating inertial quaternion rigid-body ascent")
    result = simulate_rotating_six_dof(configuration)
    context.cancellation.raise_if_cancelled()
    dataset = ResultDataset.from_simulation_result(
        result,
        metadata={
            "translation_frame": "planet-centred inertial",
            "attitude": "Hamilton q_ib, scalar first",
            "reporting_frames": "ECEF and local NED",
        },
    )
    quaternion_error = float(np.max(np.abs(dataset.channels["quaternion_norm"] - 1.0)))
    dry_mass_kg = configuration.six_dof.base.vehicle.mass_properties.dry_mass_kg
    minimum_mass_kg = float(np.min(dataset.channels["mass_kg"]))
    requirements = (
        RequirementOutcome(
            "RUN-ROT6-QUAT",
            quaternion_error <= 1.0e-9,
            value=quaternion_error,
            limit=1.0e-9,
            margin=1.0e-9 - quaternion_error,
        ),
        RequirementOutcome(
            "RUN-ROT6-MASS",
            minimum_mass_kg >= dry_mass_kg - 1.0e-10,
            value=minimum_mass_kg,
            limit=dry_mass_kg,
            margin=minimum_mass_kg - dry_mass_kg,
            unit="kg",
        ),
    )
    context.report_progress(1.0, "Rotating-planet 6-DOF run complete")
    return WorkflowResult(
        dataset,
        {
            "method": "inertial translation and quaternion rotation, fixed-step RK4",
            "step_s": configuration.six_dof.step_s,
            "duration_s": configuration.six_dof.duration_s,
        },
        requirements,
    )


def _multistage_recovery_workflow(context: WorkflowContext) -> WorkflowResult:
    context.cancellation.raise_if_cancelled()
    _reject_parameters(context)
    context.report_progress(0.05, "Loading multistage/recovery configuration")
    configuration = load_multistage_recovery_configuration(context.configuration_path)
    context.report_progress(0.15, "Propagating vertical staging and recovery benchmark")
    result = simulate_configured_multistage_recovery(configuration)
    context.cancellation.raise_if_cancelled()
    dataset = ResultDataset.from_simulation_result(
        result,
        metadata={"frame": "one-axis positive-up altitude", "model": "constant-density benchmark"},
    )
    event_names = tuple(str(event["name"]) for event in dataset.events)
    mass_margin_kg = float(
        np.min(dataset.channels["mass_kg"] - dataset.channels["retained_dry_mass_floor_kg"])
    )
    continuity = configuration.vehicle.continuity_report()
    requirements = (
        RequirementOutcome(
            "RUN-STAGE-CONT",
            continuity.passed,
            value=continuity.maximum_separation_mass_residual_kg,
            limit=1.0e-10,
            margin=1.0e-10 - continuity.maximum_separation_mass_residual_kg,
            unit="kg",
        ),
        RequirementOutcome(
            "RUN-STAGE-MASS",
            mass_margin_kg >= -1.0e-10,
            value=mass_margin_kg,
            limit=0.0,
            margin=mass_margin_kg,
            unit="kg",
        ),
        RequirementOutcome(
            "RUN-RECOVERY-GROUND",
            bool(event_names) and event_names[-1] == "ground_contact",
            detail=f"last event={event_names[-1] if event_names else 'none'}",
        ),
    )
    context.report_progress(1.0, "Multistage/recovery run complete")
    return WorkflowResult(
        dataset,
        {
            "method": "vertical fixed-step RK4 with discrete mass jettison",
            "step_s": configuration.step_s,
            "maximum_time_s": configuration.maximum_time_s,
        },
        requirements,
        warnings=(
            "One-axis constant-density recovery benchmark; not a parachute certification model.",
        ),
    )


def built_in_workflow_registry() -> WorkflowRegistry:
    """Return a fresh registry containing stable public-safe built-in workflows."""
    registry = WorkflowRegistry()
    registry.register(
        WorkflowDescriptor(
            "multistage-recovery",
            "Fictional vertical staging and deployable-recovery benchmark",
            _multistage_recovery_workflow,
        )
    )
    registry.register(
        WorkflowDescriptor(
            "three-dof",
            "Deterministic fictional point-mass research-rocket ascent",
            _three_dof_workflow,
        )
    )
    registry.register(
        WorkflowDescriptor(
            "rotating-six-dof",
            "Quaternion research-rocket ascent on a synthetic rotating planet",
            _rotating_six_dof_workflow,
        )
    )
    registry.register(
        WorkflowDescriptor(
            "six-dof",
            "Closed-loop fictional quaternion rigid-body research-rocket ascent",
            _six_dof_workflow,
        )
    )
    registry.register(
        WorkflowDescriptor(
            "orbit-tour",
            "Fictional civilian capture, parking-orbit dwell, and departure tour",
            _orbit_tour_workflow,
        )
    )
    return registry


class ProjectRunService:
    """Execute registered scenarios and persist terminal evidence."""

    def __init__(
        self,
        registry: WorkflowRegistry | None = None,
        store_factory: Callable[[Path], ResultStore] = ResultStore,
    ) -> None:
        self.registry = built_in_workflow_registry() if registry is None else registry
        self._store_factory = store_factory

    def validate_workflows(self, project: EngineeringProject) -> tuple[str, ...]:
        """Return all unknown scenario workflow diagnostics without executing them."""
        available = set(self.registry.names())
        return tuple(
            f"scenario {scenario.name!r} uses unknown workflow {scenario.workflow!r}"
            for scenario in project.scenarios
            if scenario.workflow not in available
        )

    def run(
        self,
        project: EngineeringProject,
        scenario_name: str,
        *,
        cancellation: CancellationToken | None = None,
        progress: ProgressCallback | None = None,
        created: datetime | None = None,
    ) -> StoredRun:
        """Run one project scenario and store completed, failed, or cancelled evidence."""
        scenario = project.scenario(scenario_name)
        if not scenario.enabled:
            raise ValueError(f"project scenario is disabled: {scenario_name}")
        configuration_path = project.configuration_path(scenario)
        configuration_hash = file_sha256(configuration_path)
        seed = project.settings.default_seed if scenario.seed is None else scenario.seed
        token = CancellationToken() if cancellation is None else cancellation
        context = WorkflowContext(
            scenario_name=scenario.name,
            configuration_path=configuration_path,
            workspace_root=project.workspace_root,
            seed=seed,
            parameters=scenario.parameters,
            cancellation=token,
            progress=progress,
        )
        store = self._store_factory(project.result_root)
        start = perf_counter()
        try:
            token.raise_if_cancelled()
            descriptor = self.registry.get(scenario.workflow)
            result = descriptor.runner(context)
            token.raise_if_cancelled()
        except Exception as error:
            elapsed = perf_counter() - start
            status: RunStatus = (
                "cancelled" if isinstance(error, WorkflowCancellationError) else "failed"
            )
            reason = f"{type(error).__name__}: {error}"
            failure_manifest = new_run_manifest(
                project_name=project.name,
                scenario_name=scenario.name,
                workflow=scenario.workflow,
                safety_scope=project.safety_scope,
                configuration_path=scenario.configuration.as_posix(),
                configuration_sha256=configuration_hash,
                seed=seed,
                solver_settings={"method": "unavailable because execution did not complete"},
                parameters=scenario.parameters,
                status=status,
                execution_time_s=elapsed,
                failure_reason=reason,
                created=created,
            )
            failed = store.commit(failure_manifest, None)
            write_engineering_report(failed)
            raise ProjectRunError(failure_manifest.run_id, status, reason) from error

        elapsed = perf_counter() - start
        manifest = new_run_manifest(
            project_name=project.name,
            scenario_name=scenario.name,
            workflow=scenario.workflow,
            safety_scope=project.safety_scope,
            configuration_path=scenario.configuration.as_posix(),
            configuration_sha256=configuration_hash,
            seed=seed,
            solver_settings=result.solver_settings,
            parameters=scenario.parameters,
            status="completed",
            execution_time_s=elapsed,
            warnings=result.warnings,
            requirements=result.requirements,
            events=result.dataset.events,
            maxima=result.dataset.maxima,
            created=created,
        )
        stored = store.commit(manifest, result.dataset)
        write_engineering_report(stored)
        return stored


def scenario_seed(project: EngineeringProject, scenario: ProjectScenario) -> int:
    """Resolve one scenario seed independently for UI previews and auditing."""
    return project.settings.default_seed if scenario.seed is None else scenario.seed
