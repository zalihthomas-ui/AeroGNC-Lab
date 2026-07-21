"""Tests for the interactive mission planner: pure model + Tk construction smoke."""

import pytest

from aerognc.mission.mission import HomePosition
from aerognc.mission.waypoint import WaypointAction
from aerognc.visualisation.mission_planner_map import PlannerModel


def _model() -> PlannerModel:
    return PlannerModel(HomePosition(39.925, 32.8369, 0.0))


# --- projection --------------------------------------------------------------


def test_home_projects_to_canvas_centre() -> None:
    model = _model()
    px, py = model.geo_to_pixel(model.home.latitude_deg, model.home.longitude_deg)
    assert px == pytest.approx(model.center_px[0], abs=1e-6)
    assert py == pytest.approx(model.center_px[1], abs=1e-6)


def test_pixel_geo_round_trip() -> None:
    model = _model()
    lat, lon = model.pixel_to_geo(600.0, 300.0)
    px, py = model.geo_to_pixel(lat, lon)
    assert px == pytest.approx(600.0, abs=1e-3)
    assert py == pytest.approx(300.0, abs=1e-3)


def test_north_is_up_east_is_right() -> None:
    model = _model()
    # A point above centre (smaller py) should be north of home (higher latitude).
    lat_up, _ = model.pixel_to_geo(model.center_px[0], model.center_px[1] - 100.0)
    lat_home = model.home.latitude_deg
    assert lat_up > lat_home
    _, lon_right = model.pixel_to_geo(model.center_px[0] + 100.0, model.center_px[1])
    assert lon_right > model.home.longitude_deg


def test_zoom_keeps_anchor_ground_point_fixed() -> None:
    model = _model()
    anchor = (700.0, 250.0)
    before = model.pixel_to_geo(*anchor)
    model.zoom(2.0, anchor_px=anchor)
    after = model.pixel_to_geo(*anchor)
    assert after[0] == pytest.approx(before[0], abs=1e-6)
    assert after[1] == pytest.approx(before[1], abs=1e-6)


# --- edits & undo/redo -------------------------------------------------------


def test_add_move_delete() -> None:
    model = _model()
    model.add_waypoint_geo(39.930, 32.845)
    assert len(model.waypoints) == 1
    model.move_waypoint_geo(0, 39.931, 32.846)
    assert model.waypoints[0].latitude_deg == pytest.approx(39.931)
    model.delete_waypoint(0)
    assert not model.waypoints


def test_duplicate_and_reorder() -> None:
    model = _model()
    model.add_waypoint_geo(39.930, 32.845)
    model.add_waypoint_geo(39.940, 32.850)
    model.duplicate_waypoint(0)
    assert len(model.waypoints) == 3
    assert model.waypoints[1].name.endswith("_copy")
    model.reorder_waypoint(2, 0)
    assert model.waypoints[0].name == "WP2"


def test_undo_redo_round_trip() -> None:
    model = _model()
    model.add_waypoint_geo(39.930, 32.845)
    model.add_waypoint_geo(39.940, 32.850)
    assert len(model.waypoints) == 2
    model.undo()
    assert len(model.waypoints) == 1
    model.undo()
    assert len(model.waypoints) == 0
    model.redo()
    assert len(model.waypoints) == 1


def test_set_home_reanchors_projection() -> None:
    model = _model()
    model.set_home_geo(40.0, 33.0)
    px, py = model.geo_to_pixel(40.0, 33.0)
    assert px == pytest.approx(model.center_px[0], abs=1e-6)
    assert py == pytest.approx(model.center_px[1], abs=1e-6)


def test_update_fields_and_action() -> None:
    model = _model()
    model.add_waypoint_geo(39.930, 32.845)
    model.update_waypoint_fields(
        0, action=WaypointAction.LOITER, loiter_radius_m=120.0, airspeed_mps=22.0
    )
    assert model.waypoints[0].action is WaypointAction.LOITER
    assert model.waypoints[0].loiter_radius_m == pytest.approx(120.0)


def test_nearest_waypoint_selection() -> None:
    model = _model()
    model.add_waypoint_geo(39.930, 32.845)
    px, py = model.geo_to_pixel(39.930, 32.845)
    assert model.nearest_waypoint(px + 3, py + 3) == 0
    assert model.nearest_waypoint(px + 500, py) is None


# --- mission build / IO ------------------------------------------------------


def test_build_and_export_import_round_trip(tmp_path) -> None:
    model = _model()
    model.add_waypoint_geo(39.927, 32.840)
    model.update_waypoint_fields(0, altitude_m=120.0)
    model.add_waypoint_geo(39.925, 32.8369)
    model.update_waypoint_fields(1, action=WaypointAction.RETURN_HOME, altitude_m=100.0)
    mission = model.build_mission("planner_test").validate()
    assert len(mission.waypoints) == 2

    path = tmp_path / "planner.mission.yaml"
    model.export_mission(str(path), "planner_test")
    fresh = PlannerModel()
    fresh.import_mission(str(path))
    assert len(fresh.waypoints) == 2
    assert fresh.waypoints[1].action is WaypointAction.RETURN_HOME


def test_validation_issues_reports_problems() -> None:
    model = _model()
    model.add_waypoint_geo(39.93, 32.84)
    model.update_waypoint_fields(0, airspeed_mps=999.0)  # outside envelope
    assert any("envelope" in issue for issue in model.validation_issues())


# --- playback controller (pure) ---------------------------------------------


def test_playback_controller_advances_and_finishes() -> None:
    from aerognc.visualisation.mission_planner_map import PlaybackController

    samples = list(range(10))
    playback = PlaybackController(samples, speed=3.0)
    assert playback.current() == 0
    assert not playback.finished
    playback.advance(3)
    assert playback.index == 3
    playback.advance(100)  # clamps to the last sample
    assert playback.index == 9
    assert playback.finished
    assert playback.progress() == pytest.approx(1.0)
    playback.reset()
    assert playback.index == 0 and not playback.playing


def test_playback_controller_handles_empty_and_speed_floor() -> None:
    from aerognc.visualisation.mission_planner_map import PlaybackController

    empty = PlaybackController([])
    assert empty.finished
    assert empty.current() is None
    assert empty.progress() == pytest.approx(1.0)
    assert PlaybackController([1, 2], speed=0.1).speed >= 1.0


# --- animation + 3D dashboard ------------------------------------------------


def _short_mission_model() -> PlannerModel:
    model = _model()
    model.add_waypoint_geo(39.928, 32.838)
    model.add_waypoint_geo(39.925, 32.8369)
    model.update_waypoint_fields(1, action=WaypointAction.RETURN_HOME, altitude_m=100.0)
    return model


def test_planner_simulation_playback_moves_aircraft() -> None:
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    try:
        root.withdraw()
        from aerognc.visualisation.mission_planner_map import InteractiveMissionPlanner

        planner = InteractiveMissionPlanner(root, _short_mission_model())
        planner._run()  # runs the internal simulation synchronously
        assert planner._playback is not None and len(planner._playback) > 0
        assert planner.view.actual_track_px  # flown track projected to pixels
        planner._playback.playing = True
        planner._animate()  # advance one animation step
        planner._stop_animation()
        assert planner.canvas.find_withtag("aircraft")  # moving glyph drawn
        assert planner.hud_var.get()  # live HUD populated
    finally:
        root.destroy()


def test_waypoint_mission_3d_dashboard_renders(tmp_path) -> None:
    from aerognc.simulation.waypoint_mission import run_waypoint_mission
    from aerognc.visualisation.waypoint_mission import plot_waypoint_mission

    result = run_waypoint_mission(_short_mission_model().build_mission())
    out = plot_waypoint_mission(result, tmp_path / "dash.png")
    assert out.is_file() and out.stat().st_size > 0


def test_waypoint_mission_replay_gif_renders(tmp_path) -> None:
    from aerognc.simulation.waypoint_mission import run_waypoint_mission
    from aerognc.visualisation.waypoint_mission import save_mission_replay_gif

    result = run_waypoint_mission(_short_mission_model().build_mission())
    out = save_mission_replay_gif(result, tmp_path / "replay.gif", max_frames=12, fps=8)
    assert out.is_file() and out.stat().st_size > 0


def test_planner_window_constructs_and_draws() -> None:
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available for Tk")
    try:
        root.withdraw()
        from aerognc.visualisation.mission_planner_map import InteractiveMissionPlanner

        model = _model()
        model.add_waypoint_geo(39.930, 32.845)
        planner = InteractiveMissionPlanner(root, model)
        planner._refresh()
        planner.redraw()
        root.update_idletasks()
        # The canvas should contain drawn items (route, markers, circles).
        assert len(planner.canvas.find_all()) > 0
    finally:
        root.destroy()
