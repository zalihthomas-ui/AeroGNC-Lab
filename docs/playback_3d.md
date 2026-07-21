# Quaternion 6-DOF 3D Simulation Player

The `play-3d` workflow runs the configured nonlinear quaternion 6-DOF simulation and
opens an interactive three-dimensional engineering dashboard. The plotted vehicle is
not a decorative animation: its position comes from the propagated North/East/down
state and its body orientation comes from the normalised Hamilton quaternion at the
displayed simulation time. The numerical trajectory is solved before playback, so
pause, seek, speed, and camera operations cannot alter the source result.

## Start the 3D player

From the repository root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m aerognc.cli play-3d `
  --config configs\six_dof_nominal.yaml
```

The default is a slowly rotating orbit camera at 2x playback speed. Use `--speed 1`
for approximately real-time presentation, `--repeat` to loop, or select an initial
view with `--camera chase`, `top`, `side`, or `free`.

## Scene conventions and display

The scene uses East on the plotted x-axis, North on y, and positive altitude on z.
This is a display mapping of the documented local NED navigation frame; the dynamics
remain NED and are not redefined by the plot. The orange vehicle centreline follows
body +x (forward). Short red, green, and blue lines show body +x, +y (right), and +z
(down), respectively. The vehicle glyph is intentionally enlarged and is not drawn
to geometric scale.

The dashboard shows completed and remaining trajectory, ground track, vertical
projection, burnout marker, quaternion attitude, live SI telemetry, Euler angles,
body rates, aerodynamic angles, attitude-tracking error, and synchronized altitude
and attitude-error histories.

## Controls and cameras

| Control | Action |
|---|---|
| Space or Play/Pause | Pause or resume the calculated flight |
| Timeline slider | Seek to any logged 6-DOF time |
| Left / Right | Seek backward or forward 0.5 simulation seconds |
| Up / Down | Double or halve playback speed within safe limits |
| `R` or Restart | Return to the initial state and resume |
| `C` or Camera button | Cycle orbit, chase, top, side, and free views |
| Mouse drag in Free mode | Rotate the Matplotlib 3D camera |

Orbit keeps the entire trajectory framed while changing azimuth. Chase follows a
window around the current vehicle position. Top and side are fixed inspection views.
Free retains the user's mouse-selected azimuth and elevation.

## Deterministic GIF export

To export without opening a desktop window:

```powershell
.\.venv\Scripts\python.exe -m aerognc.cli play-3d `
  --config configs\six_dof_nominal.yaml `
  --camera orbit --speed 4 --fps 20 `
  --save-gif results\playback\six_dof_3d.gif --no-window
```

The export is sampled deterministically from the immutable result log. Animated GIFs
can be large and are ignored outside the compact `results/reference/` set.

## Scope and limitations

This is a simulation player, not a manually piloted game or a real-time hardware
loop. The nominal 6-DOF demonstration covers eight seconds of controlled fictional
research-rocket ascent; it ends before apogee and ground impact. It uses synthetic
aerodynamic, propulsion, mass, wind, actuator, guidance, and control parameters. It
contains no target, interception, terminal-homing, or operational engagement model.
