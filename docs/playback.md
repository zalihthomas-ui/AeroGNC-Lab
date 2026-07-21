# Interactive Simulation Playback

The `play` workflow turns a completed deterministic 3-DOF trajectory into a desktop
flight player. The simulation is solved first with the same custom RK4, environment,
vehicle, and event models used by the normal CLI; the player then reads the immutable
result log. Pausing or seeking therefore cannot change the engineering result.

## Start the player

From the repository root on Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m aerognc.cli play `
  --config configs\three_dof_nominal.yaml
```

The default speed is 4x, so the 31.8 s fictional flight takes approximately eight
seconds to play when the computer keeps up with the requested 30 frames per second.
Use `--speed 1` for approximately real-time playback, `--speed 8` for a faster review,
or `--repeat` to restart automatically after impact.

## Display and controls

The window contains a moving vehicle direction marker, completed trajectory trail,
full-path reference, live flight phase, telemetry, event status, altitude/speed
history, time cursor, and seek slider.

| Control | Action |
|---|---|
| Space or Play/Pause | Pause or resume |
| Timeline slider | Seek to any logged simulation time |
| Left / Right | Seek backward or forward one simulation second |
| Up / Down | Double or halve playback speed |
| `0.5x` / `2x` buttons | Halve or double playback speed |
| `R` or Restart | Return to launch and resume |
| Close window | End playback without modifying results |

Live telemetry uses SI units and includes time, altitude, ground range, vertical and
total speed, Mach number, dynamic pressure, mass, thrust, and flight-path angle.
Burnout, apogee, and ground impact are marked in both the trajectory and event panel.

## Animated GIF export

To save and also open the interactive player:

```powershell
.\.venv\Scripts\python.exe -m aerognc.cli play `
  --config configs\three_dof_nominal.yaml `
  --speed 8 --save-gif results\playback\nominal_flight.gif
```

For export without a desktop window:

```powershell
.\.venv\Scripts\python.exe -m aerognc.cli play `
  --config configs\three_dof_nominal.yaml `
  --speed 8 --fps 20 `
  --save-gif results\playback\nominal_flight.gif --no-window
```

GIF creation uses Matplotlib's Pillow writer; Pillow is an explicit bounded runtime
dependency. Transient animations outside `results/reference/` are ignored by Git
because they can be large.

## Scope and limitations

This is a playback of calculated results, not a manually piloted game and not a
real-time hardware loop. Controller-in-the-loop parameter changes during playback
would require restarting the numerical propagation and are intentionally not hidden
behind the timeline controls. The vehicle and all displayed values remain fictional
and synthetic; the player contains no target, interception, or terminal-guidance
interface.
