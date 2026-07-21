# Desktop UI Architecture Decision

## Decision

AeroGNC-Lab retains its native Tk/ttk Simulation Workbench for the first publishable
release. The numerical and project services remain independent of Tk so this is a
reversible delivery decision, not an architectural lock-in. PySide6/Qt is not added
as a dependency, and the local-web comparison remains an explicitly non-production
prototype that does not call an engineering solver.
In other words, the prototype is not connected to the engineering solver.

The current workbench already gives a user labelled SI-unit fields, verified presets,
hover help, validation messages, background execution, cooperative cancellation,
3D playback, project/run history, comparison, and report access. Replacing it now
would add a second application stack. Direct usability feedback did identify a real
problem in the existing information architecture: workflows were named by internal
engineering concepts, specialist fields appeared before the user understood the
question, and result panels could fall below the minimum-height window. The retained
Tk application was therefore redesigned and visually checked before reconsidering a
framework migration.

## Compared options

| Criterion | Retained Tk/ttk workbench | PySide6/Qt candidate | Local-web prototype |
|---|---|---|---|
| Accessibility | Standard labelled widgets support keyboard traversal and native focus. Canvas plots need text summaries, and no screen-reader certification is claimed. | Rich widget/accessibility APIs could support a more sophisticated desktop product, but AeroGNC-Lab has not installed or tested them. | Semantic landmarks, explicit labels, helper text, visible focus, `aria-live`, responsive layout, and reduced-motion handling are demonstrated. Browser/screen-reader testing is still required. |
| Deployment | Python's Tk binding is already present in the verified environment; the one-click batch launcher has no extra UI package. | Adds a large binary dependency and platform packaging/signing work. PySide6 was not installed during this decision, so no startup result is invented. | The prototype uses one offline HTML file and a standard-library localhost server. A production version would still need browser launch, port lifecycle, origin/security policy, and API packaging. |
| 3D capability | Existing Matplotlib 3D rocket, planetary-tour, and catalog views work and share the tested Python data objects. Interaction is adequate rather than game-engine grade. | Qt/OpenGL integration could provide higher-performance scenes, picking, and docking, at greater implementation and verification cost. | WebGL could provide strong 3D interaction, but the prototype intentionally uses only an illustrative canvas. Selecting a WebGL engine would add a separately licensed JavaScript stack and new numerical/UI boundary tests. |
| Licensing | AeroGNC-Lab remains MIT-licensed; Python/Tcl/Tk distribution terms must still accompany packaged runtimes. | PySide/Qt distribution choices and LGPL/GPL/commercial obligations require a packaging-specific review before adoption. | This dependency-free prototype adds no third-party browser library. Future framework and WebGL dependencies would require a new licence review. |
| Startup and memory | A hidden eight-page widget skeleton was measured successfully. See the scoped reference measurement below. | Availability was checked, but construction was not executed because PySide6 is absent. | A real ephemeral localhost server returned the complete prototype, then shut down. See the scoped measurement below. |
| Maintenance | One Python UI, one service layer, and existing tests. Tk-specific code remains isolated in `visualisation/workbench.py`. | A migration would duplicate widgets/tests during transition and require platform-specific Qt maintenance. | A production client would introduce HTML/CSS/JavaScript, an HTTP/API boundary, browser compatibility, and additional security maintenance. |

## Comparison prototype

[`workbench_local_web.html`](prototypes/workbench_local_web.html) is an offline form
shell for input-comprehension and accessibility evaluation. It includes a verified
preset selector, plain-language help, explicit units, numeric bounds, responsive
layout, keyboard focus styling, a live validation status, and an illustrative canvas.
Both the page header and its generated status state that it is not connected to the
engineering solver. Consequently, it cannot be confused with simulation evidence.

The repeatable probe is:

```powershell
python scripts/ui_capability_probe.py --output results/diagnostics/ui_capability_probe.json
```

On the Windows/Python 3.13 release environment on 2026-07-20, the latest hidden Tk
skeleton constructed in 0.087 s with a 0.973 MB `tracemalloc` peak. A separate probe
constructed all eight production pages, loaded the bundled five-scenario project and
the 6,324-row local catalog, and reached idle in 5.782 s with a 78.425 MB peak. The
standard-library web prototype served 6,572 bytes over `127.0.0.1` in 0.045 s with a
0.247 MB peak. PySide6 was unavailable and was not executed. These values are
machine-specific construction diagnostics: `tracemalloc` observes Python allocations
rather than full process resident memory, and none is a startup requirement or a
rendering benchmark. The JSON record preserves full precision and the exact scope.

## Accessibility and usability actions retained

The production workbench now begins with the questions a user can answer, explains
the input-to-equations-to-results flow, and runs either verified example with one
action. Rocket and planetary pages expose basic values first; specialist numerical,
orientation, orbit, propulsion, and limit values remain behind explicit disclosures.
Pages scroll at the minimum supported size and move to a plain-language explanation
after a run. Rocket, satellite, aircraft, and planetary-trip pages expose only their
basic questions first and keep specialist fields behind explicit disclosures.
Purpose-labelled tabs separate solvers, saved evidence, read-only astronomy context,
and engineering checks. Editable values retain explicit units, validation occurs
before execution, and long calculations remain off the event thread.

This is not an accessibility conformance claim. A release intended for broad public
deployment should add keyboard-only task testing, screen-reader testing on target
operating systems, colour-contrast automation, zoom/high-DPI checks, and user studies
with both new and experienced simulation users.

## Revisit triggers

The decision should be reopened only when evidence justifies migration. Useful
triggers are measured Matplotlib interaction limits on representative catalog sizes,
a requirement for accessible widgets that Tk cannot meet, cross-platform packaging
failures, a need for multi-window docking, or repeated user-test failures in the
current input flow. A candidate must then run the same scenario/project service,
preserve deterministic outputs, pass input-contract tests, and demonstrate a clear
accessibility or performance improvement before Tk is removed.
