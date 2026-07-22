# Security Policy

## Supported versions

AeroGNC-Lab is pre-1.0 research software. Security fixes target the latest commit on
`main` and the current 0.8.x release line; older snapshots are not maintained.

## Reporting a vulnerability

Please do not publish exploit details in a regular issue. Use GitHub's
[private vulnerability reporting](https://github.com/zalihthomas-ui/AeroGNC-Lab/security/advisories/new)
to share the affected component, reproduction steps, impact, and any suggested fix.
Remove credentials, personal data, proprietary vehicle information, and sensitive
mission data from every report and attachment.

Maintainers will acknowledge a complete report, assess its scope, and coordinate a
fix and disclosure through the private advisory. This policy covers the repository's
software, parsers, local file handling, dependency configuration, and localhost test
interfaces. It does not turn the project into certified, hardened, or operational
flight software.

## Public-safety scope

Reports and proposed fixes must preserve the project's fictional, civilian,
public-safe scope described in [CONTRIBUTING.md](CONTRIBUTING.md). Do not submit
classified, proprietary, weapon-system, interception, terminal-homing, or engagement
information.
