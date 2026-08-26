# Security & Responsible Disclosure Policy

Sorbed processes medical images. A vulnerability here can expose protected
health information (PHI) or produce misleading clinical output. We take that
seriously and we ask you to as well.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.x     | :white_check_mark: (pre-release; security fixes on `main`) |

Until a 1.0 release, security fixes land on `main` and in the most recent
tagged release.

## Reporting a Vulnerability

**Please do not open a public issue for security vulnerabilities.**

Report privately through one of:

1. GitHub's **Private Vulnerability Reporting** (Security tab → "Report a
   vulnerability"), or
2. Email **32umutkk@gmail.com** with the subject line `SECURITY: Sorbed`.

Include, where possible:

- A description of the vulnerability and its impact
- Steps to reproduce or a proof of concept
- Affected version/commit
- Whether the issue can expose PHI, model weights, or system credentials

We aim to acknowledge reports within **72 hours** and to provide a remediation
timeline within **7 days**. We will credit reporters who wish to be named once a
fix is released.

## Scope

In scope:

- Remote code execution, path traversal, deserialization, or SSRF in the API,
  CLI, or image decoders
- PHI leakage: unintended persistence, logging, or transmission of patient
  images or metadata (DICOM tags in particular)
- Model-weight supply-chain issues (tampered checksums, unsafe pickle loading)
- Authentication/authorization flaws in the optional server deployment

Out of scope:

- The inherent clinical limitations of the model (see `DISCLAIMER.md`) — these
  are documented, not vulnerabilities
- Denial of service from intentionally huge uploads when rate limiting is
  disabled by the operator
- Findings that require a compromised host or physical access

## Handling of Patient Data

Sorbed is designed to run locally and to **not** transmit images off the host by
default. If you find any code path that violates that principle, treat it as a
security vulnerability and report it privately.
