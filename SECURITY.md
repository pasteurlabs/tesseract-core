# Security Policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately via GitHub's private vulnerability reporting:
<https://github.com/pasteurlabs/tesseract-core/security/advisories/new>.
Do not open a public issue. We aim to acknowledge reports within 5 business days.

## Security model

See [Security considerations](https://docs.pasteurlabs.ai/projects/tesseract-core/latest/content/security/)
for our threat model and guidance on building, running, and serving Tesseracts.

Tesseracts are trusted code. The container provides reproducibility, not a sandbox.
The HTTP API is intended for trusted networks.
