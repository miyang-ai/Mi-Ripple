# Security policy

## Supported versions

Security fixes are applied to the latest release.

## Reporting

Please report suspected vulnerabilities privately through GitHub's
“Report a vulnerability” feature for this repository. Do not open a public issue
until the maintainers have responded.

Never include API keys, private images, signed URLs, or production traces in a
report. Use synthetic inputs where possible.

## Untrusted images

Applications exposing Mi-Ripple to untrusted uploads must enforce file-size,
pixel-count, format, timeout, memory, and concurrency limits before invoking the
library. The command-line research implementation does not provide a hardened
upload service.
