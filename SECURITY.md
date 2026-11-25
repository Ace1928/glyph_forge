# Security policy

## Supported versions

Security fixes are provided for the latest minor release in the 1.x line. Users
should upgrade to the newest patch before reporting an issue that may already be
resolved.

| Version | Security support |
| --- | --- |
| 1.x | Supported |
| 0.x | Best effort; upgrade recommended |

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability.

Use GitHub's private security-advisory flow for this repository when available.
If that is not possible, email <ace1928@gmail.com> with the subject
`Glyph Forge security report`.

Include, when possible:

- affected version and operating system;
- a minimal reproduction or proof of concept;
- expected and observed behavior;
- potential impact and required preconditions;
- any known workaround;
- whether public disclosure is already planned.

Do not include credentials, private media, authentication cookies, or unrelated
personal data. A maintainer will aim to acknowledge a complete report within
three business days. Remediation and disclosure timing depends on severity,
affected platforms, and the availability of a tested fix.

## Security boundaries

Glyph Forge processes untrusted media through Pillow, OpenCV, ffmpeg, browsers,
and platform capture APIs. Keep these dependencies current and process hostile
files in an appropriately restricted environment.

Important safeguards in the 1.x design include:

- package import has no filesystem, logging, signal, network, or subprocess side
  effects;
- network and screen-capture CLI flows require an explicit confirmation;
- subprocess argument lists are used without `shell=True`;
- user filenames are escaped in generated gallery HTML;
- streaming lookup tables use validated NPZ with `allow_pickle=False`;
- cache and configuration writes use temporary files and atomic replacement;
- persistent user configuration receives owner-only permissions on POSIX;
- optional browser, network, capture, audio, and TUI dependencies are isolated;
- dependency scanning runs in CI.

Glyph Forge does not require exported browser cookies. If a user explicitly
passes a yt-dlp cookie file, its confidentiality remains the user's
responsibility. Cookie files and common secret formats are excluded by the
repository ignore rules.

## Disclosure

The project prefers coordinated disclosure after a patch is available. Security
researchers are credited with permission. Public advisories will describe
affected versions, impact, remediation, and relevant upgrade instructions while
avoiding unnecessary exposure of user data.
