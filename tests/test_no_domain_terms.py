"""CI guardrail: the core must contain ZERO domain semantics (BRIEF.md §6, §16).

This greps the installable package for a denylist of domain terms and fails on any hit.
Matching is whole-word and case-insensitive, so generic engineering vocabulary that
merely *contains* a domain word as a substring is not flagged (e.g. ``ballistic`` is fine
even though ``ball`` is denylisted; ``\bball\b`` does not match inside ``ballistic``).

Never weaken this test. Extend the denylist as new domains tempt their way in.
"""

from __future__ import annotations

import re
from pathlib import Path

# Whole-word domain terms that must never appear anywhere in the core package.
DENYLIST = [
    "ball",
    "player",
    "court",
    "tennis",
    "padel",
    "racket",
    "racquet",
    "sport",
    "soccer",
    "football",
    "basketball",
    "baseball",
    "hockey",
    "cricket",
    "goalkeeper",
    "referee",
    "puck",
    "shuttlecock",
    "net",
]

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "src" / "trackphysics"

_PATTERN = re.compile(r"\b(" + "|".join(re.escape(t) for t in DENYLIST) + r")\b", re.IGNORECASE)


def _source_files() -> list[Path]:
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def test_package_root_exists() -> None:
    assert PACKAGE_ROOT.is_dir(), f"expected package at {PACKAGE_ROOT}"
    assert _source_files(), "no source files found to scan"


def test_no_domain_terms_in_core() -> None:
    offenders: list[str] = []
    for path in _source_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = _PATTERN.search(line)
            if match:
                rel = path.relative_to(PACKAGE_ROOT.parent.parent)
                term = match.group(0)
                offenders.append(f"{rel}:{lineno}: domain term {term!r} -> {line.strip()}")
    assert not offenders, "domain terms leaked into core:\n" + "\n".join(offenders)
