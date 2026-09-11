"""Download policy helpers for publication/revision datasets.

The public CLI accepts legacy booleans as well as ``auto``.  ``auto`` only
permits datasets explicitly marked safe for unattended download; large or
terms-sensitive datasets require an explicit opt-in or manual preparation.
"""
from __future__ import annotations

import argparse

DOWNLOAD_MODES = ("auto", "yes", "no")


def parse_download_mode(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return "yes"
    if text in {"0", "false", "no", "n", "off"}:
        return "no"
    if text == "auto":
        return "auto"
    raise argparse.ArgumentTypeError("download must be one of: auto, yes/true, no/false")


def normalize_download_mode(value) -> str:
    if value is None:
        return "no"
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return "yes"
    if text in {"0", "false", "no", "n", "off", ""}:
        return "no"
    if text == "auto":
        return "auto"
    raise ValueError(f"Unknown download mode: {value!r}")


def should_download(value, policy: str) -> bool:
    """Resolve a CLI download value against a dataset policy.

    Policies:
    - safe_auto: auto/yes may download
    - large_auto: only explicit yes may download
    - manual: never auto-download (licence/auth/manual-layout datasets)
    """
    mode = normalize_download_mode(value)
    policy = str(policy or "manual").lower()
    if policy == "manual":
        return False
    if mode == "yes":
        return True
    if mode == "auto":
        return policy == "safe_auto"
    return False
