"""File handling for analyses of licensed cohort data."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]


def require_data_access():
    """Require the analyst to acknowledge their own data-use permissions."""
    if os.environ.get("COHORT_DATA_ACCESS") != "authorised":
        raise PermissionError(
            "Set COHORT_DATA_ACCESS=authorised only after obtaining the required "
            "cohort permissions for your account and computing environment."
        )


def external(path):
    """Keep cohort data, fitted objects and analysis outputs outside Git."""
    result = Path(path).expanduser().resolve()
    if result == REPOSITORY or REPOSITORY in result.parents:
        raise ValueError("Data and analysis outputs must be outside the repository")
    return result


def fresh_output(path):
    result = external(path)
    os.umask(0o077)
    result.mkdir(mode=0o700, parents=True, exist_ok=False)
    return result


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()
