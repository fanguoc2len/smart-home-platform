#!/usr/bin/env python3
"""Fail when private runtime artifacts are tracked by Git."""

from __future__ import annotations

import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]

BLOCKED_EXACT = {
    ".env",
    "registered.json",
    "tempCodeRunnerFile.py",
    "transcription.txt",
    "client_secrets.json",
    "mycreds.txt",
}

BLOCKED_PREFIXES = (
    "__pycache__/",
    ".venv/",
    ".vscode/",
    "registered_images/",
    "cuda_sdk_lib/",
    "esp32_homekit/",
)

BLOCKED_SUFFIXES = (
    ".pyc",
    ".pyo",
    ".m4a",
    ".wav",
)


def tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return [line for line in result.stdout.splitlines() if line]


def is_blocked(path: str) -> bool:
    name = pathlib.PurePosixPath(path).name
    return (
        path in BLOCKED_EXACT
        or name in BLOCKED_EXACT
        or path.startswith(BLOCKED_PREFIXES)
        or path.endswith(BLOCKED_SUFFIXES)
        or (name.startswith("namtran") and name.endswith(".jpg"))
    )


def main() -> int:
    failures = [path for path in tracked_files() if is_blocked(path)]
    if failures:
        print("Private or generated files are tracked:")
        for path in failures:
            print(f"  - {path}")
        print("Keep these files local and remove them from the Git index.")
        return 1

    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
