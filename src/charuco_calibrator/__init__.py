"""Public package for the ChArUco calibrator GUI."""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    from .cli import main as cli_main

    return cli_main(argv)


__all__ = ["main"]
