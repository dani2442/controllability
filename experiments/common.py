"""CLI and output utilities shared by all experiments."""

from __future__ import annotations

import argparse


def parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--quality", choices=("quick", "paper"), default="quick",
                   help="quick is intended for iteration; paper uses finer grids")
    return p


def tex_num(value: float, digits: int = 3) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-2 or abs(value) >= 1e3:
        return f"\\num{{{value:.{digits}e}}}"
    return f"\\num{{{value:.{digits}g}}}"
