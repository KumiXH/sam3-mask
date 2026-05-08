from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sam3-mask")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--labels",
        nargs="+",
        help="Override prompt labels with one or more values.",
    )
    parser.add_argument(
        "--save-cutout",
        dest="save_cutout",
        action="store_true",
        help="Enable alpha-cutout export.",
    )
    parser.add_argument(
        "--no-save-cutout",
        dest="save_cutout",
        action="store_false",
        help="Disable alpha-cutout export.",
    )
    parser.add_argument("--backend", choices=["dummy", "sam3"], help="Model backend to use.")
    parser.add_argument("--device", help="Runtime device, such as cpu, cuda, or cuda:0.")
    parser.add_argument("--checkpoint", help="Path to the model checkpoint.")
    parser.add_argument("--label-mode", choices=["single", "multi"], help="Object label assignment mode.")
    parser.set_defaults(save_cutout=None)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if args.labels is not None:
        overrides["prompts.labels"] = list(args.labels)
    if args.save_cutout is not None:
        overrides["output.save_cutout"] = args.save_cutout
    if args.backend:
        overrides["model.backend"] = args.backend
    if args.device:
        overrides["model.device"] = args.device
    if args.checkpoint:
        overrides["model.checkpoint"] = args.checkpoint
    if args.label_mode:
        overrides["prompts.label_mode"] = args.label_mode
    return overrides
