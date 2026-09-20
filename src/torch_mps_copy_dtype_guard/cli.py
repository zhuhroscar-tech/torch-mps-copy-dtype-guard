"""Command-line interface: run the from-scratch diagnosis of the MPS
copy/cast dtype bug against the currently installed torch build,
using the shared semantic-color design system.
"""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-mps-copy-dtype-guard",
        description=(
            "Diagnose whether the currently installed torch build silently "
            "writes zeros when transferring an MPS tensor into a CPU "
            "float64 or complex128 destination via .to()/.copy_() "
            "(pytorch/pytorch#197715), and verify the safe_to()/"
            "safe_copy_() guards produce the correct, CPU-oracle-matching "
            "result on every exercised device. Runs on CPU always; runs "
            "on MPS only when this host's MPS device passes a real "
            "allocation probe (not just torch.backends.mps.is_available())."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-mps-copy-dtype-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["guard_fully_correct"] else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields(
        [
            ("torch version", report["torch_version"]),
            ("mps functional on this host", "yes" if report["mps_functional"] else "no (see notes below)"),
        ]
    )

    if report["any_native_silently_wrong"]:
        print(status_headline(style, "fail", "MPS-to-CPU float64/complex128 silent copy bug reproduced on this host"))
    else:
        print(status_headline(style, "info", "no silent copy bug reproduced on this host's exercised devices"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "guard matches the CPU-oracle conversion on every exercised device"))
    else:
        print(status_headline(style, "fail", "guard did NOT match the expected contract on at least one device"))

    section("per-dtype results (native .to()/.copy_() vs guarded vs CPU oracle)")
    for c in report["cases"]:
        if not c["ran"]:
            print_fields([(c["dtype_name"], f"skipped: {c['skip_reason']}")])
            continue
        to_flag = "SILENT-WRONG" if c["native_to_matches_oracle"] is False else "ok"
        copy_flag = "SILENT-WRONG" if c["native_copy_matches_oracle"] is False else "ok"
        guard_to_flag = "guard-ok" if c["guard_to_matches_oracle"] else "GUARD-FAILED"
        guard_copy_flag = "guard-ok" if c["guard_copy_matches_oracle"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    c["dtype_name"],
                    f"native.to={to_flag:12s}  native.copy_={copy_flag:12s}  "
                    f"guard.to={guard_to_flag:12s}  guard.copy_={guard_copy_flag:12s}",
                )
            ]
        )

    return 0 if report["guard_fully_correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
