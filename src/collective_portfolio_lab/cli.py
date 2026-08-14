"""CLI entry point for collective-portfolio-lab."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from collective_portfolio_lab.core import (
    cara_expected_product_benchmark,
    detect_non_monotonic,
    run_binary_example,
    sweep_continuous,
    validate_bounds,
)
from collective_portfolio_lab.plots import plot_figure1_style


def cmd_binary_example(args) -> None:
    results = run_binary_example()
    out = [
        {
            "gamma2": r["gamma2"],
            "risky_expected_utility": r["risky_expected_utility"],
            "safe_utility": r["safe_utility"],
            "difference": r["difference"],
            "choice": r["choice"],
        }
        for r in results
    ]
    if args.format == "json":
        print(json.dumps(out, indent=None))
    else:
        for rec in out:
            msg = (
                f"gamma2={rec['gamma2']} risky={rec['risky_expected_utility']} "
                f"safe={rec['safe_utility']} choice={rec['choice']}"
            )
            print(msg)


def cmd_reproduce(args) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Binary CSV
    binary_path = out_dir / "binary_example.csv"
    with open(binary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gamma2", "risky_expected_utility", "safe_utility", "difference", "choice"])
        for r in run_binary_example():
            writer.writerow([r["gamma2"], r["risky_expected_utility"], r["safe_utility"], r["difference"], r["choice"]])

    # Continuous sweeps
    # CARA: A1=2, A2 from 0.5 to 8 with 60 points
    cara_range = np.linspace(0.5, 8.0, 60)
    cara_result = sweep_continuous(
        "cara",
        param1_fixed=2.0,
        param_range2=cara_range,
        lambda1=0.5,
        lambda2=0.5,
        mu=0.1,
        sigma=0.2,
        rf=0.0,
        W=2.0,
        seed=42,
        n_sims=20000,
    )
    # CRRA: gamma1=4, gamma2 from 0.5 to 12 with 60 points
    crra_range = np.linspace(0.5, 12.0, 60)
    crra_result = sweep_continuous(
        "crra",
        param1_fixed=4.0,
        param_range2=crra_range,
        lambda1=0.5,
        lambda2=0.5,
        mu=0.1,
        sigma=0.2,
        rf=0.0,
        W=2.0,
        seed=42,
        n_sims=20000,
    )

    # CSV outputs
    for name, res in [("figure1_cara", cara_result), ("figure1_crra", crra_result)]:
        csv_path = out_dir / f"{name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["risk_aversion_2", "individual_1_share", "individual_2_share", "household_share"])
            for i in range(len(res["risk_aversion_2"])):
                writer.writerow(
                    [
                        res["risk_aversion_2"][i],
                        float(np.clip(res["individual_1_share"][i], 0.0, 1.0)),
                        float(np.clip(res["individual_2_share"][i], 0.0, 1.0)),
                        float(np.clip(res["household_share"][i], 0.0, 1.0)),
                    ]
                )

    # PNG outputs
    plot_figure1_style(cara_result, out_dir / "figure1_cara.png")
    plot_figure1_style(crra_result, out_dir / "figure1_crra.png")

    # Non-monotonicity detection
    cara_nonmono = detect_non_monotonic(cara_result["household_share"])
    crra_nonmono = detect_non_monotonic(crra_result["household_share"])

    # Analytic benchmark comparison for CARA at midpoint
    benchmark = cara_expected_product_benchmark(
        A1=2.0,
        A2=float(np.median(cara_range)),
        lambda1=0.5,
        lambda2=0.5,
        mu=0.1,
        sigma=0.2,
        W=2.0,
        rf=0.0,
    )

    # Summary JSON
    summary = {
        "model_parameters": {
            "mu": 0.1,
            "sigma": 0.2,
            "rf": 0.0,
            "wealth": 2.0,
            "lambda1": 0.5,
            "lambda2": 0.5,
            "simulation_count": 20000,
            "seed": 42,
            "points": 60,
        },
        "non_monotonicity": {
            "cara_detected": bool(cara_nonmono),
            "crra_detected": bool(crra_nonmono),
        },
        "benchmark": {
            "expected_product_cara_midpoint": benchmark,
        },
        "notes": "Independent implementation inspired by arXiv:2608.12411v1. Not official author code.",
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def cmd_scan(args) -> None:
    utility = args.utility
    n_points = args.points
    alpha_lb = args.lower
    alpha_ub = args.upper
    output_path = args.output

    validate_bounds(alpha_lb, alpha_ub)

    if utility not in ("cara", "crra"):
        raise ValueError("utility must be 'cara' or 'crra'")

    if utility == "cara":
        param_range = np.linspace(alpha_lb, alpha_ub, n_points)
        result = sweep_continuous(
            "cara",
            param1_fixed=2.0,
            param_range2=param_range,
            lambda1=0.5,
            lambda2=0.5,
            mu=0.1,
            sigma=0.2,
            rf=0.0,
            W=2.0,
            seed=args.seed,
            n_sims=args.simulations,
        )
    else:
        param_range = np.linspace(alpha_lb, alpha_ub, n_points)
        result = sweep_continuous(
            "crra",
            param1_fixed=4.0,
            param_range2=param_range,
            lambda1=0.5,
            lambda2=0.5,
            mu=0.1,
            sigma=0.2,
            rf=0.0,
            W=2.0,
            seed=args.seed,
            n_sims=args.simulations,
        )

    rows = []
    for i in range(len(param_range)):
        rows.append(
            {
                "risk_aversion_2": float(result["risk_aversion_2"][i]),
                "individual_1_share": float(result["individual_1_share"][i]),
                "individual_2_share": float(result["individual_2_share"][i]),
                "household_share": float(result["household_share"][i]),
            }
        )

    if args.format == "json":
        print(json.dumps(rows, indent=2))
    else:
        csv_path = Path(output_path) if output_path else None
        if csv_path:
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["risk_aversion_2", "individual_1_share", "individual_2_share", "household_share"])
                for r in rows:
                    writer.writerow(
                        [r["risk_aversion_2"], r["individual_1_share"], r["individual_2_share"], r["household_share"]]
                    )
            print(f"Saved CSV to {csv_path}")
        else:
            for r in rows:
                msg = (
                    f"{r['risk_aversion_2']}, {r['individual_1_share']}, "
                    f"{r['individual_2_share']}, {r['household_share']}"
                )
                print(msg)


def main():
    parser = argparse.ArgumentParser(
        description="Collective portfolio choice toolkit inspired by arXiv:2608.12411v1.",
    )
    sub = parser.add_subparsers(dest="command")

    # binary-example
    bin_parser = sub.add_parser("binary-example", help="Reproduce binary CRRA gamble.")
    bin_parser.add_argument("--format", choices=["json", "text"], default="json")
    bin_parser.set_defaults(func=cmd_binary_example)

    # reproduce
    rep_parser = sub.add_parser("reproduce", help="Generate CSV, PNG, and JSON artifacts.")
    rep_parser.add_argument("--output-dir", required=True)
    rep_parser.set_defaults(func=cmd_reproduce)

    # scan
    scan_parser = sub.add_parser("scan", help="Custom parameter sweep.")
    scan_parser.add_argument("--utility", choices=["cara", "crra"], default="cara")
    scan_parser.add_argument("--lower", type=float, default=0.5)
    scan_parser.add_argument("--upper", type=float, default=8.0)
    scan_parser.add_argument("--points", type=int, default=60)
    scan_parser.add_argument("--seed", type=int, default=42)
    scan_parser.add_argument("--simulations", type=int, default=20000)
    scan_parser.add_argument("--format", choices=["json", "csv"], default="json")
    scan_parser.add_argument("--output", type=str, default="scan.csv")
    scan_parser.set_defaults(func=cmd_scan)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
