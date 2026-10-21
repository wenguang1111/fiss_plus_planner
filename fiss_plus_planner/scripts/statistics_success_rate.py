import argparse
import csv
import math
from pathlib import Path


MEASUREMENTS_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "measurements"
)


DEFAULT_CSVS = {
    "FISS+": MEASUREMENTS_DIR / "measurement_FISS+.csv",
    "FOP+": MEASUREMENTS_DIR / "measurement_FOP+.csv",
    "FOP+ random": MEASUREMENTS_DIR / "measurement_FOP+_random.csv",
    "FOP+ WM": MEASUREMENTS_DIR / "measurement_FOP+_WM.csv",
    # "Sparse CVAE": MEASUREMENTS_DIR / "measurement_Sparse_CVAE.csv",
    # "Sparse RLoffline": MEASUREMENTS_DIR / "measurement_Sparse_RLoffline.csv",
    # "Sparse WorldModel": MEASUREMENTS_DIR / "measurement_Sparse_WorldModel.csv",
    # "Sparse WM_selection": MEASUREMENTS_DIR / "measurement_Sparse_WM_selection.csv",
    # "Sparse CVAE random": MEASUREMENTS_DIR / "measurement_Sparse_CVAE_random.csv",
}


REJECTION_COLUMNS = {
    "Dynamic": "last_cycle_rejected_dynamic",
    "Offroad": "last_cycle_rejected_offroad",
    "Collision": "last_cycle_rejected_collision",
}


def parse_success(value: str) -> bool:
    normalized = value.strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    raise ValueError(f"Invalid success value: {value!r}")


def parse_int(value: str) -> int:
    return int(float(value.strip()))


def parse_optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None


def calculate_statistics(csv_path: Path):
    total = 0
    succeeded = 0
    failed = 0
    horizon_records = 0
    reached_final_time = 0
    horizon_failures = 0
    runtime_values = []
    runtime_squared_values = []
    final_trajectory_cost_values = []

    # Total number of rejected trajectories in the
    # last cycle of failed scenarios.
    failed_rejection_totals = {
        rejection_type: 0
        for rejection_type in REJECTION_COLUMNS
    }

    # Number of failed scenarios in which each
    # rejection type occurred at least once.
    failed_scenarios_with_rejection = {
        rejection_type: 0
        for rejection_type in REJECTION_COLUMNS
    }

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise ValueError(
                f"CSV has no header: {csv_path}"
            )

        reader.fieldnames = [
            field.strip()
            for field in reader.fieldnames
        ]
        has_horizon_column = "reached_final_time" in reader.fieldnames
        runtime_column = next(
            (name for name in reader.fieldnames if name.lower().startswith("average runtime")),
            None,
        )
        normalized_columns = {name.lower(): name for name in reader.fieldnames}
        final_cost_column = next(
            (
                normalized_columns[name]
                for name in ("final_trajectory_cost", "final_trajector_cost")
                if name in normalized_columns
            ),
            None,
        )

        required_columns = {"scenario"}

        if "success" not in reader.fieldnames:
            required_columns.update(
                {
                    "steps",
                    "step_number_for_break",
                }
            )

        required_columns.update(
            REJECTION_COLUMNS.values()
        )

        missing_columns = (
            required_columns.difference(reader.fieldnames)
        )

        if missing_columns:
            raise ValueError(
                f"CSV is missing required columns "
                f"{sorted(missing_columns)}: {csv_path}"
            )

        for row in reader:

            row = {
                key.strip(): value
                for key, value in row.items()
                if key is not None
            }

            if not row.get("scenario"):
                continue

            total += 1
            if runtime_column:
                runtime = parse_optional_float(row.get(runtime_column))
                if runtime is not None and math.isfinite(runtime):
                    runtime_values.append(runtime)
                    runtime_squared_values.append(runtime * runtime)
            if final_cost_column:
                final_cost = parse_optional_float(row.get(final_cost_column))
                if final_cost is not None:
                    final_trajectory_cost_values.append(final_cost)

            # ---------------------------------------------------------
            # Determine whether the scenario succeeded
            # ---------------------------------------------------------

            if "success" in row:
                succeeded_row = parse_success(
                    row["success"]
                )
            else:
                steps = parse_int(row["steps"])
                break_step = parse_int(
                    row["step_number_for_break"]
                )

                succeeded_row = (
                    break_step == 0
                    or break_step >= steps
                )

            if has_horizon_column and (row.get("reached_final_time") or "").strip():
                horizon_records += 1
                if parse_success(row["reached_final_time"]):
                    reached_final_time += 1
                    if not succeeded_row:
                        horizon_failures += 1

            # ---------------------------------------------------------
            # Success / failure
            # ---------------------------------------------------------

            if succeeded_row:
                succeeded += 1
                continue

            failed += 1

            # ---------------------------------------------------------
            # Rejection statistics
            #
            # Only the LAST cycle of FAILED scenarios
            # is considered.
            # ---------------------------------------------------------

            for rejection_type, column in REJECTION_COLUMNS.items():

                rejected = parse_int(row[column])

                failed_rejection_totals[
                    rejection_type
                ] += rejected

                if rejected > 0:
                    failed_scenarios_with_rejection[
                        rejection_type
                    ] += 1

    success_percentage = (
        succeeded / total * 100.0
        if total
        else 0.0
    )

    failure_percentage = (
        failed / total * 100.0
        if total
        else 0.0
    )

    total_rejected = sum(
        failed_rejection_totals.values()
    )

    # Percentage of all rejected trajectories belonging
    # to each rejection category.
    rejection_percentages = {}

    for rejection_type, count in failed_rejection_totals.items():
        rejection_percentages[rejection_type] = (
            count / total_rejected * 100.0
            if total_rejected
            else 0.0
        )

    # Percentage of failed scenarios affected by
    # each rejection type.
    scenario_rejection_percentages = {}

    for (
        rejection_type,
        count,
    ) in failed_scenarios_with_rejection.items():

        scenario_rejection_percentages[rejection_type] = (
            count / failed * 100.0
            if failed
            else 0.0
        )

    return {
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "success_percentage": success_percentage,
        "failure_percentage": failure_percentage,
        "horizon_records": horizon_records,
        "reached_final_time": reached_final_time,
        "horizon_failures": horizon_failures,
        "average_runtime": sum(runtime_values) / len(runtime_values) if runtime_values else None,
        "runtime_standard_deviation": (
            math.sqrt(max(
                sum(runtime_squared_values) / len(runtime_squared_values)
                - (sum(runtime_values) / len(runtime_values)) ** 2,
                0.0,
            ))
            if runtime_values else None
        ),
        "runtime_records": len(runtime_values),
        "average_final_trajectory_cost": (
            sum(final_trajectory_cost_values) / len(final_trajectory_cost_values)
            if final_trajectory_cost_values else None
        ),
        "final_trajectory_cost_records": len(final_trajectory_cost_values),
        "failed_rejection_totals": failed_rejection_totals,
        "failed_scenarios_with_rejection":
            failed_scenarios_with_rejection,
        "total_rejected": total_rejected,
        "rejection_percentages": rejection_percentages,
        "scenario_rejection_percentages":
            scenario_rejection_percentages,
    }


def print_comparison(results):
    print()
    print("=" * 100)
    print("SCENARIO SUCCESS COMPARISON")
    print("=" * 100)

    print(
        f"{'Method':<25}"
        f"{'Total':>10}"
        f"{'Success':>10}"
        f"{'Failed':>10}"
        f"{'Success %':>12}"
        f"{'Avg runtime (s)':>18}"
        f"{'Runtime SD (s)':>18}"
        f"{'Avg final cost':>18}"
    )

    print("-" * 100)

    for method, stats in results.items():
        print(
            f"{method:<25}"
            f"{stats['total']:>10}"
            f"{stats['succeeded']:>10}"
            f"{stats['failed']:>10}"
            f"{stats['success_percentage']:>11.2f}%"
            f"{stats['average_runtime'] if stats['average_runtime'] is not None else float('nan'):>18.4f}"
            f"{stats['runtime_standard_deviation'] if stats['runtime_standard_deviation'] is not None else float('nan'):>18.4f}"
            f"{stats['average_final_trajectory_cost'] if stats['average_final_trajectory_cost'] is not None else float('nan'):>18.4f}"
        )

    print()
    print("SIMULATION HORIZON COMPLETION")
    print("-" * 100)
    print(
        f"{'Method':<25}"
        f"{'Recorded':>12}"
        f"{'Reached horizon':>20}"
        f"{'Horizon, goal not reached':>30}"
    )
    print("-" * 100)
    for method, stats in results.items():
        reached = stats["reached_final_time"]
        reached_without_goal = stats["horizon_failures"]
        recorded = stats["horizon_records"]
        print(
            f"{method:<25}"
            f"{recorded:>12}"
            f"{reached:>20}"
            f"{reached_without_goal:>30}"
        )
    if any(stats["horizon_records"] == 0 for stats in results.values()):
        print("Horizon counts are unavailable for CSV files without the reached_final_time column.")

    # -------------------------------------------------------------
    # Rejection statistics
    # -------------------------------------------------------------

    print()
    print("=" * 100)
    print("REJECTION STATISTICS - FAILED SCENARIOS")
    print("=" * 100)

    print(
        "\nThe rejection statistics consider only the LAST cycle "
        "of failed scenarios."
    )

    # -------------------------------------------------------------
    # Total rejected trajectories
    # -------------------------------------------------------------

    print()
    print("TOTAL REJECTED TRAJECTORIES")
    print("-" * 100)

    print(
        f"{'Method':<25}"
        f"{'Dynamic':>15}"
        f"{'Offroad':>15}"
        f"{'Collision':>15}"
        f"{'Total':>15}"
    )

    print("-" * 100)

    for method, stats in results.items():

        rejected = stats["failed_rejection_totals"]

        print(
            f"{method:<25}"
            f"{rejected['Dynamic']:>15}"
            f"{rejected['Offroad']:>15}"
            f"{rejected['Collision']:>15}"
            f"{stats['total_rejected']:>15}"
        )

    # -------------------------------------------------------------
    # Percentage of rejection types
    # -------------------------------------------------------------

    print()
    print("PERCENTAGE OF REJECTED TRAJECTORIES BY TYPE")
    print("-" * 100)

    print(
        f"{'Method':<25}"
        f"{'Dynamic':>15}"
        f"{'Offroad':>15}"
        f"{'Collision':>15}"
    )

    print("-" * 100)

    for method, stats in results.items():

        percentages = stats["rejection_percentages"]

        print(
            f"{method:<25}"
            f"{percentages['Dynamic']:>14.2f}%"
            f"{percentages['Offroad']:>14.2f}%"
            f"{percentages['Collision']:>14.2f}%"
        )

    # -------------------------------------------------------------
    # Percentage of failed scenarios affected
    # -------------------------------------------------------------

    print()
    print(
        "PERCENTAGE OF FAILED SCENARIOS AFFECTED "
        "BY EACH REJECTION TYPE"
    )
    print("-" * 100)

    print(
        f"{'Method':<25}"
        f"{'Dynamic':>15}"
        f"{'Offroad':>15}"
        f"{'Collision':>15}"
    )

    print("-" * 100)

    for method, stats in results.items():

        percentages = (
            stats["scenario_rejection_percentages"]
        )

        print(
            f"{method:<25}"
            f"{percentages['Dynamic']:>14.2f}%"
            f"{percentages['Offroad']:>14.2f}%"
            f"{percentages['Collision']:>14.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare success and rejection statistics "
            "for FOP+ variants, FISS+, and sparse planner variants."
        )
    )

    parser.add_argument(
            "--fiss",
            type=Path,
            default=DEFAULT_CSVS["FISS+"],
            help=(
                "FISS+ measurement CSV "
                f"(default: {DEFAULT_CSVS['FISS+' ]})"
            ),
    )

    parser.add_argument(
        "--fopplus",
        type=Path,
        default=DEFAULT_CSVS["FOP+"],
        help=f"FOP+ measurement CSV (default: {DEFAULT_CSVS['FOP+']})",
    )

    parser.add_argument(
        "--fopplus-random",
        type=Path,
        default=DEFAULT_CSVS["FOP+ random"],
        help=f"FOP+ random-selection measurement CSV (default: {DEFAULT_CSVS['FOP+ random']})",
    )

    parser.add_argument(
        "--fopplus-wm",
        type=Path,
        default=DEFAULT_CSVS["FOP+ WM"],
        help=f"FOP+ world-model-selection measurement CSV (default: {DEFAULT_CSVS['FOP+ WM']})",
    )   

    # parser.add_argument(
    #     "--cvae", 
    #     type=Path,
    #     default=DEFAULT_CSVS["Sparse CVAE"],
    #     help=(
    #         "Sparse CVAE measurement CSV "
    #         f"(default: {DEFAULT_CSVS['Sparse CVAE']})"
    #     ),
    # )

    # parser.add_argument(
    #     "--worldmodel",
    #     type=Path,
    #     default=DEFAULT_CSVS["Sparse WorldModel"],
    #     help=(
    #         "Sparse WorldModel measurement CSV "
    #         f"(default: {DEFAULT_CSVS['Sparse WorldModel']})"
    #     ),
    # )

    # parser.add_argument(
    #     "--rloffline",
    #     type=Path,
    #     default=DEFAULT_CSVS["Sparse RLoffline"],
    #     help=(
    #         "Sparse RLoffline measurement CSV "
    #         f"(default: {DEFAULT_CSVS['Sparse RLoffline']})"
    #     ),
    # )

    # parser.add_argument(
    #     "--wmselection",
    #     type=Path,
    #     default=DEFAULT_CSVS["Sparse WM_selection"],
    #     help=(
    #         "Sparse WM_selection measurement CSV "
    #         f"(default: {DEFAULT_CSVS['Sparse WM_selection']})"
    #     ),
    # )

    # parser.add_argument(
    #     "--random-selection",
    #     type=Path,
    #     default=None,
    #     help=(
    #         "Sparse CVAE random-selection measurement CSV "
    #         f"(default: include {DEFAULT_CSVS['Sparse CVAE random']} if it exists)"
    #     ),
    # )

    args = parser.parse_args()

    csv_files = {
        "FISS+": args.fiss,
        "FOP+": args.fopplus,
        "FOP+ random": args.fopplus_random,
        "FOP+ WM": args.fopplus_wm,
        # "Sparse CVAE": args.cvae,
        # "Sparse RLoffline": args.rloffline,
        # "Sparse WorldModel": args.worldmodel,
        # "Sparse WM_selection": args.wmselection,
    }

    # sparse_random_csv = args.random_selection or DEFAULT_CSVS["Sparse CVAE random"]
    # if args.random_selection is not None or sparse_random_csv.is_file():
    #     csv_files["Sparse CVAE random"] = sparse_random_csv
    # else:
    #     print(f"Skipping Sparse CVAE random: default CSV not found: {sparse_random_csv}")

    results = {}

    for method, csv_path in csv_files.items():

        csv_path = csv_path.resolve()

        if not csv_path.is_file():
            raise FileNotFoundError(
                f"{method} CSV not found: {csv_path}"
            )

        print(
            f"Processing {method}: {csv_path}"
        )

        results[method] = calculate_statistics(
            csv_path
        )

    print_comparison(results)


if __name__ == "__main__":
    main()


