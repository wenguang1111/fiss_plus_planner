import argparse
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Set default planner names here for easy modification.
PLANNER_A_NAME = "Sparse"
PLANNER_B_NAME = "FISS+"


@dataclass
class ScenarioMetrics:
    avg_cost: float
    max_cost: float
    avg_runtime: float


def _to_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_measurements(csv_path: str) -> Dict[str, ScenarioMetrics]:
    data: Dict[str, ScenarioMetrics] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            scenario = row.get("scenario")
            if not scenario:
                continue
            data[scenario] = ScenarioMetrics(
                avg_cost=_to_float(row.get("average_cost")),
                max_cost=_to_float(row.get("max_cost")),
                avg_runtime=_to_float(row.get("average runtime_plan [s]")),
            )
    return data


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(metrics: List[ScenarioMetrics]) -> Tuple[float, float, float]:
    avg_costs = [m.avg_cost for m in metrics]
    max_costs = [m.max_cost for m in metrics]
    runtimes = [m.avg_runtime for m in metrics]
    return mean(avg_costs), max(max_costs) if max_costs else 0.0, mean(runtimes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare planner measurements.")
    parser.add_argument("--planner_a", default=PLANNER_A_NAME, help="Planner A name for measurement_{name}.csv")
    parser.add_argument("--planner_b", default=PLANNER_B_NAME, help="Planner B name for measurement_{name}.csv")
    parser.add_argument("--csv_a", default=None, help="Explicit CSV path for planner A")
    parser.add_argument("--csv_b", default=None, help="Explicit CSV path for planner B")
    args = parser.parse_args()

    repo_dir = os.getcwd()
    measurements_dir = os.path.join(repo_dir, "data", "measurements")

    csv_a = args.csv_a or os.path.join(measurements_dir, f"measurement_{args.planner_a}.csv")
    csv_b = args.csv_b or os.path.join(measurements_dir, f"measurement_{args.planner_b}.csv")

    if not os.path.exists(csv_a):
        raise FileNotFoundError(f"CSV A not found: {csv_a}")
    if not os.path.exists(csv_b):
        raise FileNotFoundError(f"CSV B not found: {csv_b}")

    data_a = load_measurements(csv_a)
    data_b = load_measurements(csv_b)

    scenarios_a = set(data_a.keys())
    scenarios_b = set(data_b.keys())
    common = sorted(scenarios_a & scenarios_b)

    print(f"{args.planner_a}_planner -> {csv_a}")
    print(f"{args.planner_b}_planner -> {csv_b}")
    print(f"{args.planner_a}_planner total scenarios: {len(scenarios_a)}")
    print(f"{args.planner_b}_planner total scenarios: {len(scenarios_b)}")
    print(f"Common scenarios: {len(common)}")

    if not common:
        return

    metrics_a = [data_a[s] for s in common]
    metrics_b = [data_b[s] for s in common]

    a_avg_cost, a_max_cost, a_avg_runtime = summarize(metrics_a)
    b_avg_cost, b_max_cost, b_avg_runtime = summarize(metrics_b)
    a_max_runtime = max((m.avg_runtime for m in metrics_a), default=0.0)
    b_max_runtime = max((m.avg_runtime for m in metrics_b), default=0.0)

    a_better_cost = sum(1 for s in common if data_a[s].avg_cost < data_b[s].avg_cost)

    if b_avg_cost == 0.0:
        avg_cost_diff_pct = 0.0
    else:
        avg_cost_diff_pct = (a_avg_cost - b_avg_cost) / b_avg_cost * 100.0

    print(
        f"{args.planner_a}_planner cost < {args.planner_b}_planner cost count: "
        f"{a_better_cost} / {len(common)}"
    )
    print(f"{args.planner_a}_planner avg cost: {a_avg_cost:.6f}, max cost: {a_max_cost:.6f}")
    print(f"{args.planner_b}_planner avg cost: {b_avg_cost:.6f}, max cost: {b_max_cost:.6f}")
    print(
        f"{args.planner_a}_planner avg cost vs {args.planner_b}_planner avg cost: "
        f"{avg_cost_diff_pct:.2f}%"
    )
    print(
        f"{args.planner_a}_planner avg runtime: {a_avg_runtime:.6f}, "
        f"max runtime: {a_max_runtime:.6f}"
    )
    print(
        f"{args.planner_b}_planner avg runtime: {b_avg_runtime:.6f}, "
        f"max runtime: {b_max_runtime:.6f}"
    )


if __name__ == "__main__":
    main()
