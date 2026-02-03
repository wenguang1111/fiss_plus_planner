"""
Script to find scenarios that exist in file A but not in file B.
The output is a CSV file named Scenarios_{Method}_not_Run.csv containing the scenario names.
"""

import pandas as pd
from pathlib import Path

# ============== Configuration ==============
# Path to the first CSV file (reference file with all scenarios)
FILE_A = "/home/wenguang/workplace/fiss_plus_planner/fiss_plus_planner/data/measurements/1_Sample/measurement_FISS+.csv"

# Path to the second CSV file (file to compare against)
FILE_B = "/home/wenguang/workplace/fiss_plus_planner/fiss_plus_planner/data/measurements/1_Sample/measurement_Sparse_64.csv"

# Method name for the output file
METHOD_NAME = "Sparse_Optimized"

# Output directory (set to None to use the same directory as FILE_A)
OUTPUT_DIR = None
# ===========================================


def main():
    # Read both CSV files
    df_a = pd.read_csv(FILE_A)
    df_b = pd.read_csv(FILE_B)

    # Get scenario names from both files
    scenarios_a = set(df_a["scenario"].tolist())
    scenarios_b = set(df_b["scenario"].tolist())

    # Find scenarios in A but not in B
    scenarios_not_in_b = scenarios_a - scenarios_b

    # Determine output directory
    if OUTPUT_DIR:
        output_dir = Path(OUTPUT_DIR)
    else:
        output_dir = Path(FILE_A).parent

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create output file
    output_file = output_dir / f"Scenarios_{METHOD_NAME}_not_Run.csv"

    # Write result to CSV
    result_df = pd.DataFrame({"scenario": sorted(list(scenarios_not_in_b))})
    result_df.to_csv(output_file, index=False)

    print(f"Found {len(scenarios_not_in_b)} scenarios in file A but not in file B")
    print(f"Results written to: {output_file}")

    if scenarios_not_in_b:
        print("\nScenarios not run:")
        for scenario in sorted(scenarios_not_in_b):
            print(f"  - {scenario}")


if __name__ == "__main__":
    main()
