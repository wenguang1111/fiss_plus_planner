import argparse
import os
import csv
import pandas as pd

if __name__ == '__main__':
    repo_dir = os.getcwd()
    parser = argparse.ArgumentParser(description='Compare two measurement CSV files (FOP vs FISS+)')
    
    # New parameters for two CSV files
    parser.add_argument('--csv_file_1', type=str, 
                       default=os.path.join(repo_dir, 'data/measurements/summarize/measurement_FOP_10.csv'),
                       help='Path to first measurement CSV file (e.g., FOP)')
    parser.add_argument('--csv_file_2', type=str,
                       default=os.path.join(repo_dir, 'data/measurements/summarize/measurement_FISS+_5.csv'),
                       help='Path to second measurement CSV file (e.g., FISS+)')
    parser.add_argument('--output_file', type=str,
                       default=os.path.join(repo_dir, 'data/measurements/summarize/comparison_result.csv'),
                       help='Output CSV file with comparison results')
    
    args = parser.parse_args()
    
    file_FOP = args.csv_file_1
    file_FISS = args.csv_file_2
    output_file = args.output_file
    
    # Determine planner names from file paths
    planner_FOP = os.path.basename(file_FOP).replace('measurement_', '').replace('.csv', '')
    planner_FISS = os.path.basename(file_FISS).replace('measurement_', '').replace('.csv', '')
    
    print(f"Comparing {planner_FOP} vs {planner_FISS}")
    print(f"File 1: {file_FOP}")
    print(f"File 2: {file_FISS}")
    
    # Check if files exist
    if not os.path.exists(file_FOP):
        print(f"Error: File not found: {file_FOP}")
        exit(1)
    if not os.path.exists(file_FISS):
        print(f"Error: File not found: {file_FISS}")
        exit(1)
    
    # Load both CSV files
    try:
        df_FOP = pd.read_csv(file_FOP)
        df_fiss = pd.read_csv(file_FISS)
        print(f"\nLoaded {planner_FOP}: {len(df_FOP)} scenarios")
        print(f"Loaded {planner_FISS}: {len(df_fiss)} scenarios")
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        exit(1)
    
    # Find common scenarios
    scenarios_FOP = set(df_FOP['scenario'])
    scenarios_FISS = set(df_fiss['scenario'])
    common_scenarios = scenarios_FOP.intersection(scenarios_FISS)
    
    print(f"\nFound {len(common_scenarios)} common scenarios")
    
    # Prepare output data
    output_data = {
        'Scenario': [],
        f'{planner_FOP}_Runtime': [],
        f'{planner_FISS}_Runtime': [],
        'Runtime_Difference': [],
        f'{planner_FOP}_Cost': [],
        f'{planner_FISS}_Cost': [],
        'Cost_Difference': [],
        f'{planner_FOP}_Better_in_Both': []
    }
    
    runtime_better_scenarios = []
    cost_better_scenarios = []
    better_in_both_scenarios = []
    
    # Compare scenarios
    for scenario in sorted(common_scenarios):
        row_FOP = df_FOP[df_FOP['scenario'] == scenario].iloc[0]
        row_FISS = df_fiss[df_fiss['scenario'] == scenario].iloc[0]
        
        runtime_FOP = row_FOP['average runtime_plan [s]']
        runtime_FISS = row_FISS['average runtime_plan [s]']
        cost_FOP = row_FOP['average_cost']
        cost_FISS = row_FISS['average_cost']
        
        # Check if planner_1 has better runtime
        runtime_better = runtime_FOP < runtime_FISS
        # Check if planner_1 has lower cost
        cost_better = cost_FOP < cost_FISS
        # Check if better in both metrics
        better_in_both = runtime_better and cost_better
        
        # Store scenario names if conditions are met
        if runtime_better:
            runtime_better_scenarios.append(scenario)
        if cost_better:
            cost_better_scenarios.append(scenario)
        if better_in_both:
            better_in_both_scenarios.append(scenario)
        
        # Add to detailed output
        output_data['Scenario'].append(scenario)
        output_data[f'{planner_FOP}_Runtime'].append(runtime_FOP)
        output_data[f'{planner_FISS}_Runtime'].append(runtime_FISS)
        output_data['Runtime_Difference'].append(runtime_FISS - runtime_FOP)
        output_data[f'{planner_FOP}_Cost'].append(cost_FOP)
        output_data[f'{planner_FISS}_Cost'].append(cost_FISS)
        output_data['Cost_Difference'].append(cost_FISS - cost_FOP)
        output_data[f'{planner_FOP}_Better_in_Both'].append('Yes' if better_in_both else 'No')
    
    # Create a simplified output CSV with just the two comparison columns
    simple_output_data = {
        f'Runtime_{planner_FOP}_Better': runtime_better_scenarios,
        f'Lower_Cost_{planner_FOP}': cost_better_scenarios
    }
    
    # Pad the lists to have same length
    max_len = max(len(runtime_better_scenarios), len(cost_better_scenarios))
    simple_output_data[f'Runtime_{planner_FOP}_Better'].extend([''] * (max_len - len(runtime_better_scenarios)))
    simple_output_data[f'Lower_Cost_{planner_FOP}'].extend([''] * (max_len - len(cost_better_scenarios)))
    
    # Create simple comparison dataframe
    simple_df = pd.DataFrame(simple_output_data)
    
    # Create detailed comparison dataframe
    detailed_df = pd.DataFrame(output_data)
    
    # Save output files
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save simple comparison
    simple_output_path = output_file.replace('.csv', '_simple.csv')
    simple_df.to_csv(simple_output_path, index=False)
    print(f"\nSimple comparison saved to: {simple_output_path}")
    
    # Save detailed comparison
    detailed_df.to_csv(output_file, index=False)
    print(f"Detailed comparison saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\nTotal common scenarios: {len(common_scenarios)}")
    print(f"\n{planner_FOP} has better runtime in: {len(runtime_better_scenarios)} scenarios ({len(runtime_better_scenarios)/len(common_scenarios)*100:.1f}%)")
    print(f"{planner_FOP} has lower cost in: {len(cost_better_scenarios)} scenarios ({len(cost_better_scenarios)/len(common_scenarios)*100:.1f}%)")
    print(f"{planner_FOP} is better in BOTH metrics: {len(better_in_both_scenarios)} scenarios ({len(better_in_both_scenarios)/len(common_scenarios)*100:.1f}%)")
    
    # Print statistics
    print(f"\nRuntime Statistics (in seconds):")
    print(f"  {planner_FOP} - Mean: {df_FOP['average runtime_plan [s]'].mean():.4f}, Std: {df_FOP['average runtime_plan [s]'].std():.4f}")
    print(f"  {planner_FISS} - Mean: {df_fiss['average runtime_plan [s]'].mean():.4f}, Std: {df_fiss['average runtime_plan [s]'].std():.4f}")
    
    print(f"\nCost Statistics:")
    print(f"  {planner_FOP} - Mean: {df_FOP['average_cost'].mean():.4f}, Std: {df_FOP['average_cost'].std():.4f}")
    print(f"  {planner_FISS} - Mean: {df_fiss['average_cost'].mean():.4f}, Std: {df_fiss['average_cost'].std():.4f}")
    
    if len(runtime_better_scenarios) > 0:
        print(f"\nScenarios where {planner_FOP} has better runtime:")
        for scenario in runtime_better_scenarios[:10]:  # Show first 10
            print(f"  - {scenario}")
        if len(runtime_better_scenarios) > 10:
            print(f"  ... and {len(runtime_better_scenarios) - 10} more")
    
    if len(cost_better_scenarios) > 0:
        print(f"\nScenarios where {planner_FOP} has lower cost:")
        for scenario in cost_better_scenarios[:10]:  # Show first 10
            print(f"  - {scenario}")
        if len(cost_better_scenarios) > 10:
            print(f"  ... and {len(cost_better_scenarios) - 10} more")
