# This file is used to categorize the sampling parameters for the planner.
# Its used to calculate the average d,v,t of each scenario which exsist in one specific folder.
# With the average d,v,t and standard deviation of d,v,t, we can categorize the scenarios

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil
import random

# Define input and output paths here
INPUT_FOLDER = '/home/wenguang/workspace/fiss_plus_planner/fiss_plus_planner/data/output'
OUTPUT_FILE = '/home/wenguang/workspace/fiss_plus_planner/fiss_plus_planner/data/measurements/dtv_stats.csv'
OUTPUT_PLOT_DIR = '/home/wenguang/workspace/fiss_plus_planner/fiss_plus_planner/data/measurements'

# Scenario selection settings
SCENARIO_SOURCE_FOLDER = '/home/wenguang/workspace/fiss_plus_planner/fiss_plus_planner/data/FOP_Scenarios_transformed_AllGoThrough'
SCENARIO_OUTPUT_FOLDER = '/home/wenguang/workspace/fiss_plus_planner/fiss_plus_planner/data/demo/selected_scenarios'
NUM_RANDOM_SAMPLES_LOW_D = 400  # Number of random samples from |d_mean| < 0.1

def calculate_scenario_durations(input_folder: str, plot_dir: str, min_duration: float = 5.0, max_duration: float = 15.0):
    """
    Calculate duration for each scenario CSV as row_count / 10 (seconds).

    Args:
        input_folder: Path to the input folder containing scenario CSV files
        plot_dir: Directory to save duration distribution plot
        min_duration: Lower bound of duration filter (seconds, inclusive)
        max_duration: Upper bound of duration filter (seconds, inclusive)
    """
    durations = []

    for filename in os.listdir(input_folder):
        if not filename.endswith('.csv'):
            continue

        filepath = os.path.join(input_folder, filename)
        scenario_name = filename.replace('_FOP_CPP_traj_log.csv', '').replace('.csv', '')

        try:
            df = pd.read_csv(filepath)
            if df.empty:
                print(f"Warning: {filename} is empty, skipping duration")
                continue

            duration = len(df) / 10.0
            durations.append({'scenario': scenario_name, 'duration': duration})
        except Exception as e:
            print(f"Error calculating duration for {filename}: {e}")
            continue

    if not durations:
        print("No valid scenario durations calculated")
        return

    duration_df = pd.DataFrame(durations).sort_values(by='duration', ascending=False).reset_index(drop=True)
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(duration_df['duration'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    plt.xlabel('Duration (s)')
    plt.ylabel('Frequency')
    plt.title('Distribution of scenario duration')
    plt.grid(True, alpha=0.3)

    mean_val = duration_df['duration'].mean()
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}s')
    plt.legend()

    plot_path = os.path.join(plot_dir, 'scenario_duration_distribution.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Duration distribution plot saved to: {plot_path}")
    selected_duration_df = duration_df[
        (duration_df['duration'] >= min_duration) & (duration_df['duration'] <= max_duration)
    ]
    print(f"Scenarios with duration in [{min_duration}, {max_duration}] s: {len(selected_duration_df)}")
    print(
        f"Duration summary -> count: {len(duration_df)}, "
        f"min: {duration_df['duration'].min():.4f}, "
        f"max: {duration_df['duration'].max():.4f}, "
        f"mean: {duration_df['duration'].mean():.4f}"
    )
    return set(selected_duration_df['scenario'].tolist())

def calculate_stats(input_folder: str, output_file: str, selected_scenarios: set = None):
    """
    Calculate the mean and standard deviation of d, t, v for all CSV files in the input folder.
    
    Args:
        input_folder: Path to the input folder containing CSV files
        output_file: Path to the output CSV file
        selected_scenarios: Optional set of scenario names to include
    """
    results = []
    
    # Iterate through all CSV files in the folder
    for filename in os.listdir(input_folder):
        if not filename.endswith('.csv'):
            continue
        
        filepath = os.path.join(input_folder, filename)
        
        try:
            # Read CSV file
            df = pd.read_csv(filepath)
            
            # Check if all required columns exist
            required_cols = ['d', 't', 'v']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: {filename} is missing required columns, skipping")
                continue
            
            # Extract scenario name (remove suffix)
            scenario_name = filename.replace('_FOP_CPP_traj_log.csv', '').replace('.csv', '')
            if selected_scenarios is not None and scenario_name not in selected_scenarios:
                continue
            
            # Calculate mean and standard deviation
            d_mean = df['d'].mean()
            d_std = df['d'].std()
            t_mean = df['t'].mean()
            t_std = df['t'].std()
            v_mean = df['v'].mean()
            v_std = df['v'].std()
            
            results.append({
                'scenario': scenario_name,
                'd_mean': d_mean,
                'd_std': d_std,
                't_mean': t_mean,
                't_std': t_std,
                'v_mean': v_mean,
                'v_std': v_std
            })
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    # Create result DataFrame and save to file
    if results:
        result_df = pd.DataFrame(results)
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        result_df.to_csv(output_file, index=False)
        print(f"Successfully processed {len(results)} files, results saved to: {output_file}")
    else:
        print("No valid CSV files found")

def plot_d_mean_distribution(output_file: str, plot_dir: str):
    """
    Generate a histogram of d_mean distribution from the stats CSV file.
    
    Args:
        output_file: Path to the stats CSV file
        plot_dir: Directory to save the plot
    """
    # Read the stats CSV file
    df = pd.read_csv(output_file)
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['d_mean'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('d_mean')
    plt.ylabel('Frequency')
    plt.title('Distribution of d_mean')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_val = df['d_mean'].mean()
    std_val = df['d_mean'].std()
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(plot_dir, 'd_mean_distribution.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved to: {plot_path}")

def plot_v_mean_distribution(output_file: str, plot_dir: str):
    """
    Generate a histogram of v_mean distribution from the stats CSV file.
    
    Args:
        output_file: Path to the stats CSV file
        plot_dir: Directory to save the plot
    """
    # Read the stats CSV file
    df = pd.read_csv(output_file)
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['v_mean'], bins=50, edgecolor='black', alpha=0.7, color='green')
    plt.xlabel('v_mean')
    plt.ylabel('Frequency')
    plt.title('Distribution of v_mean')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_val = df['v_mean'].mean()
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(plot_dir, 'v_mean_distribution.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved to: {plot_path}")

def plot_t_mean_distribution(output_file: str, plot_dir: str):
    """
    Generate a histogram of t_mean distribution from the stats CSV file.
    
    Args:
        output_file: Path to the stats CSV file
        plot_dir: Directory to save the plot
    """
    # Read the stats CSV file
    df = pd.read_csv(output_file)
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['t_mean'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('t_mean')
    plt.ylabel('Frequency')
    plt.title('Distribution of t_mean')
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    mean_val = df['t_mean'].mean()
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(plot_dir, 't_mean_distribution.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved to: {plot_path}")

def select_scenarios(
    stats_file: str,
    source_folder: str,
    output_folder: str,
    input_folder: str,
    min_duration: float = 5.0,
    max_duration: float = 15.0,
    low_d_multiplier: float = 15
):
    """
    Select scenarios with this logic:
    1) Keep only scenarios with duration in [min_duration, max_duration] seconds,
       where duration = row_count / 10.
    2) Select all scenarios with 0.1 <= |d_mean| < 0.4 (count = N).
    3) From scenarios with |d_mean| < 0.1, randomly sample ceil(low_d_multiplier * N).

    Args:
        stats_file: Path to the stats CSV file
        source_folder: Path to the folder containing original scenario files
        output_folder: Path to save selected scenario files
        input_folder: Path to the folder containing scenario CSV logs
        min_duration: Lower bound of scenario duration (seconds, inclusive)
        max_duration: Upper bound of scenario duration (seconds, inclusive)
        low_d_multiplier: Sampling multiplier for |d_mean| < 0.1 group
    """
    # Read the stats CSV file
    df = pd.read_csv(stats_file)
    df['d_abs'] = df['d_mean'].abs()

    # Filter by duration first: duration = row_count / 10
    duration_filtered_scenarios = set()
    for filename in os.listdir(input_folder):
        if not filename.endswith('.csv'):
            continue

        filepath = os.path.join(input_folder, filename)
        scenario_name = filename.replace('_FOP_CPP_traj_log.csv', '').replace('.csv', '')
        try:
            scenario_df = pd.read_csv(filepath)
            if scenario_df.empty:
                continue
            duration = len(scenario_df) / 10.0
            if min_duration <= duration <= max_duration:
                duration_filtered_scenarios.add(scenario_name)
        except Exception:
            continue

    df = df[df['scenario'].isin(duration_filtered_scenarios)].copy()

    # Group A: 0.1 <= |d_mean| < 0.4 (all selected)
    mid_d_df = df[(df['d_abs'] >= 0.1) & (df['d_abs'] < 0.4)].copy()
    selected_mid_d = mid_d_df['scenario'].tolist()
    n_mid = len(selected_mid_d)

    # Group B: |d_mean| < 0.1 (random sample of ceil(low_d_multiplier * N))
    low_d_df = df[df['d_abs'] < 0.1].copy()
    target_low_count = int(np.ceil(low_d_multiplier * n_mid))

    if target_low_count > 0 and len(low_d_df) > 0:
        if len(low_d_df) >= target_low_count:
            selected_low_d = random.sample(low_d_df['scenario'].tolist(), target_low_count)
        else:
            selected_low_d = low_d_df['scenario'].tolist()
            print(
                f"Warning: Only {len(low_d_df)} scenarios with |d_mean| < 0.1, "
                f"less than requested {target_low_count}. Taking all."
            )
    else:
        selected_low_d = []

    selected_scenarios = selected_mid_d + selected_low_d

    print(f"=== Scenario Selection ===")
    print(f"Duration filter [{min_duration}, {max_duration}] s: {len(df)} scenarios")
    print(f"Selected all with 0.1 <= |d_mean| < 0.4: {len(selected_mid_d)}")
    print(f"Selected random from |d_mean| < 0.1: {len(selected_low_d)} (target={target_low_count})")
    print(f"Total selected scenarios: {len(selected_scenarios)}")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Copy selected scenario files
    copied_count = 0
    not_found_count = 0
    
    for scenario_name in selected_scenarios:
        # Try different possible filename patterns
        possible_files = [
            f"{scenario_name}.xml",
            f"{scenario_name}.json",
            scenario_name
        ]
        
        found = False
        for filename in possible_files:
            src_path = os.path.join(source_folder, filename)
            if os.path.exists(src_path):
                dst_path = os.path.join(output_folder, filename)
                shutil.copy2(src_path, dst_path)
                copied_count += 1
                found = True
                break
        
        if not found:
            not_found_count += 1
            if not_found_count <= 5:  # Only print first 5 warnings
                print(f"Warning: Scenario file not found for {scenario_name}")
    
    if not_found_count > 5:
        print(f"... and {not_found_count - 5} more scenarios not found")
    
    print(f"\nSuccessfully copied {copied_count} scenario files to: {output_folder}")
    print(f"Scenarios not found: {not_found_count}")
    return selected_scenarios

def plot_selected_scenarios_distributions(
    stats_file: str,
    input_folder: str,
    selected_scenarios: list,
    plot_dir: str
):
    """
    Plot distributions for the final selected scenarios:
    1) d_mean distribution
    2) duration distribution, where duration = row_count / 10
    """
    if not selected_scenarios:
        print("No selected scenarios found, skip selected-scenarios plots")
        return

    os.makedirs(plot_dir, exist_ok=True)
    selected_set = set(selected_scenarios)

    # d_mean distribution for selected scenarios
    stats_df = pd.read_csv(stats_file)
    selected_stats_df = stats_df[stats_df['scenario'].isin(selected_set)].copy()
    if len(selected_stats_df) > 0:
        plt.figure(figsize=(10, 6))
        plt.hist(selected_stats_df['d_mean'], bins=50, edgecolor='black', alpha=0.7, color='teal')
        plt.xlabel('d_mean')
        plt.ylabel('Frequency')
        plt.title('Distribution of d_mean (selected scenarios)')
        plt.grid(True, alpha=0.3)
        mean_val = selected_stats_df['d_mean'].mean()
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        plt.legend()

        d_plot_path = os.path.join(plot_dir, 'selected_scenarios_d_mean_distribution.png')
        plt.savefig(d_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Selected scenarios d_mean distribution plot saved to: {d_plot_path}")
    else:
        print("No selected scenarios found in stats file, skip selected d_mean plot")

    # duration distribution for selected scenarios
    selected_durations = []
    for filename in os.listdir(input_folder):
        if not filename.endswith('.csv'):
            continue
        scenario_name = filename.replace('_FOP_CPP_traj_log.csv', '').replace('.csv', '')
        if scenario_name not in selected_set:
            continue
        filepath = os.path.join(input_folder, filename)
        try:
            scenario_df = pd.read_csv(filepath)
            if scenario_df.empty:
                continue
            selected_durations.append(len(scenario_df) / 10.0)
        except Exception:
            continue

    if selected_durations:
        duration_series = pd.Series(selected_durations)
        plt.figure(figsize=(10, 6))
        plt.hist(duration_series, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        plt.xlabel('Duration (s)')
        plt.ylabel('Frequency')
        plt.title('Distribution of duration (selected scenarios)')
        plt.grid(True, alpha=0.3)
        mean_val = duration_series.mean()
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}s')
        plt.legend()

        duration_plot_path = os.path.join(plot_dir, 'selected_scenarios_duration_distribution.png')
        plt.savefig(duration_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Selected scenarios duration distribution plot saved to: {duration_plot_path}")
    else:
        print("No valid duration data for selected scenarios, skip selected duration plot")

def print_d_distribution_by_bins(output_file: str):
    """
    Print the number of scenarios in different |d_mean| bins:
    [0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.8], [0.8, +inf]
    
    Args:
        output_file: Path to the stats CSV file
    """
    # Read the stats CSV file
    df = pd.read_csv(output_file)
    df['d_abs'] = df['d_mean'].abs()
    
    # Define bins
    bins = [
        (0.0, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.8),
        (0.8, float('inf'))
    ]
    
    print("=" * 50)
    print("Distribution of scenarios by |d_mean| bins")
    print("=" * 50)
    
    total_scenarios = len(df)
    print(f"Total scenarios: {total_scenarios}\n")
    
    for low, high in bins:
        if high == float('inf'):
            count = len(df[(df['d_abs'] >= low)])
            bin_label = f"[{low}, +inf)"
        else:
            count = len(df[(df['d_abs'] >= low) & (df['d_abs'] < high)])
            bin_label = f"[{low}, {high})"
        
        percentage = (count / total_scenarios * 100) if total_scenarios > 0 else 0
        print(f"|d_mean| in {bin_label}: {count} scenarios ({percentage:.2f}%)")
    
    print("=" * 50)
    
    # Print scenarios with 0.3 <= |d_mean| < 0.5
    mid_d_scenarios = df[(df['d_abs'] >= 0.3) & (df['d_abs'] < 0.5)]['scenario'].tolist()
    if mid_d_scenarios:
        print(f"\nScenarios with 0.3 <= |d_mean| < 0.5:")
        for scenario in mid_d_scenarios:
            d_val = df[df['scenario'] == scenario]['d_mean'].values[0]
            print(f"  - {scenario} (d_mean = {d_val:.4f})")

if __name__ == '__main__':
    scenarios_in_duration_range = calculate_scenario_durations(INPUT_FOLDER, OUTPUT_PLOT_DIR, 5.0, 15.0)
    calculate_stats(INPUT_FOLDER, OUTPUT_FILE, scenarios_in_duration_range)
    plot_d_mean_distribution(OUTPUT_FILE, OUTPUT_PLOT_DIR)
    plot_v_mean_distribution(OUTPUT_FILE, OUTPUT_PLOT_DIR)
    plot_t_mean_distribution(OUTPUT_FILE, OUTPUT_PLOT_DIR)
    
    # Print distribution by d_mean bins
    # print_d_distribution_by_bins(OUTPUT_FILE)
    
    # Select scenarios based on duration and d_mean rules
    selected_scenarios = select_scenarios(
        OUTPUT_FILE, SCENARIO_SOURCE_FOLDER, SCENARIO_OUTPUT_FOLDER, INPUT_FOLDER
    )
    plot_selected_scenarios_distributions(
        OUTPUT_FILE, INPUT_FOLDER, selected_scenarios, OUTPUT_PLOT_DIR
    )
