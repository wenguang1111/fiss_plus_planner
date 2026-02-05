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
SCENARIO_SOURCE_FOLDER = '/home/wenguang/workspace/fiss_plus_planner/fiss_plus_planner/data/demo/FOP_Scenarios_transformed'
SCENARIO_OUTPUT_FOLDER = '/home/wenguang/workspace/fiss_plus_planner/fiss_plus_planner/data/demo/selected_scenarios'
NUM_RANDOM_SAMPLES_LOW_D = 1500  # Number of random samples from |d_mean| < 0.1

def calculate_stats(input_folder: str, output_file: str):
    """
    Calculate the mean and standard deviation of d, t, v for all CSV files in the input folder.
    
    Args:
        input_folder: Path to the input folder containing CSV files
        output_file: Path to the output CSV file
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

def select_scenarios(stats_file: str, source_folder: str, output_folder: str, num_samples_low_d: int):
    """
    Select scenarios based on |d_mean| distribution:
    - Randomly pick num_samples_low_d scenarios from |d_mean| < 0.1
    - Pick all scenarios with |d_mean| >= 0.1
    
    Args:
        stats_file: Path to the stats CSV file
        source_folder: Path to the folder containing original scenario files
        output_folder: Path to save selected scenario files
        num_samples_low_d: Number of random samples from |d_mean| < 0.1
    """
    # Read the stats CSV file
    df = pd.read_csv(stats_file)
    df['d_abs'] = df['d_mean'].abs()
    
    # Split into two groups
    low_d_scenarios = df[df['d_abs'] < 0.1]['scenario'].tolist()
    high_d_scenarios = df[df['d_abs'] >= 0.1]['scenario'].tolist()
    
    print(f"=== Scenario Selection ===")
    print(f"Total scenarios with |d_mean| < 0.1: {len(low_d_scenarios)}")
    print(f"Total scenarios with |d_mean| >= 0.1: {len(high_d_scenarios)}")
    
    # Randomly sample from low_d group
    if len(low_d_scenarios) > num_samples_low_d:
        random.seed(42)  # For reproducibility
        selected_low_d = random.sample(low_d_scenarios, num_samples_low_d)
    else:
        selected_low_d = low_d_scenarios
        print(f"Warning: Only {len(low_d_scenarios)} scenarios available, taking all")
    
    # Combine selected scenarios
    selected_scenarios = selected_low_d + high_d_scenarios
    
    print(f"Selected from |d_mean| < 0.1: {len(selected_low_d)}")
    print(f"Selected from |d_mean| >= 0.1: {len(high_d_scenarios)}")
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

if __name__ == '__main__':
    calculate_stats(INPUT_FOLDER, OUTPUT_FILE)
    # plot_d_mean_distribution(OUTPUT_FILE, OUTPUT_PLOT_DIR)
    # plot_v_mean_distribution(OUTPUT_FILE, OUTPUT_PLOT_DIR)
    # plot_t_mean_distribution(OUTPUT_FILE, OUTPUT_PLOT_DIR)
    
    # Select scenarios based on d_mean distribution
    select_scenarios(OUTPUT_FILE, SCENARIO_SOURCE_FOLDER, SCENARIO_OUTPUT_FOLDER, NUM_RANDOM_SAMPLES_LOW_D)
