# This file is used to categorize the sampling parameters for the planner.
# Its used to calculate the average d,v,t of each scenario which exsist in one specific folder.
# With the average d,v,t and standard deviation of d,v,t, we can categorize the scenarios

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define input and output paths here
INPUT_FOLDER = '/home/wenguang/workspace/fiss_plus_planner/fiss_plus_planner/data/output'
OUTPUT_FILE = '/home/wenguang/workspace/fiss_plus_planner/fiss_plus_planner/data/measurements/dtv_stats.csv'
OUTPUT_PLOT_DIR = '/home/wenguang/workspace/fiss_plus_planner/fiss_plus_planner/data/measurements'

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

if __name__ == '__main__':
    calculate_stats(INPUT_FOLDER, OUTPUT_FILE)
    plot_d_mean_distribution(OUTPUT_FILE, OUTPUT_PLOT_DIR)
    plot_v_mean_distribution(OUTPUT_FILE, OUTPUT_PLOT_DIR)
    plot_t_mean_distribution(OUTPUT_FILE, OUTPUT_PLOT_DIR)
