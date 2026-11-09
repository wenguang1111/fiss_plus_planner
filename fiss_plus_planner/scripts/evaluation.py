import argparse
import os
import json
import yaml
import csv

from fiss_plus_planner.planners.benchmark.planning import planning
from demo_cr import MEASUREMENT_CSV_HEADER, append_measurement_to_csv

if __name__ == '__main__':
    repo_dir = os.getcwd()
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--cfg_file', type=str, default=os.path.join(repo_dir, 'cfgs/demo_config.yaml'), help='specify the config file for the demo')
    args = parser.parse_args()
    
    with open(args.cfg_file, 'r') as file:
        cfg = yaml.safe_load(file)
        file.close()

    save_measurments = cfg['SAVE_MEASUREMENTS']
        
    output_dir = os.path.join(os.getcwd(), cfg['OUTPUT_DIR'])
    input_dir = os.path.join(os.getcwd(), cfg['INPUT_DIR'])
    measurement_dir = os.path.join(os.getcwd(), cfg['MEASUREMENTS_DIR'])
    name_planner = cfg['PLANNER']
    
    if save_measurments:
        os.makedirs(measurement_dir, exist_ok=True)
        csv_path = os.path.join(measurement_dir, 'measurement_' + name_planner + '.csv')
        
        existing_scenarios = set()
        if os.path.exists(csv_path):
            with open(csv_path, 'r', newline='', encoding='utf-8') as csv_file:
                reader = csv.DictReader(csv_file)
                expected_fields = next(csv.reader([MEASUREMENT_CSV_HEADER]))
                if reader.fieldnames is not None and reader.fieldnames != expected_fields:
                    raise ValueError(
                        f'{csv_path} has a different header than the current measurement format; '
                        'move it away or delete it before resuming.'
                    )
                if reader.fieldnames is not None:
                    for row in reader:
                        existing_scenarios.add(row['scenario'])
        else:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
                csv_file.write(MEASUREMENT_CSV_HEADER)
    
    if cfg['FILES']:
        # Only run the specified scenario files under the input directory
        for i, file in enumerate(cfg['FILES']):
            if save_measurments and file in existing_scenarios:
                print(f"Skipping {file} (already processed)")
                continue
            
            measurement = planning(cfg, output_dir, input_dir, file)
            
            if save_measurments and measurement is not None:
                append_measurement_to_csv(csv_path, file, measurement)
    else:
        for i, file in enumerate(os.listdir(input_dir)):
            if save_measurments and file in existing_scenarios:
                print(f"Skipping {file} (already processed)")
                continue
            
            measurement = planning(cfg, output_dir, input_dir, file)
            
            if save_measurments and measurement is not None:
                append_measurement_to_csv(csv_path, file, measurement)