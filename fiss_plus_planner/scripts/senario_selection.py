import argparse
import os
import json
import yaml

from fiss_plus_planner.planners.benchmark.planning import planning

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
    measurements = []
    name_planner = cfg['PLANNER']

    Folder = os.path.join(input_dir, "FOP_Scenarios")
    if cfg['FILES']:
        # Only run the specified scenario files under the input directory
        for i, file in enumerate(cfg['FILES']):
            measurement = planning(cfg, output_dir, input_dir, file)
            measurements.append((file, measurement))
    else:
        # Read all scenario files under the input directory
        for i, file in enumerate(os.listdir(input_dir)):
            print("Processing scenario file:", file)
            measurement = planning(cfg, output_dir, input_dir, file)
            measurements.append((file, measurement))
            if measurement is not None and measurement.success:
                # copy the file into a separate folder success_dir
                os.makedirs(Folder, exist_ok=True)
                src_file = os.path.join(input_dir, file)
                dst_file = os.path.join(Folder, file)
                print("Copying successful scenario file:", src_file, "to", dst_file)
                os.system(f"cp {src_file} {dst_file}")
            
    # for measurment in measurements:
    #     if measurement is not None and measurement.success:
    #             # copy the file into a separate folder success_dir
    #             os.makedirs(Folder, exist_ok=True)
    #             src_file = os.path.join(input_dir, file)
    #             dst_file = os.path.join(Folder, file)
    #             print("Copying successful scenario file:", src_file, "to", dst_file)
    #             os.system(f"cp {src_file} {dst_file}")