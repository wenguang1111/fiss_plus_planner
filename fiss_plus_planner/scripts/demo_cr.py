import argparse
import os
import json
import yaml

from fiss_plus_planner.planners.benchmark.planning import planning


def readExsistedScenarios(existing_entries):
    """
    Convert existing output entries to scenario file names (e.g. DEU_xxx.xml).
    """
    existing_scenarios = set()
    for entry in existing_entries:
        scenario_name = os.path.splitext(entry)[0]
        if scenario_name:
            existing_scenarios.add(f"{scenario_name}.xml")
    return existing_scenarios


def append_measurement_to_csv(csv_path, file, measurement):
    if not hasattr(measurement, 'best_traj_costs'):
        return 
    max_cost = max(measurement.best_traj_costs) if measurement.best_traj_costs else 0.0
    runtime_history_str = json.dumps(measurement.runtime_history)
    intervention_percent = (
        measurement.num_FOP_intervention / measurement.step_number * 100
        if measurement.step_number else 0.0
    )
    with open(csv_path, 'a', newline='') as csv_file:
        csv_file.write(
            f'{file},{measurement.step_number},{measurement.average_runtime},"{runtime_history_str}",'
            f'{measurement.num_trajs_generated},{measurement.num_trajs_validated},'
            f'{measurement.num_collison_checks},'
            f'{measurement.average_cost},{max_cost},{measurement.final_traj_cost}, {measurement.time_step_have_to_break},{measurement.num_FOP_intervention},{intervention_percent},{measurement.success},'
            f'{measurement.num_rejected_dynamic},{measurement.num_rejected_offroad},{measurement.num_rejected_collision},'
            f'{measurement.last_cycle_num_rejected_dynamic},{measurement.last_cycle_num_rejected_offroad},{measurement.last_cycle_num_rejected_collision}\n'
        )


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
    data_collection = cfg['Collect_Data_For_ML']
    reading_dir = '/home/wenguang/workplace/fiss_plus_planner/fiss_plus_planner/data/output/imgs'
    if data_collection and os.path.isdir(reading_dir):
        exsited_files = readExsistedScenarios(os.listdir(reading_dir))
    else:
        exsited_files = set()

    if save_measurments:
        os.makedirs(measurement_dir, exist_ok=True)
        csv_path = os.path.join(measurement_dir, 'measurement_' + name_planner + '.csv')
        with open(csv_path, 'w', newline='') as csv_file:
            csv_file.write(
                'scenario,steps,average runtime_plan [s],runtime history [s],num_trajs_generated,num_trajs_validated,'
                'num_collision_checks,'
                'average_cost,max_cost,final_trajector_cost, step_number_for_break, num_FOP_intervence_for_SP, Percent_FOP_Intervence, success,'
                'rejected_dynamic_per_cycle,rejected_offroad_per_cycle,rejected_collision_per_cycle,'
                'last_cycle_rejected_dynamic,last_cycle_rejected_offroad,last_cycle_rejected_collision\n'
            )

    if cfg['FILES']:
        # Only run the specified scenario files under the input directory
        for i, file in enumerate(cfg['FILES']):
            measurement = planning(cfg, output_dir, input_dir, file)
            if save_measurments:
                append_measurement_to_csv(csv_path, file, measurement)
    else:
        # Read all scenario files under the input directory
        for i, file in enumerate(os.listdir(input_dir)):
            if file in exsited_files:
                print(f"Skipping {file} as it already exists in the reading_dir.")
                continue
            else:
                print(f"Processing {file}...")
                measurement = planning(cfg, output_dir, input_dir, file)
                if save_measurments:
                    append_measurement_to_csv(csv_path, file, measurement)
