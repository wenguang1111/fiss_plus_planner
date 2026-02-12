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
    if cfg['FILES']:
        # Only run the specified scenario files under the input directory
        for i, file in enumerate(cfg['FILES']):
            measurement = planning(cfg, output_dir, input_dir, file)
            measurements.append((file, measurement))
    else:
        # Read all scenario files under the input directory
        for i, file in enumerate(os.listdir(input_dir)):
            measurement = planning(cfg, output_dir, input_dir, file)
            measurements.append((file, measurement))
    
    if save_measurments:
        t_cvae_timesteps, t_dense_timesteps, t_total_timesteps = 0, 0, 0
        os.makedirs(measurement_dir, exist_ok=True)
        csv_path = os.path.join(measurement_dir, 'measurement_' + name_planner + '.csv')
        with open(csv_path, 'w', newline='') as csv_file:
            csv_file.write(
                'scenario,steps,average runtime_plan [s],runtime history [s],num_trajs_generated,num_trajs_validated,'
                'num_collision_checks,average_cost,max_cost, step_number_for_break, cvae_samples, dense_samples, total_samples, success\n'
            )
            for file, measurement in measurements:
                if measurement is None:
                    continue
                max_cost = max(measurement.best_traj_costs) if measurement.best_traj_costs else 0.0
                runtime_history_str = json.dumps(measurement.runtime_history)
                csv_file.write(
                    f'{file},{measurement.step_number},{measurement.average_runtime},"{runtime_history_str}",'
                    f'{measurement.num_trajs_generated},{measurement.num_trajs_validated},'
                    f'{measurement.num_collison_checks},{measurement.average_cost},{max_cost}, {measurement.time_step_have_to_break},' 
                    f'{measurement.cvae_timesteps}, {measurement.dense_timesteps}, {measurement.total_timesteps}, {measurement.success}\n'
                )
                t_cvae_timesteps += measurement.cvae_timesteps
                t_dense_timesteps += measurement.dense_timesteps
                t_total_timesteps += measurement.total_timesteps
                
        print(f"Total timesteps planned: {t_total_timesteps}")
        print(f"Timesteps planned with CVAE: {t_cvae_timesteps}")
        print(f"Timesteps planned with dense sampling: {t_dense_timesteps}")
        print(f"Percentage of CVAE sampling: {t_cvae_timesteps/t_total_timesteps*100:.2f}")
        print(f"Percentage of dense sampling: {t_dense_timesteps/t_total_timesteps*100:.2f}")
                