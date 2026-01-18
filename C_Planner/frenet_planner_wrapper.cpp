// #include "frenet_planner_wrapper.h"
// #include <iostream>
// #include <new>

// // Version string
// static const char* VERSION = "1.0.0";

// // Global planner instance storage
// static std::vector<Frenet_Planner*> planner_instances;

// // Create planner instance
// PlannerHandle create_planner(const CSettingParameters* settings, 
//                             const CVehicleParams* vehicle,
//                             const double* obstacles_array,
//                             const int* num_vertices_array,
//                             int num_time_steps,
//                             int num_obstacles,
//                             int max_vertices) {
//     try {
//         SettingParameters cpp_settings(settings->num_width, settings->num_speed, settings->num_t);
//         cpp_settings.tick_t = settings->tick_t;
//         cpp_settings.max_road_width = settings->max_road_width;
//         cpp_settings.highest_speed = settings->highest_speed;
//         cpp_settings.lowest_speed = settings->lowest_speed;
//         cpp_settings.min_t = settings->min_t;
//         cpp_settings.max_t = settings->max_t;
//         cpp_settings.check_obstacle = settings->check_obstacle != 0;
//         cpp_settings.check_boundary = settings->check_boundary != 0;
        
//         VehicleParams cpp_vehicle;
//         cpp_vehicle.l = vehicle->l;
//         cpp_vehicle.w = vehicle->w;
//         cpp_vehicle.a = vehicle->a;
//         cpp_vehicle.b = vehicle->b;
//         cpp_vehicle.T_f = vehicle->T_f;
//         cpp_vehicle.T_r = vehicle->T_r;
//         cpp_vehicle.max_speed = vehicle->max_speed;
//         cpp_vehicle.max_accel = vehicle->max_accel;
//         cpp_vehicle.max_steering_angle = vehicle->max_steering_angle;
//         cpp_vehicle.max_steering_rate = vehicle->max_steering_rate;
        
//         Frenet_Planner* planner = new Frenet_Planner(
//             cpp_settings, 
//             cpp_vehicle,
//             obstacles_array,
//             num_vertices_array,
//             num_time_steps,
//             num_obstacles,
//             max_vertices
//         );
//         planner_instances.push_back(planner);
        
//         return (PlannerHandle)planner;
//     } catch (const std::exception& e) {
//         std::cerr << "Error creating planner: " << e.what() << std::endl;
//         return nullptr;
//     }
// }

// // Destroy planner instance
// void destroy_planner(PlannerHandle handle) {
//     if (handle == nullptr) return;
    
//     Frenet_Planner* planner = (Frenet_Planner*)handle;
    
//     auto it = std::find(planner_instances.begin(), planner_instances.end(), planner);
//     if (it != planner_instances.end()) {
//         planner_instances.erase(it);
//     }
    
//     delete planner;
// }

// // Generate Frenet frame
// void generate_frenet_frame(PlannerHandle handle, const double* centerline_pts, int num_points, int pts_dim) {
//     if (handle == nullptr) {
//         std::cerr << "Invalid planner handle" << std::endl;
//         return;
//     }
    
//     Frenet_Planner* planner = (Frenet_Planner*)handle;
//     planner->generate_frenet_frame(centerline_pts, num_points, pts_dim);
// }

// // Main planning function - simplified interface
// CFrenetTrajectory plan(
//     PlannerHandle handle,
//     const CFrenetState* frenet_state,
//     double max_target_speed,
//     int time_step_now
// ) {
//     CFrenetTrajectory result = {};
//     result.is_valid = 0;
    
//     if (handle == nullptr) {
//         std::cerr << "Invalid planner handle" << std::endl;
//         return result;
//     }
    
//     try {
//         Frenet_Planner* planner = (Frenet_Planner*)handle;
        
//         // Convert C state to C++ state
//         FrenetState cpp_state;
//         cpp_state.t = frenet_state->t;
//         cpp_state.s = frenet_state->s;
//         cpp_state.s_d = frenet_state->s_d;
//         cpp_state.s_dd = frenet_state->s_dd;
//         cpp_state.s_ddd = frenet_state->s_ddd;
//         cpp_state.d = frenet_state->d;
//         cpp_state.d_d = frenet_state->d_d;
//         cpp_state.d_dd = frenet_state->d_dd;
//         cpp_state.d_ddd = frenet_state->d_ddd;
        
//         // Call planning with simplified interface
//         FrenetTrajectory traj = planner->plan(cpp_state, max_target_speed, time_step_now);
        
//         // Check if trajectory is valid
//         if (traj.x.empty()) {
//             return result;
//         }
        
//         result.trajectory_size = traj.x.size();
//         result.cost_final = traj.cost_final;
//         result.is_valid = 1;
        
//         // Allocate and copy trajectory data
//         result.x = new double[result.trajectory_size];
//         result.y = new double[result.trajectory_size];
//         result.yaw = new double[result.trajectory_size];
//         result.t = new double[result.trajectory_size];
//         result.s = new double[result.trajectory_size];
//         result.s_d = new double[result.trajectory_size];
//         result.s_dd = new double[result.trajectory_size];
//         result.s_ddd = new double[result.trajectory_size];
//         result.d = new double[result.trajectory_size];
//         result.d_d = new double[result.trajectory_size];
//         result.d_dd = new double[result.trajectory_size];
//         result.d_ddd = new double[result.trajectory_size];
        
//         std::copy(traj.x.begin(), traj.x.end(), result.x);
//         std::copy(traj.y.begin(), traj.y.end(), result.y);
//         std::copy(traj.yaw.begin(), traj.yaw.end(), result.yaw);
//         std::copy(traj.t.begin(), traj.t.end(), result.t);
//         std::copy(traj.s.begin(), traj.s.end(), result.s);
//         std::copy(traj.s_d.begin(), traj.s_d.end(), result.s_d);
//         std::copy(traj.s_dd.begin(), traj.s_dd.end(), result.s_dd);
//         std::copy(traj.s_ddd.begin(), traj.s_ddd.end(), result.s_ddd);
//         std::copy(traj.d.begin(), traj.d.end(), result.d);
//         std::copy(traj.d_d.begin(), traj.d_d.end(), result.d_d);
//         std::copy(traj.d_dd.begin(), traj.d_dd.end(), result.d_dd);
//         std::copy(traj.d_ddd.begin(), traj.d_ddd.end(), result.d_ddd);
        
//         return result;
//     } catch (const std::exception& e) {
//         std::cerr << "Error during planning: " << e.what() << std::endl;
//         return result;
//     }
// }

// // Free trajectory memory
// void free_trajectory(CFrenetTrajectory* traj) {
//     if (traj == nullptr) return;
    
//     delete[] traj->x;
//     delete[] traj->y;
//     delete[] traj->yaw;
//     delete[] traj->t;
//     delete[] traj->s;
//     delete[] traj->s_d;
//     delete[] traj->s_dd;
//     delete[] traj->s_ddd;
//     delete[] traj->d;
//     delete[] traj->d_d;
//     delete[] traj->d_dd;
//     delete[] traj->d_ddd;
    
//     traj->x = nullptr;
//     traj->y = nullptr;
//     traj->yaw = nullptr;
//     traj->t = nullptr;
//     traj->s = nullptr;
//     traj->s_d = nullptr;
//     traj->s_dd = nullptr;
//     traj->s_ddd = nullptr;
//     traj->d = nullptr;
//     traj->d_d = nullptr;
//     traj->d_dd = nullptr;
//     traj->d_ddd = nullptr;
// }

// // Set number of threads
// void set_num_threads(int num_threads) {
//     // TODO: Implement thread pool configuration
//     // This can be extended later for OpenMP or thread pool management
// }

// // Get version
// const char* get_version() {
//     return VERSION;
// }
