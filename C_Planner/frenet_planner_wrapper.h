// #ifndef FRENET_PLANNER_WRAPPER_H
// #define FRENET_PLANNER_WRAPPER_H

// #include "Frenet_Planner.h"
// #include <vector>
// #include <cstring>

// // C-style wrapper for Python ctypes/CFFI compatibility
// extern "C" {
//     // Opaque handles for C++ objects
//     typedef void* PlannerHandle;
    
//     // ===== Settings Creation =====
//     struct CSettingParameters {
//         double tick_t;
//         double max_road_width;
//         int num_width;
//         double highest_speed;
//         double lowest_speed;
//         int num_speed;
//         double min_t;
//         double max_t;
//         int num_t;
//         int check_obstacle;
//         int check_boundary;
//     };
    
//     // ===== Vehicle Parameters =====
//     struct CVehicleParams {
//         double l;
//         double w;
//         double a;
//         double b;
//         double T_f;
//         double T_r;
//         double max_speed;
//         double max_accel;
//         double max_steering_angle;
//         double max_steering_rate;
//     };
    
//     // ===== Frenet State =====
//     struct CFrenetState {
//         double t;
//         double s;
//         double s_d;
//         double s_dd;
//         double s_ddd;
//         double d;
//         double d_d;
//         double d_dd;
//         double d_ddd;
//     };
    
//     // ===== Trajectory Output =====
//     struct CFrenetTrajectory {
//         int trajectory_size;
//         double* x;
//         double* y;
//         double* yaw;
//         double* t;
//         double* s;
//         double* s_d;
//         double* s_dd;
//         double* s_ddd;
//         double* d;
//         double* d_d;
//         double* d_dd;
//         double* d_ddd;
//         double cost_final;
//         int is_valid;
//     };
    
//     // ===== Planner Creation/Destruction =====
//     PlannerHandle create_planner(const CSettingParameters* settings, 
//                                 const CVehicleParams* vehicle,
//                                 const double* obstacles_array,
//                                 const int* num_vertices_array,
//                                 int num_time_steps,
//                                 int num_obstacles,
//                                 int max_vertices);
//     void destroy_planner(PlannerHandle handle);
    
//     // ===== Frenet Frame Generation =====
//     void generate_frenet_frame(PlannerHandle handle, const double* centerline_pts, int num_points, int pts_dim);
    
//     // ===== Main Planning - Simplified Interface =====
//     CFrenetTrajectory plan(
//         PlannerHandle handle,
//         const CFrenetState* frenet_state,
//         double max_target_speed,
//         int time_step_now
//     );
    
//     // ===== Memory Management =====
//     void free_trajectory(CFrenetTrajectory* traj);
    
//     // ===== Utility Functions =====
//     void set_num_threads(int num_threads);
//     const char* get_version();
// }

// #endif // FRENET_PLANNER_WRAPPER_H
