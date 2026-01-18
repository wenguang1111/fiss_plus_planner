#ifndef FRENET_PLANNER_EXAMPLE_H
#define FRENET_PLANNER_EXAMPLE_H

#include "Frenet_Planner.h"
#include <iostream>
#include <vector>

// Example usage of the Frenet Optimal Planner
class FrenetPlannerExample {
public:
    static void example_basic_planning() {
        // Create planner settings
        SettingParameters settings(5, 5, 5);  // 5 width, 5 speed, 5 time samples
        settings.tick_t = 0.1f;
        settings.max_road_width = 3.5f;
        settings.highest_speed = 13.4f;
        settings.lowest_speed = 0.0f;
        settings.min_t = 3.0f;
        settings.max_t = 5.0f;
        
        // Create vehicle parameters
        VehicleParams vehicle;
        vehicle.l = 4.5f;              // 4.5m length
        vehicle.w = 1.8f;              // 1.8m width
        vehicle.a = 1.35f;             // CoG distance from base_link
        vehicle.b = 3.15f;             // CoG distance to front
        vehicle.T_f = 1.6f;            // Front track width
        vehicle.T_r = 1.6f;            // Rear track width
        vehicle.max_speed = 13.4f;     // m/s
        vehicle.max_accel = 5.0f;      // m/ss
        vehicle.max_steering_angle = 0.785f;  // ~45 degrees
        vehicle.max_steering_rate = 1.0f;     // rad/s
        
        // Create planner instance
        Frenet_Planner planner(settings, vehicle);
        
        // Setup centerline (road path)
        std::vector<float> centerline_x = {0, 10, 20, 30, 40, 50};
        std::vector<float> centerline_y = {0, 0.5, 1.0, 0.5, 0, -0.5};
        
        float centerline_pts[12] = {0, 0, 10, 0.5, 20, 1.0, 30, 0.5, 40, 0, 50, -0.5};
        planner.generate_frenet_frame(centerline_pts, 6, 2);
        
        // Create initial Frenet state
        FrenetState frenet_state;
        frenet_state.t = 0.0f;
        frenet_state.s = 5.0f;         // 5m along the road
        frenet_state.s_d = 10.0f;      // 10 m/s longitudinal velocity
        frenet_state.s_dd = 0.0f;      // no longitudinal acceleration
        frenet_state.d = 0.0f;         // centered on road
        frenet_state.d_d = 0.0f;       // no lateral velocity
        frenet_state.d_dd = 0.0f;      // no lateral acceleration
        
        // Plan trajectory without obstacles
        FrenetTrajectory best_traj = planner.plan(frenet_state, 12.0f);
        
        if (!best_traj.x.empty()) {
            std::cout << "Best trajectory found!" << std::endl;
            std::cout << "Cost: " << best_traj.cost_final << std::endl;
            std::cout << "Duration: " << best_traj.t.back() << " seconds" << std::endl;
            std::cout << "Final position: (" << best_traj.x.back() << ", " 
                      << best_traj.y.back() << ")" << std::endl;
        } else {
            std::cout << "No feasible trajectory found!" << std::endl;
        }
    }
    
    static void example_with_obstacles() {
        // Setup planner
        SettingParameters settings(3, 3, 3);
        VehicleParams vehicle;
        vehicle.l = 4.5f;
        vehicle.w = 1.8f;
        vehicle.max_speed = 13.4f;
        vehicle.max_accel = 5.0f;
        
        Frenet_Planner planner(settings, vehicle);
        
        // Simple straight road
        float centerline_pts[6] = {0, 0, 50, 0};
        planner.generate_frenet_frame(centerline_pts, 2, 2);
        
        // Create obstacle array
        // Format: [num_time_steps][num_obstacles][max_vertices][2]
        int num_time_steps = 50;
        int num_obstacles = 2;
        int max_vertices = 4;
        
        float obstacles_array[50 * 2 * 4 * 2];  // Flat array
        int num_vertices_array[50 * 2];         // Vertex counts
        
        // Initialize obstacle 1: stationary box at (25, 0.5)
        for (int t = 0; t < num_time_steps; t++) {
            // Obstacle 1 - rectangle centered at (25, 0.5)
            float ox = 25.0f;
            float oy = 0.5f;
            float ol = 2.0f;  // length
            float ow = 1.0f;  // width
            
            int base_idx = (t * num_obstacles + 0) * max_vertices * 2;
            // Front-left
            obstacles_array[base_idx + 0] = ox + ol/2;
            obstacles_array[base_idx + 1] = oy + ow/2;
            // Front-right
            obstacles_array[base_idx + 2] = ox + ol/2;
            obstacles_array[base_idx + 3] = oy - ow/2;
            // Rear-right
            obstacles_array[base_idx + 4] = ox - ol/2;
            obstacles_array[base_idx + 5] = oy - ow/2;
            // Rear-left
            obstacles_array[base_idx + 6] = ox - ol/2;
            obstacles_array[base_idx + 7] = oy + ow/2;
            
            num_vertices_array[t * num_obstacles + 0] = 4;
            
            // Obstacle 2 - another box at (40, -0.7)
            ox = 40.0f;
            oy = -0.7f;
            base_idx = (t * num_obstacles + 1) * max_vertices * 2;
            obstacles_array[base_idx + 0] = ox + ol/2;
            obstacles_array[base_idx + 1] = oy + ow/2;
            obstacles_array[base_idx + 2] = ox + ol/2;
            obstacles_array[base_idx + 3] = oy - ow/2;
            obstacles_array[base_idx + 4] = ox - ol/2;
            obstacles_array[base_idx + 5] = oy - ow/2;
            obstacles_array[base_idx + 6] = ox - ol/2;
            obstacles_array[base_idx + 7] = oy + ow/2;
            
            num_vertices_array[t * num_obstacles + 1] = 4;
        }
        
        // Plan with obstacles
        FrenetState frenet_state;
        frenet_state.s = 2.0f;
        frenet_state.s_d = 10.0f;
        frenet_state.d = 0.0f;
        frenet_state.d_d = 0.0f;
        frenet_state.d_dd = 0.0f;
        frenet_state.s_dd = 0.0f;
        
        FrenetTrajectory best_traj = planner.plan(
            frenet_state,
            12.0f,
            obstacles_array,
            num_vertices_array,
            num_time_steps,
            num_obstacles,
            max_vertices,
            0
        );
        
        if (!best_traj.x.empty()) {
            std::cout << "Trajectory avoiding obstacles found!" << std::endl;
            std::cout << "Final lateral position: " << best_traj.d.back() << " m" << std::endl;
        } else {
            std::cout << "No collision-free trajectory found!" << std::endl;
        }
    }
    
    static void print_trajectory(const FrenetTrajectory& traj, int num_points = 5) {
        std::cout << "\n=== Trajectory Summary ===" << std::endl;
        std::cout << "Total points: " << traj.x.size() << std::endl;
        std::cout << "Cost: " << traj.cost_final << std::endl;
        std::cout << "\nSample points:" << std::endl;
        
        int step = std::max(1, (int)traj.x.size() / num_points);
        for (size_t i = 0; i < traj.x.size(); i += step) {
            printf("  [%.2f] x=%.3f, y=%.3f, yaw=%.3f, speed=%.2f, lateral=%.2f\n",
                   traj.t[i], traj.x[i], traj.y[i], traj.yaw[i], traj.s_d[i], traj.d[i]);
        }
        
        // Print final point
        if (!traj.x.empty()) {
            size_t i = traj.x.size() - 1;
            printf("  [%.2f] x=%.3f, y=%.3f, yaw=%.3f, speed=%.2f, lateral=%.2f\n",
                   traj.t[i], traj.x[i], traj.y[i], traj.yaw[i], traj.s_d[i], traj.d[i]);
        }
    }
};

#endif // FRENET_PLANNER_EXAMPLE_H
