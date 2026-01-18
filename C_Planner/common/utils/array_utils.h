#ifndef ARRAY_UTILS_H
#define ARRAY_UTILS_H

#include <vector>
#include "common/scenario/frenet.h"

// Prepare trajectory array for collision checking
// Input: list of trajectories
// Output: flat array of shape [num_trajs, max_length, 3] where each point is (x, y, yaw)
//         and array of trajectory lengths
struct TrajectoryArray {
    std::vector<double> data;         // flat array [num_trajs * max_length * 3]
    std::vector<int> lengths;        // trajectory lengths [num_trajs]
    int num_trajs;
    int max_length;
};

TrajectoryArray prepare_trajectory_array(const std::vector<FrenetTrajectory>& fplist);

// Extract obstacle polygons from flat array
// Array layout: [num_time_steps][num_obstacles][max_vertices][2]
// num_vertices layout: [num_time_steps][num_obstacles]
struct ObstacleArray {
    std::vector<double> data;          // flat array
    std::vector<int> num_vertices;    // actual vertex counts
    int num_time_steps;
    int num_obstacles;
    int max_vertices;
};

// Helper function to access trajectory array
inline double get_trajectory_point(const TrajectoryArray& traj_array, 
                                  int traj_idx, int point_idx, int coord) {
    int idx = traj_idx * traj_array.max_length * 3 + point_idx * 3 + coord;
    return traj_array.data[idx];
}

// Helper function to access obstacle vertex
inline double get_obstacle_vertex(const ObstacleArray& obs_array,
                                 int time_step, int obs_idx, int vertex_idx, int coord) {
    int base_idx = (time_step * obs_array.num_obstacles + obs_idx) * obs_array.max_vertices * 2;
    int idx = base_idx + vertex_idx * 2 + coord;
    return obs_array.data[idx];
}

// Helper function to get vertex count
inline int get_vertex_count(const ObstacleArray& obs_array, int time_step, int obs_idx) {
    return obs_array.num_vertices[time_step * obs_array.num_obstacles + obs_idx];
}

#endif // ARRAY_UTILS_H
