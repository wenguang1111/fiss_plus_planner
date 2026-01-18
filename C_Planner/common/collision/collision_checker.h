#ifndef COLLISION_CHECKER_H
#define COLLISION_CHECKER_H

#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include "../scenario/frenet.h"

// Polygon collision detection helper functions
bool point_in_polygon(const Eigen::Vector2d& point, const std::vector<Eigen::Vector2d>& polygon);

bool segments_intersect(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2,
                       const Eigen::Vector2d& p3, const Eigen::Vector2d& p4);

bool aabb_collision(const std::vector<Eigen::Vector2d>& poly1,
                   const std::vector<Eigen::Vector2d>& poly2);

bool polygon_collision(const std::vector<Eigen::Vector2d>& poly1,
                      const std::vector<Eigen::Vector2d>& poly2);

// Compute vehicle polygon at given position and orientation
std::vector<Eigen::Vector2d> compute_vehicle_polygon(double x, double y, double yaw,
                                                     double vehicle_length, double vehicle_width);

// Main single-threaded collision checker
// Returns vector of trajectories that have no collisions
std::vector<FrenetTrajectory> check_collision(
    const std::vector<FrenetTrajectory>& trajs,
    const double* obstacles_array,
    const int* num_vertices_array,
    int num_time_steps,
    int num_obstacles,
    int max_vertices,
    double vehicle_length,
    double vehicle_width,
    int time_step_now = 0,
    int check_resolution = 1
);

#endif // COLLISION_CHECKER_H
