#ifndef COLLISION_CHECKER_H
#define COLLISION_CHECKER_H

#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include "../scenario/frenet.h"
#include "../scenario/obstacles.h"

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

// Penetration depth of two convex polygons: the smallest overlap of their projections
// over all edge normals (separating axis theorem). 0 when they are separated.
double polygon_penetration_depth(const std::vector<Eigen::Vector2d>& poly1,
                                 const std::vector<Eigen::Vector2d>& poly2);

struct TrajectoryCollision {
    bool collided = false;
    int colliding_steps = 0;     // number of checked steps that hit an obstacle
    double violation = 0.0;      // (1/N) * sum_k (depth_k / vehicle_width)^2, see .cpp
};

// Checks the ego footprint at each trajectory step against every obstacle polygon at
// time step time_step_now + k. With stop_at_first = true the check returns at the
// first collision (the violation then only reflects that step).
TrajectoryCollision check_trajectory_collision(const FrenetTrajectory& traj,
                                               const ObstacleView& obstacles,
                                               double vehicle_length,
                                               double vehicle_width,
                                               int time_step_now,
                                               bool stop_at_first);

#endif // COLLISION_CHECKER_H
