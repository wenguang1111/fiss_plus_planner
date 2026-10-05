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

// Index of the first trajectory step whose ego footprint overlaps an obstacle polygon
// at time step time_step_now + k, or -1 if the trajectory is collision-free. The check
// stops at the first collision; later steps are never examined.
int first_collision_step(const FrenetTrajectory& traj,
                         const ObstacleView& obstacles,
                         double vehicle_length,
                         double vehicle_width,
                         int time_step_now);

#endif // COLLISION_CHECKER_H
