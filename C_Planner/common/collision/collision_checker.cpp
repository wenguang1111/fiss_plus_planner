#include "collision_checker.h"
#include <algorithm>

bool point_in_polygon(const Eigen::Vector2d& point, const std::vector<Eigen::Vector2d>& polygon) {
    double x = point.x();
    double y = point.y();
    int n = polygon.size();
    bool inside = false;
    
    double p1x = polygon[0].x();
    double p1y = polygon[0].y();
    
    for (int i = 1; i <= n; i++) {
        double p2x = polygon[i % n].x();
        double p2y = polygon[i % n].y();
        
        if (y > std::min(p1y, p2y)) {
            if (y <= std::max(p1y, p2y)) {
                if (x <= std::max(p1x, p2x)) {
                    double xinters = p1x;  // Default value
                    if (p1y != p2y) {
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x;
                    }
                    if (p1x == p2x || x <= xinters) {
                        inside = !inside;
                    }
                }
            }
        }
        p1x = p2x;
        p1y = p2y;
    }
    
    return inside;
}

bool segments_intersect(const Eigen::Vector2d& p1, const Eigen::Vector2d& p2,
                       const Eigen::Vector2d& p3, const Eigen::Vector2d& p4) {
    auto ccw = [](const Eigen::Vector2d& A, const Eigen::Vector2d& B, const Eigen::Vector2d& C) {
        return (C.y() - A.y()) * (B.x() - A.x()) > (B.y() - A.y()) * (C.x() - A.x());
    };
    
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) && ccw(p1, p2, p3) != ccw(p1, p2, p4);
}

bool aabb_collision(const std::vector<Eigen::Vector2d>& poly1,
                   const std::vector<Eigen::Vector2d>& poly2) {
    double min_x1 = poly1[0].x(), max_x1 = poly1[0].x();
    double min_y1 = poly1[0].y(), max_y1 = poly1[0].y();
    
    for (const auto& p : poly1) {
        min_x1 = std::min(min_x1, p.x());
        max_x1 = std::max(max_x1, p.x());
        min_y1 = std::min(min_y1, p.y());
        max_y1 = std::max(max_y1, p.y());
    }
    
    double min_x2 = poly2[0].x(), max_x2 = poly2[0].x();
    double min_y2 = poly2[0].y(), max_y2 = poly2[0].y();
    
    for (const auto& p : poly2) {
        min_x2 = std::min(min_x2, p.x());
        max_x2 = std::max(max_x2, p.x());
        min_y2 = std::min(min_y2, p.y());
        max_y2 = std::max(max_y2, p.y());
    }
    
    if (max_x1 < min_x2 || max_x2 < min_x1) return false;
    if (max_y1 < min_y2 || max_y2 < min_y1) return false;
    
    return true;
}

bool polygon_collision(const std::vector<Eigen::Vector2d>& poly1,
                      const std::vector<Eigen::Vector2d>& poly2) {
    if (!aabb_collision(poly1, poly2)) {
        return false;
    }
    
    // Check if any point of poly1 is inside poly2
    for (const auto& p : poly1) {
        if (point_in_polygon(p, poly2)) {
            return true;
        }
    }
    
    // Check if any point of poly2 is inside poly1
    for (const auto& p : poly2) {
        if (point_in_polygon(p, poly1)) {
            return true;
        }
    }
    
    // Check if any edges intersect
    for (size_t i = 0; i < poly1.size(); i++) {
        for (size_t j = 0; j < poly2.size(); j++) {
            const auto& p1 = poly1[i];
            const auto& p2 = poly1[(i + 1) % poly1.size()];
            const auto& p3 = poly2[j];
            const auto& p4 = poly2[(j + 1) % poly2.size()];
            
            if (segments_intersect(p1, p2, p3, p4)) {
                return true;
            }
        }
    }
    
    return false;
}

std::vector<Eigen::Vector2d> compute_vehicle_polygon(double x, double y, double yaw,
                                                     double vehicle_length, double vehicle_width) {
    std::vector<Eigen::Vector2d> corner_offsets = {
        {vehicle_length / 2.0, vehicle_width / 2.0},
        {vehicle_length / 2.0, -vehicle_width / 2.0},
        {-vehicle_length / 2.0, -vehicle_width / 2.0},
        {-vehicle_length / 2.0, vehicle_width / 2.0}
    };
    
    double cos_yaw = std::cos(yaw);
    double sin_yaw = std::sin(yaw);
    
    std::vector<Eigen::Vector2d> polygon;
    for (const auto& offset : corner_offsets) {
        double rotated_x = offset.x() * cos_yaw - offset.y() * sin_yaw;
        double rotated_y = offset.x() * sin_yaw + offset.y() * cos_yaw;
        
        polygon.push_back({rotated_x + x, rotated_y + y});
    }
    
    return polygon;
}

int first_collision_step(const FrenetTrajectory& traj,
                         const ObstacleView& obstacles,
                         double vehicle_length,
                         double vehicle_width,
                         int time_step_now) {
    if (obstacles.empty()) {
        return -1;
    }

    const int traj_len = static_cast<int>(std::min({traj.x.size(), traj.y.size(), traj.yaw.size()}));
    const int max_steps = std::min(traj_len, obstacles.num_time_steps - time_step_now);
    std::vector<Eigen::Vector2d> obs_poly;

    for (int k = 0; k < max_steps; ++k) {
        const auto ego_poly = compute_vehicle_polygon(traj.x[k], traj.y[k], traj.yaw[k],
                                                      vehicle_length, vehicle_width);
        const int time_step = time_step_now + k;
        for (int j = 0; j < obstacles.num_obstacles; ++j) {
            const int n = obstacles.vertex_count(time_step, j);
            if (n <= 0) continue;
            const double* verts = obstacles.polygon(time_step, j);
            obs_poly.clear();
            for (int v = 0; v < n; ++v) {
                obs_poly.emplace_back(verts[2 * v], verts[2 * v + 1]);
            }
            if (polygon_collision(ego_poly, obs_poly)) {
                return k;
            }
        }
    }
    return -1;
}
