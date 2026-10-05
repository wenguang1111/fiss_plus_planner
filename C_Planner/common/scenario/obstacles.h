#ifndef OBSTACLES_H
#define OBSTACLES_H

#include <algorithm>

// Read-only view of the pre-processed obstacle prediction arrays owned by Python:
//   vertices:     [num_time_steps][num_obstacles][max_vertices][2]
//   num_vertices: [num_time_steps][num_obstacles]
struct ObstacleView {
    const double* vertices = nullptr;
    const int* num_vertices = nullptr;
    int num_time_steps = 0;
    int num_obstacles = 0;
    int max_vertices = 0;

    bool empty() const {
        return vertices == nullptr || num_vertices == nullptr ||
               num_time_steps <= 0 || num_obstacles <= 0 || max_vertices <= 0;
    }

    // Number of valid vertices of obstacle j at time step t (0 if absent).
    int vertex_count(int t, int j) const {
        return std::min(num_vertices[t * num_obstacles + j], max_vertices);
    }

    // Pointer to the first (x, y) pair of obstacle j at time step t.
    const double* polygon(int t, int j) const {
        return vertices + static_cast<long>(t * num_obstacles + j) * max_vertices * 2;
    }
};

#endif // OBSTACLES_H
