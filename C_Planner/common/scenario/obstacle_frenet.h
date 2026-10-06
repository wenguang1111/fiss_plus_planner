#ifndef OBSTACLE_FRENET_H
#define OBSTACLE_FRENET_H

#include <vector>

// Extent of every obstacle polygon in the Frenet frame of the reference line, computed
// once per reference frame in Python: [time step][obstacle] -> (s_min, s_max, l_min, l_max),
// NaN where the obstacle is absent. Used to find obstacles ahead in the ego path.
struct ObstacleFrenetBounds {
    std::vector<double> data;
    int num_time_steps = 0;
    int num_obstacles = 0;

    bool empty() const { return data.empty(); }
    const double* at(int t, int j) const {
        return data.data() + 4L * (static_cast<long>(t) * num_obstacles + j);
    }
};

#endif // OBSTACLE_FRENET_H
