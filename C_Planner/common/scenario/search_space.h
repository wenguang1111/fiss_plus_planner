#ifndef SEARCH_SPACE_H
#define SEARCH_SPACE_H

#include <algorithm>
#include <array>
#include "frenet.h"

// Admissible region of the sampling parameters z = (d, s_d, t), shared by all planners.
// to_unit / from_unit map z to [0, 1]^3, so sampling distributions can use one scale
// for all three dimensions.
struct SearchSpace {
    double d_min, d_max;   // terminal lateral offset [m]
    double v_min, v_max;   // terminal longitudinal speed [m/s]
    double t_min, t_max;   // planning horizon [s]

    std::array<double, 3> to_unit(const SamplingParam& z) const {
        return {unit(z.d, d_min, d_max), unit(z.s_d, v_min, v_max), unit(z.t, t_min, t_max)};
    }

    SamplingParam from_unit(const std::array<double, 3>& u) const {
        return SamplingParam(d_min + u[0] * (d_max - d_min),
                             v_min + u[1] * (v_max - v_min),
                             t_min + u[2] * (t_max - t_min));
    }

private:
    static double unit(double x, double lo, double hi) {
        return hi - lo > 1e-12 ? std::clamp((x - lo) / (hi - lo), 0.0, 1.0) : 0.5;
    }
};

#endif // SEARCH_SPACE_H
