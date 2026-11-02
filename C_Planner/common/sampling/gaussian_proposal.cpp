#include "gaussian_proposal.h"
#include <algorithm>
#include <cmath>

std::vector<Unit3> GaussianProposal::sample(int n, std::mt19937_64& rng, bool include_mean) const {
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<Unit3> points(std::max(n, 0));
    for (int i = 0; i < n; ++i) {
        if (include_mean && i == 0) {
            points[i] = mean;
            continue;
        }
        for (int dim = 0; dim < 3; ++dim) {
            points[i][dim] = std::clamp(mean[dim] + stddev[dim] * normal(rng), 0.0, 1.0);
        }
    }
    return points;
}

void GaussianProposal::fit(const std::vector<Unit3>& points, const std::vector<double>& weights,
                           Unit3& mean, Unit3& stddev) {
    double weight_sum = 0.0;
    mean = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < points.size(); ++i) {
        weight_sum += weights[i];
        for (int dim = 0; dim < 3; ++dim) mean[dim] += weights[i] * points[i][dim];
    }
    Unit3 var{0.0, 0.0, 0.0};
    for (int dim = 0; dim < 3; ++dim) mean[dim] /= weight_sum;
    for (size_t i = 0; i < points.size(); ++i) {
        for (int dim = 0; dim < 3; ++dim) {
            const double diff = points[i][dim] - mean[dim];
            var[dim] += weights[i] * diff * diff;
        }
    }
    for (int dim = 0; dim < 3; ++dim) stddev[dim] = std::sqrt(var[dim] / weight_sum);
}
