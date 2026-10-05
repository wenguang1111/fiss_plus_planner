#ifndef GAUSSIAN_PROPOSAL_H
#define GAUSSIAN_PROPOSAL_H

#include <array>
#include <random>
#include <vector>

// Point of the search space in unit coordinates, see SearchSpace::to_unit
using Unit3 = std::array<double, 3>;

// Diagonal Gaussian proposal over unit coordinates, used by the iterative sampling
// planners (CEM, MPPI).
struct GaussianProposal {
    Unit3 mean{0.5, 0.5, 0.5};
    Unit3 stddev{0.3, 0.3, 0.3};

    // n samples clipped to [0, 1]^3; with include_mean the first sample is the mean itself.
    std::vector<Unit3> sample(int n, std::mt19937_64& rng, bool include_mean) const;

    // Weighted mean and per-dimension standard deviation of points (weights need not
    // be normalized but must not all be zero).
    static void fit(const std::vector<Unit3>& points, const std::vector<double>& weights,
                    Unit3& mean, Unit3& stddev);
};

#endif // GAUSSIAN_PROPOSAL_H
