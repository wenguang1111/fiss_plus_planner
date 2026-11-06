#ifndef MPPI_PLANNER_H
#define MPPI_PLANNER_H

#include "Iterative_Sampling_Planner.h"

struct MPPISettings : IterativeSamplingSettings {
    double temperature = 0.1;    // gamma, applied to the per-iteration normalized cost
    double learning_rate = 0.7;  // eta of the mean / covariance update
};

// MPPI-style update over the FOP sampling parameters: the feasible samples are weighted by
// exp(-J / gamma) (MPPI, Williams et al. 2017; infeasible samples have weight 0) and the proposal
// moves towards the weighted mean and covariance with learning rate eta, the update MMD-OPT
// (Sharma & Singh 2025, Eq. 30) uses for Frenet behavioural inputs. J is normalized to [0, 1] per
// iteration because it differs by orders of magnitude between scenarios.
class MPPI_Planner : public Iterative_Sampling_Planner {
public:
    MPPISettings mppi_settings;

    MPPI_Planner(const SettingParameters& settings_param,
                 const MPPISettings& mppi_param,
                 const VehicleParams& vehicle_param,
                 const double* obs_array,
                 const int* num_verts,
                 int n_time_steps,
                 int n_obstacles,
                 int max_verts);

protected:
    const IterativeSamplingSettings& iteration_settings() const override { return mppi_settings; }
    void update_proposal(GaussianProposal& proposal,
                         const std::vector<Unit3>& units,
                         const std::vector<SampleEval>& evals) const override;
};

#endif // MPPI_PLANNER_H
