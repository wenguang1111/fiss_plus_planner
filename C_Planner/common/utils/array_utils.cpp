#include "array_utils.h"

TrajectoryArray prepare_trajectory_array(const std::vector<FrenetTrajectory>& fplist) {
    TrajectoryArray result;
    
    result.num_trajs = fplist.size();
    if (result.num_trajs == 0) {
        result.max_length = 0;
        return result;
    }
    
    // Find max length
    result.max_length = 0;
    for (const auto& traj : fplist) {
        result.max_length = std::max(result.max_length, (int)traj.x.size());
    }
    
    // Allocate flat array: [num_trajs][max_length][3]
    result.data.resize(result.num_trajs * result.max_length * 3, 0.0f);
    result.lengths.resize(result.num_trajs);
    
    // Fill array with trajectory data
    for (int i = 0; i < result.num_trajs; i++) {
        const auto& traj = fplist[i];
        result.lengths[i] = traj.x.size();
        
        for (size_t j = 0; j < traj.x.size(); j++) {
            int idx = i * result.max_length * 3 + j * 3;
            result.data[idx] = traj.x[j];           // x
            result.data[idx + 1] = traj.y[j];       // y
            result.data[idx + 2] = traj.yaw[j];     // yaw
        }
    }
    
    return result;
}
