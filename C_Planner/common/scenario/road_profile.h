#ifndef ROAD_PROFILE_H
#define ROAD_PROFILE_H

#include <vector>

// Lane width and drivable-road extents along the reference line. Distances are in
// the same reference frame as trajectory s. Positive extents measure from the
// reference line to the left/right same-direction road edge.
class RoadProfile {
public:
    // Throws std::invalid_argument unless s is strictly increasing, spans [0, s_end]
    // and all arrays have the same length (>= 2) with nonnegative finite values.
    void set(const std::vector<double>& s,
             const std::vector<double>& lane_width,
             const std::vector<double>& left_extent,
             const std::vector<double>& right_extent,
             double s_end);
    void clear();
    bool empty() const { return s_.empty(); }

    // Linearly interpolated geometry at s; false outside the profile.
    bool at(double s, double& lane_width, double& left, double& right) const;

private:
    std::vector<double> s_, lane_widths_, left_, right_;
};

#endif // ROAD_PROFILE_H
