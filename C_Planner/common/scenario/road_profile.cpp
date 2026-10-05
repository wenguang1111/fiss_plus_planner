#include "road_profile.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

void RoadProfile::set(const std::vector<double>& s,
                      const std::vector<double>& lane_width,
                      const std::vector<double>& left_extent,
                      const std::vector<double>& right_extent,
                      double s_end) {
    if (s.size() < 2 || lane_width.size() != s.size() ||
        left_extent.size() != s.size() || right_extent.size() != s.size()) {
        throw std::invalid_argument("Road profile requires a reference frame and equally sized arrays of at least two points");
    }
    for (size_t i = 0; i < s.size(); ++i) {
        if (!std::isfinite(s[i]) || (i > 0 && s[i] <= s[i - 1]) ||
            !std::isfinite(lane_width[i]) || lane_width[i] < 0.0 ||
            !std::isfinite(left_extent[i]) || left_extent[i] < 0.0 ||
            !std::isfinite(right_extent[i]) || right_extent[i] < 0.0) {
            throw std::invalid_argument("Road profile needs increasing finite s and nonnegative finite widths/extents");
        }
    }
    if (std::abs(s.front()) > 1e-6 || std::abs(s.back() - s_end) > 1e-6) {
        throw std::invalid_argument("Road profile must span the reference frame's s range");
    }
    s_ = s;
    lane_widths_ = lane_width;
    left_ = left_extent;
    right_ = right_extent;
}

void RoadProfile::clear() {
    s_.clear();
    lane_widths_.clear();
    left_.clear();
    right_.clear();
}

bool RoadProfile::at(double s, double& lane_width, double& left, double& right) const {
    if (s_.empty() || !std::isfinite(s) || s < s_.front() - 1e-6 || s > s_.back() + 1e-6) {
        return false;
    }
    s = std::clamp(s, s_.front(), s_.back());
    size_t upper = std::upper_bound(s_.begin(), s_.end(), s) - s_.begin();
    upper = std::min(upper, s_.size() - 1);
    size_t lower = upper - 1;
    double ratio = (s - s_[lower]) / (s_[upper] - s_[lower]);
    auto interpolate = [lower, upper, ratio](const std::vector<double>& values) {
        return values[lower] + ratio * (values[upper] - values[lower]);
    };
    lane_width = interpolate(lane_widths_);
    left = interpolate(left_);
    right = interpolate(right_);
    return true;
}
