#include "cubic_spline.h"
#include <algorithm>
#include <Eigen/Dense>

// CubicSpline1D Implementation
CubicSpline1D::CubicSpline1D(const std::vector<double>& x_coords, const std::vector<double>& y_coords) {
    x = x_coords;
    y = y_coords;
    nx = x.size();
    a = y;
    
    // Calculate h (differences)
    std::vector<double> h(nx - 1);
    for (int i = 0; i < nx - 1; i++) {
        h[i] = x[i + 1] - x[i];
        if (h[i] < 0) {
            throw std::invalid_argument("x coordinates must be sorted in ascending order");
        }
    }
    
    // Calculate coefficient c
    std::vector<double> A_vec = calc_A(h);
    std::vector<double> B_vec = calc_B(h, a);
    
    Eigen::MatrixXd A_matrix = Eigen::Map<Eigen::MatrixXd>(A_vec.data(), nx, nx);
    Eigen::VectorXd b_vector = Eigen::Map<Eigen::VectorXd>(B_vec.data(), nx);
    Eigen::VectorXd c_vector = A_matrix.colPivHouseholderQr().solve(b_vector);
    c.resize(nx);
    for (int i = 0; i < nx; i++) {
        c[i] = c_vector(i);
    }
    
    // Calculate spline coefficients b and d
    for (int i = 0; i < nx - 1; i++) {
        double d_coef = (c[i + 1] - c[i]) / (3.0 * h[i]);
        double b_coef = (a[i + 1] - a[i]) / h[i] - h[i] / 3.0 * (2.0 * c[i] + c[i + 1]);
        d.push_back(d_coef);
        b.push_back(b_coef);
    }
}

int CubicSpline1D::search_index(double x_val) {
    auto it = std::lower_bound(x.begin(), x.end(), x_val);
    int idx = std::distance(x.begin(), it) - 1;
    return std::max(0, idx);
}

double CubicSpline1D::calc_position(double x_val) {
    if (x_val < x[0] || x_val > x[nx - 1]) {
        return NAN;
    }
    
    int i = search_index(x_val);
    double dx = x_val - x[i];
    double position = a[i] + b[i] * dx + c[i] * dx * dx + d[i] * dx * dx * dx;
    return position;
}

double CubicSpline1D::calc_first_derivative(double x_val) {
    if (x_val < x[0] || x_val > x[nx - 1]) {
        return NAN;
    }
    
    int i = search_index(x_val);
    double dx = x_val - x[i];
    double dy = b[i] + 2.0 * c[i] * dx + 3.0 * d[i] * dx * dx;
    return dy;
}

double CubicSpline1D::calc_second_derivative(double x_val) {
    if (x_val < x[0] || x_val > x[nx - 1]) {
        return NAN;
    }
    
    int i = search_index(x_val);
    double dx = x_val - x[i];
    double ddy = 2.0 * c[i] + 6.0 * d[i] * dx;
    return ddy;
}

std::vector<double> CubicSpline1D::calc_A(const std::vector<double>& h) {
    std::vector<double> A_data(nx * nx, 0.0);
    A_data[0] = 1.0;  // A[0,0] = 1.0
    
    for (int i = 0; i < nx - 1; i++) {
        if (i != (nx - 2)) {
            A_data[(i + 1) * nx + (i + 1)] = 2.0 * (h[i] + h[i + 1]);
        }
        A_data[(i + 1) * nx + i] = h[i];
        A_data[i * nx + (i + 1)] = h[i];
    }
    
    A_data[0 * nx + 1] = 0.0;
    A_data[(nx - 1) * nx + (nx - 2)] = 0.0;
    A_data[(nx - 1) * nx + (nx - 1)] = 1.0;
    
    return A_data;
}

std::vector<double> CubicSpline1D::calc_B(const std::vector<double>& h, const std::vector<double>& a_vals) {
    std::vector<double> B(nx, 0.0);
    for (int i = 0; i < nx - 2; i++) {
        B[i + 1] = 3.0 * (a_vals[i + 2] - a_vals[i + 1]) / h[i + 1] 
                 - 3.0 * (a_vals[i + 1] - a_vals[i]) / h[i];
    }
    return B;
}

// CubicSpline2D Implementation
CubicSpline2D::CubicSpline2D(const std::vector<double>& x_coords, const std::vector<double>& y_coords) {
    s = calc_s(x_coords, y_coords);
    sx = new CubicSpline1D(s, x_coords);
    sy = new CubicSpline1D(s, y_coords);
}

CubicSpline2D::~CubicSpline2D() {
    if (sx != nullptr) delete sx;
    if (sy != nullptr) delete sy;
}

std::vector<double> CubicSpline2D::calc_s(const std::vector<double>& x, const std::vector<double>& y) {
    std::vector<double> result;
    result.push_back(0.0);
    
    for (size_t i = 0; i < x.size() - 1; i++) {
        double dx = x[i + 1] - x[i];
        double dy = y[i + 1] - y[i];
        double distance = std::sqrt(dx * dx + dy * dy);
        ds.push_back(distance);
        result.push_back(result.back() + distance);
    }
    
    return result;
}

std::pair<double, double> CubicSpline2D::calc_position(double s_val) {
    double x = sx->calc_position(s_val);
    double y = sy->calc_position(s_val);
    return std::make_pair(x, y);
}

double CubicSpline2D::calc_yaw(double s_val) {
    double dx = sx->calc_first_derivative(s_val);
    double dy = sy->calc_first_derivative(s_val);
    return std::atan2(dy, dx);
}

double CubicSpline2D::calc_curvature(double s_val) {
    double dx = sx->calc_first_derivative(s_val);
    double ddx = sx->calc_second_derivative(s_val);
    double dy = sy->calc_first_derivative(s_val);
    double ddy = sy->calc_second_derivative(s_val);
    
    double numerator = ddy * dx - ddx * dy;
    double denominator = std::pow(dx * dx + dy * dy, 1.5);
    
    if (denominator == 0.0) return 0.0;
    return numerator / denominator;
}
