#ifndef CUBIC_SPLINE_H
#define CUBIC_SPLINE_H

#include <vector>
#include <cmath>

class CubicSpline1D {
public:
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> a, b, c, d;
    int nx;
    
    CubicSpline1D(const std::vector<double>& x_coords, const std::vector<double>& y_coords);
    
    double calc_position(double x_val);
    double calc_first_derivative(double x_val);
    double calc_second_derivative(double x_val);
    
private:
    int search_index(double x_val);
    std::vector<double> calc_A(const std::vector<double>& h);
    std::vector<double> calc_B(const std::vector<double>& h, const std::vector<double>& a_vals);
};

class CubicSpline2D {
public:
    std::vector<double> s;
    std::vector<double> ds;
    CubicSpline1D* sx;
    CubicSpline1D* sy;
    
    CubicSpline2D(const std::vector<double>& x_coords, const std::vector<double>& y_coords);
    ~CubicSpline2D();
    
    std::pair<double, double> calc_position(double s_val);
    double calc_yaw(double s_val);
    double calc_curvature(double s_val);
    
private:
    std::vector<double> calc_s(const std::vector<double>& x, const std::vector<double>& y);
};

#endif // CUBIC_SPLINE_H
