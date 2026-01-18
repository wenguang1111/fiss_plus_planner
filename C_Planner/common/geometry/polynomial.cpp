#include "polynomial.h"

// Quartic Polynomial Implementation
QuarticPolynomial::QuarticPolynomial(double xs, double vxs, double axs, double vxe, double axe, double time) {
    a0 = xs;
    a1 = vxs;
    a2 = axs / 2.0;
    
    // Solve linear system: A * x = b
    // [3*t^2, 4*t^3] [a3]   [vxe - a1 - 2*a2*t]
    // [6*t,  12*t^2] [a4] = [axe - 2*a2]
    
    Eigen::Matrix2d A;
    A(0, 0) = 3 * time * time;
    A(0, 1) = 4 * time * time * time;
    A(1, 0) = 6 * time;
    A(1, 1) = 12 * time * time;
    
    Eigen::Vector2d b;
    b(0) = vxe - a1 - 2 * a2 * time;
    b(1) = axe - 2 * a2;
    
    Eigen::Vector2d x = A.colPivHouseholderQr().solve(b);
    a3 = x(0);
    a4 = x(1);
}

double QuarticPolynomial::calc_point(double t) {
    return a0 + a1 * t + a2 * t * t + a3 * t * t * t + a4 * t * t * t * t;
}

double QuarticPolynomial::calc_first_derivative(double t) {
    return a1 + 2 * a2 * t + 3 * a3 * t * t + 4 * a4 * t * t * t;
}

double QuarticPolynomial::calc_second_derivative(double t) {
    return 2 * a2 + 6 * a3 * t + 12 * a4 * t * t;
}

double QuarticPolynomial::calc_third_derivative(double t) {
    return 6 * a3 + 24 * a4 * t;
}

// Quintic Polynomial Implementation
QuinticPolynomial::QuinticPolynomial(double xs, double vxs, double axs, double xe, double vxe, double axe, double time) {
    a0 = xs;
    a1 = vxs;
    a2 = axs / 2.0;
    
    // Solve linear system: A * x = b
    // [t^3,    t^4,     t^5   ] [a3]   [xe - a0 - a1*t - a2*t^2]
    // [3*t^2,  4*t^3,   5*t^4 ] [a4] = [vxe - a1 - 2*a2*t]
    // [6*t,    12*t^2,  20*t^3] [a5]   [axe - 2*a2]
    
    Eigen::Matrix3d A;
    A(0, 0) = time * time * time;
    A(0, 1) = time * time * time * time;
    A(0, 2) = time * time * time * time * time;
    A(1, 0) = 3 * time * time;
    A(1, 1) = 4 * time * time * time;
    A(1, 2) = 5 * time * time * time * time;
    A(2, 0) = 6 * time;
    A(2, 1) = 12 * time * time;
    A(2, 2) = 20 * time * time * time;
    
    Eigen::Vector3d b;
    b(0) = xe - a0 - a1 * time - a2 * time * time;
    b(1) = vxe - a1 - 2 * a2 * time;
    b(2) = axe - 2 * a2;
    
    Eigen::Vector3d x = A.colPivHouseholderQr().solve(b);
    a3 = x(0);
    a4 = x(1);
    a5 = x(2);
}

double QuinticPolynomial::calc_point(double t) {
    return a0 + a1 * t + a2 * t * t + a3 * t * t * t + a4 * t * t * t * t + a5 * t * t * t * t * t;
}

double QuinticPolynomial::calc_first_derivative(double t) {
    return a1 + 2 * a2 * t + 3 * a3 * t * t + 4 * a4 * t * t * t + 5 * a5 * t * t * t * t;
}

double QuinticPolynomial::calc_second_derivative(double t) {
    return 2 * a2 + 6 * a3 * t + 12 * a4 * t * t + 20 * a5 * t * t * t;
}

double QuinticPolynomial::calc_third_derivative(double t) {
    return 6 * a3 + 24 * a4 * t + 60 * a5 * t * t;
}
