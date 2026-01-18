#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include <vector>
#include <cmath>
#include <Eigen/Dense>

class QuarticPolynomial {
public:
    double a0, a1, a2, a3, a4;
    
    QuarticPolynomial(double xs, double vxs, double axs, double vxe, double axe, double time);
    
    double calc_point(double t);
    double calc_first_derivative(double t);
    double calc_second_derivative(double t);
    double calc_third_derivative(double t);
};

class QuinticPolynomial {
public:
    double a0, a1, a2, a3, a4, a5;
    
    QuinticPolynomial(double xs, double vxs, double axs, double xe, double vxe, double axe, double time);
    
    double calc_point(double t);
    double calc_first_derivative(double t);
    double calc_second_derivative(double t);
    double calc_third_derivative(double t);
};

#endif // POLYNOMIAL_H
