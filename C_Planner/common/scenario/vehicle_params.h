#ifndef VEHICLE_PARAMS_H
#define VEHICLE_PARAMS_H

// Vehicle parameters struct
struct VehicleParams {
    double l;          // length [m]
    double w;          // width [m]
    double a;          // distance from base_link to CoG [m]
    double b;          // distance from CoG to front_link [m]
    double T_f;        // front track width [m]
    double T_r;        // rear track width [m]
    double max_speed;  // maximum speed [m/s]
    double max_accel;  // maximum acceleration [m/ss]
    double max_steering_angle;  // maximum steering angle [rad]
    double max_steering_rate;   // maximum steering rate [rad/s]
};

#endif // VEHICLE_PARAMS_H
