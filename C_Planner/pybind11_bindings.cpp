#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "Frenet_Planner.h"
#include "common/scenario/frenet.h"

namespace py = pybind11;

PYBIND11_MODULE(frenet_planner_cpp, m) {
    m.doc() = "Frenet Optimal Planner C++ extension";

    // Bind SettingParameters struct
    py::class_<SettingParameters>(m, "SettingParameters")
        .def(py::init<int, int, int>(), 
             py::arg("num_width") = 5,
             py::arg("num_speed") = 5,
             py::arg("num_t") = 5)
        .def_readwrite("tick_t", &SettingParameters::tick_t)
        .def_readwrite("max_road_width", &SettingParameters::max_road_width)
        .def_readwrite("num_width", &SettingParameters::num_width)
        .def_readwrite("highest_speed", &SettingParameters::highest_speed)
        .def_readwrite("lowest_speed", &SettingParameters::lowest_speed)
        .def_readwrite("num_speed", &SettingParameters::num_speed)
        .def_readwrite("min_t", &SettingParameters::min_t)
        .def_readwrite("max_t", &SettingParameters::max_t)
        .def_readwrite("num_t", &SettingParameters::num_t)
        .def_readwrite("check_obstacle", &SettingParameters::check_obstacle)
        .def_readwrite("check_boundary", &SettingParameters::check_boundary);

    // Bind VehicleParams struct
    py::class_<VehicleParams>(m, "VehicleParams")
        .def(py::init<>())
        .def_readwrite("l", &VehicleParams::l)
        .def_readwrite("w", &VehicleParams::w)
        .def_readwrite("a", &VehicleParams::a)
        .def_readwrite("b", &VehicleParams::b)
        .def_readwrite("T_f", &VehicleParams::T_f)
        .def_readwrite("T_r", &VehicleParams::T_r)
        .def_readwrite("max_speed", &VehicleParams::max_speed)
        .def_readwrite("max_accel", &VehicleParams::max_accel)
        .def_readwrite("max_steering_angle", &VehicleParams::max_steering_angle)
        .def_readwrite("max_steering_rate", &VehicleParams::max_steering_rate);

    // Bind FrenetState struct
    py::class_<FrenetState>(m, "FrenetState")
        .def(py::init<>())
        .def_readwrite("t", &FrenetState::t)
        .def_readwrite("s", &FrenetState::s)
        .def_readwrite("s_d", &FrenetState::s_d)
        .def_readwrite("s_dd", &FrenetState::s_dd)
        .def_readwrite("s_ddd", &FrenetState::s_ddd)
        .def_readwrite("d", &FrenetState::d)
        .def_readwrite("d_d", &FrenetState::d_d)
        .def_readwrite("d_dd", &FrenetState::d_dd)
        .def_readwrite("d_ddd", &FrenetState::d_ddd);

    // Bind SamplingParam struct
    py::class_<SamplingParam>(m, "SamplingParam")
        .def(py::init<double, double, double>(),
             py::arg("d") = 0.0,
             py::arg("s_d") = 0.0,
             py::arg("t") = 0.0)
        .def_readwrite("d", &SamplingParam::d)
        .def_readwrite("s_d", &SamplingParam::s_d)
        .def_readwrite("t", &SamplingParam::t);

    // Bind FrenetTrajectory struct
    py::class_<FrenetTrajectory>(m, "FrenetTrajectory")
        .def(py::init<>())
        .def_readwrite("cost_final", &FrenetTrajectory::cost_final)
        .def_readwrite("is_generated", &FrenetTrajectory::is_generated)
        .def_readwrite("constraint_passed", &FrenetTrajectory::constraint_passed)
        .def_readwrite("collision_passed", &FrenetTrajectory::collision_passed)
        // Frenet frame coordinates
        .def_readwrite("t", &FrenetTrajectory::t)
        .def_readwrite("s", &FrenetTrajectory::s)
        .def_readwrite("s_d", &FrenetTrajectory::s_d)
        .def_readwrite("s_dd", &FrenetTrajectory::s_dd)
        .def_readwrite("s_ddd", &FrenetTrajectory::s_ddd)
        .def_readwrite("d", &FrenetTrajectory::d)
        .def_readwrite("d_d", &FrenetTrajectory::d_d)
        .def_readwrite("d_dd", &FrenetTrajectory::d_dd)
        .def_readwrite("d_ddd", &FrenetTrajectory::d_ddd)
        // World frame coordinates
        .def_readwrite("x", &FrenetTrajectory::x)
        .def_readwrite("y", &FrenetTrajectory::y)
        .def_readwrite("yaw", &FrenetTrajectory::yaw)
        .def_readwrite("ds", &FrenetTrajectory::ds)
        .def_readwrite("c", &FrenetTrajectory::c)
        .def_readwrite("c_d", &FrenetTrajectory::c_d)
        .def_readwrite("c_dd", &FrenetTrajectory::c_dd)
        // Sampling parameters
        .def_readwrite("sampling_param", &FrenetTrajectory::sampling_param);

    // Bind TimingStats struct
    py::class_<TimingStats>(m, "TimingStats")
        .def(py::init<>())
        .def_readwrite("generation_ms", &TimingStats::generation_ms)
        .def_readwrite("transform_ms", &TimingStats::transform_ms)
        .def_readwrite("constraint_ms", &TimingStats::constraint_ms)
        .def_readwrite("collision_ms", &TimingStats::collision_ms)
        .def_readwrite("cost_ms", &TimingStats::cost_ms)
        .def_readwrite("total_ms", &TimingStats::total_ms);

    // Bind PlanStats struct
    py::class_<PlanStats>(m, "PlanStats")
        .def(py::init<>())
        .def_readwrite("num_search_iterations", &PlanStats::num_search_iterations)
        .def_readwrite("num_trajs_generated", &PlanStats::num_trajs_generated)
        .def_readwrite("num_global_transforms", &PlanStats::num_global_transforms)
        .def_readwrite("num_rejected_transform", &PlanStats::num_rejected_transform)
        .def_readwrite("num_constraint_checks", &PlanStats::num_constraint_checks)
        .def_readwrite("num_constraint_passed", &PlanStats::num_constraint_passed)
        .def_readwrite("num_rejected_dynamic", &PlanStats::num_rejected_dynamic)
        .def_readwrite("num_rejected_offroad", &PlanStats::num_rejected_offroad)
        .def_readwrite("num_collision_checks", &PlanStats::num_collision_checks)
        .def_readwrite("num_collision_free", &PlanStats::num_collision_free)
        .def_property_readonly("num_rejected_collision", &PlanStats::num_rejected_collision)
        .def_readwrite("num_cost_evaluations", &PlanStats::num_cost_evaluations)
        .def_readwrite("num_FOP_intervention", &PlanStats::num_FOP_intervention)
        .def_readwrite("timing", &PlanStats::timing);

    // Bind CostBreakdown struct (weighted terms of Eq. 4 / Eq. 7-8)
    py::class_<CostBreakdown>(m, "CostBreakdown")
        .def(py::init<>())
        .def_readwrite("time", &CostBreakdown::time)
        .def_readwrite("velocity", &CostBreakdown::velocity)
        .def_readwrite("acceleration", &CostBreakdown::acceleration)
        .def_readwrite("jerk", &CostBreakdown::jerk)
        .def_readwrite("lane_center", &CostBreakdown::lane_center)
        .def_readwrite("obstacle", &CostBreakdown::obstacle)
        .def("running", &CostBreakdown::running)
        .def("total", &CostBreakdown::total);

    py::class_<ConstraintViolation>(m, "ConstraintViolation")
        .def(py::init<>())
        .def_readwrite("speed", &ConstraintViolation::speed)
        .def_readwrite("acceleration", &ConstraintViolation::acceleration)
        .def_readwrite("road", &ConstraintViolation::road)
        .def_readwrite("collision", &ConstraintViolation::collision)
        .def_readwrite("transform", &ConstraintViolation::transform)
        .def("total", &ConstraintViolation::total);

    py::enum_<Rejection>(m, "Rejection")
        .value("NONE", Rejection::kNone)
        .value("TRANSFORM", Rejection::kTransform)
        .value("DYNAMIC", Rejection::kDynamic)
        .value("OFFROAD", Rejection::kOffroad)
        .value("COLLISION", Rejection::kCollision);

    py::class_<EvaluationResult>(m, "EvaluationResult")
        .def(py::init<>())
        .def_readwrite("feasible", &EvaluationResult::feasible)
        .def_readwrite("rejection", &EvaluationResult::rejection)
        .def_readwrite("violation", &EvaluationResult::violation)
        .def_readwrite("has_cost", &EvaluationResult::has_cost)
        .def_readwrite("cost", &EvaluationResult::cost);

    m.def("ranks_before", &ranks_before, py::arg("a"), py::arg("b"));

    // Cost weights (defaults: CommonRoad WX1)
    py::class_<CostWeights>(m, "CostWeights")
        .def(py::init<>())
        .def_readwrite("w_T", &CostWeights::w_T)
        .def_readwrite("w_V", &CostWeights::w_V)
        .def_readwrite("w_A", &CostWeights::w_A)
        .def_readwrite("w_J", &CostWeights::w_J)
        .def_readwrite("w_D", &CostWeights::w_D)
        .def_readwrite("w_dist", &CostWeights::w_dist)
        .def_readwrite("w_LC", &CostWeights::w_LC);

    // Scenario-level evaluation of the executed trajectory, Eq. (7) + (8)
    m.def("scenario_cost",
          [](const FrenetTrajectory& executed, const std::vector<double>& v_des, double dt,
             py::array_t<double, py::array::c_style | py::array::forcecast> obstacles_array,
             py::array_t<int, py::array::c_style | py::array::forcecast> num_vertices_array,
             const CostWeights& weights) {
              auto obs_buf = obstacles_array.request();
              auto num_buf = num_vertices_array.request();
              if (obs_buf.ndim != 4 || obs_buf.shape[3] != 2 || num_buf.ndim != 2 ||
                  num_buf.shape[0] != obs_buf.shape[0] || num_buf.shape[1] != obs_buf.shape[1]) {
                  throw py::value_error("obstacles_array must be (T, O, V, 2) and num_vertices_array (T, O)");
              }
              ObstacleView obstacles;
              obstacles.vertices = static_cast<const double*>(obs_buf.ptr);
              obstacles.num_vertices = static_cast<const int*>(num_buf.ptr);
              obstacles.num_time_steps = static_cast<int>(obs_buf.shape[0]);
              obstacles.num_obstacles = static_cast<int>(obs_buf.shape[1]);
              obstacles.max_vertices = static_cast<int>(obs_buf.shape[2]);
              return CostFunction(weights).scenario_cost(executed, v_des, dt, obstacles);
          },
          py::arg("executed"), py::arg("v_des"), py::arg("dt"),
          py::arg("obstacles_array"), py::arg("num_vertices_array"),
          py::arg("weights") = CostWeights());

    // Bind Frenet_Planner class
    py::class_<Frenet_Planner>(m, "FrenetPlanner")
        .def(py::init([](const SettingParameters& settings,
                         const VehicleParams& vehicle,
                         py::array_t<double> obstacles_array,
                         py::array_t<int> num_vertices_array,
                         int num_time_steps,
                         int num_obstacles,
                         int max_vertices) {
            auto obs_buf = obstacles_array.request();
            auto num_verts_buf = num_vertices_array.request();
            return new Frenet_Planner(
                settings,
                vehicle,
                static_cast<double*>(obs_buf.ptr),
                static_cast<int*>(num_verts_buf.ptr),
                num_time_steps,
                num_obstacles,
                max_vertices
            );
        }),
             py::arg("settings"),
             py::arg("vehicle"),
             py::arg("obstacles_array"),
             py::arg("num_vertices_array"),
             py::arg("num_time_steps"),
             py::arg("num_obstacles"),
             py::arg("max_vertices"))
        .def("generate_frenet_frame", 
             [](Frenet_Planner& self, py::array_t<double, py::array::c_style | py::array::forcecast> centerline_pts) {
                auto buf = centerline_pts.request();
                if (buf.ndim != 2 || buf.shape[0] < 2 || buf.shape[1] != 2) {
                    throw py::value_error("centerline_pts must have shape (N, 2), N >= 2");
                }
                int num_points = buf.shape[0];
                int pts_dim = buf.shape[1];
                self.generate_frenet_frame(static_cast<double*>(buf.ptr), num_points, pts_dim);
             },
             py::arg("centerline_pts"))
        .def("set_road_profile", &Frenet_Planner::set_road_profile,
             py::arg("s"), py::arg("lane_width"), py::arg("left_extent"), py::arg("right_extent"))
        .def("plan",
             [](Frenet_Planner& self,
                const FrenetState& frenet_state,
                double max_target_speed,
                int time_step_now,
                int num_threads,
                double desired_speed) {
                auto best_traj = self.plan(frenet_state, max_target_speed, time_step_now, num_threads,
                                           desired_speed);
               return best_traj;
             },
             py::arg("frenet_state"),
             py::arg("max_target_speed"),
             py::arg("time_step_now") = 0,
             py::arg("num_threads") =1,
             py::arg("desired_speed") = -1.0)
        .def("best_traj_generation",
             [](Frenet_Planner& self,
                const FrenetState& frenet_state,
                const std::vector<std::tuple<double, double, double>>& samples,
                double max_target_speed,
                int time_step_now,
                int num_threads,
                double desired_speed) {
                return self.best_traj_generation(
                    frenet_state,
                    samples,
                    max_target_speed,
                    time_step_now,
                    num_threads,
                    desired_speed
                );
             },
             py::arg("frenet_state"),
             py::arg("samples"),
             py::arg("max_target_speed"),
             py::arg("time_step_now") = 0,
             py::arg("num_threads") = 1,
             py::arg("desired_speed") = -1.0)
        .def("get_samples", &Frenet_Planner::get_samples, py::arg("current_s") = 0.0)
        .def("evaluate_samples", &Frenet_Planner::evaluate_samples,
             py::arg("frenet_state"), py::arg("samples"), py::arg("max_target_speed"),
             py::arg("time_step_now") = 0, py::arg("full_violation") = true,
             py::arg("desired_speed") = -1.0)
        .def("getAllSuccessfulTrajectories", &Frenet_Planner::getAllSuccessfulTrajectories)
        .def("get_stats", &Frenet_Planner::get_stats)
        .def_readwrite("settings", &Frenet_Planner::settings)
        .def_readwrite("vehicle_params", &Frenet_Planner::vehicle_params)
        .def_readwrite("best_traj", &Frenet_Planner::best_traj);
}
