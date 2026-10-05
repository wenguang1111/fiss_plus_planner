#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "Fiss_Plus_Planner.h"
#include "common/scenario/frenet.h"

namespace py = pybind11;

PYBIND11_MODULE(fiss_plus_planner_cpp, m) {
    m.doc() = "FISS+ Planner C++ extension";

    // Only register FISS+ specific types to avoid conflicts with frenet_planner_cpp

    // Bind FissPlusPlannerSettings
    py::class_<FissPlusPlannerSettings>(m, "FissPlusPlannerSettings")
        .def(py::init<int, int, int, int>(),
             py::arg("num_width") = 5,
             py::arg("num_speed") = 5,
             py::arg("num_t") = 5,
             py::arg("refine_iters") = 3)
        .def_readwrite("tick_t", &FissPlusPlannerSettings::tick_t)
        .def_readwrite("max_road_width", &FissPlusPlannerSettings::max_road_width)
        .def_readwrite("num_width", &FissPlusPlannerSettings::num_width)
        .def_readwrite("highest_speed", &FissPlusPlannerSettings::highest_speed)
        .def_readwrite("lowest_speed", &FissPlusPlannerSettings::lowest_speed)
        .def_readwrite("num_speed", &FissPlusPlannerSettings::num_speed)
        .def_readwrite("min_t", &FissPlusPlannerSettings::min_t)
        .def_readwrite("max_t", &FissPlusPlannerSettings::max_t)
        .def_readwrite("num_t", &FissPlusPlannerSettings::num_t)
        .def_readwrite("check_obstacle", &FissPlusPlannerSettings::check_obstacle)
        .def_readwrite("check_boundary", &FissPlusPlannerSettings::check_boundary)
        .def_readwrite("refine_trajectory", &FissPlusPlannerSettings::refine_trajectory)
        .def_readwrite("max_refine_iters", &FissPlusPlannerSettings::max_refine_iters)
        .def_readwrite("has_time_limit", &FissPlusPlannerSettings::has_time_limit)
        .def_readwrite("time_limit", &FissPlusPlannerSettings::time_limit)
        .def_readwrite("decaying_factor", &FissPlusPlannerSettings::decaying_factor)
        .def_readwrite("w_heuristic", &FissPlusPlannerSettings::w_heuristic)
        .def_readwrite("vis_all_candidates", &FissPlusPlannerSettings::vis_all_candidates);

    // Bind Fiss_Plus_Planner class
    py::class_<Fiss_Plus_Planner>(m, "FissPlusPlanner")
        .def(py::init([](const FissPlusPlannerSettings& settings,
                         py::object vehicle,
                         py::array_t<double> obstacles_array,
                         py::array_t<int> num_vertices_array,
                         int num_time_steps,
                         int num_obstacles,
                         int max_vertices) {
            VehicleParams vp;
            vp.l = vehicle.attr("l").cast<double>();
            vp.w = vehicle.attr("w").cast<double>();
            vp.a = vehicle.attr("a").cast<double>();
            vp.b = vehicle.attr("b").cast<double>();
            vp.T_f = vehicle.attr("T_f").cast<double>();
            vp.T_r = vehicle.attr("T_r").cast<double>();
            vp.max_speed = vehicle.attr("max_speed").cast<double>();
            vp.max_accel = vehicle.attr("max_accel").cast<double>();
            vp.max_steering_angle = vehicle.attr("max_steering_angle").cast<double>();
            vp.max_steering_rate = vehicle.attr("max_steering_rate").cast<double>();
            
            auto obs_buf = obstacles_array.request();
            auto num_verts_buf = num_vertices_array.request();
            return new Fiss_Plus_Planner(
                settings,
                vp,
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
             [](Fiss_Plus_Planner& self, py::array_t<double> centerline_pts) {
                auto buf = centerline_pts.request();
                int num_points = buf.shape[0];
                int pts_dim = buf.shape[1];
                self.generate_frenet_frame(static_cast<double*>(buf.ptr), num_points, pts_dim);
             },
             py::arg("centerline_pts"))
        .def("set_road_profile", &Fiss_Plus_Planner::set_road_profile,
             py::arg("s"), py::arg("lane_width"), py::arg("left_extent"), py::arg("right_extent"))
        .def("plan",
             [](Fiss_Plus_Planner& self,
                py::object frenet_state,
                double max_target_speed,
                int time_step_now,
                double desired_speed) {
                FrenetState fs;
                fs.t = frenet_state.attr("t").cast<double>();
                fs.s = frenet_state.attr("s").cast<double>();
                fs.s_d = frenet_state.attr("s_d").cast<double>();
                fs.s_dd = frenet_state.attr("s_dd").cast<double>();
                fs.s_ddd = frenet_state.attr("s_ddd").cast<double>();
                fs.d = frenet_state.attr("d").cast<double>();
                fs.d_d = frenet_state.attr("d_d").cast<double>();
                fs.d_dd = frenet_state.attr("d_dd").cast<double>();
                fs.d_ddd = frenet_state.attr("d_ddd").cast<double>();
                
                FrenetTrajectory traj = self.plan(fs, max_target_speed, time_step_now, desired_speed);
                
                py::dict d;
                d["t"] = traj.t;
                d["s"] = traj.s;
                d["s_d"] = traj.s_d;
                d["s_dd"] = traj.s_dd;
                d["s_ddd"] = traj.s_ddd;
                d["d"] = traj.d;
                d["d_d"] = traj.d_d;
                d["d_dd"] = traj.d_dd;
                d["d_ddd"] = traj.d_ddd;
                d["x"] = traj.x;
                d["y"] = traj.y;
                d["yaw"] = traj.yaw;
                d["ds"] = traj.ds;
                d["c"] = traj.c;
                d["c_d"] = traj.c_d;
                d["c_dd"] = traj.c_dd;
                d["cost_final"] = traj.cost_final;
                d["is_generated"] = traj.is_generated;
                d["idx"] = std::vector<int>{traj.idx[0], traj.idx[1], traj.idx[2]};
                py::dict sampling_param_dict;
                sampling_param_dict["d"] = traj.sampling_param.d;
                sampling_param_dict["s_d"] = traj.sampling_param.s_d;
                sampling_param_dict["t"] = traj.sampling_param.t;
                d["sampling_param"] = sampling_param_dict;
                return d;
             },
             py::arg("frenet_state"),
             py::arg("max_target_speed"),
             py::arg("time_step_now") = 0,
             py::arg("desired_speed") = -1.0)
        .def("getAllSuccessfulTrajectories", 
             [](Fiss_Plus_Planner& self) {
                auto trajs = self.getAllSuccessfulTrajectories();
                py::list result;
                for (const auto& traj : trajs) {
                    py::dict d;
                    d["t"] = traj.t;
                    d["s"] = traj.s;
                    d["s_d"] = traj.s_d;
                    d["s_dd"] = traj.s_dd;
                    d["s_ddd"] = traj.s_ddd;
                    d["d"] = traj.d;
                    d["d_d"] = traj.d_d;
                    d["d_dd"] = traj.d_dd;
                    d["d_ddd"] = traj.d_ddd;
                    d["x"] = traj.x;
                    d["y"] = traj.y;
                    d["yaw"] = traj.yaw;
                    d["ds"] = traj.ds;
                    d["c"] = traj.c;
                    d["c_d"] = traj.c_d;
                    d["c_dd"] = traj.c_dd;
                    d["cost_final"] = traj.cost_final;
                    d["is_generated"] = traj.is_generated;
                    d["idx"] = std::vector<int>{traj.idx[0], traj.idx[1], traj.idx[2]};
                    py::dict sampling_param_dict;
                    sampling_param_dict["d"] = traj.sampling_param.d;
                    sampling_param_dict["s_d"] = traj.sampling_param.s_d;
                    sampling_param_dict["t"] = traj.sampling_param.t;
                    d["sampling_param"] = sampling_param_dict;
                    result.append(d);
                }
                return result;
             })
        .def("get_stats", 
             [](Fiss_Plus_Planner& self) {
                auto stats = self.get_stats();
                py::dict d;
                d["num_search_iterations"] = stats.num_search_iterations;
                d["num_trajs_generated"] = stats.num_trajs_generated;
                d["num_global_transforms"] = stats.num_global_transforms;
                d["num_rejected_transform"] = stats.num_rejected_transform;
                d["num_constraint_checks"] = stats.num_constraint_checks;
                d["num_constraint_passed"] = stats.num_constraint_passed;
                d["num_rejected_dynamic"] = stats.num_rejected_dynamic;
                d["num_rejected_offroad"] = stats.num_rejected_offroad;
                d["num_collision_checks"] = stats.num_collision_checks;
                d["num_collision_free"] = stats.num_collision_free;
                d["num_rejected_collision"] = stats.num_rejected_collision();
                d["num_cost_evaluations"] = stats.num_cost_evaluations;
                d["num_FOP_intervention"] = stats.num_FOP_intervention;
                py::dict timing;
                timing["sampling_ms"] = stats.timing.sampling_ms;
                timing["generation_ms"] = stats.timing.generation_ms;
                timing["transform_ms"] = stats.timing.transform_ms;
                timing["constraint_ms"] = stats.timing.constraint_ms;
                timing["collision_ms"] = stats.timing.collision_ms;
                timing["cost_ms"] = stats.timing.cost_ms;
                timing["total_ms"] = stats.timing.total_ms;
                d["timing"] = timing;
                return d;
             })
        .def_readwrite("fiss_settings", &Fiss_Plus_Planner::fiss_settings);
}
