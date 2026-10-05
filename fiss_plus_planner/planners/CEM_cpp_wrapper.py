from fiss_plus_planner.planners.FOP_cpp_wrapper import FOP_CPP_Wrapper, frenet_planner_cpp


class CEM_CPP_Wrapper(FOP_CPP_Wrapper):
    """FOP backend with a cross-entropy sampler over (d, s_d, t), see C_Planner/CEM_Planner.h.

    `cem_cfg` sets fields of the C++ CEMSettings by name, e.g. {"num_iterations": 4,
    "population": 250, "elite_fraction": 0.1, "smoothing": 0.7, "init_std": 0.3, "seed": 0}.
    """

    def __init__(self, *args, cem_cfg: dict = None, **kwargs):
        self.cem_cfg = cem_cfg or {}
        super().__init__(*args, **kwargs)

    def _create_cpp_planner(self, cpp_settings, cpp_vehicle, *obstacle_args):
        cem_settings = frenet_planner_cpp.CEMSettings()
        for name, value in self.cem_cfg.items():
            setattr(cem_settings, name, value)
        return frenet_planner_cpp.CEMPlanner(cpp_settings, cem_settings, cpp_vehicle, *obstacle_args)
