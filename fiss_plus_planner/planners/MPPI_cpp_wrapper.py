from fiss_plus_planner.planners.FOP_cpp_wrapper import FOP_CPP_Wrapper, frenet_planner_cpp


class MPPI_CPP_Wrapper(FOP_CPP_Wrapper):
    """FOP backend with an MPPI-style sampler over (d, s_d, t), see C_Planner/MPPI_Planner.h.

    `mppi_cfg` sets fields of the C++ MPPISettings by name, e.g. {"num_iterations": 4,
    "population": 250, "temperature": 0.1, "learning_rate": 0.7, "init_std": 0.3, "seed": 0}.
    """

    def __init__(self, *args, mppi_cfg: dict = None, **kwargs):
        self.mppi_cfg = mppi_cfg or {}
        super().__init__(*args, **kwargs)

    def _create_cpp_planner(self, cpp_settings, cpp_vehicle, *obstacle_args):
        mppi_settings = frenet_planner_cpp.MPPISettings()
        for name, value in self.mppi_cfg.items():
            setattr(mppi_settings, name, value)
        return frenet_planner_cpp.MPPIPlanner(cpp_settings, mppi_settings, cpp_vehicle, *obstacle_args)
