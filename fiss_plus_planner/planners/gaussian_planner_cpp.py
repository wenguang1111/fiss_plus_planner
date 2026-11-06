"""Gaussian Planner: the Sparse Planner (SparsePlanner_CPP) with the CVAE replaced by a conditional Gaussian / Gaussian
mixture p([t, d, v] | 3 BEV frames) (planners/sparse_planning/cond_gaussian.py, trained by scripts/train_sp_sampler.py
on the same SP data as the paper CVAE). Everything else is the Sparse Planner's: the BEV images and their paper
preprocessing, N samples per cycle evaluated by the C++ FOP pipeline (best_traj_generation), the FOP grid fallback,
the timing (image generation is input, the inference from the H2D copy on is planning time). Samples are not clipped
to the FOP grid range (as the CVAE samples)."""
from pathlib import Path

import numpy as np
import torch

from fiss_plus_planner.planners.common.vehicle.vehicle import Vehicle
from fiss_plus_planner.planners.sparse_planner_cpp import SparsePlannerSettings_CPP, SparsePlanner_CPP
from fiss_plus_planner.planners.sparse_planning.cond_gaussian import CondGaussian
from fiss_plus_planner.planners.sparse_planning.cvae_paper import CVAE_EfficientPaper


class GaussianPlannerSettings_CPP(SparsePlannerSettings_CPP):
    def __init__(self, num_width: int = 5, num_speed: int = 5, num_t: int = 5, scenario_dir: str = "", scenario_file: str = ""):
        super().__init__(num_width, num_speed, num_t, scenario_dir, scenario_file)
        # checkpoint of train_sp_sampler.py (best.pt: model state, args with K / min_std, target statistics)
        self.gaussian_weights = Path(__file__).parent.parent / "data/output/sp_sampler/runs/gaussian_s0/best.pt"


class GaussianSampler:
    """Loads a train_sp_sampler.py checkpoint; the interface of CVAE_EfficientPaper used by SparsePlanner_CPP
    (condition_tensor, generate_from_tensor, img_dim, frame_size, device)."""

    def __init__(self, device: torch.device, weights_path: str):
        self.device, self.model_path = device, weights_path
        self.img_dim, self.frame_size = 256, 3
        ckpt = torch.load(weights_path, map_location=device)
        args = ckpt["args"]
        self.n_components = 1 if args["model"] == "gaussian" else args["K"]
        self.model = CondGaussian(n_components=self.n_components, min_std=args["min_std"], pretrained=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(device).eval()
        self.TARGET_MEAN, self.TARGET_STD = ckpt["target_mean"], ckpt["target_std"]
        # the same preprocessing as the paper CVAE (initialize_transforms does not use the instance)
        self.imgs_transforms = CVAE_EfficientPaper.initialize_transforms(self, self.img_dim)

    def condition_tensor(self, imgs_list) -> torch.Tensor:
        return CVAE_EfficientPaper.condition_tensor(self, imgs_list)

    def generate_from_tensor(self, imgs_tensor: torch.Tensor, num_samples: int) -> np.ndarray:
        """(num_samples, 3) array of [t, d, v] for a preprocessed condition tensor."""
        with torch.inference_mode():
            x = self.model.generate(imgs_tensor.to(self.device), num_samples).cpu().numpy().astype(np.float64)
        return x * np.array(self.TARGET_STD) + np.array(self.TARGET_MEAN)


class GaussianGraphSampler:
    """GaussianSampler's inference as one CUDA graph for a fixed num_samples (as CVAEGraphSampler): encoder, head and
    sampling from noise buffers (normal eps, uniform for the mixture component), optionally in fp16 autocast for the
    network (the sampling arithmetic in fp32)."""

    def __init__(self, sampler: GaussianSampler, num_samples: int, half: bool = True):
        self.device, self.num_samples, self.half = sampler.device, num_samples, half
        self.model = sampler.model.eval()
        self.mean = np.array(sampler.TARGET_MEAN, dtype=np.float64)
        self.std = np.array(sampler.TARGET_STD, dtype=np.float64)
        img_dim, frames = sampler.img_dim, sampler.frame_size
        self.cond_host = torch.zeros(1, frames, img_dim, img_dim).pin_memory()
        self.out_host = torch.empty(num_samples, 3).pin_memory()
        self.cond_dev = torch.zeros(1, frames, img_dim, img_dim, device=self.device)
        self.eps = torch.zeros(num_samples, 3, device=self.device)
        self.uniform = torch.full((num_samples, sampler.n_components), 0.5, device=self.device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(0)

        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.inference_mode():
            with torch.cuda.stream(stream):
                for _ in range(3):  # warm up outside the capture
                    self._forward()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out_dev = self._forward()

    def _forward(self) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
            tokens = self.model.encode(self.cond_dev)
        return self.model.sample_from_noise(tokens, self.eps, self.uniform)

    def set_condition(self, imgs_tensor: torch.Tensor):
        self.cond_host.copy_(imgs_tensor)

    def sample(self) -> np.ndarray:
        """(num_samples, 3) array of [t, d, v] for the condition of set_condition."""
        with torch.inference_mode():
            self.cond_dev.copy_(self.cond_host, non_blocking=True)
            self.eps.normal_(generator=self.generator)
            self.uniform.uniform_(generator=self.generator)
            self.graph.replay()
            self.out_host.copy_(self.out_dev, non_blocking=True)
            torch.cuda.current_stream(self.device).synchronize()
        return self.out_host.numpy().astype(np.float64) * self.std + self.mean


class GaussianTRTSampler:
    """GaussianGraphSampler's inference as a TensorRT engine (as CVAETRTSampler for the CVAE): ONNX export of
    encoder (fp16 copy of the model) -> head and sampling from the noise buffers (fp32, as the autocast graph) for a
    fixed num_samples; TensorRT 11 builds strongly typed networks, so the precision is that of the exported model.
    Per cycle: H2D copy of the fp16 condition, noise, one engine execution, D2H copy of the samples. The engine is
    built on first use and cached in cache_dir (per weights file, num_samples, TensorRT version and GPU)."""

    def __init__(self, sampler: GaussianSampler, num_samples: int, cache_dir: str):
        import copy, hashlib, os
        import tensorrt as trt
        self.device, self.num_samples = sampler.device, num_samples
        self.mean = np.array(sampler.TARGET_MEAN, dtype=np.float64)
        self.std = np.array(sampler.TARGET_STD, dtype=np.float64)
        img_dim, frames, K = sampler.img_dim, sampler.frame_size, sampler.n_components

        weights = hashlib.sha1(open(sampler.model_path, "rb").read()).hexdigest()[:12]
        gpu = torch.cuda.get_device_name(self.device).replace(" ", "_")
        os.makedirs(cache_dir, exist_ok=True)
        engine_path = os.path.join(cache_dir, f"gauss_k{K}_{weights}_n{num_samples}_trt{trt.__version__}_{gpu}.engine")
        logger = trt.Logger(trt.Logger.ERROR)
        if not os.path.exists(engine_path):
            onnx_path = engine_path.replace(".engine", ".onnx")
            head = copy.deepcopy(sampler.model).eval().requires_grad_(False).float()
            enc = copy.deepcopy(sampler.model.cond).eval().requires_grad_(False).half()

            class Forward(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.head, self.enc = head, enc

                def forward(self, cond, eps, uniform):
                    return self.head.sample_from_noise(self.enc(cond).float(), eps, uniform)

            example = (torch.zeros(1, frames, img_dim, img_dim, device=self.device, dtype=torch.float16),
                       torch.zeros(num_samples, 3, device=self.device),
                       torch.full((num_samples, K), 0.5, device=self.device))
            with torch.no_grad():
                torch.onnx.export(Forward(), example, onnx_path, input_names=["cond", "eps", "uniform"],
                                  output_names=["out"], opset_version=17, dynamo=False)
            builder = trt.Builder(logger)
            network = builder.create_network(0)
            parser = trt.OnnxParser(network, logger)
            if not parser.parse(open(onnx_path, "rb").read()):
                raise RuntimeError([parser.get_error(i) for i in range(parser.num_errors)])
            open(engine_path, "wb").write(builder.build_serialized_network(network, builder.create_builder_config()))
            os.remove(onnx_path)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(open(engine_path, "rb").read())
        self.context = self.engine.create_execution_context()

        self.cond_host = torch.zeros(1, frames, img_dim, img_dim, dtype=torch.float16).pin_memory()
        self.out_host = torch.empty(num_samples, 3).pin_memory()
        self.cond_dev = torch.zeros(1, frames, img_dim, img_dim, device=self.device, dtype=torch.float16)
        self.eps = torch.zeros(num_samples, 3, device=self.device)
        self.uniform = torch.full((num_samples, K), 0.5, device=self.device)
        self.out_dev = torch.empty(num_samples, 3, device=self.device)
        for name, tensor in [("cond", self.cond_dev), ("eps", self.eps), ("uniform", self.uniform), ("out", self.out_dev)]:
            if name == "uniform" and name not in [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]:
                continue  # K = 1: the export drops the unused input
            self.context.set_tensor_address(name, tensor.data_ptr())
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(0)
        for _ in range(3):  # warm up
            self.sample()

    def set_condition(self, imgs_tensor: torch.Tensor):
        self.cond_host.copy_(imgs_tensor)

    def sample(self) -> np.ndarray:
        """(num_samples, 3) array of [t, d, v] for the condition of set_condition."""
        stream = torch.cuda.current_stream(self.device)
        with torch.inference_mode():
            self.cond_dev.copy_(self.cond_host, non_blocking=True)
            self.eps.normal_(generator=self.generator)
            self.uniform.uniform_(generator=self.generator)
            self.context.execute_async_v3(stream.cuda_stream)
            self.out_host.copy_(self.out_dev, non_blocking=True)
            stream.synchronize()
        return self.out_host.numpy().astype(np.float64) * self.std + self.mean


class GaussianPlanner_CPP(SparsePlanner_CPP):
    def __init__(self, planner_settings: GaussianPlannerSettings_CPP, ego_vehicle: Vehicle,
                 obstacles_array=None, obstacles_num_vertices=None, number_threads: int = 1, runtime_measurement: bool = False):
        if planner_settings.sampler != "cvae":
            raise ValueError("GaussianPlanner_CPP samples from its model (SP_SAMPLER cvae)")
        super().__init__(planner_settings, ego_vehicle, obstacles_array, obstacles_num_vertices, number_threads, runtime_measurement)

    def load_sampler_model(self):
        return GaussianSampler(self.settings.device, str(self.settings.gaussian_weights))

    def build_graph_sampler(self):
        if self.settings.device.type != "cuda" or self.settings.cvae_backend not in ("graph_fp16", "graph_fp32", "trt_fp16"):
            raise ValueError("GaussianPlanner_CPP: SP_CVAE_BACKEND graph_fp16 / graph_fp32 / trt_fp16 (CUDA) or eager")
        if self.settings.cvae_backend == "trt_fp16":
            return GaussianTRTSampler(self.cvae_efficient_model, self.settings.num_samples, str(self.settings.trt_cache_dir))
        return GaussianGraphSampler(self.cvae_efficient_model, self.settings.num_samples,
                                    half=self.settings.cvae_backend == "graph_fp16")
