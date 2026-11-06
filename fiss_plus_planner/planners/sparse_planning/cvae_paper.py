import numpy as np
import torch
from torchvision import transforms

from CVAE_efficient_sampling.CVAE import CVAE_Efficient
from CVAE_efficient_sampling.normalizer import Normalizer


class CVAE_EfficientPaper(CVAE_Efficient):
    """The CVAE sampler as it ran for the measurements of the Sparse Planner paper (2026-02-28, laptop working copy
    in ~/backup/workplace/fiss_plus_planner/CVAE_efficient_sampling/{CVAE,attn_cvae}.py): Resize to img_dim, the
    target statistics below, and the condition image encoded once for all samples (the submodule's attnCVAE.generate
    encodes num_samples copies: 246 ms instead of 2 ms for 1000 samples, same result). The submodule (b382f22 and
    later) has other statistics ([4.37, -0.0023, 10.87]), which make the same weights sample faster and more often
    infeasible end states."""

    # Dataset statistics [t, d, v]
    TARGET_MEAN = [3.76989018, -1.97695157e-3, 8.92879236]
    TARGET_STD = [0.95395955, 0.14355468, 4.87223273]

    def initialize_transforms(self, img_dim: int) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((img_dim, img_dim)),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Lambda(lambda x: 1.0 - x),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def condition_tensor(self, imgs_list) -> torch.Tensor:
        """Image preprocessing: the frames as one (1, frame_size, img_dim, img_dim) CPU tensor."""
        return torch.cat([self.imgs_transforms(img).unsqueeze(0) for img in imgs_list[0:self.frame_size]], dim=1)

    def generate_samples(self, imgs_list, num_samples):
        return self.generate_from_tensor(self.condition_tensor(imgs_list), num_samples).tolist()

    def generate_from_tensor(self, imgs_tensor: torch.Tensor, num_samples: int) -> np.ndarray:
        """(num_samples, 3) array of [t, d, v] for a preprocessed condition tensor."""
        self.model.eval()
        normalizer = Normalizer()
        with torch.inference_mode():
            parameters_normalized = self.generate(imgs_tensor.to(self.device), num_samples)
            normalizer.load_from_stats(mean=self.TARGET_MEAN, std=self.TARGET_STD)
            return normalizer.inverse_transform_targets(parameters_normalized.cpu().numpy())

    def generate(self, c: torch.Tensor, batch: int) -> torch.Tensor:
        """attnCVAE.generate of the paper's working copy: condition tokens computed once and repeated."""
        model = self.model
        cond_tokens = model.cond(c.to(self.device)).repeat(batch, 1, 1)
        mu, logvar = model.pri(cond_tokens)
        z = model.reparameterize(mu, logvar)
        return model.dec(model.lat(z), cond_tokens)


class CVAEGraphSampler:
    """The same inference as CVAE_EfficientPaper.generate (one encoding of the condition, prior, decoder for
    num_samples latents) captured as one CUDA graph for a fixed num_samples, optionally in fp16 autocast: per cycle
    only the H2D copy of the condition, the latent noise, one graph replay and the D2H copy of the samples are
    launched (eager fp32 1.23 ms -> 0.64 ms for 64 samples on an RTX 5070 Ti; fp16 changes the samples by at most
    0.002 s / 0.2 mm / 0.017 m/s). Other random draws than the eager version, so not bit-identical to it."""

    def __init__(self, cvae: CVAE_EfficientPaper, num_samples: int, half: bool = True):
        self.device = cvae.device
        self.num_samples = num_samples
        self.half = half
        self.model = cvae.model.eval()
        self.mean = np.array(cvae.TARGET_MEAN, dtype=np.float64)
        self.std = np.array(cvae.TARGET_STD, dtype=np.float64)
        img_dim, frames = cvae.img_dim, cvae.frame_size

        self.cond_host = torch.zeros(1, frames, img_dim, img_dim).pin_memory()
        self.out_host = torch.empty(num_samples, 3).pin_memory()
        self.cond_dev = torch.zeros(1, frames, img_dim, img_dim, device=self.device)
        self.eps = torch.zeros(num_samples, cvae.latent_dim, device=self.device)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(0)

        stream = torch.cuda.Stream(self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.inference_mode():
            with torch.cuda.stream(stream):
                for _ in range(3):  # warm up (cuDNN / cuBLAS handles) outside the capture
                    self._forward()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.out_dev = self._forward()

    def _forward(self) -> torch.Tensor:
        model = self.model
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.half):
            cond_tokens = model.cond(self.cond_dev)
            mu, logvar = model.pri(cond_tokens)
            z = mu + self.eps * torch.exp(0.5 * logvar)
            return model.dec(model.lat(z), cond_tokens.expand(self.num_samples, -1, -1)).float()

    def set_condition(self, imgs_tensor: torch.Tensor):
        """Image preprocessing: the condition into pinned host memory (not part of the inference)."""
        self.cond_host.copy_(imgs_tensor)

    def sample(self) -> np.ndarray:
        """(num_samples, 3) array of [t, d, v] for the condition of set_condition."""
        with torch.inference_mode():
            self.cond_dev.copy_(self.cond_host, non_blocking=True)
            self.eps.normal_(generator=self.generator)
            self.graph.replay()
            self.out_host.copy_(self.out_dev, non_blocking=True)
            torch.cuda.current_stream(self.device).synchronize()
        return self.out_host.numpy().astype(np.float64) * self.std + self.mean


class CVAETRTSampler:
    """The inference of CVAEGraphSampler as a TensorRT engine of the fp16 model (fp16 ONNX export of cond -> pri ->
    z = mu + eps*exp(0.5*logvar) -> dec for a fixed num_samples; TensorRT 11 builds strongly typed networks, so the
    precision is that of the exported model). Per cycle: H2D copy of the fp16 condition, latent noise, one engine
    execution, D2H copy of the samples. GPU time on an RTX 5070 Ti, N = 1 / 64 / 1024: 0.18 / 0.20 / 0.55 ms (graph_fp16
    0.62 / 0.66 / 1.21 ms); samples within 0.002 s / 0.6 mm / 0.019 m/s of the fp32 model for the same noise. The engine
    is built on first use and cached in cache_dir (per weights file, num_samples, TensorRT version and GPU)."""

    def __init__(self, cvae: CVAE_EfficientPaper, num_samples: int, cache_dir: str):
        import copy, hashlib, os
        import tensorrt as trt
        self.device = cvae.device
        self.num_samples = num_samples
        self.mean = np.array(cvae.TARGET_MEAN, dtype=np.float64)
        self.std = np.array(cvae.TARGET_STD, dtype=np.float64)
        img_dim, frames = cvae.img_dim, cvae.frame_size

        weights = hashlib.sha1(open(cvae.model_path, "rb").read()).hexdigest()[:12]
        gpu = torch.cuda.get_device_name(self.device).replace(" ", "_")
        os.makedirs(cache_dir, exist_ok=True)
        engine_path = os.path.join(cache_dir, f"cvae_{weights}_n{num_samples}_trt{trt.__version__}_{gpu}.engine")
        logger = trt.Logger(trt.Logger.ERROR)
        if not os.path.exists(engine_path):
            onnx_path = engine_path.replace(".engine", ".onnx")
            model = copy.deepcopy(cvae.model).eval().requires_grad_(False).half()

            class Forward(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.m = model

                def forward(self, cond, eps):
                    tokens = self.m.cond(cond)
                    mu, logvar = self.m.pri(tokens)
                    return self.m.dec(self.m.lat(mu + eps * torch.exp(0.5 * logvar)), tokens.expand(num_samples, -1, -1))

            example = (torch.zeros(1, frames, img_dim, img_dim, device=self.device, dtype=torch.float16),
                       torch.zeros(num_samples, cvae.latent_dim, device=self.device, dtype=torch.float16))
            with torch.no_grad():
                torch.onnx.export(Forward(), example, onnx_path, input_names=["cond", "eps"], output_names=["out"],
                                  opset_version=17, dynamo=False)
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
        self.out_host = torch.empty(num_samples, 3, dtype=torch.float16).pin_memory()
        self.cond_dev = torch.zeros(1, frames, img_dim, img_dim, device=self.device, dtype=torch.float16)
        self.eps = torch.zeros(num_samples, cvae.latent_dim, device=self.device, dtype=torch.float16)
        self.out_dev = torch.empty(num_samples, 3, device=self.device, dtype=torch.float16)
        for name, tensor in [("cond", self.cond_dev), ("eps", self.eps), ("out", self.out_dev)]:
            self.context.set_tensor_address(name, tensor.data_ptr())
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(0)
        for _ in range(3):  # warm up
            self.sample()

    def set_condition(self, imgs_tensor: torch.Tensor):
        """Image preprocessing: the condition into pinned host memory as fp16 (not part of the inference)."""
        self.cond_host.copy_(imgs_tensor)

    def sample(self) -> np.ndarray:
        """(num_samples, 3) array of [t, d, v] for the condition of set_condition."""
        stream = torch.cuda.current_stream(self.device)
        with torch.inference_mode():
            self.cond_dev.copy_(self.cond_host, non_blocking=True)
            self.eps.normal_(generator=self.generator)
            self.context.execute_async_v3(stream.cuda_stream)
            self.out_host.copy_(self.out_dev, non_blocking=True)
            stream.synchronize()
        return self.out_host.numpy().astype(np.float64) * self.std + self.mean
