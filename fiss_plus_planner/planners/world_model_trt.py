"""TensorRT backend for the world model used by FopPlusWMSelectionPlanner.

The PyTorch model is exported to ONNX under fp16 autocast, so the graph carries the same fp16/fp32
casts as the PyTorch inference path; TensorRT (strongly typed) then keeps exactly those precisions.
The built engine is cached next to the weights and rebuilt when the weights, TensorRT version,
GPU or maximum batch size change.
"""
import tempfile
from pathlib import Path

import tensorrt as trt
import torch
from torch import nn


class _NextStateOnly(nn.Module):
    """Wraps the world model so the exported graph has one fp32 output: the predicted next BEV."""

    def __init__(self, world_model: nn.Module):
        super().__init__()
        self.world_model = world_model

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        prediction, _ = self.world_model(state, action)
        return prediction.float()


class TensorRTWorldModel:
    """Predicts the next BEV frame from (state, action) with a TensorRT engine."""

    STATE_INPUT = "state"
    ACTION_INPUT = "action"
    OUTPUT = "next_state"

    def __init__(self, world_model: nn.Module, weights_path: Path, state_shape: tuple,
                 action_shape: tuple, max_batch_size: int, device: torch.device):
        """state_shape is (C, H, W) and action_shape is (horizon, 2), both without the batch dim."""
        if device.type != "cuda":
            raise ValueError("TensorRTWorldModel needs a CUDA device")
        self.device = device
        self.max_batch_size = max_batch_size
        self._logger = trt.Logger(trt.Logger.WARNING)

        engine_path = self._engine_path(Path(weights_path), max_batch_size)
        if not engine_path.exists() or engine_path.stat().st_mtime < Path(weights_path).stat().st_mtime:
            self._build_engine(world_model, engine_path, state_shape, action_shape)

        runtime = trt.Runtime(self._logger)
        self._engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine {engine_path}")
        self._context = self._engine.create_execution_context()
        # TensorRT synchronizes extra on the default stream, so inference runs on its own stream.
        self._stream = torch.cuda.Stream(device)

    def __call__(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """state: (B, C, H, W) fp32, action: (B, horizon, 2) fp32 on the CUDA device; returns (B, 1, H, W) fp32."""
        batch_size = state.shape[0]
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch {batch_size} exceeds the engine's maximum of {self.max_batch_size}")

        state = state.contiguous().float()
        action = action.contiguous().float()
        self._context.set_input_shape(self.STATE_INPUT, tuple(state.shape))
        self._context.set_input_shape(self.ACTION_INPUT, tuple(action.shape))
        output = torch.empty(tuple(self._context.get_tensor_shape(self.OUTPUT)),
                             dtype=torch.float32, device=self.device)

        self._context.set_tensor_address(self.STATE_INPUT, state.data_ptr())
        self._context.set_tensor_address(self.ACTION_INPUT, action.data_ptr())
        self._context.set_tensor_address(self.OUTPUT, output.data_ptr())

        caller_stream = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(caller_stream)  # inputs are written on the caller's stream
        if not self._context.execute_async_v3(self._stream.cuda_stream):
            raise RuntimeError("TensorRT inference failed")
        caller_stream.wait_stream(self._stream)  # the caller reads the output afterwards
        for tensor in (state, action, output):
            tensor.record_stream(self._stream)
        return output

    def _engine_path(self, weights_path: Path, max_batch_size: int) -> Path:
        major, minor = torch.cuda.get_device_capability(self.device)
        return weights_path.with_name(
            f"{weights_path.stem}.trt{trt.__version__}.sm{major}{minor}.b{max_batch_size}.engine"
        )

    def _build_engine(self, world_model: nn.Module, engine_path: Path,
                      state_shape: tuple, action_shape: tuple) -> None:
        print(f"Building TensorRT engine {engine_path.name} (one-time, may take a few minutes)...")
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path = Path(tmp_dir) / "world_model.onnx"
            self._export_onnx(world_model, onnx_path, state_shape, action_shape)
            serialized = self._compile_onnx(onnx_path, state_shape, action_shape)
        engine_path.write_bytes(serialized)

    def _export_onnx(self, world_model: nn.Module, onnx_path: Path,
                     state_shape: tuple, action_shape: tuple) -> None:
        wrapper = _NextStateOnly(world_model).eval()
        example_state = torch.rand((1, *state_shape), device=self.device)
        example_action = torch.zeros((1, *action_shape), device=self.device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            torch.onnx.export(
                wrapper,
                (example_state, example_action),
                str(onnx_path),
                input_names=[self.STATE_INPUT, self.ACTION_INPUT],
                output_names=[self.OUTPUT],
                dynamic_axes={self.STATE_INPUT: {0: "batch"}, self.ACTION_INPUT: {0: "batch"},
                              self.OUTPUT: {0: "batch"}},
                opset_version=17,
                dynamo=False,
            )

    def _compile_onnx(self, onnx_path: Path, state_shape: tuple, action_shape: tuple) -> bytes:
        builder = trt.Builder(self._logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(network, self._logger)
        if not parser.parse_from_file(str(onnx_path)):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"Failed to parse the world-model ONNX graph:\n{errors}")

        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        opt_batch = self.max_batch_size
        profile.set_shape(self.STATE_INPUT, (1, *state_shape), (opt_batch, *state_shape),
                          (self.max_batch_size, *state_shape))
        profile.set_shape(self.ACTION_INPUT, (1, *action_shape), (opt_batch, *action_shape),
                          (self.max_batch_size, *action_shape))
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")
        return bytes(serialized)
