import socket

import ray
import torch

try:
    from vllm.worker.worker import Worker
except ModuleNotFoundError:
    # vLLM v1 API layout (e.g., v0.15.x)
    from vllm.v1.worker.gpu_worker import Worker

from orz.exp_engine.parallels.orz_distributed_c10d import orz_init_process_group


def _coerce_dtype(dtype):
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        name = dtype.replace("torch.", "")
        if hasattr(torch, name):
            candidate = getattr(torch, name)
            if isinstance(candidate, torch.dtype):
                return candidate
    raise TypeError(f"Unsupported dtype representation: {dtype!r}")


class WorkerWrap(Worker):
    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend="nccl"):
        """Init torch process group for model weights update"""
        assert torch.distributed.is_initialized(), "default torch process group must be initialized"
        assert group_name != "", "group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_group = orz_init_process_group(
            backend=backend,
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=rank,
            group_name=group_name,
        )
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def update_weight(self, name, dtype, shape, empty_cache=False):
        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        # if torch.distributed.get_rank() == 0:
        #     print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        dtype = _coerce_dtype(dtype)
        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        # vLLM v1 can OOM when allocating an extra full-size tensor per parameter.
        # Prefer in-place broadcast into existing model parameter storage.
        target_param = None
        for param_name, param in self.model_runner.model.named_parameters():
            if param_name == name:
                target_param = param
                break

        if target_param is not None:
            torch.distributed.broadcast(target_param.data, 0, group=self._model_update_group)
            return

        # Fallback path for models that don't expose named_parameters() as expected.
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        torch.distributed.broadcast(weight, 0, group=self._model_update_group)
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight

    def update_weight_internal_with_cuda_ipc(self, name, dtype, shape, cudaipc_handler, empty_cache=False):
        dtype = _coerce_dtype(dtype)
        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        weight = cudaipc_handler.rebuild().clone()
        # weight = rebuild_tensor_from_handles(cudaipc_handler).clone()
        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight

    def get_ip_and_port(self):
        master_address = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]
        return master_address, master_port

    def free_weight(
        self,
    ):
        self.model_runner.model.to("meta")

    def free_cache_engine(
        self,
    ):
        self.cache_engine = None
        if hasattr(self, "gpu_cache"):
            self.gpu_cache = None

    def init_cache_engine(
        self,
    ):
        if hasattr(super(), "_init_cache_engine"):
            if self.cache_engine is None and (not hasattr(self, "gpu_cache") or self.gpu_cache is None):
                super()._init_cache_engine()


class OffloadableVLLMWorker(WorkerWrap):
    """Monkey patch for the vLLM worker to manipulate the model parameters.

    This class will replace the original Worker class as VLLMAccelerated-
    InferenceModelWorker is imported, inspired by `OpenRLHF`.
    """

    def offload_cpu(self):
        assert self.model_config.enforce_eager, "Must use eager mode to offload!"
        if hasattr(self, "sleep"):
            # vLLM v1 workers provide built-in sleep/wake_up APIs.
            self.sleep(level=2)
            torch.cuda.empty_cache()
            return

        for param in self.model_runner.model.parameters():
            param.meta_tensor = param.data.to("meta")
            param.data = torch.Tensor([])

        self.cache_engine = None
        self.gpu_cache = None
        torch.cuda.empty_cache()

    def load_gpu(self):
        assert self.model_config.enforce_eager, "Must use eager mode to offload!"
        if hasattr(self, "wake_up"):
            # Wake up all resources in vLLM v1 path.
            self.wake_up()
            return

        for param in self.model_runner.model.parameters():
            param.data = torch.empty_like(param.meta_tensor, device="cuda")
            param.meta_tensor = None
        if hasattr(self, "_init_cache_engine") and self.cache_engine is None and (
            not hasattr(self, "gpu_cache") or self.gpu_cache is None
        ):
            super()._init_cache_engine()
