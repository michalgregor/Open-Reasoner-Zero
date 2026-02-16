import re

try:
    from packaging.version import Version
except ImportError:  # pragma: no cover - packaging is expected in most environments
    Version = None


def _version_gte(current: str, target: str) -> bool:
    if Version is not None:
        return Version(current) >= Version(target)
    current_nums = tuple(int(x) for x in re.findall(r"\d+", current))
    target_nums = tuple(int(x) for x in re.findall(r"\d+", target))
    max_len = max(len(current_nums), len(target_nums))
    current_nums = current_nums + (0,) * (max_len - len(current_nums))
    target_nums = target_nums + (0,) * (max_len - len(target_nums))
    return current_nums >= target_nums


class LLMActor:
    def __init__(self, *args, **kwargs):
        import vllm

        self.__version__ = vllm.__version__
        assert _version_gte(self.__version__, "0.4.1"), "OpenRLHF only supports vLLM >= 0.4.1"

        self.use_gpu_executor = kwargs["tensor_parallel_size"] == 1

        # See https://github.com/vllm-project/vllm/blob/main/vllm/executor/gpu_executor.py
        if self.use_gpu_executor:
            if _version_gte(vllm.__version__, "0.15.0"):
                # vLLM v1 resolves worker class from qualified name.
                kwargs[
                    "worker_cls"
                ] = "orz.exp_engine.accelerators.inference.vllm_worker_wrap.OffloadableVLLMWorker"
            else:
                from .vllm_worker_wrap import OffloadableVLLMWorker

                vllm.worker.worker.Worker = OffloadableVLLMWorker
        else:
            # RayGPUExecutor
            # See the patch https://github.com/vllm-project/vllm/commit/479d69fad0538f04cb22bf13e76ff91cfeb8a4e5
            kwargs["worker_use_ray"] = True

            if _version_gte(vllm.__version__, "0.6.5"):
                # https://github.com/vllm-project/vllm/pull/10555
                kwargs[
                    "worker_cls"
                ] = "orz.exp_engine.accelerators.inference.vllm_worker_wrap.OffloadableVLLMWorker"
            else:
                RayWorkerWrapperPath = vllm.executor.ray_utils

                class RayWorkerWrapper(RayWorkerWrapperPath.RayWorkerWrapper):
                    def __init__(self, *args, **kwargs) -> None:
                        kwargs[
                            "worker_module_name"
                        ] = "orz.exp_engine.accelerators.inference.vllm_worker_wrap"
                        kwargs["worker_class_name"] = "OffloadableVLLMWorker"
                        super().__init__(*args, **kwargs)

                RayWorkerWrapperPath.RayWorkerWrapper = RayWorkerWrapper

        kwargs["enforce_eager"] = True
        self.llm = vllm.LLM(*args, **kwargs)
        self.scheduler_config = getattr(self.llm.llm_engine, "scheduler_config", None)
        self.model_config = getattr(self.llm.llm_engine, "model_config", None)
        self.cache_config = getattr(self.llm.llm_engine, "cache_config", None)
        self.lora_config = getattr(self.llm.llm_engine, "lora_config", None)
        self.parallel_config = getattr(self.llm.llm_engine, "parallel_config", None)

    def generate(self, *args, **kwargs):
        llm_engine = self.llm.llm_engine
        if hasattr(llm_engine, "wake_up"):
            # Ensure all engine resources are restored before generation.
            # Some attention backends (e.g. Triton) can crash if any tensor
            # remains in a slept/offloaded state.
            llm_engine.wake_up()
        # vLLM >= 0.15 removed `prompt_token_ids=` from LLM.generate().
        # Convert old call style to token-prompt objects for compatibility.
        if "prompt_token_ids" in kwargs and len(args) == 0 and "prompts" not in kwargs:
            prompt_token_ids = kwargs.pop("prompt_token_ids")
            kwargs["prompts"] = [{"prompt_token_ids": ids} for ids in prompt_token_ids]
        return self.llm.generate(*args, **kwargs)

    def _run_worker_method(self, method_name, *args, **kwargs):
        llm_engine = self.llm.llm_engine
        model_executor = getattr(llm_engine, "model_executor", None)

        if model_executor is not None and hasattr(model_executor, "driver_worker"):
            driver_worker = model_executor.driver_worker
            if hasattr(driver_worker, method_name):
                return getattr(driver_worker, method_name)(*args, **kwargs)

        if model_executor is not None and hasattr(model_executor, "_run_workers"):
            return model_executor._run_workers(method_name, *args, **kwargs)

        if hasattr(llm_engine, "collective_rpc"):
            return llm_engine.collective_rpc(method_name, args=args, kwargs=kwargs)

        raise RuntimeError(f"Cannot dispatch worker method '{method_name}' on this vLLM version")

    def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self._run_worker_method(
            "init_process_group", master_address, master_port, rank_offset, world_size, group_name, backend
        )

    def get_ip_and_port(self):
        return self._run_worker_method("get_ip_and_port")

    def offload_to_cpu(self):
        llm_engine = self.llm.llm_engine
        if hasattr(llm_engine, "sleep"):
            llm_engine.sleep(level=2)
            return
        return self._run_worker_method("offload_cpu")

    def backload_to_gpu(self, tags=None):
        llm_engine = self.llm.llm_engine
        if hasattr(llm_engine, "wake_up"):
            llm_engine.wake_up(tags=tags)
            self._rebuild_scheduler_if_needed()
            return
        self._run_worker_method("load_gpu")
        self._rebuild_scheduler_if_needed()

    def _rebuild_scheduler_if_needed(self):
        llm_engine = self.llm.llm_engine

        # Prefer engine-provided (newer vLLM) hooks when available.
        for method_name in ("reset_scheduler", "_reset_scheduler", "init_scheduler", "_init_scheduler"):
            method = getattr(llm_engine, method_name, None)
            if callable(method):
                method()
                return

        # Fall back to the legacy private scheduler API used in older vLLM versions.
        try:
            from vllm.core.scheduler import Scheduler
        except Exception:
            return

        async_callbacks = getattr(llm_engine, "async_callbacks", None)
        if self.parallel_config is None or self.model_config is None:
            return
        llm_engine.scheduler = [
            Scheduler(
                self.scheduler_config,
                self.cache_config,
                self.lora_config,
                self.parallel_config.pipeline_parallel_size,
                async_callbacks[v_id]
                if (self.model_config.use_async_output_proc and async_callbacks is not None)
                else None,
            )
            for v_id in range(self.parallel_config.pipeline_parallel_size)
        ]

    def update_weight(self, name, dtype, shape, empty_cache=False):
        self.stop_remote_worker_execution_loop()

        return self._run_worker_method("update_weight", name, dtype, shape, empty_cache)

    def update_weight_internal_with_cuda_ipc(self, name, dtype, shape, cudaipc_handler, empty_cache=False):
        return self._run_worker_method(
            "update_weight_internal_with_cuda_ipc", name, dtype, shape, cudaipc_handler, empty_cache
        )

    def stop_remote_worker_execution_loop(self):
        # Fix error for using 2 communication group
        # https://github.com/vllm-project/vllm/commit/eb6d3c264d0cd8e44dec16bca7947fbe96415ce9#diff-e1ad69e38e033accddfa5480ec808c4740eb39244d1ef51cc3407e20dde8cfd4
        if _version_gte(self.__version__, "0.4.3"):
            model_executor = getattr(self.llm.llm_engine, "model_executor", None)
            if model_executor is not None and hasattr(model_executor, "stop_remote_worker_execution_loop"):
                model_executor.stop_remote_worker_execution_loop()

    def get_gpu_memory(self):
        """获取当前Actor使用的GPU内存"""
        import torch

        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated() / 1024**2  # 转换为MB

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        stats = {}
        model_executor = self.llm.llm_engine.model_executor
        if hasattr(model_executor, "driver_worker"):
            model_runner = model_executor.driver_worker.model_runner
        else:
            raise RuntimeError("get_weight_statistics requires a local driver_worker on this vLLM backend")
        for name, param in model_runner.model.named_parameters():
            # 计算关键统计信息
            tensor_stats = {
                "mean": param.mean().item(),
                "std": param.std().item(),
                "norm": param.norm().item(),
                "shape": tuple(param.shape),
                # 可选：计算一些极值
                "max": param.max().item(),
                "min": param.min().item(),
            }
            stats[name] = tensor_stats
        return stats
