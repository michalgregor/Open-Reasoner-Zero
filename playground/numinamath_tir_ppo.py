# GPU instance 24 (not lambda)
# install Anaconda
# curl -O https://repo.anaconda.com/archive/Anaconda3-2025.12-2-Linux-x86_64.sh
# bash ./Anaconda3-2025.12-2-Linux-x86_64.sh -b -p $HOME/anaconda3
# source ~/anaconda3/bin/activate
# conda init --all

# pip install torch torchvision
# pip install math_verify

# build vllm from source:
# pip install -U pip setuptools wheel ninja packaging
# sudo apt-get update
# sudo apt-get install -y build-essential cmake
# pip install "vllm @ git+https://github.com/vllm-project/vllm.git@v0.15.1"

# pip install loguru omegaconf hydra-core ray transformers datasets deepspeed peft accelerate tensorboard
# MAX_JOBS=8 pip install flash-attn --no-build-isolation

# git clone https://github.com/michalgregor/Open-Reasoner-Zero.git
# pip install -e . # for ORZ

# tensorboard:
# ssh -L 6006:localhost:6006 lambda
# cd ~/Open-Reasoner-Zero/; tensorboard --port 6006 --logdir orz_logs/numinamath_tir_ppo


"""
PPO experiment for AI-MO/NuminaMath-TIR using Open-Reasoner-Zero.

Run:
  python -m playground.numinamath_tir_ppo

Common overrides:
  python -m playground.numinamath_tir_ppo exp.pretrain=Qwen/Qwen2.5-1.5B-Instruct
  python -m playground.numinamath_tir_ppo exp.train_split='train[:1%]'
"""

import asyncio
import copy
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Awaitable, Callable, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from omegaconf.listconfig import ListConfig
from typing_extensions import override

from orz.exps.examples.ppo.ppo_base_exp import BasePPOExp, BasePPOExpConfig
from orz.ppo import PromptDataset
from orz.ppo.tools.math_utils import is_equal, solution2answer
from playground.orz_7b_ppo import CustomRewardTrainer


FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
EXECUTOR = ThreadPoolExecutor(max_workers=32)


class NuminaTrainDataset(PromptDataset):
    def __init__(self, *args, system_prompt: str, **kwargs):
        self.system_prompt = system_prompt.strip()
        super().__init__(*args, **kwargs)

    def process_dialogue(self, dialogue: dict):
        prompt = dialogue["problem"]
        answer = dialogue["solution"]

        bos_token = ""
        if self.tokenizer.bos_token_id is not None:
            bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])

        prompt_text = f"{bos_token}{self.system_prompt}\nUser: {prompt}\nAssistant: <think>"
        return prompt_text, {"answer": answer}


class NuminaEvalDataset(PromptDataset):
    def __init__(self, *args, system_prompt: str, file_name: str, **kwargs):
        self.system_prompt = system_prompt.strip()
        self.file_name = file_name
        super().__init__(*args, **kwargs)

    def process_dialogue(self, dialogue: dict):
        prompt = dialogue["problem"]
        answer = dialogue["solution"]

        bos_token = ""
        if self.tokenizer.bos_token_id is not None:
            bos_token = self.tokenizer.decode([self.tokenizer.bos_token_id])

        prompt_text = f"{bos_token}{self.system_prompt}\nUser: {prompt}\nAssistant: <think>"
        return prompt_text, {"answer": answer, "file_name": self.file_name}


class NuminaRewardTrainer(CustomRewardTrainer):
    # Mirror TRL-style think-format reward: reward 1 when completion contains a valid
    # <think> ... </think> block and at least one non-whitespace answer token after it.
    _THINK_FORMAT_PATTERN = re.compile(r"^\s*<think>.*?</think>\s*\S[\s\S]*$", re.DOTALL)

    @staticmethod
    def _extract_final_answer(response: str) -> str:
        boxed_matches = re.findall(r"(\\boxed\{.*?\})", response, flags=re.DOTALL)
        if boxed_matches:
            return boxed_matches[-1]
        return response

    @classmethod
    def _think_format_reward(cls, response: str) -> float:
        candidate = response
        if "<think>" not in candidate:
            # Prompt is prefilled with "Assistant: <think>", so completions often start
            # inside the think block. Prefix to evaluate with TRL-style format semantics.
            candidate = f"<think>{candidate}"
        return 1.0 if cls._THINK_FORMAT_PATTERN.match(candidate) else 0.0

    @override
    async def custom_reward_fn(
        self,
        prompts: List[str],
        outputs: List[Any],
        extras: List[dict],
        reward_model_fn: Callable[[List[str], List[str]], Awaitable[torch.Tensor]],
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        scores: List[float] = []
        format_rewards: List[float] = []
        accuracy_rewards: List[float] = []
        responses: List[str] = []
        pass_at_n_dict = defaultdict(list)
        num_tokens: List[int] = []

        for output in outputs:
            responses.append(output["response"])
        output_tokens = self._tokenize(responses, self.cfg.generate_max_len, padding=False)["input_ids"]

        self.writer.add_text(
            "generated_raws",
            (
                f"prompts: {prompts[0]}\n\noutputs: {outputs[0]['response']}\n\nfinal_answer: "
                f"{outputs[0]['final_answer']}\n\nis_correct: {outputs[0]['iscorrect']}\n\n"
                f"stop_reason: {outputs[0]['stop_reason']}\n\nresponse_token: {len(output_tokens[0])}"
            ),
            self.global_step,
        )

        for idx in range(len(outputs)):
            prompt, output, out_token = prompts[idx], outputs[idx], output_tokens[idx]
            iscorrect = bool(output["iscorrect"])
            response = output["response"]
            response_token = len(out_token)

            # TRL-style component rewards:
            # - accuracy_reward: 1.0 if answer is correct else 0.0
            # - think_format_reward: 1.0 if output format follows think tags else 0.0
            accuracy_reward = 1.0 if iscorrect else 0.0
            format_reward = self._think_format_reward(response)
            score = self.cfg.accuracy_reward_weight * accuracy_reward + self.cfg.format_reward_weight * format_reward

            output["accuracy_reward"] = accuracy_reward
            output["format_reward"] = format_reward
            output["scalar_reward"] = score

            accuracy_rewards.append(accuracy_reward)
            format_rewards.append(format_reward)
            scores.append(score)
            pass_at_n_dict[prompt].append(score)
            num_tokens.append(response_token)

        num_tokens_arr = np.array(num_tokens, dtype=np.float32)
        scores_arr = np.array(scores)
        correct_tokens_arr = np.array([]) if np.all(scores_arr <= 0.0) else np.array(num_tokens_arr[scores_arr > 0.0])
        incorrect_tokens_arr = (
            np.array([]) if np.all(scores_arr > 0.0) else np.array(num_tokens_arr[scores_arr <= 0.0])
        )

        # Keep optional GRPO normalization behavior for compatibility.
        if self.cfg.use_grpo:
            self.writer.add_scalar("grpo_raw_reward", float(np.mean(scores)) if scores else 0.0, self.global_step)
            for i, prompt in enumerate(prompts):
                prompt_scores = pass_at_n_dict[prompt]
                centered = scores[i] - float(np.mean(prompt_scores))
                std = float(np.std(prompt_scores))
                if std > 0:
                    centered /= std
                scores[i] = centered
                outputs[i]["scalar_reward"] = centered

        def dump_results(local_prompts, local_outputs, local_scores):
            saved = []
            for prompt, output, score in zip(local_prompts, local_outputs, local_scores):
                saved.append(dict(prompt=prompt, score=score, outputs=output))
            json.dump(
                saved,
                open(os.path.join(self.cfg.save_path, f"iter{self.global_step}_generation_results.json"), "w"),
                ensure_ascii=False,
                indent=2,
            )

        asyncio.get_event_loop().run_in_executor(
            EXECUTOR, dump_results, copy.deepcopy(prompts), copy.deepcopy(outputs), copy.deepcopy(scores)
        )

        log_dict = {
            "avg_scalar_reward": float(np.mean(scores)) if scores else 0.0,
            "avg_accuracy_reward": float(np.mean(accuracy_rewards)) if accuracy_rewards else 0.0,
            "avg_format_reward": float(np.mean(format_rewards)) if format_rewards else 0.0,
            "avg_pass_at_n": sum(1 for v in pass_at_n_dict.values() if np.sum(v) > 0) / len(pass_at_n_dict),
            "avg_num_tokens": np.mean(num_tokens_arr).item() if len(num_tokens_arr) else 0.0,
            "std_num_tokens": np.std(num_tokens_arr).item() if len(num_tokens_arr) else 0.0,
            "avg_correct_num_tokens": 0 if len(correct_tokens_arr) == 0 else np.mean(correct_tokens_arr).item(),
            "std_correct_num_tokens": 0 if len(correct_tokens_arr) == 0 else np.std(correct_tokens_arr).item(),
            "avg_incorrect_num_tokens": 0 if len(incorrect_tokens_arr) == 0 else np.mean(incorrect_tokens_arr).item(),
            "std_incorrect_num_tokens": 0 if len(incorrect_tokens_arr) == 0 else np.std(incorrect_tokens_arr).item(),
        }
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.global_step)
        logger.info(",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()]))

        if len(correct_tokens_arr) > 0:
            self.writer.add_histogram("correct_response_length", correct_tokens_arr, self.global_step)
        if len(incorrect_tokens_arr) > 0:
            self.writer.add_histogram("incorrect_response_length", incorrect_tokens_arr, self.global_step)

        score_tensors = []
        for score, output_token in zip(scores, output_tokens):
            score_tensor = torch.zeros(len(output_token))
            if len(output_token) > 0:
                score_tensor[-1] = score
            score_tensors.append(score_tensor)

        res_prompts = []
        res_responses = []
        res_score_tensors = []
        for prompt, response, score_tensor in zip(prompts, responses, score_tensors):
            if len(response) > 0:
                res_prompts.append(prompt)
                res_responses.append(response)
                res_score_tensors.append(score_tensor)

        return res_prompts, res_responses, res_score_tensors

    @override
    @torch.no_grad()
    async def generate_vllm(
        self,
        gen_func: Callable[[List[str]], Awaitable[List[str | Any]]],
        prompts: List[str],
        extras: List[dict],
        **kwargs,
    ) -> List[str | Any]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            max_tokens=self.cfg.generate_max_len,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
            stop=list(self.cfg.stop),
        )
        responses, stop_reasons = await gen_func(
            prompts=prompts, sampling_params=sampling_params, use_tqdm=False, truncate_prompt=True
        )

        final_answers = [self._extract_final_answer(response) for response in responses]
        equal_tasks = [
            is_equal(solution2answer(extra["answer"]), solution2answer(final_answer), None)
            for extra, final_answer in zip(extras, final_answers)
        ]
        equal_results = await asyncio.gather(*equal_tasks)

        results = []
        for response, final_answer, stop_reason, iscorrect in zip(
            responses, final_answers, stop_reasons, equal_results
        ):
            results.append(
                {
                    "response": response,
                    "iscorrect": iscorrect,
                    "stop_reason": stop_reason,
                    "final_answer": final_answer,
                }
            )
        return results


@dataclass
class PPOExpConfig(BasePPOExpConfig):
    # Mirrors trl_experiments/configs/base_config.yaml
    dataset_name: str = "AI-MO/NuminaMath-TIR"
    train_split: str = "train[:5%]"
    eval_split: str = "test[:5%]"
    test_split: str = "test[5%:15%]"
    system_prompt: str = (
        "A conversation between user and assistant. The user asks a question, and "
        "the assistant solves it. The assistant first thinks about the reasoning "
        "process in the mind and then provides the user with the answer. The "
        "reasoning process and answer are enclosed within <think></think> tags, "
        "i.e., <think>\\nThis is my reasoning.\\n</think>\\nThis is my answer."
    )

    # Resource settings: default to single-node 1-GPU (can be overridden)
    total_num_nodes: int = 1
    ref_num_nodes: int = total_num_nodes
    ref_num_gpus_per_node: int = 1
    actor_num_nodes: int = total_num_nodes
    actor_num_gpus_per_node: int = 1
    critic_num_nodes: int = total_num_nodes
    critic_num_gpus_per_node: int = 1
    colocate_all: bool = True
    colocate_critic_reward: bool = True
    colocate_actor_ref: bool = True
    vllm_num_engines: int = total_num_nodes
    vllm_tensor_parallel_size: int = 1
    adam_offload: bool = False


    zero_stage: int = 3 # >0 probably make sense only on multiple GPUs, but otherwise errors out
    


    # Paths and model
    pretrain: Optional[str] = "Qwen/Qwen2.5-0.5B-Instruct"
    reward_pretrain: Optional[str] = None
    save_interval: int = 50
    ckpt_path: str = f"orz_ckpt/{FILE_NAME}"
    save_path: str = f"orz_ckpt/{FILE_NAME}"
    tensorboard_log_dir: str = f"orz_logs/{FILE_NAME}"

    # Keep empty because this experiment loads HF data directly.
    prompt_data: ListConfig = ListConfig([])
    eval_prompt_data: ListConfig = ListConfig([])
    prompt_data_probs: ListConfig = ListConfig([1.0])

    # PPO settings aligned to your TRL run shape.
    use_compute_reward_fn: bool = True
    use_orm_score: bool = False
    accuracy_reward_weight: float = 1.0
    format_reward_weight: float = 1.0
    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 5e-6
    num_warmup_steps: int = 50
    prompt_max_len: int = 128
    generate_max_len: int = 256
    max_len: int = 512
    packing_max_len: int = prompt_max_len + generate_max_len

    enable_prefix_caching: bool = True
    update_ref_every_epoch: bool = True
    advantage_normalize: bool = True

    num_episodes: int = 1
    max_epochs: int = 1
    rollout_batch_size: int = 8
    train_batch_size: int = 8
    n_samples_per_prompt: int = 4
    micro_rollout_batch_size: int = 8
    policy_update_steps: int = 1
    critic_update_steps: int = 12
    micro_train_batch_size: int = 1
    micro_forward_batch_size: int = 1
    freezing_actor_steps: int = -1

    init_kl_coef: float = 0.0
    kl_loss_coef: float = 0.0
    use_kl_loss: bool = True
    use_kl_estimator_k3: bool = True
    use_reference_model: bool = False

    enable_eval: bool = True
    eval_interval: int = 10

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop: ListConfig = ListConfig(["User:", "Human:", "Assistant:"])

    use_grpo: bool = False
    gpu_memory_utilization: float = 0.75
    attention_backend: Optional[str] = "TRITON_ATTN"
    critic_pretrain: Optional[str] = ""
    gamma: float = 1.0
    lambd: float = 1.0


    # not sure that this is what we want, but deepspeed complains about fp32 currently
    grad_accum_dtype: str = "bf16"



class PPOExp(BasePPOExp):
    @cached_property
    def trainer(self):
        vllm_engines = self.create_inference_engine()
        return NuminaRewardTrainer(
            cfg=self.cfg,
            strategy=self.strategy,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            vllm_engines=vllm_engines,
            colocate_pg=self.get_colocate_pg,
        )

    @override
    @cached_property
    def train_dataset(self):
        import datasets

        ds = datasets.load_dataset(self.cfg.dataset_name, split=self.cfg.train_split)
        dialogues = [{"problem": row["problem"], "solution": row["solution"]} for row in ds]

        logger.info(f"Loaded {len(dialogues)} train samples from {self.cfg.dataset_name}:{self.cfg.train_split}")
        return NuminaTrainDataset(
            dialogues,
            self.tokenizer,
            self.cfg.prompt_max_len,
            self.strategy,
            pretrain_mode=False,
            num_processors=1,
            system_prompt=self.cfg.system_prompt,
        )

    @override
    @cached_property
    def eval_dataset(self):
        import datasets

        ds = datasets.load_dataset(self.cfg.dataset_name, split=self.cfg.eval_split)
        dialogues = [{"problem": row["problem"], "solution": row["solution"]} for row in ds]

        logger.info(f"Loaded {len(dialogues)} eval samples from {self.cfg.dataset_name}:{self.cfg.eval_split}")
        return NuminaEvalDataset(
            dialogues,
            self.tokenizer,
            self.cfg.prompt_max_len,
            self.strategy,
            pretrain_mode=False,
            num_processors=1,
            system_prompt=self.cfg.system_prompt,
            file_name="numinamath_tir_valid_5pct",
        )


if __name__ == "__main__":
    exp = PPOExp().set_cfg(PPOExpConfig())

    if not exp.cfg.use_grpo and not exp.cfg.critic_pretrain:
        exp.cfg.critic_pretrain = exp.cfg.pretrain

    logger.info(exp.get_cfg_as_str(exp.cfg))

    os.makedirs(exp.cfg.save_path, exist_ok=True)
    os.makedirs(exp.cfg.tensorboard_log_dir, exist_ok=True)
    os.makedirs(exp.cfg.ckpt_path, exist_ok=True)

    with open(os.path.join(exp.cfg.save_path, "run_config.json"), "w") as f:
        json.dump(exp.cfg.__dict__, f, indent=2, ensure_ascii=False, default=str)

    asyncio.run(exp.run())
