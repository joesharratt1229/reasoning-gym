import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union

from datasets import Dataset
from trl.trainer.grpo_trainer import RewardFunc
from vllm import LLM, SamplingParams  # type: ignore

import reasoning_gym
from reasoning_gym.utils import SYSTEM_PROMPTS

from .utils import preprocess_dataset


class BaseEnv(ABC):

    def __init__(self, **kwargs: Any):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.logger = logging.getLogger(f"verifiers.envs.{self.__class__.__name__}")
        self.tokenizer = None

    @abstractmethod
    def get_dataset(self, **kwargs: Any) -> Dataset:
        pass

    @abstractmethod
    def get_rubric(self, **kwargs: Any) -> List[RewardFunc]:
        pass

    @abstractmethod
    def generate(
        self,
        prompts: List[List[Dict[str, Any]]],
        llm: LLM,
        sampling_params: SamplingParams,
        output_type: str = "ids",
        **kwargs: Any,
    ) -> Union[List[Sequence[int]], List[str], List[List[Dict[str, Any]]]]:
        pass


class ReasongGymEnv(BaseEnv):
    def __init__(self, dataset_name, seed, size, tokenizer, system_prompt: Optional[str] = None, **kwargs: Any):
        super().__init__(**kwargs)
        sampling_args = {
            "skip_special_tokens": False,
            "space_between_special_tokens": False,
        }

        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = SYSTEM_PROMPTS["default"]

        self.dataset = reasoning_gym.create_dataset(dataset_name, seed=seed, size=size)
        self.sampling_args = sampling_args
        self.seed = seed
        self.size = size
        self.tokenizer = tokenizer
        self.developer_role = "system"

    def get_dataset(self, **kwargs):
        dataset = preprocess_dataset(
            self.dataset, self.seed, self.size, self.tokenizer, self.system_prompt, self.developer_role
        )
        return dataset

    def _format_reward(self, completions, **kwargs):
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"
        matches = [re.match(regex, completion, flags=re.DOTALL) for completion in completions]
        return [1.0 if match else 0.0 for match in matches]

    def _accuracy_reward(self, completions, metadata, **kwargs):
        answers = [extract_answer(completion) for completion in completions]
        return [self.dataset.score_answer(answer, entry=obj) for (answer, obj) in zip(answers, metadata)]

    def get_rubric(self, **kwargs):
        return [self._format_reward, self._accuracy_reward]

    def generate(
        self,
        prompts: List[List[Dict[str, Any]]],
        llm: LLM,
        sampling_params: SamplingParams,
        output_type: str = "ids",
        **kwargs: Any,
    ) -> Union[List[Sequence[int]], List[str], List[List[Dict[str, Any]]]]:

        custom_sp = sampling_params.clone()
        for k, v in self.sampling_args.items():
            setattr(custom_sp, k, v)

        states = [{"messages": m, "prompt_ids": [], "completion_ids": []} for m in prompts]

        # get completions
        completions = llm.chat(prompts, sampling_params=custom_sp, use_tqdm=False)  # type: ignore
        return completions
