#!/usr/bin/env python

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from tqdm import tqdm
from vllm import LLM, SamplingParams

import reasoning_gym
from reasoning_gym.utils import SYSTEM_PROMPTS, extract_answer


@dataclass
class DatasetConfig:
    dataset: str
    size: Optional[int] = None
    seed: Optional[int] = None
    params: Dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class CategoryConfig:
    category: str
    datasets: List[DatasetConfig]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CategoryConfig":
        datasets = [DatasetConfig(**d) for d in data["datasets"]]
        return cls(category=data["category"], datasets=datasets)


@dataclass
class EvalConfig:
    model_path: str
    max_tokens: int
    temperature: float
    top_p: float
    output_dir: str
    save_metadata: bool
    save_full_results: bool
    categories: List[CategoryConfig]

    # Optional: you can provide a system prompt name (looked up in SYSTEM_PROMPTS)
    developer_prompt: Optional[str] = None
    developer_role: str = "system"

    # How many times each question is evaluated
    eval_repeats: int = 1

    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        categories = [CategoryConfig.from_dict(cat) for cat in data["categories"]]
        data["categories"] = categories
        return cls(**data)


class LocalModelEvaluator:
    def __init__(
        self,
        model_path: str,
        config: EvalConfig,
        device: str = "cuda:0",
        batch_size: int = 1,
        verbose: bool = False,
    ):
        self.config = config
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose

        # Load vLLM
        self.llm = LLM(model=model_path)
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
        )

        self.start_time = datetime.now()
        # If you have a system prompt, retrieve it from SYSTEM_PROMPTS
        self.developer_prompt = None
        if self.config.developer_prompt:
            self.developer_prompt = SYSTEM_PROMPTS[self.config.developer_prompt]
        self.developer_role = self.config.developer_role

    def get_model_response(self, question: str) -> str:
        """
        Generates a single response to the given question and returns the
        raw text of that response.
        """
        # Build chat prompt
        chat = []
        if self.developer_prompt:
            chat.append({"role": self.developer_role, "content": self.developer_prompt})
        chat.append({"role": "user", "content": question})

        prompt = self.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        # Generate with vLLM
        outputs = self.llm.generate(prompt, self.sampling_params, use_tqdm=False)
        response = outputs[0].outputs[0].text

        if self.verbose:
            print(f"[Prompt]\n{question}\n[Response]\n{response}\n{'-'*60}")

        return response

    def process_entry(self, question: str, expected_answer: Any) -> Dict[str, Any]:
        all_completions = []
        for _ in range(self.config.eval_repeats):
            try:
                raw = self.get_model_response(question)
                ans = extract_answer(raw)
                try:
                    score = 1 if int(ans) == int(expected_answer) else 0
                except (ValueError, TypeError):
                    score = 0
                    
                all_completions.append(
                    {"model_answer": ans, "full_model_response": raw, "score": score}
                )
            except Exception as e:
                all_completions.append(
                    {"model_answer": None, "full_model_response": "", "score": 0, "error": str(e)}
                )

        scores = [c["score"] for c in all_completions]
        return {
            "question": question,
            "expected_answer": str(expected_answer),
            "best_score": max(scores, default=0),
            "mean_score": sum(scores) / len(scores) if scores else 0.0,
            "completions": all_completions,
        }

    def evaluate_dataset(self, category_name: str, cfg: DatasetConfig, question_col: str, answer_col: str) -> Dict[str, Any]:
        # Load dataset via pandas
        df = pd.read_parquet(cfg.dataset)
        questions = df[question_col].tolist()
        answers = df[answer_col].tolist()

        results = []
        for q, a in tqdm(zip(questions, answers), total=len(questions), desc=f"Processing {cfg.dataset}"):
            results.append(self.process_entry(q, a))

        avg = sum(r["mean_score"] for r in results) / len(results) if results else 0.0
        return {
            "name": cfg.dataset,
            "category": category_name,
            "average_score": avg,
            "total_examples": len(results),
            "config": {**cfg.params, "size": cfg.size, "seed": cfg.seed},
            "results": results,
        }

    def evaluate_all(self, question_col, answer_col) -> Dict[str, Any]:
        cat_results = []
        for cat in self.config.categories:
            ds_results = []
            for ds in cat.datasets:
                ds_results.append(self.evaluate_dataset(cat.category, ds, question_col, answer_col))
            cat_results.append({"name": cat.category, "datasets": ds_results})

        return {
            "metadata": {
                "timestamp": self.start_time.isoformat(),
                "model": self.config.model_path,
                "device": self.device,
                "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "eval_repeats": self.config.eval_repeats,
            },
            "categories": cat_results,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--category")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--question-col", default="question")
    parser.add_argument("--answer-col", default='answer')
    args = parser.parse_args()

    config = EvalConfig.from_yaml(args.config)
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.category:
        config.categories = [c for c in config.categories if c.category == args.category]
        if not config.categories:
            print(f"Category '{args.category}' not found.")
            return 1

    evaluator = LocalModelEvaluator(
        model_path=config.model_path,
        config=config,
        device=args.device,
        batch_size=args.batch_size,
        verbose=args.verbose,
    )

    results = evaluator.evaluate_all(args.question_col, args.answer_col)

    out_dir = Path(config.output_dir) / f"vllm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
