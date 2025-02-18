from importlib.util import find_spec
from typing import Any, Dict, List, Tuple, Union

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def is_liger_available() -> bool:
    return find_spec("liger_kernel") is not None


def get_model(model_name: str, model_kwargs: Union[Dict[str, Any], None] = None) -> Any:
    if model_kwargs is None:
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
    if is_liger_available():
        print("Using Liger kernel")
        from liger_kernel.transformers import AutoLigerKernelForCausalLM  # type: ignore

        return AutoLigerKernelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)


def get_tokenizer(model_name: str) -> Any:
    if "Instruct" in model_name:
        return AutoTokenizer.from_pretrained(model_name)
    else:
        try:
            return AutoTokenizer.from_pretrained(model_name + "-Instruct")
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            # check if tokenizer has chat_template attribute
            if hasattr(tokenizer, "chat_template"):
                return tokenizer
            else:
                raise ValueError(
                    f"Tokenizer for model {model_name} does not have chat_template attribute, \
                                  and could not find a tokenizer with the same name as the model with suffix \
                                 '-Instruct'. Please provide a tokenizer with the chat_template attribute."
                )


def get_model_and_tokenizer(model_name: str, model_kwargs: Union[Dict[str, Any], None] = None) -> Tuple[Any, Any]:
    model = get_model(model_name, model_kwargs)
    tokenizer = get_tokenizer(model_name)
    return model, tokenizer


def format_prompt(
    prompt: str,
    system_prompt: str | None = None,
    few_shot: List[Dict[str, str]] | None = None,
    fewshot_prob: float = 1.0,
) -> List[Dict[str, str]]:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def preprocess_dataset(dataset, seed, size, tokenizer, developer_prompt, developer_role):
    formatted_data = [
        {
            "prompt": format_prompt(item["question"], developer_prompt, developer_role),
            "answer": item["answer"],
            "metadata": item["metadata"],
        }
        for item in dataset
    ]

    # Convert list of dicts to Hugging Face Dataset
    return Dataset.from_list(formatted_data)
