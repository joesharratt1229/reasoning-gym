from src.config import get_default_grpo_config
from src.env import ReasongGymEnv
from src.trainer import GRPOVerifierTrainer
from src.utils import get_model_and_tokenizer

model_name = "Qwen/Qwen2.5-Math-1.5B"
model, tokenizer = get_model_and_tokenizer(model_name)
dataset_name = "chain_sum"
env = ReasongGymEnv(dataset_name=dataset_name, seed=42, size=100, tokenizer=tokenizer)

reward_funcs = env.get_rubric()
dataset = env.get_dataset()

training_args = get_default_grpo_config(run_name="qwen-math-1.5b-verifier", num_gpus=1)
trainer = GRPOVerifierTrainer(
    model, processing_class=tokenizer, reward_funcs=reward_funcs, args=training_args, train_dataset=dataset, env=env
)

trainer.train()
