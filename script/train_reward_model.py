from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerBase,
    HfArgumentParser,
    AdamW,
)
from transformers.utils import PaddingStrategy
from deepspeed.runtime.lr_schedules import WarmupDecayLR
from typing import Optional, Union, List, Dict, Any
import evaluate
from dataclasses import dataclass, field
import torch.nn as nn
import numpy as np
import wandb
import multiprocessing
cpu_cores = multiprocessing.cpu_count()

# python -m torch.distributed.launch --nproc_per_node=8 train_reward_model.py \
# --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --gradient_accumulation_steps=2 \
# --model_name=xlm-roberta-base --bf16 --deepspeed=../config/reward_model_deepspeed_config.json

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    num_train_epochs: Optional[int] = field(default=2)
    resume_from_checkpoint: Optional[bool] = field(default=False)
    #multigpu stuff
    local_rank: Optional[int] = field(default=0)
    deepspeed: Optional[str] = field(default=None)
    per_device_train_batch_size: Optional[int] = field(default=8)
    per_device_eval_batch_size: Optional[int] = field(default=8)
    gradient_accumulation_steps: Optional[int] = field(default=2)
    #lr stuff
    max_learning_rate: Optional[float] = field(default=5e-5)
    min_learning_rate: Optional[float] = field(default=0.)
    weight_decay: Optional[float] = field(default=0.001)
    warmup_ratio: Optional[float] = field(default=0.1)
    #logging stuff
    wandb_project: Optional[str] = field(default="php_reward_model")
    logging_steps: Optional[int] = field(default=50)
    #eval stuff
    eval_steps: Optional[int] = field(default=500)
    #model and dataset
    model_name: Optional[str] = field(default="xlm-roberta-base")
    dataset_name: Optional[str] = field(default="pythainlp/php_reward")
    better_column: Optional[str] = field(default="human_ref_1st")
    worse_column: Optional[str] = field(default="human_ref_2nd")
    train_split_name: Optional[str] = field(default="train") 
    eval_split_name: Optional[str] = field(default="test") 
    #tokenizer stuff
    max_length: Optional[int] = field(default=512)
    #half precision stuff
    bf16: Optional[bool] = field(default=True,)

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# initialize wandb with project and run names
wandb.init(project=script_args.wandb_project, 
           name=f"{script_args.wandb_project}_{wandb.util.generate_id()}")

# Load the human comparisons dataset for tuning the reward model.
ds = load_dataset(script_args.dataset_name)
# #debug
# ds['train'] = ds['train'].select([i for i in range(1000)])
# ds['test'] = ds['test'].select([i for i in range(100)])

# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
training_args = TrainingArguments(
    output_dir=f"{script_args.model_name}_reward_model",
    learning_rate=script_args.max_learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    warmup_ratio=script_args.warmup_ratio,
    evaluation_strategy="steps",
    eval_steps=script_args.eval_steps,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=script_args.logging_steps,
    save_strategy="steps",
    save_steps=script_args.eval_steps,
    load_best_model_at_end=True,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    label_names=[],
    remove_unused_columns=False
)

# Load the value-head model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
model = AutoModelForSequenceClassification.from_pretrained(script_args.model_name, num_labels=1)

# Need to do this for gpt2, because it doesn't have an official pad token.
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Tokenize the dataset.
def preprocess_function(examples):
    tokenized_j = tokenizer(examples[script_args.better_column], 
                            truncation=True)
    tokenized_k = tokenizer(examples[script_args.worse_column], 
                            truncation=True)
    return {
        "input_ids_j": tokenized_j["input_ids"],
        "attention_mask_j": tokenized_j["attention_mask"],
        "input_ids_k": tokenized_k["input_ids"],
        "attention_mask_k": tokenized_k["attention_mask"],
    }

tokenized_ds = ds.map(preprocess_function, batched=True, 
                      num_proc=cpu_cores, 
                      remove_columns=None)

# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = script_args.max_length
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        features_j = []
        features_k = []
        for feature in features:
            features_j.append({"input_ids": feature["input_ids_j"], 
                               "attention_mask": feature["attention_mask_j"]})
            features_k.append({"input_ids": feature["input_ids_k"], 
                               "attention_mask": feature["attention_mask_k"]})
        batch_j = self.tokenizer.pad(
            features_j,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_k = self.tokenizer.pad(
            features_k,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_j": batch_j["input_ids"],
            "attention_mask_j": batch_j["attention_mask"],
            "input_ids_k": batch_k["input_ids"],
            "attention_mask_k": batch_k["attention_mask"],
            "return_loss": True,
        }
        return batch


# Define the metric that we'll use for validation.
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_j and rewards_k.
    # We want to see how much of the time rewards_j > rewards_k.
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape) #first one always better
    return accuracy.compute(predictions=predictions, references=labels)


class RewardTrainer(Trainer):
    # Define how to compute the reward loss.
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_j = model(input_ids=inputs["input_ids_j"], 
                          attention_mask=inputs["attention_mask_j"])[0]
        rewards_k = model(input_ids=inputs["input_ids_k"], 
                          attention_mask=inputs["attention_mask_k"])[0]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        params = self.get_model().parameters()
        optimizer = AdamW(params, lr=self.args.max_learning_rate, 
                          weight_decay=self.args.weight_decay,
                          bias_correction=True)
        total_steps = num_training_steps
        warmup_steps = int(self.args.warmup_ratio*total_steps)
        scheduler = WarmupDecayLR(optimizer, total_num_steps=total_steps,
                                  warmup_min_lr=script_args.min_learning_rate,
                                  warmup_max_lr=script_args.max_learning_rate,
                                  warmup_num_steps=warmup_steps,)
        return optimizer, scheduler

# Train the model, woohoo.
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds[script_args.train_split_name],
    eval_dataset=tokenized_ds[script_args.eval_split_name],
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer),
)

trainer.train(script_args.resume_from_checkpoint)

# Push to the hub so you can share it with people :D
# model.push_to_hub(f"{script_args.model_name}_reward_model")
# tokenizer.push_to_hub(f"{script_args.model_name}_reward_model")