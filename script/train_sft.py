from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizerBase,
    HfArgumentParser,
    AdamW,
    default_data_collator,
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

# python -m torch.distributed.launch --nproc_per_node=8 train_sft.py \
# --per_device_train_batch_size=1 --per_device_eval_batch_size=1 --gradient_accumulation_steps=4 \
# --model_name=facebook/xglm-1.7B --bf16 --deepspeed=../config/sft_deepspeed_config.json

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
    max_learning_rate: Optional[float] = field(default=1e-5)
    min_learning_rate: Optional[float] = field(default=0.)
    weight_decay: Optional[float] = field(default=0.001)
    warmup_ratio: Optional[float] = field(default=0.1)
    #logging stuff
    wandb_project: Optional[str] = field(default="php_sft_model")
    logging_steps: Optional[int] = field(default=50)
    #eval stuff
    eval_steps: Optional[int] = field(default=500)
    #model and dataset
    model_name: Optional[str] = field(default="facebook/xglm-1.7B")
    dataset_name: Optional[str] = field(default="pythainlp/php_sft")
    question_column: Optional[str] = field(default="question")
    answer_column: Optional[str] = field(default="answer")
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
# # #debug
ds['train'] = ds['train'].select([i for i in range(1000)])
ds['test'] = ds['test'].select([i for i in range(10)])

# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
training_args = TrainingArguments(
    output_dir=f"{script_args.model_name}_sft_model",
    learning_rate=script_args.max_learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    warmup_ratio=script_args.warmup_ratio,
    evaluation_strategy="steps",
    eval_steps=script_args.eval_steps,
    metric_for_best_model="rougeL",
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
model = AutoModelForCausalLM.from_pretrained(script_args.model_name)

# Tokenize the dataset.
def preprocess_function(examples):
    tokenized_question = tokenizer(examples[script_args.question_column], 
                            truncation=True, 
                            padding="max_length",
                            max_length=script_args.max_length,
                            )
    tokenized_answer = tokenizer(examples[script_args.answer_column], 
                            truncation=True,
                            padding="max_length",
                            max_length=script_args.max_length,
                            )
    return {
        "input_ids": tokenized_question["input_ids"],
        "attention_mask": tokenized_question["attention_mask"],
        "labels": tokenized_answer["input_ids"],
    }

tokenized_ds = ds.map(preprocess_function, 
                      batched=True, 
                      num_proc=cpu_cores, 
                      remove_columns=ds[script_args.train_split_name].column_names)

#use rouge for metric; not really representative but ok
rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    labels_ids = eval_preds.label_ids
    pred_ids = eval_preds.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    result = rouge.compute(predictions=pred_str, references=label_str)
    result = {'rougeL': result['rouge-l'].mid.fmeasure}
    return result

# Create a preprocessing function to extract out the proper logits from the model output
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

class SFTTrainer(Trainer):
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
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds[script_args.train_split_name],
    eval_dataset=tokenized_ds[script_args.eval_split_name],
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

trainer.train(script_args.resume_from_checkpoint)

# Push to the hub so you can share it with people :D
# model.push_to_hub(f"{script_args.model_name}_reward_model")
# tokenizer.push_to_hub(f"{script_args.model_name}_reward_model")