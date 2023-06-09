# kill all processes on GPUs
# sudo fuser -v /dev/nvidia* -k
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig, 
    get_peft_model,
    prepare_model_for_int8_training,
)
from typing import Optional, Union, List, Dict, Any
import evaluate
from dataclasses import dataclass, field
import torch.nn as nn
import numpy as np
import wandb
import multiprocessing
import copy

cpu_cores = multiprocessing.cpu_count()
free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
max_memory = f"{free_in_GB-2}GB"
n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    num_train_epochs: Optional[int] = field(default=3)
    per_device_train_batch_size: Optional[int] = field(default=2)
#     per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=64)
    #lr stuff
    max_learning_rate: Optional[float] = field(default=2e-5)
    min_learning_rate: Optional[float] = field(default=0.)
    weight_decay: Optional[float] = field(default=0.)
    warmup_ratio: Optional[float] = field(default=0.00025)
    #logging stuff
    wandb_project: Optional[str] = field(default="wangchanglm-7.5B-lora-singleworld")
    logging_steps: Optional[int] = field(default=1)
    #model and dataset
    model_name: Optional[str] = field(default="facebook/xglm-7.5B")
    dataset_name: Optional[str] = field(default="pythainlp/final_training_set_v1")
    qa_column: Optional[str] = field(default="text")
    context_start_str: Optional[str] = field(default="<context>:")
    question_start_str: Optional[str] = field(default="<human>:")
    answer_start_str: Optional[str] = field(default="<bot>:")
    ignore_index: Optional[int] = field(default=-100)
    train_split_name: Optional[str] = field(default="train") 
    eval_split_name: Optional[str] = field(default="test") 
    #tokenizer stuff
    max_length: Optional[int] = field(default=512)
    #save stuff
    adapter_name: Optional[str] = field(default='facebook/adapter-wangchanglm-7.5B-lora-singleworld')

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# initialize wandb with project and run names
wandb.init(project=script_args.wandb_project, 
           name=f"{script_args.wandb_project}_{wandb.util.generate_id()}")

# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
training_args = TrainingArguments(
    output_dir=f"{script_args.model_name}",
    learning_rate=script_args.max_learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
#     per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    warmup_ratio=script_args.warmup_ratio,
#     evaluation_strategy="epoch",
#     metric_for_best_model="loss",
#     greater_is_better=False,
    logging_steps=script_args.logging_steps,
    save_strategy="epoch",
    load_best_model_at_end=False,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    remove_unused_columns=False,
    fp16=True
)

##################################################################################################
# MODEL AND TOKENIZER
##################################################################################################

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)

# #not 8bit
# model = AutoModelForCausalLM.from_pretrained(
#     script_args.model_name,
#     device_map='auto',
#     max_memory=max_memory,
# )

#8bit
quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map='auto',
    quantization_config=quantization_config,
    max_memory=max_memory)

#needed for single world model parallel
model.is_parallelizable = True
model.model_parallel = True

print(model)

##################################################################################################
# ADAPTER PREP
##################################################################################################

#8bit
model = prepare_model_for_int8_training(model)

# #not 8bit
# for param in model.parameters():
#   param.requires_grad = False  # freeze the model - train adapters later
#   if param.ndim == 1:
#     # cast the small parameters (e.g. layernorm) to fp32 for stability
#     param.data = param.data.to(torch.float32)
# # model.gradient_checkpointing_enable()  # reduce number of stored activations
# # model.model.decoder.project_in = lambda x: x.requires_grad_(True)
# class CastOutputToFloat(nn.Sequential):
#   def forward(self, x): return super().forward(x).to(torch.float32)
# model.lm_head = CastOutputToFloat(model.lm_head)
                              
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

config = LoraConfig(
    r=128,
    lora_alpha=32, 
    target_modules=[
        "q_proj", "v_proj",
        "k_proj", "out_proj", "fc1", "fc2",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, config)
print_trainable_parameters(model)

##################################################################################################
# DATASET
##################################################################################################

ds = load_dataset(script_args.dataset_name)
# Preprocess the dataset.
def mask_labels(l, context_cue, human_cue, bot_cue):
    result = []
    i = 0
    while i < len(l):
        if (l[i:i+len(human_cue)] == human_cue)|((l[i:i+len(context_cue)] == context_cue)):
            while l[i:i+len(bot_cue)] != bot_cue:
                result.append(-100)
                i += 1
        else:
            result.append(l[i])
            i += 1
    return result
        
def preprocess_function(example):
    tokenized_qa = tokenizer(example[script_args.qa_column]+tokenizer.eos_token, 
                            truncation=True, 
                            padding="max_length",
                            max_length=script_args.max_length,
                            add_special_tokens=False
                            )
    labels = copy.deepcopy(tokenized_qa['input_ids'])
    labels = mask_labels(labels, 
              tokenizer(script_args.context_start_str, add_special_tokens=False)['input_ids'],
              tokenizer(script_args.question_start_str, add_special_tokens=False)['input_ids'],
              tokenizer(script_args.answer_start_str, add_special_tokens=False)['input_ids']
             )
    labels = [script_args.ignore_index if i==tokenizer.pad_token_id else i for i in labels]
    return {
        "input_ids": tokenized_qa["input_ids"],
        "attention_mask": tokenized_qa["attention_mask"],
        "labels": labels,
    }

tokenized_ds = ds.map(preprocess_function, 
                      batched=False, 
                      num_proc=cpu_cores, 
                      remove_columns=ds[script_args.train_split_name].column_names)

##################################################################################################
# TRAINER
##################################################################################################

trainer = Trainer(
    model=model, 
    train_dataset=tokenized_ds[script_args.train_split_name],
    eval_dataset=tokenized_ds[script_args.eval_split_name],
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
model.config.use_cache = True # reenable before saving       
model.save_pretrained(script_args.adapter_name)
                              
#testing inference
context = "นางสาวสิริธิดา พนมวัน ณ อยุธยา ผู้ช่วยผู้ว่าการ สายกำกับระบบการชำระเงินและคุ้มครองผู้ใช้บริการทางการเงิน ธนาคารแห่งประเทศไทย (ธปท.) ชี้แจงกรณีข่าวข้อมูลประชาชนรั่วไหล ว่า ธนาคารแห่งประเทศไทย ร่วมกับศูนย์ประสานงานด้านความมั่นคงปลอดภัยเทคโนโลยีสารสนเทศภาคการธนาคาร (TB-CERT) ภายใต้สมาคมธนาคารไทย (TBA) ในการตรวจสอบระบบของธนาคารแล้ว ไม่พบข้อมูลรั่วไหลจากธนาคาร นอกจากนี้ ข้อมูลที่รั่วไหลออกไปดังกล่าว ไม่สามารถนำไปใช้ทำธุรกรรมทางการเงินผ่าน mobile banking ได้ เนื่องจากยังต้องใช้เครื่องโทรศัพท์มือถือของผู้ใช้บริการ ซึ่งต้องมีรหัสส่วนตัวในการเข้าใช้ รวมทั้งจะต้องยืนยันตัวตนอีกครั้งในการทำธุรกรรม"

batch = tokenizer(f"<context>: {context} <human>: ทำไมข้อมูลที่รั่วไหลนำไปใช้ไม่ได้ <bot>:", 
                  return_tensors='pt')
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
model.eval()
with torch.cuda.amp.autocast():
  output_tokens = model.generate(**batch, max_new_tokens=512)
print('\n\n', tokenizer.decode(output_tokens[0], skip_special_tokens=True))