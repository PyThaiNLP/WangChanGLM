# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

import sys
sys.path.append('../')

from dataclasses import dataclass, field
from typing import Optional
import jsonlines
import pandas as pd
import time

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, pipeline, AdamW, BitsAndBytesConfig

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler

import openai
openai.organization = "XXXXXX"
openai.api_key = "XXXXXXX"
tokenizer_name = 'facebook/xglm-7.5B'

free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
max_memory = f"{free_in_GB-2}GB"
n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPT2 model on the IMDB dataset using PPO
# (proximal policy optimization).
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################

########################################################################
# NOTE for to train with a 8-bit model a more recent version of
# transformers is required, full dependecies for this example:
# pip install  bitsandbytes datasets accelerate loralib
# pip install  git+https://github.com/huggingface/transformers.git@main
# pip install peft
########################################################################

# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `accelerator_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.

# Define and parse arguments.

# pythainlp/wangchanglm-7.5B-sft-adapter-merged
# /workspace/deprecated-ChomGPT/chatgpt_reward_model
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(
        default="pythainlp/wangchanglm-7.5B-sft-adapter-merged", metadata={"help": "the model name"}
    )
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=2, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )

print(f"Loading script....")
# parser = HfArgumentParser(ScriptArguments)
# script_args = parser.parse_args_into_dataclasses()[0]
script_args = ScriptArguments

config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
)

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(config, dataset_name="pythainlp/final_training_set_v1_rlhf", input_min_text_length=4, input_max_text_length=5114):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,add_special_tokens = False)
    ds = load_dataset(dataset_name)["train"]
    

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["prompt"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

print(f"Buliding data....")
# We retrieve the dataloader by calling the `build_dataset` function.
dataset = build_dataset(config)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer.
print(f"Loading model....")
# quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
pretrained_model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=torch.float16,
    device_map='auto',
    max_memory=max_memory)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name,add_special_tokens = False)

"""### Apply LoRA
Here comes the magic with `peft`! Let's load a `PeftModel` and specify that we are going to use low-rank adapters (LoRA) using `get_peft_model` utility function from `peft`.
"""

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


lora_config = LoraConfig(
    r=128,
    lora_alpha=32,
    target_modules=[
        "q_proj", "v_proj",
        "k_proj", "out_proj", "fc1", "fc2",
    ], 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

pretrained_model = prepare_model_for_int8_training(pretrained_model, output_embedding_layer_name="embed_out")
pretrained_model = get_peft_model(pretrained_model, lora_config)

model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained_model)

model.gradient_checkpointing_disable = model.pretrained_model.gradient_checkpointing_disable
model.gradient_checkpointing_enable = model.pretrained_model.gradient_checkpointing_enable

print_trainable_parameters(model)

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config, model, ref_model=None, tokenizer=tokenizer, dataset=dataset, data_collator=collator, optimizer=optimizer
)

# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": -1,
}
output_min_length = 4
output_max_length = 128
output_length_sampler = LengthSampler(output_min_length, output_max_length)

print(f"Training stage....")

# We then build the reward pipeline with ChatGPT
def chatgpt_reward(queries,predicted_respones):
    results = []
    for q,pr in zip(queries,predicted_respones):
        prompt_text = f'''Please rate the helpfulness, relevance, accuracy, level of details of the response of <bot> to a query by <human>. The overall score is on a scale of 0 to 1 ('answer only the overall score no other texts'), where a higher score indicates better overall performance . 

    {q.replace('</s> ','')}{pr}

    Overall Score:'''
        while True:
            try:
                response = openai.ChatCompletion.create(
                  model='gpt-3.5-turbo',
                  messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt_text},
                    ]
                )
            except:
                print(f"API failed... trying again in 3 minutes ")
                time.sleep(180)
                continue     
            break

        answer = response["choices"][0]["message"]["content"]
        if answer[:4] == 'Help':
            answer = answer[10:]
        final_answer = ''
        for ch in answer:
            if ch in ['0','1','2','3','4','5','6','7','8','9','.']:
                final_answer+=ch
            else:
                break
        try:
            if final_answer[-1] == '.':
                final_answer = final_answer[:-1]
        except:
            results.append(0.0) # when ChatGPT said "can't give the score. We give 0 to the response"
            continue
            
        results.append(float(final_answer))
        
    if len(results) != len(queries):
        raise Exception(f"Len of reward and query is not equal {len(results)} vs {len(queries)}")
    return results

df_chatpgt = pd.DataFrame([[0,0,0]],columns=['input','output','score'])
start_iter = 0
for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    print(f"Iteration:{epoch+1}")
    if (epoch+1) >= start_iter:
        query_tensors = batch["input_ids"]

        model.gradient_checkpointing_disable()
        model.pretrained_model.config.use_cache = True
        # Get response from Causal LM
        print(f"Generating...")
        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            with torch.cuda.amp.autocast():
                response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch["response"] = [tokenizer.decode(r.squeeze()).split('</s>')[0] for r in response_tensors]

        # Compute sentiment score
        print(f"Rewarding...")
        chatgpt_labels = chatgpt_reward(batch["query"],batch["response"])
        rewards = [torch.tensor(output) for output in chatgpt_labels]

        # save score to log file
        df_input = []
        for index in range(len(batch["response"])):
            df_chatpgt.loc[len(df_chatpgt)] = [batch['query'][index],batch["response"][index],chatgpt_labels[index]]
        df_chatpgt.to_csv('chatgpt_reward_model/reward.csv', index=False)

        # Run PPO step
        model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False

        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

        print(f"Saving model...")
        try:
            model.push_to_hub("HF Repo",private=True,max_shard_size="400MB")
        except:
            continue
        

