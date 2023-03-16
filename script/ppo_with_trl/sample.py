# The code can run with DGX v100 1 gpu
import torch
from tqdm import tqdm
import pandas as pd

from transformers import pipeline, AutoTokenizer,AutoModelForCausalLM
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed
from trl.core import LengthSampler
import bitsandbytes as bnb

import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True # fixed full disk in DGX

config = PPOConfig(
    model_name="facebook/xglm-564M",#
    learning_rate=1.41e-5,
    log_with="tensorboard", # change to wandb
    batch_size=16, # batch_size for training!!!
    accelerator_kwargs={"logging_dir":"./report"} # close if use wandb
)

sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16
}
# unclose if use wandb
# import wandb
# wandb.init()

def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
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
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, split='train')
    ds = ds.rename_columns({'text': 'review'})
    ds = ds.filter(lambda x: len(x["review"])>200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[:input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type='torch')
    return ds

dataset = build_dataset(config)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

model = AutoModelForCausalLM.from_pretrained(config.model_name)#,torch_dtype=torch.float16)#,load_in_8bit=True, device_map="auto",torch_dtype=torch.float16)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
model.train()
ref_model = create_reference_model(model)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
#tokenizer.pad_token = tokenizer.eos_token # close because XGLM has pad_token_id

# tokenizer.pad_token = tokenizer.eos_token
#optimizer = bnb.optim.Adam8bit(model.parameters(), lr=config.learning_rate)

ppo_trainer = PPOTrainer(config, model, ref_model=ref_model, tokenizer=tokenizer, dataset=dataset, data_collator=collator)#,optimizer=optimizer)

device = "cpu"
if ppo_trainer.accelerator.num_processes == 1:
   device = 0 if torch.cuda.is_available() else "cpu" # to avoid a `pipeline` bug
sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)

output_min_length = 4
output_max_length = 16
output_length_sampler = LengthSampler(output_min_length, output_max_length)


generation_kwargs = {
    "min_length":-1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id":tokenizer.pad_token_id#.eos_token_id # close because XGLM has pad_token_id
}


for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    query_tensors = batch['input_ids']

    #### Get response from model
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    #### Compute sentiment score
    texts = [q + r for q,r in zip(batch['query'], batch['response'])]
    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

    #### Run PPO step 
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

model.save_pretrained("./xglm-imdb") # save to the moon
tokenizer.save_pretrained("./xglm-imdb") # save to the moon
ref_model.save_pretrained("./xglm-imdb-ref_model") # maybe error
