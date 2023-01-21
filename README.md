# ChomGPT
ChomGPT - ChatGPT-like model for Thai

### How to install trl

1. Clone Transformer Reinforcement Learning (trl)

> git clone https://github.com/lvwerra/trl.git

2. copy the patch to trln and apply the patch

> cp trl_patch/0001-add-stop-token-commits.patch trl/0001-add-stop-token-commits.patch

> cd trl

> git apply 0001-add-stop-token-commits.patch

3. Install

> pip install -e .

### How 8 bit to load (Big) model

1. Install bitsandbytes by CUDA; choices: {cuda92, cuda 100, cuda101, cuda102, cuda110, cuda111, cuda113} and replace XXX with the respective number

> pip install bitsandbytes-cudaXXX

2. copy the patch to trln and apply the patch

> cp trl_patch/0002-use_8bit.patch trl/0002-use_8bit.patch

> cd trl

> git apply 0002-use_8bit.patch

3. Update the load model

> model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained).to(device)

to

> model = AutoModelForCausalLMWithValueHead.from_pretrained(pretrained, load_in_8bit=True).to(device)

and

> ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer)

to

> ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer, use_8bit=True)

4. Done
