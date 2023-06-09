# WangChanGLM 🐘 - The Multilingual Instruction-Following Model

[Blog](https://medium.com/@iwishcognitivedissonance/wangchanglm-the-thai-turned-multilingual-instruction-following-model-7aa9a0f51f5f) | [Codes](https://github.com/pythainlp/wangchanglm) | [Demo](https://colab.research.google.com/github/pythainlp/WangChanGLM/blob/main/demo/WangChanGLM_v0_1_demo.ipynb) 

![](https://i.imgur.com/RL5Mqs3.png)

WangChanGLM is a multilingual, instruction-finetuned Facebook XGLM-7.5B using open-source, commercially permissible datasets (LAION OIG chip2 and infill_dbpedia, DataBricks Dolly v2, OpenAI TL;DR, and Hello-SimpleAI HC3; about 400k examples), released under CC-BY SA 4.0. The models are trained to perform a subset of instruction-following tasks we found most relevant namely: reading comprehension, brainstorming, and creative writing. We provide the weights for a model finetuned on an English-only dataset ([wangchanglm-7.5B-sft-en](https://huggingface.co/pythainlp/wangchanglm-7.5B-sft-en)) and another checkpoint further finetuned on Google-Translated Thai dataset ([wangchanglm-7.5B-sft-enth](https://huggingface.co/pythainlp/wangchanglm-7.5B-sft-enth)). We perform Vicuna-style evaluation using both humans and ChatGPT (in our case, `gpt-3.5-turbo` since we are still on the waitlist for `gpt-4`) and observe some discrepancies between the two types of annoators. All training and evaluation codes are shared under the [Apache-2.0 license](https://github.com/pythainlp/wangchanglm/blob/main/LICENSE) in our Github, as well as datasets and model weights on [HuggingFace](https://huggingface.co/pythainlp). In a similar manner to [Dolly v2](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm), we use only use open-source, commercially permissive pretrained models and datasets, our models are neither restricted by non-commercial clause like models that use LLaMA as base nor non-compete clause like models that use self-instruct datasets from ChatGPT. See our live demo [here](https://colab.research.google.com/github/pythainlp/WangChanGLM/blob/main/demo/WangChanGLM_v0_1_demo.ipynb).

## Models
We provide various versions of our models as follows:
* [pythainlp/wangchanglm-7.5B-sft-en](https://huggingface.co/pythainlp/wangchanglm-7.5B-sft-en)
* [pythainlp/wangchanglm-7.5B-sft-enth](https://huggingface.co/pythainlp/wangchanglm-7.5B-sft-enth)

Sharded versions used in demo:
* [pythainlp/wangchanglm-7.5B-sft-en-sharded](https://huggingface.co/pythainlp/wangchanglm-7.5B-sft-en-sharded)
* [pythainlp/wangchanglm-7.5B-sft-enth-sharded](https://huggingface.co/pythainlp/wangchanglm-7.5B-sft-enth-sharded)

## Training Sets

We provide our training sets as follows:
* [pythainlp/final_training_set_v1](https://huggingface.co/datasets/pythainlp/final_training_set_v1)
* [pythainlp/final_training_set_v1_enth](https://huggingface.co/datasets/pythainlp/final_training_set_v1_enth)

## Finetuning

### Multi-world LoRA

We finetuned XGLM-7.5B on 4 V100 GPU (32GB VARM) with the hyperparameters described in `script/train_sft_peft_multi_world.py`.

```
python -m torch.distributed.launch --nproc_per_node=4 train_sft_peft_multi_world.py \
--per_device_train_batch_size 1 --gradient_accumulation_steps 32 \ #effective batch size = 128 (4 GPUs * 1 batch size * 32 gradient accumulation)
--wandb_project your_project_name \
--model_name facebook/xglm-7.5B \
--dataset_name pythainlp/final_training_set_v1 \ 
--adapter_name save_adapter_to
```

The adapter is merged to the main weights with the script from [lvwerra/trl](https://github.com/lvwerra/trl/blob/main/examples/sentiment/scripts/gpt-neox-20b_peft/merge_peft_adapter.py).

### Single-world LoRA

It is possible to finetune XGLM-7.5B on a single 32GB-VRAM GPU or multiple GPUs with a smaller VRAM with the hyperparameters described in `script/train_sft_peft_single_world.py`.

```
python train_sft_peft_single_world.py \
--per_device_train_batch_size 2 --gradient_accumulation_steps 64 \ #effective batch size = 128 (1 GPU * 2 batch size * 64 gradient accumulation)
--wandb_project your_project_name \
--model_name facebook/xglm-7.5B \
--dataset_name pythainlp/final_training_set_v1 \ 
--adapter_name save_adapter_to
```

### Full-finetuning

We also provide a script for full finetuning we experimented with a smaller model on a different set of training data.

```
python -m torch.distributed.launch --nproc_per_node=8 train_sft.py \
--per_device_train_batch_size=8 --per_device_eval_batch_size=8 --gradient_accumulation_steps=16 \
--model_name=facebook/xglm-1.7B --bf16 --deepspeed=../config/sft_deepspeed_config.json
```

## Inference

We performed inference on the OpenAssistant prompts using hyperparameters described in `script/generate_huggingface_answer.py`.

```
python generate_huggingface_answer.py --input_fname ../data/oasst1_gpt35turbo_answer.csv \
--model_name pythainlp/wangchanglm-7.5B-sft-en \
--tokenizer_name pythainlp/wangchanglm-7.5B-sft-en \
--output_fname ../data/oasst1_wangchang_sft_en_only_answer_answer.csv 
```

## Evaluation

We evaluated any pair of model answers using `gpt-3.5-turbo` as described in `script/eval_vicuna_style.py`. The entire inference and evaluation is stored in `script/infer_and_eval.sh`. The human questionnaires are stored in `data/human_questionnaire`.

## Environmental Impact

Experiments were conducted using a private infrastructure, which has a carbon efficiency of 0.432 kgCO2eq/kWh. A cumulative of 500 hours of computation was performed on hardware of type Tesla V100-SXM2-32GB (TDP of 300W). Total emissions are estimated to be 64.8 CO2eq of which 0 percents were directly offset. Estimations were conducted using the [MachineLearning Impact calculator](https://mlco2.github.io/impact#compute) presented in [lacoste2019quantifying](https://arxiv.org/abs/1910.09700).

## Bibtex
```
@software{charin_polpanumas_2023_7878101,
  author       = {Charin Polpanumas and
                  Wannaphong Phatthiyaphaibun and
                  Patomporn Payoungkhamdee and
                  Peerat Limkonchotiwat and
                  Lalita Lowphansirikul and
                  Can Udomcharoenchaikit and
                  Titipat Achakulwisut and
                  Ekapol Chuangsuwanich and
                  Sarana Nutanong},
  title        = {{WangChanGLM🐘 — The Multilingual Instruction- 
                   Following Model}},
  month        = apr,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v0.1},
  doi          = {10.5281/zenodo.7878101},
  url          = {https://doi.org/10.5281/zenodo.7878101}
}
```

## Acknowledgements

We would like to thank Huggingface for the open-source infrastructure and ecosystem they have built, especially [lvwerra](https://github.com/lvwerra/) of the [trl repository](https://github.com/lvwerra/). We give our appreciation to the open-source finetuning pioneers that come before us including but not limited to [Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [Alpaca-LoRA](https://github.com/tloen/alpaca-lora/), [GPT4All](https://github.com/nomic-ai/gpt4all), [OpenAssistant](https://github.com/LAION-AI/Open-Assistant), [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/), [Vicuna](https://vicuna.lmsys.org/), and [Dolly](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm).

## License

The source code is licensed under the [Apache-2.0 license](https://github.com/pythainlp/wangchanglm/blob/main/LICENSE). The model weights are licensed under [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Finetuning datasets are sourced from [LAION OIG chip2 and infill_dbpedia](https://huggingface.co/datasets/laion/OIG) ([Apache-2.0](https://github.com/pythainlp/wangchanglm/blob/main/LICENSE)), [DataBricks Dolly v2](https://github.com/databrickslabs/dolly) ([Apache-2.0](https://github.com/pythainlp/wangchanglm/blob/main/LICENSE)), [OpenAI TL;DR](https://github.com/openai/summarize-from-feedback) ([MIT](https://opensource.org/license/mit/)), and [Hello-SimpleAI HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) ([CC-BY SA](https://creativecommons.org/licenses/by-sa/4.0/)).


