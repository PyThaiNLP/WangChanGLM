#! /bin/bash
#infer gpt35turbo
# python generate_openai_answer.py --input_fname ../data_large/oasst1_selected.csv \
# --output_fname ../data_large/oasst1_gpt35turbo_answer.csv \
# --credentials_jsonl ../../credentials/openai.jsonl

#infer wangchang_sft_en
python generate_huggingface_answer.py --input_fname ../data/oasst1_gpt35turbo_answer.csv \
--output_fname ../data/oasst1_wangchang_sft_en_answer.csv 

#human vs gpt35turbo
python eval_vicuna_style.py --input_fname ../data/oasst1_wangchang_sft_en_answer.csv \
--output_fname ../data/eval_human_vs_gpt35turbo.csv \
--eval_col eval_score \
--answer1_col answer --answer2_col gpt35turbo_answer \
--credentials_jsonl ../../credentials/openai.jsonl

#human vs wangchang_sft_en
python eval_vicuna_style.py --input_fname ../data/oasst1_wangchang_sft_en_answer.csv \
--output_fname ../data/eval_human_vs_wangchang_sft_en.csv \
--eval_col eval_score \
--answer1_col answer --answer2_col wangchang_sft_en_answer \
--credentials_jsonl ../../credentials/openai.jsonl

#gpt35turbo vs wangchang_sft_en
python eval_vicuna_style.py --input_fname ../data/oasst1_wangchang_sft_en_answer.csv \
--output_fname ../data/eval_gpt35turbo_vs_wangchang_sft_en.csv \
--eval_col eval_score \
--answer1_col gpt35turbo_answer --answer2_col wangchang_sft_en_answer \
--credentials_jsonl ../../credentials/openai.jsonl