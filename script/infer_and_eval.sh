#! /bin/bash
# #infer gpt35turbo
# python generate_openai_answer.py --input_fname ../data_large/oasst1_selected.csv \
# --output_fname ../data_large/oasst1_gpt35turbo_answer.csv \
# --credentials_jsonl ../../credentials/openai.jsonl

# #infer wangchang_sft_en
# python generate_huggingface_answer.py --input_fname ../data/oasst1_gpt35turbo_answer.csv \
# --output_fname ../data/oasst1_wangchang_sft_en_answer.csv 

# #infer wangchang_sft_enth
# python generate_huggingface_answer.py --input_fname ../data/oasst1_wangchang_sft_en_answer.csv \
# --answer_col wangchang_sft_enth_answer --model_name pythainlp/xglm-7.5B-en_th-adapter-merged \
# --output_fname ../data/oasst1_wangchang_sft_enth_answer.csv

# #human vs gpt35turbo
# python eval_vicuna_style.py --input_fname ../data/oasst1_wangchang_sft_en_answer.csv \
# --output_fname ../data/eval_human_vs_gpt35turbo.csv \
# --eval_col eval_score \
# --answer1_col answer --answer2_col gpt35turbo_answer \
# --credentials_jsonl ../../credentials/openai.jsonl

# #human vs wangchang_sft_en
# python eval_vicuna_style.py --input_fname ../data/oasst1_wangchang_sft_en_answer.csv \
# --output_fname ../data/eval_human_vs_wangchang_sft_en.csv \
# --eval_col eval_score \
# --answer1_col answer --answer2_col wangchang_sft_en_answer \
# --credentials_jsonl ../../credentials/openai.jsonl

# #gpt35turbo vs wangchang_sft_en
# python eval_vicuna_style.py --input_fname ../data/oasst1_wangchang_sft_en_answer.csv \
# --output_fname ../data/eval_gpt35turbo_vs_wangchang_sft_en.csv \
# --eval_col eval_score \
# --answer1_col gpt35turbo_answer --answer2_col wangchang_sft_en_answer \
# --credentials_jsonl ../../credentials/openai.jsonl

# #human vs wangchang_sft_enth
# python eval_vicuna_style.py --input_fname ../data/oasst1_wangchang_sft_enth_answer.csv \
# --output_fname ../data/eval_human_vs_wangchang_sft_enth.csv \
# --eval_col eval_score \
# --answer1_col answer --answer2_col wangchang_sft_enth_answer \
# --credentials_jsonl ../../credentials/openai.jsonl

# #gpt35turbo vs wangchang_sft_enth
# python eval_vicuna_style.py --input_fname ../data/oasst1_wangchang_sft_enth_answer.csv \
# --output_fname ../data/eval_gpt35turbo_vs_wangchang_sft_enth.csv \
# --eval_col eval_score \
# --answer1_col gpt35turbo_answer --answer2_col wangchang_sft_enth_answer \
# --credentials_jsonl ../../credentials/openai.jsonl

#wangchang_sft_en vs openthaigpt
python eval_vicuna_style.py --input_fname ../data/oasst1_openthaigpt_answer.csv \
--output_fname ../data/eval_wangchang_sft_en_vs_openthaigpt.csv \
--eval_col eval_score \
--answer1_col wangchang_sft_en_answer --answer2_col openthaigpt010_answer \
--credentials_jsonl ../../credentials/openai.jsonl

#human vs openthaigpt_answer
python eval_vicuna_style.py --input_fname ../data/oasst1_openthaigpt_answer.csv \
--output_fname ../data/eval_human_vs_openthaigpt.csv \
--eval_col eval_score \
--answer1_col answer --answer2_col openthaigpt010_answer \
--credentials_jsonl ../../credentials/openai.jsonl