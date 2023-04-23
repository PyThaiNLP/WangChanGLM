from dataclasses import dataclass, field
from typing import Optional
import openai
from transformers import (
    HfArgumentParser,
)
import jsonlines
from tqdm.auto import tqdm
import pandas as pd
import logging
# Create a logger instance
logger = logging.getLogger('my_logger')
# Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
logger.setLevel(logging.DEBUG)
# Create a console handler and set its level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
# Create a formatter and set it on the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
# Add the console handler to the logger
logger.addHandler(console_handler)

# python eval_vicuna_style.py --input_fname ../data_large/oasst1_benchmark.csv \
# --output_fname ../data_large/oasst1_gpt35turbo_vs_human.csv \
# --eval_col model_human_score \
# --answer1_col answer --answer2_col answer_davinci003 \
# --credentials_jsonl ../../credentials/openai.jsonl

@dataclass
class ScriptArguments:
    input_fname: Optional[str] = field(
        metadata={"help": "path to input file with `prompt` and two `answer` columns"})
    output_fname: Optional[str] = field(metadata={"help": "path to output file"})
    credentials_jsonl: Optional[str] = field(metadata={"help": "path to credential dictionary"})
    eval_col: Optional[str] = field(metadata={"help": "column to store evaluation text"})
    model_name: Optional[str] = field(default="gpt-3.5-turbo", 
                                        metadata={"help": "openai model name"})
    prompt_col: Optional[str] = field(default="prompt",
                                      metadata={"help": "column to get prompt"})
    answer1_col: Optional[str] = field(default="gpt35turbo_answer",
                                      metadata={"help": "column to store answer1"})
    answer2_col: Optional[str] = field(default="answer",
                                      metadata={"help": "column to store answer2"})
    
    

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

credentials = []
with jsonlines.open(script_args.credentials_jsonl) as reader:
    for obj in reader:
        credentials.append(obj)
        
openai.organization = credentials[0]['organization']
openai.api_key = credentials[0]['api_key']

def get_openai_answer(prompt, model_name = script_args.model_name):
    response = openai.ChatCompletion.create(
  model=model_name,
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
)
    return response['choices'][0]['message']['content']

df = pd.read_csv(script_args.input_fname)
logger.info('Input file loaded')

#modified vicuna eval templates
vicuna_template = {"prompt_id": 1, 
                   "system_prompt": "You are a helpful and precise assistant for checking the quality of the answer.", 
                   "prompt_template": "[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{prompt}\n\n", 
                   "defaults": {
                       "prompt": "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 0 to 1, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."
                   }, 
                   "description": "Prompt for general questions", 
                   "category": "general"}

for i,row in tqdm(df.iterrows()):
    try:
        ques = df.loc[i,script_args.prompt_col]
        ans1 = df.loc[i,script_args.answer1_col]
        ans2 = df.loc[i,script_args.answer2_col]

        prompt_template = vicuna_template["prompt_template"]
        user_prompt = prompt_template.format(
            question=ques, answer_1=ans1, answer_2=ans2, **vicuna_template["defaults"]
        )
        response = openai.ChatCompletion.create(
            model=script_args.model_name,
            messages=[
                {"role": "system", "content": vicuna_template["system_prompt"]},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2, 
            max_tokens=64,
        )
        answer = response['choices'][0]['message']['content']
        df.loc[i, script_args.eval_col] = answer
    except Exception as e:
        logger.info(f"""
        {e}
        <human>:{row[script_args.prompt_col]}
        <bot1>:{row[script_args.answer1_col]}
        <bot2>:{row[script_args.answer2_col]}
        """)
        
df.to_csv(script_args.output_fname, index=False)
logger.info('Output file saved')
    