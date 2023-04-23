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

# python generate_openai_answer.py --input_fname ../data_large/oasst1_selected.csv \
# --output_fname ../data_large/oasst1_gpt35turbo_answer.csv \
# --credentials_jsonl ../../credentials/openai.jsonl

@dataclass
class ScriptArguments:
    input_fname: Optional[str] = field(metadata={"help": "path to input file with `prompt`"})
    output_fname: Optional[str] = field(metadata={"help": "path to output file"})
    credentials_jsonl: Optional[str] = field(metadata={"help": "path to credential dictionary"})
    model_name: Optional[str] = field(default="gpt-3.5-turbo", 
                                        metadata={"help": "openai model name"})
    prompt_col: Optional[str] = field(default="prompt",
                                      metadata={"help": "column to get prompt"})
    answer_col: Optional[str] = field(default="gpt35turbo_answer",
                                      metadata={"help": "column to store answer"})
    

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

for i,row in tqdm(df.iterrows()):
    try:
        answer = get_openai_answer(row[script_args.prompt_col])
        df.loc[i, script_args.answer_col] = answer
    except Exception as e:
        logger.info(f"""
        {e}
        {row[script_args.prompt_col]}
        """)

df.to_csv(script_args.output_fname, index=False)
logger.info('Output file saved')
    