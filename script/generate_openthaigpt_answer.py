from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    HfArgumentParser,
)
import openthaigpt
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

# python generate_openthaigpt_answer.py --input_fname ../data/oasst1_wangchang_sft_en_answer.csv \
# --output_fname ../data/oasst1_openthaigpt_answer.csv

@dataclass
class ScriptArguments:
    input_fname: Optional[str] = field(metadata={"help": "path to input file with `prompt`"})
    output_fname: Optional[str] = field(metadata={"help": "path to output file"})
    prompt_col: Optional[str] = field(default="prompt",
                                      metadata={"help": "column to get prompt"})
    answer_col: Optional[str] = field(default="openthaigpt010_answer",
                                      metadata={"help": "column to store answer"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

df = pd.read_csv(script_args.input_fname)
df = df[df.lang=='th'] #openthaigpt is monolingual
logger.info('Input file loaded')

for i,row in tqdm(df.iterrows()):
    try:
        #config from example notebook https://colab.research.google.com/drive/1Uds0ioOZSZrJ9m2FgW3DHlqVRFNHVRtu
        answer = openthaigpt.generate(instruction=i['text'], 
            input="",
            model_name = "kobkrit/openthaigpt-0.1.0-alpha",
            min_length=50,
            max_length=768,
            top_k=20,
            num_beams=5,
            no_repeat_ngram_size=20,
            temperature=1,
            early_stopping=True
        )
        df.loc[i, script_args.answer_col] = answer
    except Exception as e:
        logger.info(f"""
        {e}
        {row[script_args.prompt_col]}
        """)
        
df.to_csv(script_args.output_fname, index=False)
logger.info('Output file saved')
    