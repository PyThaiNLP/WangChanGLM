from dataclasses import dataclass, field
from typing import Optional
import openai
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModelForCausalLM,
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

# python generate_huggingface_answer.py --input_fname ../data/oasst1_gpt35turbo_answer.csv \
# --output_fname ../data/oasst1_wangchang_sft_en_only_answer_answer.csv 

@dataclass
class ScriptArguments:
    input_fname: Optional[str] = field(metadata={"help": "path to input file with `prompt`"})
    output_fname: Optional[str] = field(metadata={"help": "path to output file"})
    model_name: Optional[str] = field(default="pythainlp/wangchanglm-7.5B-sft-adapter-merged", 
                                        metadata={"help": "hf model name"})
    tokenizer_name: Optional[str] = field(default="facebook/xglm-7.5B", 
                                        metadata={"help": "hf tokenizer name"})
    prompt_col: Optional[str] = field(default="prompt",
                                      metadata={"help": "column to get prompt"})
    answer_col: Optional[str] = field(default="wangchang_sft_en_answer",
                                      metadata={"help": "column to store answer"})
    

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
logger.info('Tokenizer loaded')
model = AutoModelForCausalLM.from_pretrained(script_args.model_name)
model.eval()
logger.info(f'Model loaded; Memory {model.get_memory_footprint()}')

def infer_answer(prompt):
    input_text = f"<human>: {prompt} <bot>: "
    batch = tokenizer(input_text, return_tensors='pt')
    output_tokens = model.generate(**batch, 
                                   no_repeat_ngram_size=2,
                                   num_beams=5,
                                   min_length=len(batch['input_ids'])+64,
                                   max_new_tokens=512,
#                                    begin_suppress_tokens = exclude_ids,
    #                                logits_processor=[logits_processor],
                                  )
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)


df = pd.read_csv(script_args.input_fname)

logger.info('Input file loaded')

for i,row in tqdm(df.iterrows()):
    try:
        answer = infer_answer(row[script_args.prompt_col]).split('<bot>: ')[-1]
        df.loc[i, script_args.answer_col] = answer
    except Exception as e:
        logger.info(f"""
        {e}
        {row[script_args.prompt_col]}
        """)
        
df.to_csv(script_args.output_fname, index=False)
logger.info('Output file saved')
    