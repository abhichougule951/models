!pip install -qqq torch --progress-bar off
!pip install -qqq transformers --progress-bar off
!pip install -qqq datasets --progress-bar off
!pip install -qqq peft --progress-bar off
!pip install -qqq bitsandbytes --progress-bar off
!pip install -qqq trl --progress-bar off
!pip install -qqq sentencepiece --progress-bar off
!pip install -qqq einops --progress-bar off
!pip install -qqq transformers[sentencepiece] --progress-bar off
!pip install -qqq huggingface-hub --progress-bar off
!pip install -qqq wandb --progress-bar off
!pip install -qqq -U git+https://github.com/huggingface/accelerate.git --progress-bar off

HF_TOKEN = "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"
!huggingface-cli login --token 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


import json
import re
from pprint import pprint

import pandas as pd
import json
import random
import torch
from datasets import Dataset, load_dataset, DatasetDict
from peft import LoraConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    logging,
    pipeline,
    TextStreamer,
)
from trl import SFTTrainer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# MODEL_NAME = "MayurPai/Llama-2-7b-hf-fine-tuned"
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

%%time

def create_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map={"": 0},
        token=HF_TOKEN
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

model, tokenizer = create_model_and_tokenizer()
model.config.use_cache = False

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_llama2_chat_reponse(prompt, max_new_tokens=500):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, top_k=2, top_p=0.1, temperature=0.1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
   
    response_start = response.find("Response:") + len("Response:")
    if response_start == len("Response:") - 1:
        # If "Response:" is not found, return the entire response as is
        return response
    
    clean_response = response[response_start:].strip()
    
    return clean_response


%%time
previous_conversation = "Human:I want to install python in my pc. Bot: Please provide version,duration and business justification"
current_conversation = "Human:I need local admin access"

prompt = f'''
[INST]<<SYS>>
You are a decision-making bot. Your task is to analyze two conversations and determine if the Current Conversation is related to the previous conversation. If the current conversation has the same intent or current conversation is the answer to the previous conversation, respond as "Relevant". Otherwise, if its not answer its breaking flow then respond as "Not Relevant".<</SYS>>

Previous Conversation:
{previous_conversation}

Current Conversation:
{current_conversation}

[/INST]

Response:
'''

response = get_llama2_chat_reponse(prompt, max_new_tokens=100)
print(response)