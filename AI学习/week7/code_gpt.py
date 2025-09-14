import torch
from transformers import AutoTokenizer,GPT2LMHeadModel, GPT2Model
from torch.optim import AdamW

import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)


class GPTTeachingDemo:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")



