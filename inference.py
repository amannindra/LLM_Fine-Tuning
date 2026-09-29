from peft import PeftModel
import torch
from trl import SFTTrainer
from peft import LoraConfig
from main2 import load_data
from transformers import TrainingArguments, DataCollatorForSeq2Seq, AutoModelForCausalLM
from unsloth import is_bfloat16_supported, FastLanguageModel
import argparse
from pathlib import Path

base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.2-3B-Instruct"
)

parser = argparse.ArgumentParser()
parser.add_argument("--trained", required=True, type=Path)
args = parser.parse_args()

model = PeftModel.from_pretrained(
    base_model,
    args.trained
)