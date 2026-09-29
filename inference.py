from peft import PeftModel
import torch
from trl import SFTTrainer
from peft import LoraConfig
from main2 import load_data
from transformers import TrainingArguments, DataCollatorForSeq2Seq, AutoModelForCausalLM
from unsloth import is_bfloat16_supported, FastLanguageModel

base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.2-3B-Instruct"
)

model = PeftModel.from_pretrained(
    base_model,
    "outputs/checkpoint-60/adapter_model.safetensors"
)