from peft import PeftModel
import torch
from trl import SFTTrainer
from peft import LoraConfig
from main2 import load_data
from transformers import TrainingArguments, DataCollatorForSeq2Seq, AutoModelForCausalLM, AutoTokenizer
from unsloth import is_bfloat16_supported, FastLanguageModel
import argparse
from pathlib import Path

# base_model = AutoModelForCausalLM.from_pretrained(
#     "unsloth/Llama-3.2-3B-Instruct"
# )

# parser = argparse.ArgumentParser()
# parser.add_argument("--trained", required=True, type=Path)
# args = parser.parse_args()

# model = PeftModel.from_pretrained(
#     base_model,
#     args.trained
# )

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


base_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.2-3B-Instruct"
)


tokenizer = AutoTokenizer.from_pretrained(
    "unsloth/Llama-3.2-3B-Instruct"
)


location = "outputs/checkpoint-60/"


model = PeftModel.from_pretrained(
    base_model,
    location
    # args.trained
)
model = model.to(device)

print("model is loaded")

messages = [
    {
        "role": "user",
        "content": """
        Question: What causes diabetes?

        Context:
        Diabetes occurs when the body cannot properly regulate blood glucose.
        """
    }
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda:0")

print("input is loaded")


outputs = model.generate(
    inputs,
    max_new_tokens=2048,
    temperature=0.7,
)

print("output is generated")

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print(response)
