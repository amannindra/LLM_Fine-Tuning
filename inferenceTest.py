from peft import PeftModel
import torch
from trl import SFTTrainer
from peft import LoraConfig
#from main2 import load_data
from transformers import TrainingArguments, DataCollatorForSeq2Seq, AutoModelForCausalLM, AutoTokenizer
# from unsloth import is_bfloat16_supported, FastLanguageModel
import argparse
from pathlib import Path
from datasets import load_dataset
# from main2 import make_prompt


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
class LLMInference:
    def __init__(self, fineTune, max_tokens, location = "outputs/checkpoint-60/"):
        self.device = torch.device("cuda")
        self.max_tokens = max_tokens
        self.base_model = AutoModelForCausalLM.from_pretrained(
            "unsloth/Llama-3.2-3B-Instruct"
            max_tokens = self.max_tokens
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "unsloth/Llama-3.2-3B-Instruct"
        )
        self.location = location
        self.model = None
        self.fineTune = fineTune
        self.load_model(self.location)
        

        
    def load_model(self, location):
        
        if self.fineTune: 
            self.model = PeftModel.from_pretrained(
                self.base_model,
                location
                # args.trained
            )
            self.model = self.model.to(self.device)
            print(f"Loaded fine-tuned model from {location}")
        else:
            self.model = self.base_model.to(self.device)
            print(f"Loaded base model without fine-tuning")
        
    # def make_prompt(question, context) -> str:
    
    #     s = f"""Use the medical context below to answer the question. {context}. Question {question}  Respond with exactly one of these labels:
    #         yes
    #         no
    #         maybe

    #         Answer:"""
            
    #     return s
        
    def inference(self, prompt):
        # prompt = self.make_prompt(prompt)
        messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
        
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip().lower()

        # print(f"final: {response}")
        return "No thinking", response
                
# ds_art = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
# ds_unlabel = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")
# ds_label = load_dataset("qiaojin/PubMedQA", "pqa_labeled")                

# index = 0
# example = ds_art['train'][index]
# question = example["question"]

# answer = example["final_decision"]
# print(f"Test Prompt: {prompt}")

# LLM = LLMInference()

# content = LLM.run(prompt)

# if content == answer:
#     print(f"Index {index}: Correct")
#     print(f"Answer: {answer}, and got: {content}")
    
# else:
#     print(f"Index {index}: Incorrect")
#     print(f"Answer: {answer}, and got: {content}")
    

