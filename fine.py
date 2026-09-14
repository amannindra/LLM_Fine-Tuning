from unsloth import FastLanguageModel
import torch

from main2 import load_data

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3-4B",
    max_seq_length=max_seq_length,
    dtype = dtype,
    load_in_4bit =load_in_4bit
    
)

print(f"Model is loaded and Tokenizer is loaded")

model = FastLanguageModel.get_peft_model(model,
                                         r = 16, 
                                         target_modules = [
                                             "q_proj", "k_proj",
                                             "v_proj", "o_proj",
                                             "gate_proj", "up_proj", 
                                             "down_proj"]
                                    , lora_alpha = 16,
                                    lora_dropout = 0, 
                                    bias = "none",
                                    use_gradient_checkpoint = "unsloth", 
                                    random_state = 3407,
                                    use_rslora = False,
                                    loftq_config = None
                                    )


ds_art, ds_unlabel, ds_label = load_data()


EOS_Token = tokenizer.eos_token

def format_prompt_function(example):
    question = example["question"]
    answer = example["final_decision"]
    context = example["context"]["contexts"]
    
    example["text"] = f"""You are an assistant helping doctors with their questions. You are given the question and the important context you need to answer that question.
    Question: {question}
    Context: {context}
    answer: {answer}
"""
    

