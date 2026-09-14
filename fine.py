from unsloth import FastLanguageModel
import torch

from main2 import load_data

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct", # unsloth/Llama-3.2-3B-Instruct #Qwen/Qwen3-4B
    max_seq_length=max_seq_length,
    dtype = dtype,
    load_in_4bit =load_in_4bit
    
)

print(f"Model is loaded and Tokenizer is loaded")

model = FastLanguageModel.get_peft_model(  model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16, # a higher alpha value assigns more weight to the LoRA activations
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,

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
    

