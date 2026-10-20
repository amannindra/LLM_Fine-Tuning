from time import perf_counter
start = perf_counter()
from datasets import Dataset


import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    default_data_collator,
    AutoProcessor, 
    TrainingArguments,
    Trainer
)

import sys
from peft import LoraConfig, get_peft_model

from trl import SFTTrainer, SFTConfig
from singleInference import load_data

from huggingface_hub import login

login()

end = perf_counter()

elapsed = end - start
print(f"Executed in: {elapsed:.6f} seconds")
print("fine2.py Imports Loaded")




# print(ds_art)

# def create_conversation(example):
#     example = example['train']
    
#     print(example)
    
#     question = example["question"]
#     answer = example["long_answer"]
        
#     context = ""
#     for i in example["context"]["contexts"]:
#         context += i
        
#     system_message = "You are an assistant helping doctors with their questions. You are given the question and the important context you need to answer that question."        
    
#     return {
#             "messages": [
#             {"role": "system", "content": system_message},
#             {"role": "user", "content":f"{question} {context}"},
#             {"role": "assistant", "content": answer}
#         ]
#     }
    


# print(f"Type: {type(ds_art)}")
ds_art, __, __ = load_data()
data_size = 1000

test = ds_art['train'][:data_size]
print(f"Type test: {type(test)}, Keys: {test.keys()}, len(test.values()): {len(test)}")
# mtrain  = test["train"]
new_dict = []

# print("context::::::::",type(test["context"][0]), "length: ", len(test["context"][0]))



# print("context::::::::",test["context"])


# print(f'Question: {test["question"][0]}')
# print(f'Long answer: {test["long_answer"][0]}')
# # print(f'context: {test["context"][0]["contexts"]}')
# # print(f'context: {type(test["context"][0]["contexts"])}')

# L = ""
# for i in test["context"][0]["contexts"]:
#     L += i 
# print(f'context: {L}')
# sys.exit()

for i in range(data_size):
    question = test["question"][i]
    # print(f"get question: {question}")
    answer = test["long_answer"][i]
    # print(f"get answer: {question}")    
    context = ""
    for j in test["context"][i]["contexts"]:
        context += " " + j
    # print(f"get context: {context}")    
        
    system_message = "You are an assistant helping doctors with their questions. You are given the question and the important context you need to answer that question."
    
    message = {
                "prompt": [
                {"role": "system", "content": system_message},
                {"role": "user", "content":f"{question} {context}"},
            ],
            "completion": [
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
    }

    new_dict.append(message)
    

# print(new_dict)

dataset = Dataset.from_list(new_dict)
split = dataset.train_test_split(
    test_size=0.2,
    seed=42
)
train_data = split["train"]
test_data = split["test"]

# MODEL_NAME = "google/gemma-4-E2B-it"

# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


model_name = "Qwen/Qwen3-4B-Base"
device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)

processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto" 
)

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[ "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
    # lora_dropout=0.1,
    # bias="none",
    # modules_to_save=["classifier"],
)
print(f"config complete: {config}")

# peft_model = get_peft_model(model, config)

# print(f"peft_model complete: {peft_model}") 

# peft_model.print_trainable_parameters()


training_args = SFTConfig(
    output_dir="new_output",
    learning_rate=1e-3,
    per_device_train_batch_size=48,
    # per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    # completion_only_loss=True,
    bf16=True,
    max_length=2048,
    report_to="none",
    
)
# print(f"training_args complete: {training_args}") 

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    processing_class=tokenizer,
    peft_config=config,
)

trainer.train()

# trainer = Trainer(
#     model=peft_model,
#     args=training_args,
#     train_dataset=train_data,
#     eval_dataset=test_data,
# #     data_collator=data_collator,
# #     compute_metrics=compute_metrics,
# )

# print(f"trainer complete: {trainer}") 


# trainer.train()


# print(test["new_text"][0])
# print(type(test["new_text"][0]))




# print(f"Type test: {type(test)}")
# test[] = test.map(create_conversation, batched = False)



    

