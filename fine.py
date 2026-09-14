from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from peft import LoraConfig
from main2 import load_data
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported


ds_art, ds_unlabel, ds_label = load_data()

ds_art = ds_art.select(range(100))

def format_prompt_function(example):
    question = example["question"]
    answer = example["final_decision"]
    context = example["context"]["contexts"]
    
    example["text"] = f"""You are an assistant helping doctors with their questions. You are given the question and the important context you need to answer that question.
    Question: {question}
    Context: {context}
    answer: {answer}
"""
    return example



    
ds_art = ds_art.map(format_prompt_function, batched = True)

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

model = FastLanguageModel.get_peft_model( model,
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


EOS_Token = tokenizer.eos_token

trainer = SFTTrainer(model = model, 
                     train_dataset = ds_art,
                     dataset_text_field = "text",
                     max_seq_length = max_seq_length,
                     dataset_num_proc = 24,
                    args = TrainingArguments(
                        per_device_train_batch_size = 20, # The batch size per GPU/TPU core
                        gradient_accumulation_steps = 4, # Number of steps to perform befor each gradient accumulation
                        warmup_steps = 5, # Few updates with low learning rate before actual training
                        max_steps = 1, # Specifies the total number of training steps (batches) to run.
                        learning_rate = 2e-4,
                        fp16 = not is_bfloat16_supported(),
                        bf16 = is_bfloat16_supported(),
                        logging_steps = 1,
                        optim = "adamw_8bit", # Optimizer
                        weight_decay = 0.01,
                        lr_scheduler_type = "linear",
                        seed = 3407,
                        output_dir = "outputs",
                        report_to = "none", # Use this for WandB etc for observability
                    ),
                )

trainer_stats = trainer.train()