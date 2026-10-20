import torch
from trl import SFTTrainer
from peft import LoraConfig
from singleInference import load_data
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported, FastLanguageModel


ds_art, ds_unlabel, ds_label = load_data()

# ds_art['train'] = ds_art["train"].select(range(100))

def format_prompt_function(example):
    question = example["question"]
    answer = example["long_answer"]
    context = ""
    for i in example["context"]["contexts"]:
        context += i
    
    example["text"] = f"""You are an assistant helping doctors with their questions. You are given the question and the important context you need to answer that question.
    Question: {question}
    Context: {context}
    long_answer: {answer} """
    
    return example





ds_art["train"] = ds_art["train"].map(format_prompt_function, batched = True)

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
    use_gradient_checkpointing = "none", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

EOS_Token = tokenizer.eos_token

trainer = SFTTrainer(model = model, 
                     train_dataset = ds_art["train"],
                     dataset_text_field = "text",
                     max_seq_length = max_seq_length,
                     dataset_num_proc = 24,
                     args = TrainingArguments(
                        per_device_train_batch_size = 20, # The batch size per GPU/TPU core
                        gradient_accumulation_steps = 4, # Number of steps to perform befor each gradient accumulation
                        warmup_steps = 5, # Few updates with low learning rate before actual training
                        # max_steps = 60, # Specifies the total number of training steps (batches) to run.
                        num_train_epochs = 1,
                        max_steps=-1,
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

torch.cuda.reset_peak_memory_stats()

# trainer_stats = trainer.train()

peak_gb = torch.cuda.max_memory_allocated() / 1024**3

print(f"Peak allocated GPU memory: {peak_gb:.2f} GiB")

trainable_params = 0
total_params = 0

for name, param in model.named_parameters():
    total_params += param.numel()

    if param.requires_grad:
        trainable_params += param.numel()
        print(name, param.shape)

print("Trainable parameters:", trainable_params)
print("Total parameters:", total_params)
print(
    "Trainable percentage:",
    100 * trainable_params / total_params
)
