from time import perf_counter
start = perf_counter()
import json

# from unsloth import FastLanguageModel
import torch
import re

from datasets import load_dataset
from timebudget import timebudget
import argparse
import os
from inferenceTest import LLMInference
import random

end = perf_counter()

elapsed = end - start
print(f"Executed in: {elapsed:.6f} seconds")
print("Imports Loaded")







def make_prompt(question, context) -> str:
    
    s = f"""Use the medical context below to answer the question. {context}. Question {question}  Respond with exactly one of these labels:
        yes
        no
        maybe

        Answer:"""
        
    return s
    

def load_data():
    ds_art = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
    ds_unlabel = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")
    ds_label = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    print(f"Loaded Datasets: {ds_art}, {ds_unlabel}, {ds_label}")
    return ds_art, ds_unlabel, ds_label

def launch_inference(ds_art, worker_model, worker_context, index):
    print(f"Launching index: {index}")
    
    
    try:
        # print("Launching inference for example index:", index)
        example = ds_art['test'][index]
        question = example["question"]
        answer = example["final_decision"]
        contexted  = example["context"]["contexts"]
        
        print(f"Question: {question}, Answer: {answer}, Context: {contexted}")
        
        if worker_context:
            
            prompt = make_prompt(question, contexted)
        else:
            prompt = make_prompt(question, "")
            
        print(f"Prompt: {prompt}")
            
        thinking_content, content = worker_model.inference(prompt)
        
        print(f"Thinking Content: {thinking_content}, Content: {content}")
    
        if content == answer:
            print(f"Index {index}: Correct")
            print(f"Answer: {answer}, and got: {content}")
            return 1
        else:
            print(f"Index {index}: Incorrect")
            print(f"Answer: {answer}, and got: {content}")
            return -1
    except Exception:
        import traceback; traceback.print_exc()
        return 0

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, help="Number of processes to use for multiprocessing.")
    parser.add_argument("--index", type=int, default=10,
                        help="Dataset-size divisor: process len(train) // index examples (default: 1, all examples).")
    parser.add_argument("--context", type=bool, default=False, help="Whether to include context in the prompt (default: True).")
    # parser.add_argument("--checkpoint", help="Local Unsloth/LoRA checkpoint to evaluate on pqa_labeled.")
    parser.add_argument("--num-samples", type=int, default=4000,
                        help="Number of labeled examples for checkpoint evaluation (default: 100).")
    parser.add_argument("--fine-tune", type=bool, default=False, help="Whether to use fine-tuned model (default: False).")
    cli_args = parser.parse_args()
    if cli_args.index <= 0:
        parser.error("--index must be a positive integer")
    if cli_args.num_samples <= 0:
        parser.error("--num-samples must be a positive integer")
    # if cli_args.checkpoint:
    #     evaluate_checkpoint(cli_args.checkpoint, cli_args.num_samples)
    #     return

    
    print("This is the main function.")
    ds_art, ds_unlabel, ds_label = load_data()
    print('Number of CPUs in the system: {}'.format(os.cpu_count()))
    
    indexes = range(0, cli_args.index)
    print(f"Processing {len(indexes)} examples.")
    # print(f"Indexes: {list(indexes)}")

    worker_data = ds_art

    worker_model = LLMInference(cli_args.fine_tune)
    worker_context = cli_args.context
    
    count = 0
    correct = 0
    incorrect = 0
    for i in indexes:
        num = random.randint(0, len(ds_art['test']) - 1)
        output = launch_inference(ds_art, worker_model, worker_context, num)
        print(f"Output for index {num}: {output}")
        if output == 1:
            correct += 1
        elif output == -1:
            incorrect += 1
        else:
            count += 1
    
    print(f"Correct: {correct}, Incorrect: {incorrect}, No Worker: {count}, Total Processed: {len(indexes)}")

    

    
if __name__ == "__main__":     
    main()
