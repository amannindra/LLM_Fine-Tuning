
from time import perf_counter
start = perf_counter()
from vllm import LLM, SamplingParams
from time import time
from singleInference import parse_arg2

import json
import sys
# from unsloth import FastLanguageModel
import torch
import re
import numpy as np


from timebudget import timebudget
import argparse
import os
from inferenceTest import LLMInference
import random
from datasets import load_dataset


# from Parent import ParentModel
# parent = ParentModel()

end = perf_counter()

elapsed = end - start
print(f"Executed in: {elapsed:.6f} seconds")
print("Imports Loaded")


def make_prompt(question, context) -> str:
    # `context` is a list of passages, or "" for the no-context setting.
    if context:
        context = " ".join(context)
        s = f"""Use the medical context below to answer the question.
        Context: {context}
        Question: {question}
        """
    else:
        s = f"""Answer the medical question.
        Question: {question}
        """

    s += """Give a detailed explanation of your reasoning.

        Explanation: """

    return s



def generate_prompt(arr, ds_art, context = False):
    prompt = []
    print(f"arr: {arr}")
    print(f"ds_art: {ds_art}")
    print(f"context: {context}")
    for index in arr:
        
        try:
            example = ds_art['train'][index]
            question = example["question"]
            contexted = example["context"]["contexts"]

            # print(f"Question: {question}, Context: {contexted}")

            if context:
                prompt.append(make_prompt(question, contexted))
            else:
                prompt.append(make_prompt(question, ""))

            # print(f"Prompt: {prompt}")

            # thinking_content, content = worker_model.inference(prompt, max_new_tokens=max_new_tokens)

            # print(f"Thinking Content: {thinking_content}, Content: {content}")


        except Exception:
            import traceback; traceback.print_exc()
            
    return prompt




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, help="Number of processes to use for multiprocessing.")
    parser.add_argument("--index", type=int, default=1000,
                        help="Number of random examples to evaluate (default: 10).")
    parser.add_argument("--context", action="store_true", help="Include the context in the student's prompt.")
    parser.add_argument("--num-samples", type=int, default=10,
                        help="Number of labeled examples for checkpoint evaluation (default: 4000).")
    parser.add_argument("--fine-tune", action="store_true", help="Use the fine-tuned checkpoint instead of the base model.")
    parser.add_argument("--max-new-tokens", type=int, default=300,
                        help="Max length of the student's explanation (default: 300).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for choosing examples (default: 0).")
    parser.add_argument("--output", help="Write per-example results and the final summary to this .txt file.")

    cli_args = parser.parse_args()
    
    if cli_args.index <= 0:
        parser.error("--index must be a positive integer")
    if cli_args.num_samples <= 0:
        parser.error("--num-samples must be a positive integer")
    if cli_args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be a positive integer")

        
    
    ds_art = load_dataset("qiaojin/PubMedQA", "pqa_artificial")

    # ds_unlabel = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")

    # ds_label = load_dataset("qiaojin/PubMedQA", "pqa_labeled")


    print('Number of CPUs in the system: {}'.format(os.cpu_count()))
    

    sampling_params = SamplingParams(temperature=0.8,
                                     top_p=0.95,
                                     max_tokens=cli_args.max_new_tokens)
    
    # random_index = random.randint(0, cli_args(num-samples))
    
    rng = np.random.default_rng(cli_args.seed)
    
    if cli_args.num_samples >= len(ds_art["train"]):
        print("Samples larger than dataaset length")
        sys.exit()


    arr = rng.choice(len(ds_art["train"]), size=cli_args.num_samples, replace=False)
    
    print(f"arr: {arr}")
    
    context = False
    
    prompts = generate_prompt(arr, ds_art, context)
    
    
    print(f"Example of Prompt: {prompts[0]}")    
    print(f"length of prompt: {len(prompts)}")
    
    # prompts = [
    #     "Hello, my name is",
    #     "The president of the United States is",
    #     "The capital of France is",
    #     "The future of AI is",
    # ]
        
    llm = LLM(model = "unsloth/Llama-3.2-3B-Instruct")
    
    start = perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end = perf_counter()
    
    total = end - start
    
    print(f"Time spent: {total}")
    print(f"Time spend per prompt: {total/len(prompts)}")
    
    i = 0
    out = ""
    for index, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        out += f"[{index}: Prompt: {prompt!r}: Generated text: {generated_text!r}], "
        if i < 5:
            print()
            # print(out)
        i += 1
        
        
    
    with open("filename2.txt", "w") as file:
        file.write(out)

if __name__ == "__main__":
    main()