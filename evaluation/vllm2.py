
from datasets import load_dataset
from time import perf_counter
start = perf_counter()
from vllm import LLM, SamplingParams
from time import time
# from singleInference import parse_arg2
from vllm.sampling_params import StructuredOutputsParams

import json
import sys
# from unsloth import FastLanguageModel
import torch
import re
import numpy as np


from timebudget import timebudget
import argparse
import os
# from inferenceTest import LLMInference
import random

from pathlib import Path


# from Parent import ParentModel
# parent = ParentModel()

end = perf_counter()

elapsed = end - start
print(f"Executed in: {elapsed:.6f} seconds")
print("Imports Loaded")


def make_prompt(question, context, max_new_tokens) -> str:
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

    s += f"""Give a detailed explanation of your reasoning. Make sure your answer is at most {max_new_tokens} characters long.

        Explanation: """

    return s

def make_judge_prompt(question, context, answer, student, max_new_tokens = 400) -> str:
    s = f"""You are an expert doctor grading a student's answer to a medical question. You are given the question, the relevant context, the answer to the question, and the student's explanation.

    Decide whether the student's explanation and final answer are correct, based on the context and your medical knowledge.

    Question: {question}
    Context: {" ".join(context)}
    answer: {answer}

    Student's explanation: {student}

    Is the student correct? Respond with exactly one of these labels:
    yes
    no

    Answer:"""

    return s


def generate_parent_prompt(out):
    prompts = []
    for i in range(len(out)):
        example = out[i]
        question = example["Question"]
        context = example["Context"]
        answer = example["Answer"]
        final_answer = example["Final_decision"]
        student = example["Model_output"]
        max_tokens = example["Max_Content"]
        
        parent_prompt = make_judge_prompt(question, context, answer, student, max_tokens)
        prompts.append(parent_prompt)
    return prompts
        

def save_file(path, data):
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)

    with open(location, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def generate_student_prompt(arr, ds_art, context = False, max_new_tokens = 400):
    prompt = [] # {Question: }

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
                prompt.append(make_prompt(question, contexted, max_new_tokens))
            else:
                prompt.append(make_prompt(question, "", max_new_tokens))
                
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
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of labeled examples for checkpoint evaluation (default: 4000).")
    parser.add_argument("--fine-tune", action="store_true", help="Use the fine-tuned checkpoint instead of the base model.")
    parser.add_argument("--max-new-tokens", type=int, default=400,
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
    

    sampling_params = SamplingParams(temperature=0.0,
                                    #  top_p=0.95,
                                     max_tokens=cli_args.max_new_tokens)
    
    # random_index = random.randint(0, cli_args(num-samples))
    
    rng = np.random.default_rng(cli_args.seed)
    
    if cli_args.num_samples >= len(ds_art["train"]):
        print("Samples larger than dataaset length")
        sys.exit()


    arr = rng.choice(len(ds_art["train"]), size=cli_args.num_samples, replace=False)
    
    print(f"arr: {arr}")
    
    context = False
    
    prompts = generate_student_prompt(arr, ds_art, context, cli_args.max_new_tokens)
    
    
    print(f"Example of Prompt: {prompts[0]}")    
    print(f"length of prompt: {len(prompts)}")
    
    # prompts = [
    #     "Hello, my name is",
    #     "The president of the United States is",
    #     "The capital of France is",
    #     "The future of AI is",
    # ]
        
    llm = LLM(model = "Qwen/Qwen3-4B-Base")
    
    
    start = perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    end = perf_counter()
    
    total = end - start
    
    print(f"Time spent: {total}")
    print(f"Time spend per prompt: {total/len(prompts)}")
    
    i = 0
    out = [] # [{Qestion: "", Context: "", "Answer"}]
    
    for index, output in enumerate(outputs):
        # prompt = output.prompt
        generated_text = output.outputs[0].text

        example = ds_art["train"][arr[index]]
        question = example["question"]
        contexted = example["context"]["contexts"]
        answer = example["long_answer"]
        final_decision = example['final_decision']
        
        out.append({"Question": question, "Context": contexted, "Answer":
            answer, "Final_decision": final_decision,
                   "Model_output": generated_text,
                   "Max_Content": cli_args.max_new_tokens})
    
        # out += f"[{index}: Prompt: {prompt!r}: Generated text: {generated_text!r}], "
        # if i < 5:
        #     print()
        #     # print(out)
        # i += 1
        
    dir_path = Path("Qwen3_4B")
    dir_path.mkdir(parents=True, exist_ok=True)
    test_count = sum(1 for item in dir_path.iterdir() if item.is_dir() and item.name.startswith("test_"))
    test = f"test_{test_count + 1}"

    print(f"Student Model: {out[0:1]}")

    save_file(dir_path / test / "student_output.json", out)
    
    
    prompts = generate_parent_prompt(out)
    
    judge_sampling_params = SamplingParams(temperature=0.0, max_tokens=10, structured_outputs=StructuredOutputsParams(
        choice=["yes", "no"]
    ))

    start = perf_counter()
    outputs = llm.generate(prompts, judge_sampling_params)
    end = perf_counter()
    
    

    out = []
    count = 0
    for index, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        if generated_text == "yes":
            count += 1
        out.append({"Prompts": prompt, "output": generated_text})
        
    print(f"Parent Model: {out[0:1]}")
        
    print(f"Final Score: {count}/{len(outputs)}: {count/len(outputs)}")    
    
    save_file(dir_path / test / "parent_output.json", out)


    # with open("filename4.txt", "w") as file:
    #     file.write(out)

if __name__ == "__main__":
    main()
