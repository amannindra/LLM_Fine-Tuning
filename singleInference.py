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

    s += """Give a detailed explanation of your reasoning, then finish with your final answer: yes, no, or maybe.

        Explanation:"""

    return s


def make_judge_prompt(question, context, explanation) -> str:
    s = f"""You are an expert doctor grading a student's answer to a medical question. You are given the question, the relevant context, and the student's explanation.

    Decide whether the student's explanation and final answer are correct, based on the context and your medical knowledge.

    Question: {question}
    Context: {" ".join(context)}

    Student's explanation: {explanation}

    Is the student correct? Respond with exactly one of these labels:
    yes
    no

    Answer:"""

    return s


def normalize(text) -> str:
    # Reduce a free-text reply to its first word, e.g. "Yes." -> "yes".
    words = re.findall(r"[a-z]+", str(text).lower())
    return words[0] if words else ""


def load_data():
    ds_art = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
    ds_unlabel = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")
    ds_label = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    print(f"Loaded Datasets: {ds_art}, {ds_unlabel}, {ds_label}")
    return ds_art, ds_unlabel, ds_label


def launch_inference(dataset, worker_model, worker_context, index, max_new_tokens):
    """Returns the student's explanation, or None if inference failed."""
    print(f"Launching index: {index}")

    try:
        example = dataset['train'][index]
        question = example["question"]
        contexted = example["context"]["contexts"]

        print(f"Question: {question}, Context: {contexted}")

        if worker_context:
            prompt = make_prompt(question, contexted)
        else:
            prompt = make_prompt(question, "")

        print(f"Prompt: {prompt}")

        thinking_content, content = worker_model.inference(prompt, max_new_tokens=max_new_tokens)

        print(f"Thinking Content: {thinking_content}, Content: {content}")

        return content

    except Exception:
        import traceback; traceback.print_exc()
        return None


def evaluate_checkpoint(dataset, index, judge_model, explanation):
    """Ask the parent model whether the student's explanation is correct.

    The judge sees the question, context and explanation, never final_decision.
    Returns "yes", "no", or whatever else the judge said.
    """
    example = dataset['train'][index]
    question = example["question"]
    contexted = example["context"]["contexts"]

    prompt = make_judge_prompt(question, contexted, explanation)

    _, content = judge_model.inference(prompt)
    return normalize(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, help="Number of processes to use for multiprocessing.")
    parser.add_argument("--index", type=int, default=10,
                        help="Number of random examples to evaluate (default: 10).")
    parser.add_argument("--context", action="store_true", help="Include the context in the student's prompt.")
    parser.add_argument("--num-samples", type=int, default=4000,
                        help="Number of labeled examples for checkpoint evaluation (default: 4000).")
    parser.add_argument("--fine-tune", action="store_true", help="Use the fine-tuned checkpoint instead of the base model.")
    parser.add_argument("--max-new-tokens", type=int, default=300,
                        help="Max length of the student's explanation (default: 300).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for choosing examples (default: 0).")
    cli_args = parser.parse_args()
    if cli_args.index <= 0:
        parser.error("--index must be a positive integer")
    if cli_args.num_samples <= 0:
        parser.error("--num-samples must be a positive integer")
    if cli_args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be a positive integer")

    print("This is the main function.")
    ds_art, ds_unlabel, ds_label = load_data()
    print('Number of CPUs in the system: {}'.format(os.cpu_count()))

    indexes = range(0, cli_args.index)
    print(f"Processing {len(indexes)} examples.")
    print(f"Fine-tuned: {cli_args.fine_tune}, Context: {cli_args.context}")

    worker_model = LLMInference(cli_args.fine_tune)
    worker_context = cli_args.context

    from Parent import ParentModel
    parent = ParentModel()

    random.seed(cli_args.seed)
    size = len(ds_art['train'])

    failed = 0
    correct = 0
    incorrect = 0
    unclear = 0
    for i in indexes:
        num = random.randint(0, size - 1)

        explanation = launch_inference(ds_art, worker_model, worker_context, num, cli_args.max_new_tokens)
        if explanation is None:
            failed += 1
            continue

        verdict = evaluate_checkpoint(ds_art, num, parent, explanation)
        if verdict == "yes":
            correct += 1
        elif verdict == "no":
            incorrect += 1
        else:
            unclear += 1

        print(f"Output for index {num}: judge verdict {verdict!r}")

    print(f"Correct: {correct}, Incorrect: {incorrect}, Unclear: {unclear}, Failed: {failed}, Total Processed: {len(indexes)}")


if __name__ == "__main__":
    main()
