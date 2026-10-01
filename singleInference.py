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

LABELS = ("yes", "no", "maybe")


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

    s += """Respond with exactly one of these labels:
        yes
        no
        maybe

        Answer:"""

    return s


def make_prompt_with_context(question, context, answer, responce) -> str:
    s = f"""You are an assistant helping doctors with their questions. You are given the question, important context, and the correct answer.

    Your job is to say yes or no: is the doctor's answer consistent with the correct answer?

    Question: {question}
    Context: {" ".join(context)}

    Correct answer: {answer}

    Doctor's answer: {responce}

    Respond with exactly one of these labels:
    yes
    no

    Answer:"""

    return s


def normalize(text) -> str:
    # Reduce a free-text reply to its first label word, e.g. "Yes." -> "yes".
    words = re.findall(r"[a-z]+", str(text).lower())
    return words[0] if words else ""


def load_data():
    ds_art = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
    ds_unlabel = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")
    ds_label = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    print(f"Loaded Datasets: {ds_art}, {ds_unlabel}, {ds_label}")
    return ds_art, ds_unlabel, ds_label


def launch_inference(dataset, worker_model, worker_context, index):
    """Returns (model_label, gold_label), or None if inference failed."""
    print(f"Launching index: {index}")

    try:
        example = dataset['train'][index]
        question = example["question"]
        answer = example["final_decision"]
        contexted = example["context"]["contexts"]

        print(f"Question: {question}, Answer: {answer}, Context: {contexted}")

        if worker_context:
            prompt = make_prompt(question, contexted)
        else:
            prompt = make_prompt(question, "")

        print(f"Prompt: {prompt}")

        thinking_content, content = worker_model.inference(prompt)

        print(f"Thinking Content: {thinking_content}, Content: {content}")

        return normalize(content), answer

    except Exception:
        import traceback; traceback.print_exc()
        return None


def evaluate_checkpoint(dataset, index, judge_model, responce):
    """Ask the parent model whether `responce` matches the gold answer."""
    example = dataset['train'][index]
    question = example["question"]
    answer = example["final_decision"]
    contexted = example["context"]["contexts"]

    prompt = make_prompt_with_context(question, contexted, answer, responce)

    _, content = judge_model.inference(prompt)
    return normalize(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, help="Number of processes to use for multiprocessing.")
    parser.add_argument("--index", type=int, default=10,
                        help="Number of random examples to evaluate (default: 10).")
    parser.add_argument("--context", action="store_true", help="Include the context in the prompt.")
    parser.add_argument("--num-samples", type=int, default=4000,
                        help="Number of labeled examples for checkpoint evaluation (default: 4000).")
    parser.add_argument("--fine-tune", action="store_true", help="Use the fine-tuned checkpoint instead of the base model.")
    parser.add_argument("--judge", action="store_true",
                        help="Also score with the Qwen parent model as judge (needs a large GPU).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for choosing examples (default: 0).")
    cli_args = parser.parse_args()
    if cli_args.index <= 0:
        parser.error("--index must be a positive integer")
    if cli_args.num_samples <= 0:
        parser.error("--num-samples must be a positive integer")

    print("This is the main function.")
    ds_art, ds_unlabel, ds_label = load_data()
    print('Number of CPUs in the system: {}'.format(os.cpu_count()))

    indexes = range(0, cli_args.index)
    print(f"Processing {len(indexes)} examples.")
    print(f"Fine-tuned: {cli_args.fine_tune}, Context: {cli_args.context}, Judge: {cli_args.judge}")

    worker_model = LLMInference(cli_args.fine_tune)
    worker_context = cli_args.context

    parent = None
    if cli_args.judge:
        from Parent import ParentModel
        parent = ParentModel()

    random.seed(cli_args.seed)
    size = len(ds_art['train'])

    failed = 0
    correct = 0
    incorrect = 0
    judge_correct = 0
    for i in indexes:
        num = random.randint(0, size - 1)

        output = launch_inference(ds_art, worker_model, worker_context, num)
        if output is None:
            failed += 1
            continue

        label, answer = output
        if label == answer:
            correct += 1
        else:
            incorrect += 1

        if parent is not None:
            if evaluate_checkpoint(ds_art, num, parent, label) == "yes":
                judge_correct += 1

        print(f"Output for index {num}: got {label}, expected {answer}")

    print(f"Correct: {correct}, Incorrect: {incorrect}, Failed: {failed}, Total Processed: {len(indexes)}")
    if parent is not None:
        print(f"Judge says correct: {judge_correct}")


if __name__ == "__main__":
    main()
