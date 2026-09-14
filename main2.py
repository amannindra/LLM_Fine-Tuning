from time import perf_counter
start = perf_counter()
import json


from datasets import load_dataset
from multiprocessing import Pool
from timebudget import timebudget
import argparse
import os

end = perf_counter()

elapsed = end - start
print(f"Executed in: {elapsed:.6f} seconds")
print("Imports Loaded")



worker_data = None
worker_model = None
worker_context = None 


def initialize_worker(data, context):
    from LLMbase import LLM
    global worker_data, worker_model, worker_context

    worker_data = data
    worker_model = LLM()
    worker_context = context

    print("Worker model initialized")


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

def launch_inference(index):

    
    global worker_data, worker_model, worker_context
    try:
        print("Launching inference for example index:", index)
        example = worker_data['train'][index]
        question = example["question"]
        answer = example["final_decision"]
        contexted  = example["context"]["contexts"]
        
        if worker_context:
            
            prompt = make_prompt(question, contexted)
        else:
            prompt = make_prompt(question, "")
            
        thinking_content, content = worker_model.inference(prompt)
    
        if content == answer:
            print(f"Index {index}: Correct")
            print(f"Answer: {answer}, and got: {content}")
            return 1
        else:
            print(f"Index {index}: Incorrect")
            print(f"Answer: {answer}, and got: {content}")
            return -1
    except Exception as e:
        print(f"Index {index}: Error during inference: {e}")
        return 0
    


# python main2.py --processes 8 --index 3000 --context True

def evaluate_checkpoint(checkpoint, num_samples):
    from unsloth import FastLanguageModel
    import torch
    import re

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=checkpoint,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    results = []
    for example in dataset:
        # Match fine.py's raw training prompt, without the target answer.
        context = "".join(example["context"]["contexts"])
        prompt = f"""You are an assistant helping doctors with their questions. You are given the question and the important context you need to answer that question.
    Question: {example['question']}
    Context: {context}
    answer:"""
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=2032).to(model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                    pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0, inputs.input_ids.shape[1]:],
                                    skip_special_tokens=True).strip()
        match = re.match(r"^(yes|no|maybe)\b", response.lower())
        prediction = match.group(1) if match else None
        results.append(dict(pubid=example["pubid"], answer=example["final_decision"],
                            prediction=prediction, response=response,
                            correct=prediction == example["final_decision"]))
        print(f"{len(results)}/{len(dataset)}: expected={example['final_decision']} response={response!r}")
    with open("checkpoint_results.json", "w") as handle:
        json.dump({"checkpoint": checkpoint, "dataset": "pqa_labeled",
                   "results": results}, handle, indent=2)
    correct = sum(row["correct"] for row in results)
    print(f"Accuracy: {correct}/{len(results)} ({correct / len(results):.2%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, help="Number of processes to use for multiprocessing.")
    parser.add_argument("--index", type=int, default=1,
                        help="Dataset-size divisor: process len(train) // index examples (default: 1, all examples).")
    parser.add_argument("--context", type=bool)
    parser.add_argument("--checkpoint", help="Local Unsloth/LoRA checkpoint to evaluate on pqa_labeled.")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of labeled examples for checkpoint evaluation (default: 100).")
    cli_args = parser.parse_args()
    if cli_args.index <= 0:
        parser.error("--index must be a positive integer")
    if cli_args.num_samples <= 0:
        parser.error("--num-samples must be a positive integer")
    if cli_args.checkpoint:
        evaluate_checkpoint(cli_args.checkpoint, cli_args.num_samples)
        return

    
    
    print("This is the main function.")
    ds_art, ds_unlabel, ds_label = load_data()
    print('Number of CPUs in the system: {}'.format(os.cpu_count()))
    
    indexes = range(0, len(ds_art['train']) // cli_args.index)
    print(f"Processing {len(indexes)} examples.")
    print(f"Indexes: {list(indexes)}")

    
    args = ds_art

    print(cli_args.processes)
    correct = 0
    incorrect = 0
    no_worker = 0
    with Pool(processes=cli_args.processes, initializer=initialize_worker,initargs=(ds_art,worker_context)) as pool:
        result = pool.map(launch_inference, indexes)
        print(f"Result: {result}")
        correct += result.count(1)
        incorrect += result.count(-1)
        no_worker += result.count(0)
        total = correct + incorrect
        
    
    with open("results.json", "w") as f:
        json.dump(result, f)
    
    print(f"Correct: {correct}, Incorrect: {incorrect}, No Worker: {no_worker}, Total Processed: {total}")
    
    
    
    
    
    # run_complex_operations(launch_inference(), range(length), processes_pool)
    
    
    
    
    # for i in range(length):
    #     example = ds_art['train'][i]
    #     question = example["question"]
    #     answer = example["final_decision"]
    #     context = example["context"]["contexts"]
    #     prompt = make_prompt(question, context)

    #     thinking_content, content = model.inference(prompt)
        
    #     if content == answer:
    #         print("Output matches the answer.")
    #         print(f"Answer: {answer}, and got: {content}")
    #         correct += 1
    #     else:
    #         print("Output does not match the answer.")
    #         print(f"Answer: {answer}, and got: {content}")
    #         incorrect += 1
            
    #     if i % 10 == 0:
    #         print(f"Processed {i} examples.")
            
    # print(f"Correct: {correct}, Incorrect: {incorrect}, Total: {correct + incorrect}")
    # print(f"Accuracy: {correct / (correct + incorrect) * 100:.2f}%")

    
if __name__ == "__main__":     
    main()
