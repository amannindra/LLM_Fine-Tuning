from time import perf_counter
start = perf_counter()


from LLMbase import LLM
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


def initialize_worker(data):
    global worker_data, worker_model

    worker_data = data
    worker_model = LLM()

    print("Worker model initialized")


def make_prompt(question, context) -> str:
    return f"""
        Use the medical context below to answer the question.

        Context:
        {context}

        Question:
        {question}

        Respond with exactly one of these labels:
        yes
        no
        maybe

        Answer:
    """

def load_data():
    ds_art = load_dataset("qiaojin/PubMedQA", "pqa_artificial")
    ds_unlabel = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")
    ds_label = load_dataset("qiaojin/PubMedQA", "pqa_labeled")
    print(f"Loaded Datasets: {ds_art}, {ds_unlabel}, {ds_label}")
    return ds_art, ds_unlabel, ds_label

def launch_inference(args):
    i = args
    
    global worker_data, worker_model
    try:
        print("Launching inference for example index:", i)
        example = worker_data['train'][i]
        question = example["question"]
        answer = example["final_decision"]
        context = example["context"]["contexts"]
        
        prompt = make_prompt(question, context)
        thinking_content, content = worker_model.inference(prompt)
    
        if content == answer:
            print(f"Index {i}: Correct")
            print(f"Answer: {answer}, and got: {content}")
            # return 1
        else:
            print(f"Index {i}: Incorrect")
            print(f"Answer: {answer}, and got: {content}")
            # return -1
    except Exception as e:
        print(f"Index {i}: Error during inference: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, help="Number of processes to use for multiprocessing.")
    cli_args = parser.parse_args()

    
    
    print("This is the main function.")
    ds_art, ds_unlabel, ds_label = load_data()
    print('Number of CPUs in the system: {}'.format(os.cpu_count()))
    
    indexes = range(0, len(ds_art['train']) // 1000)
    print(f"Processing {len(indexes)} examples.")
    print(f"Indexes: {list(indexes)}")

    
    args = ds_art

    print(cli_args.processes)
    
    with Pool(processes=cli_args.processes, initializer=initialize_worker,initargs=(args,)) as pool:
        result = pool.map(launch_inference, indexes)
        
    correct = result.count(1)
    incorrect = result.count(-1)
    no_worker = result.count(0)
    total = correct + incorrect
    
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