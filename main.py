from time import perf_counter
start = perf_counter()


from LLMbase import LLM
from datasets import load_dataset

end = perf_counter()

elapsed = end - start
print(f"Executed in: {elapsed:.6f} seconds")
print("Imports Loaded")


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


def main():
    print("This is the main function.")
    ds_art, ds_unlabel, ds_label = load_data()
    
    correct = 0
    incorrect = 0
    for i in range(len(ds_art['train']) // 1000):
        example = ds_art['train'][0]
        
        
        question = example["question"]
        answer = example["final_decision"]
        context = example["context"]["contexts"]
        prompt = make_prompt(question, context)

        model = LLM()
        
        thinking_content, content = model.inference(prompt)
        
        if content == answer:
            print("Output matches the answer.")
            print(f"Answer: {answer}, and got: {content}")
            correct += 1
        else:
            print("Output does not match the answer.")
            print(f"Answer: {answer}, and got: {content}")
            incorrect += 1
            
            
    print(f"Correct: {correct}, Incorrect: {incorrect}, Total: {correct + incorrect}")
    print(f"Accuracy: {correct / (correct + incorrect) * 100:.2f}%")

    
if __name__ == "__main__":     
    main()