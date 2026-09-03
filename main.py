from LLMbase import LLM
from datasets import load_dataset
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
    
    example = ds_art['train'][0]
    question = example["question"]
    answer = example["final_decision"]
    context = example["context"]["contexts"]
    prompt = make_prompt(question, context)

    model = LLM()
    
    thinking_content, content = model.inference(prompt)
    
    if content == "":
        print("Output is empty.")
        print(f"Expected Answer: {answer}, but got: {content}")    
    elif content == answer:
        print("Output matches the answer.")
        print(f"Answer: {answer}, and got: {content}")
    else:
        print("Output does not match the answer.")
        print(f"Expected Answer: {answer}, but got: {content}")    



    
if __name__ == "__main__":     
    main()