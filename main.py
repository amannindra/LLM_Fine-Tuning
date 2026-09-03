from LLMbase import LLM
from datasets import load_dataset



def make_prompt(example: dict) -> str:
    question = example["question"]
    answer = example["final_decision"]
    context = example["context"]["contexts"]

    return f"""
        Use the medical context below to answer the question.

        Context:
        {context}

        Question:
        {example["question"]}

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
    return ds_art, ds_unlabel, ds_label


def main():
    print("This is the main function.")
    ds_art, ds_unlabel, ds_label = load_data()
    print(f"Prompt: {make_prompt(ds_art['train'][0])}")

    model = LLM()
    
    thinking_content, content = model.inference(make_prompt(ds_art['train'][0]))
    print(f"Thinking Content: {thinking_content}")
    print(f"Content: {content}")  


    
if __name__ == "__main__":     
    main()