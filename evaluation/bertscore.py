from bert_score import BERTScorer
import torch
import Path
import json

# Initialize scorer (uses GPU if available, or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", device=device)


with open('/home/anindra/data/LLM_Fine-Tuning/Qwen3_4B/test_4/student_output.json', 'r') as file:
    data = json.load(file)

student = []
answer = []

for i in range(len(data)):
    student.append(data[i]["Model_output"])
    answer.append(data[i]["Answer"])
    
# Candidate and reference sentences

# Compute precision, recall, and F1


P, R, F1 = scorer.score(
    cands=student, 
    refs=answer, 
    batch_size=64,
)

# f1_scores = F1.tolist()
# precision_scores = P.tolist()
# recall_scores = R.tolist()





# Print summary metrics
print(f"Processed {len(f1_scores)} pairs successfully.")
print(f"Average F1 Score: {sum(f1_scores) / len(f1_scores):.4f}")