from bert_score import BERTScorer
import torch

# Initialize scorer (uses GPU if available, or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", device=device)



# Candidate and reference sentences
cands = ["The cat sat on the mat."]
refs = ["A feline was resting on the rug."]

# Compute precision, recall, and F1
P, R, F1 = scorer.score(cands, refs)

print(f"Precision: {P.mean():.4f}")
print(f"Recall:    {R.mean():.4f}")
print(f"F1 Score:  {F1.mean():.4f}")