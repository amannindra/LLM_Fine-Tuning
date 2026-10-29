from bert_score import BERTScorer
import torch
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Initialize scorer (uses GPU if available, or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", device=device)


model_output_dir = Path("Qwen3_4B")
# test_dirs = sorted(
#     (path for path in model_output_dir.glob("test_*") if path.is_dir()),
#     key=lambda path: int(path.name.removeprefix("test_"))
# )
# if not test_dirs:
#     raise FileNotFoundError(f"No test_N directories found in {model_output_dir}")
test_dirs = Path("/home/anindra/data/LLM_Fine-Tuning/evaluation/Qwen3_4B/test_7")

input_path = test_dirs / Path("student_output.json")
if not input_path.exists():
    raise FileNotFoundError(f"Student output not found: {input_path}")


print(f"Scoring student output: {input_path}")

with input_path.open("r", encoding="utf-8") as file:
    data = json.load(file)

student = []
answer = []

for i in range(len(data)):
    student.append(data[i]["Model_output"])
    answer.append(data[i]["Answer"])



P, R, F1 = scorer.score(
    cands=student, 
    refs=answer, 
    batch_size=350,
    verbose=True
)

# f1_scores = F1.tolist()
# precision_scores = P.tolist()
# recall_scores = R.tolist()


p = P.detach().cpu().numpy()
r = R.detach().cpu().numpy()
f1 = F1.detach().cpu().numpy()

# Summary statistics
summary = {
    "count": len(f1),
    "mean_precision": float(np.mean(p)),
    "mean_recall": float(np.mean(r)),
    "mean_f1": float(np.mean(f1)),
    "median_f1": float(np.median(f1)),
    "std_f1": float(np.std(f1, ddof=1)),
    "min_f1": float(np.min(f1)),
    "p10_f1": float(np.percentile(f1, 10)),
    "p25_f1": float(np.percentile(f1, 25)),
    "p75_f1": float(np.percentile(f1, 75)),
    "p90_f1": float(np.percentile(f1, 90)),
    "max_f1": float(np.max(f1)),
}

for metric, value in summary.items():
    print(f"{metric}: {value:.4f}")

# Save per-example results
results = pd.DataFrame({
    "student_output": student,
    "reference": answer,
    "precision": p,
    "recall": r,
    "f1": f1
})

output_root = Path("bertscore")
output_root.mkdir(parents=True, exist_ok=True)
result_dirs = sorted(
    (path for path in output_root.glob("test_*") if path.is_dir()),
    key=lambda path: int(path.name.removeprefix("test_"))
)
next_result_dir = output_root / f"test_{int(result_dirs[-1].name.removeprefix('test_')) + 1 if result_dirs else 1}"
next_result_dir.mkdir()

results.to_csv(next_result_dir / "bertscore_results.csv", index=True)
summary_path = next_result_dir / "summary.json"
temporary_summary_path = summary_path.with_suffix(".json.tmp")
with temporary_summary_path.open("w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)
temporary_summary_path.replace(summary_path)
# Inspect the 20 lowest-scoring examples
print(
    results.nsmallest(20, "f1").to_string(index=True)
)



# # Print summary metrics
# print(f"Processed {len(f1_scores)} pairs successfully.")
# print(f"Average F1 Score: {sum(f1_scores) / len(f1_scores):.4f}")
