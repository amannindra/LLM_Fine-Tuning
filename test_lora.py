
from time import perf_counter
start = perf_counter()
from transformers import AutoTokenizer

from pathlib import Path

from vllm import LLM, SamplingParams
end = perf_counter()

print(f"Import time: {str(end - start)}")


# MODEL_DIR = Path("merged_adapters_new").resolve()

# print(f"MODEL_DIR: {MODEL_DIR}")
# if not (MODEL_DIR / "model.safetensors").is_file():
#     raise FileNotFoundError(f"Merged model weights not found: {MODEL_DIR / 'model-00002-of-00002.safetensors'}")
# if not (MODEL_DIR / "config.json").is_file():
#     raise FileNotFoundError(f"Model config not found: {MODEL_DIR / 'config.json'}")
t = "model-00001-of-00002.safetensors"
config = 'merged_adapters_new2/config.json'

safe = "/home/anindra/data/LLM_Fine-Tuning/merged_adapters_new2"
tokenizer = AutoTokenizer.from_pretrained("merged_adapters_new2")

prompt = """Answer the research question using your learned knowledge. Give the most accurate answer you can, and do not invent study-specific findings or supporting facts.

Question: Does uremia cause premature ageing of the T cell compartment in end-stage renal disease patients?
Write one or two concise declarative sentences that state the finding directly, like a research conclusion. Preserve uncertainty or limitations when relevant. Do not start with a yes/no label, add general background or unsupported mechanisms, or use a list.

Answer:"""

llm = LLM(
    model=safe,
    tokenizer=safe,
    dtype="auto",
    max_model_len=2048,
)
sampling_params = SamplingParams(temperature=0.0, max_tokens=200)

outputs = llm.generate([prompt], sampling_params)
print(f"Model: {safe}")
print(f"Prompt: {prompt}")
print(f"Output: {outputs[0].outputs[0].text.strip()}")
