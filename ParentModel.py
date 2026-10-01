
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

class ParentModel:
    def __init__(self, model_name: str):
        self.model_name = "google/gemma-4-26B-A4B-it"
        self.device = torch.device("cuda")
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype="auto",
            device_map=self.device 
        )

    def load_model(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def inference(self, context: str):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Write a short joke about saving RAM."},
        ]

        # Process input
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True, 
            enable_thinking=False
        )
        inputs = processor(text=text, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        # Generate output
        outputs = model.generate(**inputs, max_new_tokens=1024)
        response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

        # Parse output
        processor.parse_response(response)
        return response
    
    
model = ParentModel("google/gemma-4-26B-A4B-it")
print(model.inference("Write a short joke about saving RAM."))