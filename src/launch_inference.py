from transformers import BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# First call → initializes model
inference = QLoRAInference(
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    adapter_path="./qlora-adapter",
    bnb_config=bnb_config
)

response = inference.generate("Explain LoRA in simple terms")
print(response)

# Second call → reuses same instance (no reload 🚀)
inference2 = QLoRAInference(
    model_name="ignored",
    adapter_path="ignored"
)

print(inference is inference2)  # True