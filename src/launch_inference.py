import os

from dotenv import load_dotenv
from transformers import BitsAndBytesConfig
import torch

from config.qlora_config import QLoRAConfig
from model.qlora_inference import QLoRAInference

if __name__ == "__main__":
    load_dotenv(verbose=True)
    config = QLoRAConfig()
    config.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    config.hf_token = os.environ["HF_TOKEN"]

    bnb_config = BitsAndBytesConfig(
        torch_dtype = torch.float16,  # instead of 4bit
        # load_in_4bit = self.config._load_in_4bit,
        bnb_4bit_compute_dtype = self.config._bnb_compute_dtype,
        bnb_4bit_use_double_quant = self.config._bnb_use_double_quant,
        bnb_4bit_quant_type = self.config._bnb_quant_type,
        llm_int8_enable_fp32_cpu_offload = self.config._cpu_offload, )

    # First call → initializes model
    inference = QLoRAInference(
        model_name = config.model_name,
        adapter_path = "./qlora-adapter",
        bnb_config = bnb_config
    )

    response = inference.generate("Explain LoRA in simple terms")
    print(response)

    # Second call → reuses same instance (no reload 🚀)
    inference2 = QLoRAInference(
        model_name = "ignored",
        adapter_path = "ignored"
    )

    print(inference is inference2)  # True
