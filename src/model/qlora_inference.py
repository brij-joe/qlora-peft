import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


class QLoRAInference:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(QLoRAInference, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str,
        adapter_path: str,
        device: str = "cuda",
        bnb_config: BitsAndBytesConfig = None,
        hf_token: str = None
    ):
        # Prevent re-initialization (important for singleton)
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.model_name = model_name
        self.adapter_path = adapter_path
        self.device = device
        self.bnb_config = bnb_config
        self.hf_token = hf_token

        self.tokenizer = None
        self.model = None

        self._load_model()
        self._initialized = True

    # =========================
    # LOAD MODEL + TOKENIZER
    # =========================
    def _load_model(self):
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token
        )

        # Base model (quantized)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            quantization_config=self.bnb_config
        )

        # Attach LoRA adapter
        self.model = PeftModel.from_pretrained(
            base_model,
            self.adapter_path
        )

        self.model.eval()

    # =========================
    # GENERATE
    # =========================
    def generate(self, prompt: str, max_new_tokens: int = 100):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)