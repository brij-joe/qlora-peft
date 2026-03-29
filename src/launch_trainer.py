import os

from dotenv import load_dotenv

from config.qlora_config import QLoRAConfig
from model.qlora_trainer import QLoRATrainer


load_dotenv("c:\\temp\\.env")

if __name__ == "__main__":
    # setting a very small training parameter so that it finished quickly
    config = QLoRAConfig()
    config.batch_size = 1
    config.max_seq_length = 256
    config.per_device_train_batch_size = 2
    config.logging_steps = 50
    config.dataset_split= "train[:200]"
    config.epochs = 1
    config.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    # config.model_name = "google/gemma-2b-it"
    # config.model_name = "Qwen/Qwen1.5-1.8B-Chat"
    # config.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    # config.model_name = "microsoft/phi-2"
    config.dataset_name = "tatsu-lab/alpaca"
    config.hf_token = os.environ["HF_TOKEN"]

    # Read values
    print(f"Starting QLoRA training for model: {config.model_name}")
    print(f"Base parameters batch_size: {config.batch_size}, epochs: {config.epochs}, and max_seq: {config.max_seq_length}")
    trainer = QLoRATrainer(config)
    trainer.run()