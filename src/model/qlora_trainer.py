import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

from config.qlora_config import QLoRAConfig


class QLoRATrainer:

    def __init__(self, config: QLoRAConfig):
        self.config = config
        self.dataset = None
        self.tokenizer = None
        self.model = None
        self.tokenized_dataset = None
        self.trainer = None

    # =========================
    # DATASET
    # =========================
    def load_dataset(self):
        dataset = load_dataset(
            self.config.dataset_name, split = self.config.dataset_split, )

        def format_example(example):
            return f"""
                ### Instruction:
                {example['instruction']}
                ### Input:
                {example['input']}
                ### Response:
                {example['output']}
                """

        dataset = dataset.map(lambda x: {"text": format_example(x)})
        self.dataset = dataset

    # =========================
    # TOKENIZER
    # =========================
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, token = self.config.hf_token, )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    # =========================
    # MODEL (QLoRA)
    # =========================
    def load_model(self):
        bnb_config = BitsAndBytesConfig(
            torch_dtype = torch.float16,  # instead of 4bit
            # load_in_4bit = self.config._load_in_4bit,
            bnb_4bit_compute_dtype = self.config._bnb_compute_dtype,
            bnb_4bit_use_double_quant = self.config._bnb_use_double_quant,
            bnb_4bit_quant_type = self.config._bnb_quant_type,
            llm_int8_enable_fp32_cpu_offload = self.config._cpu_offload, )

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, quantization_config = bnb_config, device_map = "auto", token = self.config.hf_token, )

        # LoRA config
        lora_config = LoraConfig(
            r = self.config.lora_r, lora_alpha = self.config.lora_alpha,
            target_modules = self.config._lora_target_modules, lora_dropout = self.config._lora_dropout, bias = "none",
            task_type = TaskType.CAUSAL_LM, )

        # Memory optimization
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        model = get_peft_model(model, lora_config)
        self.model = model

    # =========================
    # TOKENIZATION
    # =========================
    def tokenize_dataset(self):
        def tokenize_function(example):
            return self.tokenizer(
                example["text"], truncation = True, max_length = self.config.max_seq_length, padding = "max_length", )

        tokenized = self.dataset.map(tokenize_function, batched = True)
        tokenized = tokenized.remove_columns(
            [col for col in self.dataset.column_names if col != "text"], )

        self.tokenized_dataset = tokenized

    # =========================
    # TRAINER
    # =========================
    def setup_trainer(self):
        data_collator = DataCollatorForLanguageModeling(
            tokenizer = self.tokenizer, mlm = False, )

        training_args = TrainingArguments(
            output_dir = self.config.output_dir, per_device_train_batch_size = self.config.batch_size,
            gradient_accumulation_steps = self.config._grad_accum_steps, num_train_epochs = self.config.epochs,
            learning_rate = self.config.learning_rate, fp16 = True, logging_steps = self.config._logging_steps,
            save_strategy = self.config._save_strategy, optim = self.config._optimizer, report_to = "none", )

        self.trainer = Trainer(
            model = self.model, train_dataset = self.tokenized_dataset, args = training_args,
            data_collator = data_collator, )

    # =========================
    # TRAIN
    # =========================
    def train(self):
        self.trainer.train()

    # =========================
    # SAVE
    # =========================
    def save_adapter(self):
        self.model.save_pretrained(self.config.adapter_save_path)
        print("Training complete. Adapter saved!")

    # =========================
    # PIPELINE
    # =========================
    def run(self):
        print("Starting QLoRA Training with config:")
        print(self.config)

        self.load_dataset()
        self.load_tokenizer()
        self.load_model()
        self.tokenize_dataset()
        self.setup_trainer()
        self.train()
        self.save_adapter()