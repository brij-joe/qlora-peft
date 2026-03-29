import torch


class QLoRAConfig:

    def __init__(self):
        # =========================
        # ENV
        # =========================
        self._env_path = "c:\\temp\\.env"

        # =========================
        # DATA
        # =========================
        self._dataset_name = "tatsu-lab/alpaca"
        self._dataset_split = "train[:5000]"
        self._max_seq_length = 512

        # =========================
        # MODEL
        # =========================
        self._model_name = "mistralai/Mistral-7B-Instruct-v0.3"

        # =========================
        # TOKENIZER
        # =========================
        self._use_auth_token = True

        # =========================
        # QLoRA
        # =========================
        self._load_in_4bit = True
        self._bnb_compute_dtype = torch.float16
        self._bnb_use_double_quant = True
        self._bnb_quant_type = "nf4"
        self._cpu_offload = True

        # =========================
        # LoRA
        # =========================
        self._lora_r = 8
        self._lora_alpha = 16
        self._lora_dropout = 0.05
        self._lora_target_modules = ["q_proj", "v_proj", "o_proj"]

        # =========================
        # TRAINING
        # =========================
        self._output_dir = "./qlora-output"
        self._batch_size = 1
        self._grad_accum_steps = 8
        self._epochs = 1
        self._learning_rate = 2e-4
        self._logging_steps = 50
        self._save_strategy = "epoch"
        self._optimizer = "paged_adamw_8bit"
        self._hf_token = None  # default (optional)

        # =========================
        # SAVE
        # =========================
        self._adapter_save_path = "./qlora-adapter"

    # =========================
    # DATA GETTERS / SETTERS
    # =========================
    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value):
        self._dataset_name = value

    @property
    def dataset_split(self):
        return self._dataset_split

    @dataset_split.setter
    def dataset_split(self, value):
        self._dataset_split = value

    @property
    def max_seq_length(self):
        return self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value):
        if value <= 0:
            raise ValueError("max_seq_length must be > 0")
        self._max_seq_length = value

    # =========================
    # MODEL
    # =========================
    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, value):
        self._model_name = value

    # =========================
    # TRAINING
    # =========================
    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        if value <= 0:
            raise ValueError("batch_size must be > 0")
        self._batch_size = value

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, value):
        if value <= 0:
            raise ValueError("epochs must be > 0")
        self._epochs = value

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        if value <= 0:
            raise ValueError("learning_rate must be > 0")
        self._learning_rate = value

    # =========================
    # LORA
    # =========================
    @property
    def lora_r(self):
        return self._lora_r

    @lora_r.setter
    def lora_r(self, value):
        if value <= 0:
            raise ValueError("lora_r must be > 0")
        self._lora_r = value

    @property
    def lora_alpha(self):
        return self._lora_alpha

    @lora_alpha.setter
    def lora_alpha(self, value):
        self._lora_alpha = value

    # =========================
    # OUTPUT
    # =========================
    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        self._output_dir = value

    @property
    def adapter_save_path(self):
        return self._adapter_save_path

    @adapter_save_path.setter
    def adapter_save_path(self, value):
        self._adapter_save_path = value

    # =========================
    # HF TOKEN
    # =========================
    @property
    def hf_token(self):
        return self._hf_token

    @hf_token.setter
    def hf_token(self, value):
        self._hf_token = value

    # =========================
    # UTIL
    # =========================
    def to_dict(self):
        return {
            "dataset_name": self._dataset_name,
            "dataset_split": self._dataset_split,
            "max_seq_length": self._max_seq_length,
            "model_name": self._model_name,
            "batch_size": self._batch_size,
            "epochs": self._epochs,
            "learning_rate": self._learning_rate,
            "lora_r": self._lora_r,
            "output_dir": self._output_dir,
            "adapter_save_path": self._adapter_save_path,
            "hf_token": "****" if self._hf_token else None,
        }

    def __repr__(self):
        return f"QLoRAConfig({self.to_dict()})"