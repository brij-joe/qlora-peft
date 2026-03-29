# QLoRA Parameter Efficient Fine-Tuning

A lightweight, modular implementation of **QLoRA (Quantized LoRA)** for fine-tuning LLM/SLM using Hugging Face ecosystem tools.

---

## Overview

This project demonstrates how to efficiently fine-tune a causal language model using:

*  **QLoRA** (4-bit quantization + LoRA adapters)
*  Hugging Face Transformers
*  PEFT (Parameter-Efficient Fine-Tuning)
*  Alpaca dataset (instruction tuning)

---

##  Tech Stack

* Python 3.10+
* `transformers`
* `peft`
* `bitsandbytes`
* `accelerate`
* `datasets`

---

## Project Structure

```
qlora-peft/
│
├── src/
│   ├── launch_trainer.py        # Entry point
│   ├── model/
│   │   └── qlora_trainer.py     # Training pipeline
│   ├── config/
│   │   └── qlora_config.py      # Config class
│
├── qlora-output/                # Fine-tuned model output
├── qlora-adapter/              # Saved LoRA adapters
├── requirements.txt
└── README.md
```

---

## ️ Configuration

Example configuration:

```python
QLoRAConfig({
    "dataset_name": "tatsu-lab/alpaca",
    "dataset_split": "train[:200]",
    "max_seq_length": 256,
    "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "batch_size": 1,
    "epochs": 1,
    "learning_rate": 2e-4,
    "lora_r": 8,
    "output_dir": "./qlora-output",
    "adapter_save_path": "./qlora-adapter",
    "hf_token": "YOUR_HF_TOKEN"
})
```

---

## ️ How to Run

### Clone the Repository

```bash
git clone https://github.com/your-username/qlora-peft.git
cd qlora-peft
```

---

### Install Dependencies (If GPU is available always prefer TORCH with CUDA installation)

```bash
uv venv .venv
uv init .
uv sync
```

Recommended versions(refer pyproject.toml):

```txt
transformers==4.38.2
peft==0.10.0
bitsandbytes==0.43.1
accelerate==0.27.2
datasets
trl
```

---

### Run Training

```bash
python src/launch_trainer.py
```

---

## Note:️ Important Notes (Windows Users)

QLoRA depends on CUDA kernels from `bitsandbytes`, which are **not fully supported on native Windows**.

### Common Errors

* `cquantize_blockwise_fp16_nf4 not found`
* `libbitsandbytes_cudaXXX.dll missing`

### Recommended Solutions

| Option                | Description                  |
| --------------------- | ---------------------------- |
| WSL2 (Recommended) | Run on Ubuntu inside Windows |
| Google Colab       | Quick and zero setup         |
| Disable 4-bit      | Use standard LoRA instead    |

---

## Model Used in example

* TinyLlama 1.1B Chat (lightweight, fast training)

---

## Dataset

* Alpaca instruction dataset (`tatsu-lab/alpaca`)

---

## Features

* Dynamic config class
* Modular trainer design
* LoRA adapter saving
* Hugging Face integration
* Easy extensibility

---

## Future Improvements

* [ ] Add evaluation pipeline
* [ ] Add inference script
* [ ] Multi-GPU support
* [ ] Experiment tracking (Weights & Biases)
* [ ] RAG integration

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## License

MIT License

---

## Acknowledgements

* Hugging Face
* PEFT library
* QLoRA research paper

---

## Support

If you find this project useful, consider giving it a ⭐ on GitHub!
