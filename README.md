# Bambara Fine-tuning

Fine-tuning Qwen3.5 on Bambara language using Unsloth.

## Requirements

- GPU with 5GB+ VRAM
- Python 3.10+
- PyTorch 2.0+

## Setup

```bash
pip install unsloth datasets trl
```

## Dataset

- [oza75/bambara-lm-qa](https://huggingface.co/datasets/oza75/bambara-lm-qa) - 824k Q&A pairs in Bambara

## Usage

```bash
python train.py
```

## Model

- Base: Qwen/Qwen3-0.5B
- Method: LoRA (r=16)
- Max seq length: 2048

## Hardware

Tested on:
- RTX 5090 32GB
- RTX 4090 24GB

## License

MIT
