# GPT Lyrics Model

This repository contains a simplified GPT-style language model trained on lyrics. It includes all components for configuration, training, and sampling.

---

## 📁 Project Structure

- `model.py`: Full definition of a GPT language model inspired by GPT-2.
- `train.py`: Script for training the GPT model with various configurations and distributed support.
- `configurator.py`: Minimalistic command-line configuration handler.
- `data/lyrics/prepare.py`: Script to prepare the lyrics dataset.
- `sample.py`: Generates text samples from the trained model.
- `config/`: Folder for custom configuration files.
- `out-lyrics/`: Directory where training outputs and checkpoints are saved.

---

## 📦 Installation

Install the required dependencies:

```bash
pip install torch torchvision torchaudio
pip install numpy
pip install transformers
pip install datasets
pip install tiktoken
pip install wandb
pip install tqdm
```

---

## 🛠️ Data Preparation

Prepare the lyrics data:

```bash
python data/lyrics/prepare.py
```

Download additional files and place them in `data/lyrics`:  
📎 [Google Drive Folder](https://drive.google.com/drive/folders/1Z2M8GTe9SgJ-3GiLseZ11pVcdSLyUg7C?usp=sharing)

---

## 🚀 Training

Train the GPT model on lyrics data:

```bash
python train.py config/train_lyrics.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=5000 --lr_decay_iters=5000 --dropout=0.0
```

---

## ✨ Text Generation

Generate lyrics starting from a prompt:

```bash
python sample.py --out_dir=out-lyrics --device=cpu --start="Love"
```

---

## 📌 Notes

- All configuration can be overridden via CLI using `--key=value` or config file path.
- We use a "poor man's configuration" approach via `configurator.py`.
- DDP support is included for multi-GPU training.

---

## 🧠 Credits

Inspired by:
- OpenAI's [GPT-2 TensorFlow implementation](https://github.com/openai/gpt-2)
- HuggingFace [Transformers](https://github.com/huggingface/transformers)

---
