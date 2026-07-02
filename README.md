# Paraphraser

A paraphrase generator built on Google's [PEGASUS](https://arxiv.org/abs/1912.08777) model, fine-tuned for paraphrasing (using the [`tuner007/pegasus_paraphrase`](https://huggingface.co/tuner007/pegasus_paraphrase) checkpoint from Hugging Face). Given an input sentence, it generates multiple reworded variants using beam search.

## How It Works

1. The PEGASUS model and tokenizer are loaded from Hugging Face (`tuner007/pegasus_paraphrase`).
2. Input text is tokenized and passed to `model.generate()` with beam search (`num_beams`) to produce several candidate paraphrases (`num_return_sequences`).
3. The generated token sequences are decoded back into readable text.

This was developed and run in **Google Colab**, with the model/tokenizer saved to Google Drive for reuse.

## Project Structure

```
.
├── Paraphrasing.ipynb    # Main notebook: setup, inference, save/load model
├── model/                 # Saved PEGASUS model config (see note below)
└── tokenizer/              # Saved PEGASUS tokenizer files
```

## ⚠️ Note on Model Weights

The `model/` folder in this repo only contains `config.json` and `generation_config.json` — the actual model weights (`pytorch_model.bin` or `model.safetensors`) are **not included** (likely too large for the repo / left in Google Drive). To run inference you'll need to either:

- Download the weights directly from Hugging Face (recommended, see Quick Start below), or
- Supply your own saved weights file in `model/` if you have one from a prior Colab session.

The `tokenizer/` folder does include the full tokenizer (`spiece.model`, configs), so that part is ready to use as-is.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- ~2GB free disk space / RAM for the model

### Installation

```bash
git clone <your-repo-url>
cd Paraphraser
pip install torch transformers sentencepiece sentence-splitter
```

### Quick Start (load model directly from Hugging Face)

```python
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

model_name = "tuner007/pegasus_paraphrase"
model = PegasusForConditionalGeneration.from_pretrained(model_name)
tokenizer = PegasusTokenizer.from_pretrained(model_name)

def get_response(input_text, num_return_sequences=5, num_beams=5):
    batch = tokenizer([input_text], padding=True, truncation=True, max_length=60, return_tensors="pt")
    translated = model.generate(
        **batch,
        max_length=60,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        temperature=1.5,
    )
    return tokenizer.batch_decode(translated, skip_special_tokens=True)

print(get_response("The ultimate test of your knowledge is your capacity to convey it to another."))
```

### Using the notebook

Open `Paraphrasing.ipynb` in Jupyter or Google Colab and run the cells in order. Note the notebook mounts Google Drive (`drive.mount('/content/gdrive')`) to save/load the model — remove or adapt that cell if you're not running in Colab.

## Usage Notes

- `max_length=60` limits both input and output to short sentences — longer input text will be truncated.
- Increasing `num_beams`/`num_return_sequences` gives more paraphrase variety at the cost of speed.
- `temperature=1.5` adds randomness to generation; lower it for more conservative paraphrases.

## Tech Stack

- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — PEGASUS model & tokenizer
- [SentencePiece](https://github.com/google/sentencepiece) — tokenization
- [sentence-splitter](https://pypi.org/project/sentence-splitter/) — (for splitting longer text into sentences before paraphrasing, given the 60-token limit)


