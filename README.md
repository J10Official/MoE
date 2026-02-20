# Sparse Mixture of Experts (MoE) Transformer for Text Summarization

**ANLP (M25) — Assignment 3**

This project implements a Sparse Mixture of Experts Transformer from scratch for extreme summarization on the [XSum](https://huggingface.co/datasets/EdinburghNLP/xsum) dataset. It also fine-tunes baseline models (Flan-T5 Base with LoRA, Llama 3.2 1B instruction-tuned) and compares all models using lexical, embedding-based, and human evaluation metrics.

**[Assignment (PDF)](Assignment.pdf)** | **[Report (PDF)](Report.pdf)** | **[HuggingFace Models](https://huggingface.co/J10Official)**

---

## Overview

The assignment consists of three parts:

1. **Baselines** — Run inference with BART (`facebook/bart-large-xsum`), fine-tune an encoder-decoder model on XSum, and instruction-tune an instruct model on XSum.
2. **MoE from Scratch** — Implement a Sparse MoE layer to replace FFN layers in a Transformer, with two routing algorithms (Hash Routing and Token-choice Top-k Routing) and a load balancer. Train from scratch on XSum.
3. **Analysis** — Evaluate all models using ROUGE, BLEU, BERTScore, compression ratio, extractiveness, LLM-as-judge, and human evaluation.

---

## Trained Models

All models are on HuggingFace: **[https://huggingface.co/J10Official](https://huggingface.co/J10Official)**

### MoE Models (from scratch)

| Model | Router | Load Balancer | HuggingFace Repo |
|-------|--------|:---:|---|
| Token Choice | Token Choice | ❌ | [sparse-moe-Token-Choice-NoLB](https://huggingface.co/J10Official/sparse-moe-Token-Choice-NoLB) |
| Hash | Hash | ❌ | [sparse-moe-Hash-NoLB](https://huggingface.co/J10Official/sparse-moe-Hash-NoLB) |
| Token Choice + LB | Token Choice | ✅ | [sparse-moe-Token-Choice](https://huggingface.co/J10Official/sparse-moe-Token-Choice) |
| Hash + LB | Hash | ✅ | [sparse-moe-Hash](https://huggingface.co/J10Official/sparse-moe-Hash) |
| Token Choice + GQA | Token Choice | — | [sparse-moe-Token-Choice-GQA](https://huggingface.co/J10Official/sparse-moe-Token-Choice-GQA) |
| LoRA MoE | — | — | [LoRA](https://huggingface.co/J10Official/LoRA) |

### Baseline Models (fine-tuned)

| Model | Description | HuggingFace Repo |
|-------|-------------|---|
| Flan-T5 Base + LoRA | Flan-T5 Base fine-tuned on XSum with LoRA (encoder-decoder baseline) | [flan-t5-base-xsum-lora](https://huggingface.co/J10Official/flan-t5-base-xsum-lora) |
| Llama 3.2 1B | Llama 3.2 1B instruction-tuned on XSum (instruct model baseline) | [llama-3.2-1b-xsum](https://huggingface.co/J10Official/llama-3.2-1b-xsum) |

BART (`facebook/bart-large-xsum`) is used directly for inference as a pre-trained baseline (already fine-tuned on XSum).

---

## Key Features

### MoE Transformer (from scratch)
- **Sparse MoE layer** replaces the FFN in a standard encoder-decoder Transformer
- **Two routing algorithms**: Hash Routing (deterministic, non-learnable) and Token-choice Top-k Routing (learned gating with softmax)
- **Load balancer**: Regularisation loss to encourage even expert utilisation
- **Sparse dispatch**: Only computes active experts per token
- **Expert usage visualisation**: Tracks and plots expert selection over training

### Baseline Fine-tuning
- **Flan-T5 Base** (`google/flan-t5-base`): Encoder-decoder model fine-tuned on XSum using LoRA (parameter-efficient fine-tuning)
- **Llama 3.2 1B** (`meta-llama/Llama-3.2-1B-Instruct`): Instruction model fine-tuned on XSum for summarisation
- **BART** (`facebook/bart-large-xsum`): Pre-trained baseline used directly for inference (already fine-tuned on XSum)

### Evaluation
- **Lexical**: ROUGE-1, ROUGE-2, ROUGE-L, BLEU
- **Embedding-based**: BERTScore
- **Document-level**: Compression ratio, extractiveness
- **Human evaluation**: Content relevance, coherence, fluency, factual consistency (1–5 scales)

---

## Bonus Features

### Bonus 1 — Load Balancer Comparison
Train MoE models with both routing algorithms with and without load balancer loss. Compare expert usage patterns and model performance.

### Bonus 2 — Grouped Query Attention (GQA)
Custom GQA implementation replacing PyTorch's `MultiheadAttention`. ~25% parameter reduction with `GQA_NUM_KEY_VALUE_HEADS = 4`.

### Bonus 3 — LoRA (Low-Rank Adaptation)
LoRA-based experts implemented from scratch for parameter-efficient MoE training. Configurable rank, alpha, and dropout.

---

## Project Structure

| File | Description |
|------|-------------|
| `MoE_Final.ipynb` | **Primary notebook** — MoE training, checkpointing, and evaluation pipeline |
| `moe-anlp_with_Checkpoint.ipynb` | Training notebook with checkpoint save/load support |
| `moe-anlp.ipynb` | Original baseline notebook |
| `moe.py` | Standalone Python script with full MoE pipeline |
| `checkpoints/` | Locally saved model checkpoints |

---

## MoE Model Architecture

| Parameter | Value |
|-----------|-------|
| Tokenizer | `t5-small` (SentencePiece, 32128 vocab) |
| `d_model` | 512 |
| `nhead` | 8 |
| Encoder layers | 6 |
| Decoder layers | 6 |
| Experts per MoE layer | 4 |
| `d_ff` | 1024 |
| Top-k routing | 2 |
| Dropout | 0.1 |
| Max sequence length | 512 |

**Training config**: batch size 32, lr 5e-5 with linear warmup (500 steps), load balance coefficient 0.01, XSum 100k train / 2k val / 2k test.

---

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Training
Open `MoE_Final.ipynb` in Jupyter or Kaggle and run all cells. The `Config` class at the top controls hyperparameters, router selection, and bonus feature toggles.

Alternatively, run the standalone script:
```bash
python3 moe.py
```

---

## Configuration

All settings live in the `Config` class (top of `MoE_Final.ipynb` or `moe.py`):

| Category | Key Parameters |
|----------|----------------|
| **Router** | `ROUTER_TYPES` (`['token_choice']`, `['hash']`, or both) |
| **Architecture** | `D_MODEL`, `NHEAD`, `NUM_EXPERTS`, `D_FF`, `TOP_K` |
| **Training** | `NUM_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `WARMUP_STEPS` |
| **Dataset** | `TRAIN_SAMPLES`, `VAL_SAMPLES`, `TEST_SAMPLES` |
| **HuggingFace** | `UPLOAD_TO_HF`, `HF_USERNAME` |
| **Bonus 1** | `USE_LOAD_BALANCER`, `RUN_LOAD_BALANCER_COMPARISON` |
| **Bonus 2** | `USE_CUSTOM_ATTENTION`, `GQA_NUM_KEY_VALUE_HEADS` |
| **Bonus 3** | `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT` |

---

## Hardware Requirements

Default MoE config (`d_model=512`, 6+6 layers) requires ~8 GB GPU memory. For smaller setups, reduce `D_MODEL`, layer counts, and `BATCH_SIZE`.
