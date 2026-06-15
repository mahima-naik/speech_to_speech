# LFM (Liquid Foundation Model) — Audio

This project uses **LFM2-Audio** and **LFM2.5-Audio** by [Liquid AI](https://www.liquid.ai/), an end-to-end audio foundation model for speech-to-speech, ASR, and TTS tasks. At only 1.5B parameters, it delivers real-time conversational quality on Apple Silicon via MLX.

## Architecture

LFM2-Audio uses a hybrid architecture with three main components:

1. **FastConformer Audio Encoder** (115M params) — Encodes raw audio waveforms into continuous embeddings by chunking audio into ~80ms segments and projecting into the LFM backbone's embedding space. Tokenizer-free approach avoids artifacts from discrete audio tokenization.

2. **LFM2 Backbone** (1.2B params) — A hybrid convolution + attention language model that processes interleaved text and audio tokens. Built on Liquid AI's dynamical systems architecture (not traditional transformers), designed for efficient inference on edge devices.

3. **Audio Detokenizer** — An RQ-Transformer that generates discrete audio tokens (8 codebooks, Mimi-compatible) from the backbone's output, decoded back into waveforms.

```
Audio ──→ FastConformer Encoder ──→ LFM2 Backbone ──→ RQ-Transformer ──→ Audio/Text
Text ──────────────────────────→         │
                                         └──→ Text tokens (ASR/chat)
```

## Generation Modes

| Mode | Description | Use Cases |
|---|---|---|
| **Interleaved** | Fixed interleaved pattern of text + audio tokens. Minimizes time-to-first-audio. | Real-time speech-to-speech chat |
| **Sequential** | Model decides when to switch modalities via special tokens. | ASR, TTS, transcription |

## Model Details

| Property | Value |
|---|---|
| Parameters (total) | ~1.5B (1.2B backbone + 115M encoder) |
| Audio encoder | FastConformer (canary-180m-flash based) |
| Backbone | Hybrid conv + attention (LFM2/LFM2.5) |
| Audio codec | 8 codebooks, Mimi-compatible |
| Context length | 32,768 tokens |
| Text vocab | 65,536 |
| Audio vocab | 2,049 × 8 codebooks |
| Precision | bfloat16 |
| Sample rate | 16 kHz |
| License | LFM Open License v1.0 |

## Usage

### Via `liquid-audio` (recommended)

```bash
pip install liquid-audio

# Gradio demo
liquid-audio-demo

# Python
from liquid_audio import LFM2AudioModel

model = LFM2AudioModel.from_pretrained("LiquidAI/LFM2.5-Audio-1.5B")
result = model.generate("input.wav", mode="interleaved")
```

### Via `mlx-lm` (text-only)

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/LFM2-1.2B-8bit")
response = generate(model, tokenizer, prompt="Hello")
```

## Variants

| Model | Description |
|---|---|
| LFM2-Audio-1.5B | Original end-to-end audio model |
| LFM2.5-Audio-1.5B | Updated with faster LFM-based detokenizer, better ASR/TTS |
| LFM2.5-Audio-1.5B-JP | Japanese speech-to-speech variant |

## References

- [LFM2 Technical Report](https://arxiv.org/abs/2511.23404)
- [Liquid AI](https://www.liquid.ai/) — Official site
- [Liquid4All/liquid-audio](https://github.com/Liquid4All/liquid-audio) — GitHub repository
- [Hugging Face Collection](https://huggingface.co/LiquidAI)
