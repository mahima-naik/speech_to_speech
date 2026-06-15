# LLaMA (Language Model Backbone)

This project uses **LLaMA** (Large Language Model Meta AI) as the language model backbone in multiple components. The architecture follows the standard LLaMA design with grouped-query attention, RoPE (Rotary Position Embeddings), SwiGLU activation, and RMSNorm.

## Usage in This Project

### 1. GLM-ASR Decoder (Speech-to-Text)

The GLM-ASR speech recognition model uses a **LLaMA decoder** (28 layers) to generate text transcripts from audio embeddings. Audio features extracted by a Whisper encoder are projected into LLaMA's embedding space via an MLP adapter.

```
Audio → Whisper Encoder → MLP Adapter → LLaMA Decoder → Text
```

| Component | Value |
|---|---|
| Layers | 28 |
| Hidden size | 2048 |
| Attention heads | 16 |
| KV heads | 4 |
| Vocab size | 59,264 |
| Max context | 65,536 |

### 2. LLaMA TTS Model (Text-to-Speech)

The `mlx_audio` TTS library includes a LLaMA-based text-to-speech model that combines a **LLaMA language model** (from `mlx_lm`) with a **SNAC audio codec** for neural audio compression.

```
Text → LLaMA LM → Audio Tokens → SNAC Decoder → Waveform
```

The LLaMA model generates hierarchical audio tokens which are decoded by the multi-scale SNAC codec into a 24 kHz waveform.

## Architecture

- **Grouped-Query Attention** — 16 heads with 4 KV heads for efficient inference
- **RoPE** — Rotary Position Embeddings with configurable theta (default 10,000)
- **SwiGLU MLP** — Gated activation with gate/up/down projections
- **RMSNorm** — Pre-normalization with learnable scale
- **Tied/Untied Embeddings** — Configurable weight tying

## Usage

```bash
# Via mlx_audio TTS
python -m mlx_audio.tts.generate \
  --model path/to/llama-tts-model \
  --text "Hello, this is a test."

# Via mlx_audio STT (GLM-ASR with LLaMA decoder)
python -m mlx_audio.stt.generate \
  --model path/to/glm-asr-model \
  --audio input.wav
```

## References

- [LLaMA Paper](https://arxiv.org/abs/2302.13971) — Original architecture
- [mlx-lm](https://github.com/ml-explore/mlx-lm) — MLX implementation of LLaMA
- [SNAC Codec](https://github.com/hubertsiuzdak/snac) — Multi-scale neural audio codec
