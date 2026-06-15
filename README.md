# Speech-to-Speech

An MLX-optimized speech-to-speech model based on [OmniVoice](https://github.com/k2-fsa/OmniVoice), designed for Apple Silicon (M-series chips). The model converts text to speech with natural prosody and supports multilingual output and zero-shot voice cloning.

## Contents

```
Speech_to_Speech/
├── models/
│   └── OmniVoice-MLX-bf16/       # Model weights and config (1.6 GB, bf16)
│       ├── config.json            # Model architecture (Qwen3-based LLM)
│       ├── model.safetensors      # Main model weights
│       ├── tokenizer.json         # Qwen2 tokenizer vocabulary
│       ├── tokenizer_config.json  # Tokenizer configuration
│       ├── chat_template.jinja    # Chat formatting template
│       ├── .gitattributes         # Git LFS tracking rules
│       ├── README.md              # Original model card
│       └── audio_tokenizer/       # Audio encoder/decoder
│           ├── config.json        # DAC + HuBERT config
│           ├── model.safetensors  # Audio tokenizer weights
│           └── preprocessor_config.json
└── venv/                          # Python virtual environment
```

## How It Works

The model combines three components:

1. **LLM (Qwen3-based)** — A 28-layer causal language model (1.0B params) that processes text input and generates audio token sequences.
2. **Semantic Audio Encoder (HuBERT)** — Extracts semantic speech features from audio at 16 kHz.
3. **Acoustic Audio Codec (DAC)** — A 9-codebook residual vector quantizer that reconstructs waveforms from audio tokens.

**Pipeline:**
- Text is tokenized by the Qwen2 tokenizer
- The LLM predicts 8 audio codebook tokens jointly
- Audio tokens are decoded by the audio tokenizer into a waveform
- Weighted loss (8,8,6,6,4,4,2,2) is applied across codebooks

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.9+
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) library

## Usage

### Installation

```bash
pip install mlx-audio
```

### Text-to-Speech (Basic)

```bash
python -m mlx_audio.tts.generate \
  --model Speech_to_Speech/models/OmniVoice-MLX-bf16 \
  --text "Hello, welcome to speech to speech." \
  --lang_code en
```

### Zero-Shot Voice Cloning

Provide a reference audio file to clone the speaker's voice:

```bash
python -m mlx_audio.tts.generate \
  --model Speech_to_Speech/models/OmniVoice-MLX-bf16 \
  --text "This is in the voice of the reference speaker." \
  --lang_code en \
  --ref_audio path/to/reference.wav
```

### Multilingual

```bash
python -m mlx_audio.tts.generate \
  --model Speech_to_Speech/models/OmniVoice-MLX-bf16 \
  --text "สวัสดีค่ะ ยินดีต้อนรับ" \
  --lang_code th
```

| Parameter | Description |
|---|---|
| `--model` | Path or Hugging Face repo ID of the model |
| `--text` | Input text to synthesize |
| `--lang_code` | Language code (e.g., `en`, `th`, `zh`, `ja`) |
| `--ref_audio` | Optional reference audio for voice cloning |

## Model Details

| Property | Value |
|---|---|
| Architecture | OmniVoice (Qwen3-based) |
| Parameters | ~1.0B |
| Precision | bfloat16 |
| Context length | 40,960 tokens |
| Audio codebooks | 8 (weighted: 8,8,6,6,4,4,2,2) |
| Audio tokenizer | DAC + HuBERT (16 kHz / 24 kHz) |
| License | Apache 2.0 |

## Model Variants

| Variant | Size | Compression |
|---|---|---|
| bf16 | 1.6 GB | -50% vs fp32 |
| 8-bit | 1.1 GB | -68% vs fp32 |
| 4-bit | 0.75 GB | -77% vs fp32 |

## References

- [OmniVoice](https://github.com/k2-fsa/OmniVoice) — Base model
- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — MLX audio library for Apple Silicon
- [Qwen3](https://github.com/QwenLM/Qwen3) — Base LLM architecture
