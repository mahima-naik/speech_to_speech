# GLM-ASR (Speech-to-Text)

Part of the [mlx-audio](https://github.com/Blaizzy/mlx-audio) library — this project uses **GLM-ASR** ("glm" mapped to "glmasr") as the speech recognition component for transcribing audio input.

## Architecture

GLM-ASR is a speech recognition model that combines three components:

1. **Whisper Audio Encoder** — A 32-layer Whisper-style encoder (1280-dim, 20 attention heads) that converts mel spectrograms into audio features. Supports RoPE (Rotary Position Embeddings) for positional encoding.

2. **MLP Adapter** — A 2-layer MLP (GELU activation) that projects Whisper encoder outputs into the language model's embedding space. Uses a merge factor of 4 to reduce sequence length.

3. **LLaMA Decoder** — A 28-layer LLaMA-style language model (2048 hidden size, 16 attention heads, 4 KV heads) with RoPE, SiLU activation, and RMSNorm. Generates text tokens from the fused audio-text embeddings.

## How It Works

1. Audio is converted to a mel spectrogram (128 mel bins, 16 kHz)
2. The Whisper encoder extracts audio features
3. Features are projected via the MLP adapter into LM embedding space
4. Audio embeddings are inserted into text token embeddings at `<|begin_of_audio|>` / `<|end_of_audio|>` positions
5. The LLaMA decoder autoregressively generates transcription tokens

## Usage

```bash
python -m mlx_audio.stt.generate \
  --model path/to/glm-asr-model \
  --audio input.wav
```

The model can also be used programmatically:

```python
from mlx_audio.stt.utils import load_model

model = load_model("path/to/glm-asr-model")
result = model.generate("input.wav")
print(result.text)
```

## Model Config

| Component | Config |
|---|---|
| Audio encoder | Whisper-style (32 layers, 1280 dim, 20 heads) |
| Adapter | 2-layer MLP (merge factor 4, GELU) |
| Language model | LLaMA (28 layers, 2048 hidden, 16 heads, 4 KV heads) |
| Vocab size | 59,264 |
| Max context | 65,536 tokens |
| Audio sample rate | 16 kHz |
| Mel bins | 128 |

## References

- [mlx-audio](https://github.com/Blaizzy/mlx-audio) — MLX audio library for Apple Silicon
- [Whisper](https://github.com/openai/whisper) — Audio encoder backbone
- [LLaMA](https://arxiv.org/abs/2302.13971) — Decoder language model architecture
