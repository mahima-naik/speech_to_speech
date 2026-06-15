---
license: apache-2.0
base_model: k2-fsa/OmniVoice
library_name: mlx-audio
pipeline_tag: text-to-speech
tags:
- mlx
---

OmniVoice in MLX (bf16). For [mlx-audio](https://github.com/Blaizzy/mlx-audio) on Apple Silicon.

## Usage

```bash
pip install mlx-audio
python -m mlx_audio.tts.generate \
  --model commandeaw/OmniVoice-MLX-bf16 \
  --text "สวัสดีค่ะ ยินดีต้อนรับ" \
  --lang_code th \
  --ref_audio reference.wav
```

`--ref_audio` is optional (zero-shot voice cloning); drop it for the default voice.

## Size

| variant | repo | vs base (fp32) |
|---|---|---|
| bf16 | 1.6 GB | -50% |
| 8bit | 1.1 GB | -68% |
| 4bit | 0.75 GB | -77% |
