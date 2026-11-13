# Explainer videos

One short (~2 minute) narrated video per paper in the chain. Every chart is drawn from indflex results, so the numbers on screen are the toolkit's own.

| Script | Video | Published |
|---|---|---|
| `lstn.py` | LSTN: a linear model of industrial production for demand response | [YouTube](https://youtu.be/QwHTS2DnenQ) |
| `crtn.py` | cRTN: efficient scheduling of discrete industrial processes through continuous modeling | [YouTube](https://youtu.be/TZ0LXch5aPw) |
| `psi.py` | PSI: production scheduling identification from smart meter data (needs Gurobi) | [YouTube](https://youtu.be/zhWvC3sqh4s) |
| `d3r.py` | D3R: data-driven dimension reduction for industrial load modeling | [YouTube](https://youtu.be/tLNliWvT6i8) |
| `vpp.py` | Real-time operation of virtual power plants with optimal power disaggregation | [YouTube](https://youtu.be/uDk0lgekcLo) |

## Rendering

Requires `ffmpeg`. The narration uses the [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) neural voice `af_heart` by default. Kokoro needs PyTorch, so install it in its own environment; the scripts find it at `~/.venvs/kokoro` or through `KOKORO_PYTHON`:

```bash
uv venv ~/.venvs/kokoro --python 3.12
uv pip install --python ~/.venvs/kokoro/bin/python "kokoro>=0.9.4" soundfile \
  "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
```

The model (about 330 MB) downloads from Hugging Face on first use. Then:

```bash
python tools/videos/lstn.py OUT_DIR                     # Kokoro af_heart, normal speed
python tools/videos/lstn.py OUT_DIR --voice af_bella --speed 1.05
python tools/videos/lstn.py OUT_DIR --engine say --voice "Ava (Premium)" --rate 170   # macOS voice, no Kokoro
python tools/videos/lstn.py OUT_DIR --no-captions
```

Each script writes `OUT_DIR/<name>.mp4` (1920×1080, H.264/AAC, loudness-normalized to −16 LUFS), `<name>.srt` (captions, for YouTube) and `<name>_script.md` (the narration). All sentences of a video are voiced in one batch, so the model loads once. The LSTN video measures solve times while rendering, so run it on an idle machine.

## How it works

`common.py` is a small framework. A video is a list of scenes; each scene has narration (a list of sentences) and a `draw(fig, t)` function where `t` runs from 0 to 1 over the scene. Every sentence is spoken separately, so captions are timed exactly, and the frames are drawn with matplotlib and piped to ffmpeg. A sentence can be a `(caption, spoken)` pair when the spoken form should differ (for example "cRTN" is spoken as "C R T N").
