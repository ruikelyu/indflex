# Explainer videos

One short (~2 minute) narrated video per paper in the chain. Every chart is drawn from indflex results, so the numbers on screen are the toolkit's own.

| Script | Video | Published |
|---|---|---|
| `lstn.py` | LSTN: a linear model of industrial production for demand response | [YouTube](https://youtu.be/wMm-IgvIvhM) |
| `crtn.py` | cRTN: efficient scheduling of discrete industrial processes through continuous modeling | [YouTube](https://youtu.be/RATnHWVbLwY) |
| `psi.py` | PSI: production scheduling identification from smart meter data (needs Gurobi) | [YouTube](https://youtu.be/gpxkEAigneE) |
| `d3r.py` | D3R: data-driven dimension reduction for industrial load modeling | [YouTube](https://youtu.be/ClJwRWYXZfI) |
| `vpp.py` | Real-time operation of virtual power plants with optimal power disaggregation | [YouTube](https://youtu.be/dFsv5i7JZIM) |

## Rendering

Requires macOS (the narration uses the `say` command) and `ffmpeg`.

```bash
python tools/videos/lstn.py OUT_DIR                     # default voice: Samantha, 175 words/min
python tools/videos/lstn.py OUT_DIR --voice "Ava (Premium)" --rate 170
python tools/videos/lstn.py OUT_DIR --no-captions
```

Each script writes `OUT_DIR/<name>.mp4` (1920×1080, H.264/AAC), `<name>.srt` (captions, for YouTube) and `<name>_script.md` (the narration). The LSTN video measures solve times while rendering, so run it on an idle machine.

**Better voices.** The built-in compact voices sound robotic. Free higher-quality voices are available in *System Settings → Accessibility → Spoken Content → System Voice → Manage Voices* (for example "Ava (Premium)" or "Zoe (Premium)"); list installed voices with `say -v '?'`.

## How it works

`common.py` is a small framework. A video is a list of scenes; each scene has narration (a list of sentences) and a `draw(fig, t)` function where `t` runs from 0 to 1 over the scene. Every sentence is spoken separately, so captions are timed exactly, and the frames are drawn with matplotlib and piped to ffmpeg. A sentence can be a `(caption, spoken)` pair when the spoken form should differ (for example "cRTN" is spoken as "C R T N").
