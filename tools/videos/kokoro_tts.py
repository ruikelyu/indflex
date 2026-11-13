"""Batch text-to-speech with Kokoro (https://huggingface.co/hexgrad/Kokoro-82M).

Called by common.py, usually from a separate environment that has ``kokoro`` installed:

    python kokoro_tts.py JOBS.json --voice af_heart --speed 1.0

JOBS.json is a list of ``{"text": ..., "path": ...}``; each text is written to its path as a
24 kHz mono WAV. The model is loaded once for the whole batch.
"""

import argparse
import json
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from kokoro import KPipeline  # noqa: E402

SR = 24000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()
    jobs = json.load(open(args.jobs))
    pipe = KPipeline(lang_code=args.voice[0], repo_id="hexgrad/Kokoro-82M")
    for job in jobs:
        chunks = [audio.numpy() for _, _, audio in pipe(job["text"], voice=args.voice, speed=args.speed)]
        sf.write(job["path"], np.concatenate(chunks) if chunks else np.zeros(SR // 10), SR)
    print(f"kokoro: {len(jobs)} clips ({args.voice})", flush=True)


if __name__ == "__main__":
    main()
