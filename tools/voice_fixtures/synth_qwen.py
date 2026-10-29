"""Step 1 of the scripted-utterance fixtures: Qwen3-TTS VoiceDesign speaks ``lines.json``.

The model is ``Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`` (Apache-2.0, like its speech tokenizer and
the ``qwen-tts`` package), loaded in this process. Each speaker of ``lines.json`` is a written voice
description (``instruction``); no person's voice is cloned. Run it in its own environment (the
``qwen-tts`` package pins its own torch / transformers), never the duckmind venv::

    uv venv --python 3.12 <env> && VIRTUAL_ENV=<env> uv pip install qwen-tts
    PYTHONPATH=src <env>/bin/python tools/voice_fixtures/synth_qwen.py --out <raw dir> --takes 4

The first run downloads the model from Hugging Face (``--model`` takes a local folder too); after
that ``HF_HUB_OFFLINE=1`` keeps it offline. It writes ``<raw dir>/<id>.t<k>.wav`` (the model's
24 kHz mono 16-bit; take ``k`` uses the line's seed + 1000 k; a line's ``say``, when it has
one, is what the model is given instead of its ``text``) and ``raw.json`` (the model, its
revision, what was asked, the seconds it took). ``--takes N`` makes N takes of every line: the
same sentence is not always said well, so ``asr.py --takes`` listens to every take and
``build.py`` (duckmind venv) keeps the one the recogniser understood best.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from duckmind.world import neuro_audio as A

HERE = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
LANGUAGE = "Chinese"


def device_name() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def fetch(model: str) -> tuple[Path, str]:
    """``model``'s local folder and its Hugging Face commit ("local" for a folder given)."""
    if Path(model).is_dir():
        return Path(model), "local"
    from huggingface_hub import snapshot_download

    folder = Path(snapshot_download(model))  # (the cache only, under HF_HUB_OFFLINE=1)
    return folder, folder.name


def load(folder: Path, device: str) -> Any:
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    return Qwen3TTSModel.from_pretrained(str(folder), device_map=device, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*", help="line ids (default: all)")
    parser.add_argument("--takes", type=int, default=1, help="takes per line (default 1)")
    parser.add_argument("--first-take", type=int, default=0, help="the take to start at")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default="", help="cuda:0 / mps / cpu (default: the best)")
    args = parser.parse_args()
    import torch

    spec = json.loads((HERE / "lines.json").read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    device = args.device or device_name()
    started = time.perf_counter()
    folder, commit = fetch(args.model)
    model = load(folder, device)
    print(f"{args.model} loaded on {device} in {time.perf_counter() - started:.1f} s", flush=True)
    done = []
    for line in spec["lines"]:
        if args.only and line["id"] not in args.only:
            continue
        speaker = spec["speakers"][line["speaker"]]
        for take in range(args.first_take, args.first_take + args.takes):
            seed = int(line["seed"]) + 1000 * take
            torch.manual_seed(seed)
            begun = time.perf_counter()
            wavs, rate = model.generate_voice_design(
                text=line.get("say", line["text"]),
                instruct=speaker["instruction"],
                language=LANGUAGE,
            )
            audio = np.asarray(wavs[0], dtype=np.float64)
            target = args.out / f"{line['id']}.t{take}.wav"
            A.write_wav(target, A.to_pcm16(audio), int(rate))
            info = {
                "id": line["id"],
                "take": take,
                "seed": seed,
                "said": line.get("say", line["text"]),
                "file": target.name,
                "rate": int(rate),
                "duration_sec": round(len(audio) / int(rate), 3),
                "elapsed_sec": round(time.perf_counter() - begun, 2),
            }
            done.append(info)
            print(
                f"{line['id']} take {take}: {info['duration_sec']} s audio in "
                f"{info['elapsed_sec']} s",
                flush=True,
            )
    summary = args.out / "raw.json"
    earlier = json.loads(summary.read_text())["lines"] if summary.is_file() else []
    seen = {(d["id"], d["take"]) for d in done}
    kept = [d for d in earlier if (d["id"], d["take"]) not in seen]
    summary.write_text(
        json.dumps(
            {
                "model": args.model,
                "revision": commit,
                "language": LANGUAGE,
                "lines": kept + done,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
