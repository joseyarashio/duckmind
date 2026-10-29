"""Step 1b / 3: does each take, and each stored line, say what it should?

Breeze-ASR-25 (``MediaTek-Research/Breeze-ASR-25``, Apache-2.0; Whisper large-v2 tuned for
Taiwanese Mandarin, it writes 繁體中文) transcribes, in this process through transformers'
pipeline,

* ``--takes <raw dir>``: every take ``synth_qwen.py`` made (``<id>.t<k>.wav``) → ``{"<id>.t<k>":
  transcript}``, from which ``build.py --takes-asr`` keeps the best take of each line;
* ``--fixtures <dir>``: every stored line of a built fixture folder → ``{id: transcript}``, for
  ``build.py --asr`` and the comparison with ``lines.json``.

Run it with the same environment as ``synth_qwen.py``::

    PYTHONPATH=src <env>/bin/python tools/voice_fixtures/asr.py \\
        (--takes <raw dir> | --fixtures <dir>) --out <json> [--model <id or folder>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from duckmind.world import neuro_audio as A

MODEL = "MediaTek-Research/Breeze-ASR-25"


def transcriber(model: str):  # type: ignore[no-untyped-def]
    import torch
    from transformers import pipeline

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    asr = pipeline("automatic-speech-recognition", model=model, device=device, dtype=dtype)

    def hear(path: Path) -> str:
        samples, rate = A.read_wav(path)
        audio = A.resample(A.to_float(samples).astype(np.float64), rate, A.SR)
        # (a second of silence either side: the recogniser wants some context)
        padded = np.concatenate([np.zeros(A.SR), audio, np.zeros(A.SR)]).astype(np.float32)
        result = asr(
            {"raw": padded, "sampling_rate": A.SR},
            generate_kwargs={"language": "zh", "task": "transcribe"},
        )
        return str(result["text"]).strip()

    return hear


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--takes", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    if (args.takes is None) == (args.fixtures is None):
        raise SystemExit("one of --takes or --fixtures")
    hear = transcriber(args.model)
    out = {}
    if args.takes:
        for item in json.loads((args.takes / "raw.json").read_text(encoding="utf-8"))["lines"]:
            key = f"{item['id']}.t{item['take']}"
            out[key] = hear(args.takes / item["file"])
            print(f"{key}: {out[key]!r}", flush=True)
    else:
        manifest = json.loads((args.fixtures / "manifest.json").read_text(encoding="utf-8"))
        for line in manifest["utterances"]:
            out[line["id"]] = hear(args.fixtures / line["file"])
            print(f"{line['id']}: {line['text']!r} -> {out[line['id']]!r}", flush=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
