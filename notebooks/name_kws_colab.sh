#!/usr/bin/env bash
# The duck's name detector, end to end on a Colab A100: public data, synthetic calls of the
# name, training, streaming evaluation, export. Resumable: every stage leaves a marker in
# $WORK/done and is skipped when run again. Family recordings are never used here.
#
#   nohup bash name_kws_colab.sh 蛋黃 "dan4 huang2" > /content/kws.log 2>&1 &
#
# Data (all ungated, permissive): MobvoiHotwords SLR87 (Apache-2.0), MUSAN SLR17 (CC BY 4.0),
# RIRS_NOISES SLR28 (Apache-2.0), AISHELL-1 SLR33 (Apache-2.0). About 46 GB of downloads.
set -euo pipefail

NAME="${1:?name in Chinese characters}"
PINYIN="${2:?pinyin, one syllable per character}"
WORK="${WORK:-/content/kws}"
CODE="${CODE:-/content/duckmind/src}"
VOICES="${VOICES:-1200}"
SYNTH_MINUTES="${SYNTH_MINUTES:-110}"
MIRRORS=(https://openslr.trmal.net https://openslr.elda.org https://www.openslr.org)
mkdir -p "$WORK/done" "$WORK/data"
PY="$WORK/env/bin/python"
export PYTHONPATH="$CODE" HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false

stage() { echo "=== $(date -u +%H:%M:%S) $*"; echo "$*" > "$WORK/STATUS"; }
done_() { touch "$WORK/done/$1"; }
is_done() { [ -f "$WORK/done/$1" ]; }

fetch() { # resource/path.tgz → $WORK/data/file, trying mirrors
  local path="$1" file="$WORK/data/$(basename "$1")"
  [ -f "$file.ok" ] && return 0
  for mirror in "${MIRRORS[@]}"; do
    if aria2c -q -x 16 -s 16 -k 20M --file-allocation=none -c -d "$WORK/data" \
        -o "$(basename "$1")" "$mirror/resources/$path"; then
      touch "$file.ok"; return 0
    fi
  done
  echo "download failed: $path" >&2; return 1
}

if ! is_done env; then
  stage "installing tools and the Python environment"
  apt-get -qq install -y aria2 > /dev/null
  pip -q install uv
  uv venv -q "$WORK/env" --python 3.12
  uv pip install -q --python "$PY" qwen-tts onnx onnxruntime scipy soundfile numpy pypinyin
  "$PY" -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__)"
  done_ env
fi

if ! is_done download; then
  stage "downloading public datasets in parallel (~46 GB)"
  fetch 87/mobvoi_hotword_dataset_resources.tgz
  fetch 87/mobvoi_hotword_dataset.tgz &
  fetch 17/musan.tar.gz &
  fetch 28/rirs_noises.zip &
  fetch 33/data_aishell.tgz &
  wait
  for f in mobvoi_hotword_dataset.tgz musan.tar.gz rirs_noises.zip data_aishell.tgz; do
    [ -f "$WORK/data/$f.ok" ] || { echo "missing $f" >&2; exit 1; }
  done
  done_ download
fi

if ! is_done unpack; then
  stage "unpacking"
  cd "$WORK/data"
  mkdir -p mobvoi
  tar -xzf mobvoi_hotword_dataset_resources.tgz -C mobvoi &
  tar -xzf mobvoi_hotword_dataset.tgz -C mobvoi &
  tar -xzf musan.tar.gz musan/noise musan/music &
  unzip -q -o rirs_noises.zip &
  tar -xzf data_aishell.tgz &
  wait
  # AISHELL-1 ships one tarball per speaker: all dev and test speakers plus 90 train ones
  # (about 70 h) are plenty of Mandarin that never says the name.
  cd data_aishell/wav
  # Speakers S0002–S0723 are train, S0724–S0763 dev, S0764–S0916 test.
  ls S*.tar.gz | sort > all.txt
  { head -90 all.txt; awk -F'[S.]' '$2 + 0 >= 724' all.txt; } | sort -u | xargs -P 16 -n 1 tar -xzf
  cd "$WORK/data"
  mkdir -p noise rir
  ln -sfn "$WORK/data/musan/noise" noise/musan_noise
  ln -sfn "$WORK/data/musan/music" noise/musan_music
  ln -sfn "$WORK/data/RIRS_NOISES/pointsource_noises" noise/pointsource
  ln -sfn "$WORK/data/RIRS_NOISES/simulated_rirs/smallroom" rir/smallroom
  ln -sfn "$WORK/data/RIRS_NOISES/simulated_rirs/mediumroom" rir/mediumroom
  ls -d mobvoi/*; du -sh data_aishell/wav noise/ rir/ 2>/dev/null | tail -3
  done_ unpack
fi &
UNPACK=$!

if ! is_done synth_smoke; then
  stage "synthesis smoke test (4 voices)"
  "$PY" -m duckmind.namekws.synth --name "$NAME" --pinyin "$PINYIN" --out "$WORK/synth_smoke" \
    --voices 4 --per-voice 12 --negatives-per-voice 12 --device cuda:0 --batch 32
  done_ synth_smoke
fi

if ! is_done synth; then
  stage "synthesizing $VOICES voices (stops starting batches after $SYNTH_MINUTES min)"
  "$PY" -m duckmind.namekws.synth --name "$NAME" --pinyin "$PINYIN" --out "$WORK/synth" \
    --voices "$VOICES" --per-voice 12 --negatives-per-voice 12 --device cuda:0 --batch 48 \
    --max-minutes "$SYNTH_MINUTES" --asr-filter
  done_ synth
fi

wait "$UNPACK"
is_done unpack || { echo "unpacking failed" >&2; exit 1; }

if ! is_done train; then
  stage "training and streaming evaluation"
  "$PY" -m duckmind.namekws.train \
    --mobvoi "$WORK/data/mobvoi" --synth "$WORK/synth" --noise "$WORK/data/noise" \
    --rir "$WORK/data/rir" --long "$WORK/data/data_aishell/wav" \
    --out "$WORK/out" --cache "$WORK/cache" --name "$NAME" --pinyin "$PINYIN" --device cuda:0
  done_ train
fi

stage "finished"
ls -la "$WORK/out"
