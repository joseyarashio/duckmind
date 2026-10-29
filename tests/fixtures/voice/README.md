# Scripted family lines (live voice, LV0)

Thirteen short 繁體中文 lines a family member says in the simulated home, as 16 kHz mono 16-bit wav
files plus `manifest.json` (text, speaker, default family member, intent, duration, level, the take
and seed it came from, the TTS model and its revision, and what the speech recogniser heard when it
listened to the stored file). About 0.5 MB in all.

They were spoken by Qwen3-TTS VoiceDesign (`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`, run locally;
voice *design*: each voice is a written description in `tools/voice_fixtures/lines.json`, no
person's voice is cloned) and checked by Breeze-ASR-25 (`MediaTek-Research/Breeze-ASR-25`, run
locally): of the takes made of each line (four; more for `stop_calling`), the one the recogniser
heard as exactly that sentence is kept, unless `lines.json` names the take (`stop_calling`: the
barge-in tests need mom to start clearly over the duck's call and talk past its end). Each file
starts where its speech starts. Rebuild with the three steps in `tools/voice_fixtures/`
(`synth_qwen.py`, `asr.py`, `build.py`); each file's own docstring says how. No cloud service is
involved.

Licence: the Qwen3-TTS model, its speech tokenizer and the `qwen-tts` package are Apache-2.0, which
puts no limit on the generated audio; these files are part of duckmind under its Apache-2.0
licence. Breeze-ASR-25 (Apache-2.0) only listens; nothing it makes is stored except the
transcripts in `manifest.json`.

The simulator plays them through the world's `say` stimulus (`duckmind.world.neuro_world`) into the
duck's microphone stream (`duckmind.simweb.neuro_mic`); `$DUCKMIND_VOICE_FIXTURES` names another
folder.
