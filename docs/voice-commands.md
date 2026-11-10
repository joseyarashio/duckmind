# Voice commands (hold to speak)

繁體中文：[voice-commands.zh-Hant.md](voice-commands.zh-Hant.md)

On the simulator page, next to the command box, there is a **Speak** button (🎤). Hold it, say a command to the duck, and let go. The page shows what it heard and what it took it as, for example `Heard: "sit down" → sit`, and the duck gets the command exactly as if you had typed it. It still decides for itself whether to do it.

**Everything happens on your computer.** The page records only while you hold the button and sends the clip to the duckmind server on the same machine. The server turns it into words with a local Whisper model, keeps the clip in memory for that one recognition and never writes it to disk. Nothing goes to a cloud service. See [PRIVACY.md](../PRIVACY.md).

## Turning it on

1. Install the voice packages: `pip install 'duckmind[voice]'` (or `uv sync --extra voice`). On a Mac with Apple silicon this uses [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (MIT), which also installs PyTorch (a large download); on other computers it uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT) on the CPU. [pypinyin](https://github.com/mozillazg/python-pinyin) (MIT) helps with Chinese words that sound alike.
2. Start the simulator with the neural duck (`--brain neuro`). Voice commands are on by default (`--voice-commands auto`). The first start downloads the model once from Hugging Face (Whisper medium, about 1.5 GB); after that it works offline.
3. Open the page at `http://localhost:<port>` (browsers allow the microphone only on `localhost` or HTTPS). The first press asks for microphone permission.

`--voice-commands off` turns it off. `--voice-commands mlx:<model>` or `faster:<model>` picks another Whisper model.

## What you can say

The page's language is the language it listens for (繁體中文, English, 日本語, Español, 简体中文, 한국어).

- **Commands**: the same words as the command box: sit, stand up, come here, look at me, stop, follow me, go home, and the tricks (lie down, spin, bow, shake, flip). You can add the duck's name ("Danhuang, sit").
- **Praise and scolding**: "good boy", "well done" is praise; "no", "not like that" is a scold. They work like the feedback buttons.
- **Anything else** does nothing: the page says it was not a command.

One short sentence per press, up to 10 seconds. A very short press, or one where nothing was said, is not sent to the recogniser at all.

When the recogniser writes a word that only sounds like a command (「按我」 for 「看我」), the server compares how the whole sentence sounds with the command words and takes the closest one, but only when it is clearly the closest. A longer sentence that merely contains a command-like sound is left alone.

## How well it works

Measured on 450 fixed test sentences (75 per language), spoken by a local text-to-speech model with four made-up voices, before anyone's real voice was tried. The numbers are the share of the 70 commands / praise / scolds per language that came out right, and how many of the 5 sentences not meant for the duck still made it do something.

| | 繁中 | English | 日本語 | Español | 简中 | 한국어 |
|---|---|---|---|---|---|---|
| quiet room | 82.9 % | 100 % | 95.7 % | 91.4 % | 91.4 % | 95.7 % |
| household noise, 10 dB | 81.4 % | 100 % | 91.4 % | 92.9 % | 87.1 % | 98.6 % |
| household noise, 5 dB | 80.0 % | 100 % | 92.9 % | 92.9 % | 84.3 % | 95.7 % |
| not for the duck, still acted | 0 / 5 | 0 / 5 | 1 / 5 | 0 / 5 | 0 / 5 | 0 / 5 |

Whisper medium on Apple silicon (M5 Pro); the noise is DEMAND's living-room, kitchen and laundry recordings.

**Traditional Chinese did not pass.** The target, set before the test, was at least 90 % in a quiet room for every language. Five languages passed; Traditional Chinese reached 82.9 % (58 of 70), 7 points short. The target was not lowered and the feature shipped with this known gap: in this test about one Traditional Chinese command in six was misheard, so check the "Heard" line. Traditional Chinese is the weakest because two- or three-character commands said alone are often heard as other words (part of that is the synthetic voice saying them unclearly). Synthetic voices are not real ones, so your own results will differ.

**How fast.** From letting go of the button to the duck getting the command, on a running simulator (M5 Pro, while other simulations ran): median 0.31 s, 95 % within 0.33 s with mlx-whisper medium. Without Apple silicon the default is faster-whisper **small** on the CPU: median 0.98 s, 95 % within 1.05 s on the same machine's CPU (medium took 2.8 s, too slow for a spoken command). Small is a little less accurate (quiet room, same test: 繁中 80.0 %, English 98.6 %, 日本語 91.4 %, Español 92.9 %, 简中 85.7 %, 한국어 92.9 %). A slower CPU takes longer.
