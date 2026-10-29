# Privacy

duckmind runs on your own computer (or the robot's board). Most of it never uses the
network. This page says what stays on your machine, and what leaves it when you
turn on a cloud feature.

## By default nothing leaves your machine

- The simulator, the duck's brain, its memories and its diary all run locally.
- The simulator page loads its fonts and scripts from the duckmind server itself, not
  from a CDN.
- The cloud voice is **off** by default. It needs a server started with
  `--live-voice` **and** the cloud switch on the page. The switch is off every time
  the server starts.
- `python -m duckmind.setup_cmd` downloads code and policies from GitHub and Hugging Face. It
  uploads nothing.
- Tests use recorded answers and never call a cloud service.

## What is kept on your machine

- **A duck's life folder** (the folder you pass with `--life`): its name, its saved
  state, and what it learned about people: the names the family gave, how each
  person looks to it (a number vector, not a photo), how close it feels to them,
  and notes. The folder is readable only by your user. Nothing in it is sent
  anywhere.
- **Caches** in duckmind's data folder (`duckmind.paths.data_dir()`, by default
  `~/.duckmind/data`): the duck's synthesised voice sounds, the motion table, and
  measurement logs if you record them.
- On a robot board, the service keeps its state in `/var/lib/duckmind`, readable only
  by the service user.

## When you turn on the cloud voice (Google Gemini Live)

While the switch is on, the server sends to Google's Gemini Live API:

- the microphone sound from the page (16 kHz mono),
- at most one camera picture per second from the duck's eye,
- short status text about the duck (at most every 5 seconds),
- the duck's name and the names of the people it knows (so the speech recogniser
  can hear them), and the instructions that describe the duck's character.

Transcripts, the last picture and the call list live only in the server's memory.
duckmind does not save recordings. The model's spoken answer is thrown away; only
its text and tool calls are used.

The API key is read from your system key store (on macOS the Keychain item
`duckmind-gemini-api-key`) or from the `GEMINI_API_KEY` environment variable. It is
never printed, logged or written to a file.

### Google's terms apply

Your use of Gemini is between you and Google, under the
[Gemini API Additional Terms of Service](https://ai.google.dev/gemini-api/terms).
Know this before you turn the switch on:

- **Free tier:** Google says that when you use unpaid services, including the
  unpaid quota of the Gemini API, it uses what you send and what it answers to
  provide, improve and develop its products, and that **human reviewers may read,
  annotate and process** your input and output. Google asks you not to send
  sensitive, confidential or personal information to unpaid services. A voice in a
  family home, a camera picture and people's names are personal information.
- **EEA, Switzerland and the UK:** Google's terms say: "You may use only Paid
  Services when making API Clients available to users in the European Economic
  Area, Switzerland, or the United Kingdom." If you or the people around the duck
  are there, use a paid Gemini API key only.
- Tell everyone who may be heard or seen by the duck before you turn the switch on.

## Recorded test answers

The recordings in `tests/fixtures/live/` are Gemini's answers to synthetic test
lines (made with a text-to-speech model), used for tests only. They contain no real
person's voice and no camera pictures.
