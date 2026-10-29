/* 說話 (live voice, LV5): the microphone's capture node (live_voice.js).
 * Runs on the audio thread: it only collects the input's samples and hands them to the page in blocks of
 * about 50 ms; the page turns them into 16 kHz mono 16-bit pieces and posts them to the simulator. */
class DuckMic extends AudioWorkletProcessor {
  constructor() {
    super();
    this.block = new Float32Array(Math.max(128, Math.round(sampleRate * 0.05)));
    this.fill = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel && channel.length) {
      let at = 0;
      while (at < channel.length) {
        const take = Math.min(channel.length - at, this.block.length - this.fill);
        this.block.set(channel.subarray(at, at + take), this.fill);
        this.fill += take;
        at += take;
        if (this.fill === this.block.length) {
          const out = this.block.slice(0);
          this.port.postMessage(out, [out.buffer]);
          this.fill = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("duck-mic", DuckMic);
