//! `duck-voice-stream` — the official Microduck voice held open for the simulator.
//!
//! OUR helper (duckmind a:sound, 2026-09-28), not upstream: it links upstream's `sounds` crate
//! by path and changes nothing in it. On the robot, `robotd` plays the ToF theremin by pulling
//! 10 ms blocks from a live `sounds::Stream::wheee(personality, 0)` whose pitch, level and mouth
//! opening the 50 Hz control loop sets from the hand's distance (`robotd/src/sound.rs`,
//! `robotd/src/theremin.rs`), and streams the joy-ride `wheee` as the bank's start → loop → end
//! segments. The simulator's robotd cannot open a sound card, so the simulator does the same
//! through this pipe — the same `Stream`, the same `hz_at`, the same segments, the same i16
//! conversion (`sounds::synth::to_i16`, which is also what robotd's writer thread does).
//!
//! One command per line on stdin, one answer per command on stdout:
//!
//! ```text
//! voice <seed> <wheee|choral|plain> <variant>  → ok <lo_hz> <hi_hz> <pitch_center_hz> <pitch_spread>
//! hz <position>                                → ok <hz>                 sounds::stream::hz_at
//! set <hz> <level> <open>                      → ok                      Stream::set
//! setpos <position> <level> <open>             → ok <hz>                 set(hz_at(position), …)
//! level <level>                                → ok                      Stream::set_level
//! rolloff <hz>                                 → ok                      0 = full range
//! block <n> <i16|f32>                          → ok <bytes>\n<raw little-endian samples>
//! silent                                       → ok <0|1>                Stream::is_silent
//! segments <seed> <variant>                    → ok <b1> <b2> <b3>\n<start><loop><end> (i16)
//! render <seed> <tag> <variant>                → ok <bytes>\n<i16 samples>  (a bank recipe)
//! seed_from_id <text…>                         → ok <u32>                sounds::seed_from_id
//! show <seed>                                  → ok <json>               the personality
//! quit
//! ```
//!
//! Anything wrong is `err <why>` on one line; the helper carries on.

use std::io::{self, BufRead, BufWriter, Write};

use sounds::{Personality, Stream};

struct Voice {
    personality: Personality,
    stream: Stream,
}

fn parse<T: std::str::FromStr>(word: Option<&str>, what: &str) -> Result<T, String> {
    word.ok_or_else(|| format!("missing {what}"))?
        .parse::<T>()
        .map_err(|_| format!("bad {what}"))
}

fn samples_i16(block: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(block.len() * 2);
    for sample in sounds::synth::to_i16(block) {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

fn samples_f32(block: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(block.len() * 4);
    for sample in block {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    bytes
}

fn show(p: &Personality) -> String {
    format!(
        "{{\"seed\":{},\"pitch_center_hz\":{},\"register\":{},\"pitch_spread\":{},\
         \"vibrato_rate_hz\":{},\"vibrato_depth\":{},\"quackiness\":{},\"breath\":{},\
         \"speed\":{}}}",
        p.seed,
        p.pitch_center_hz,
        p.register,
        p.pitch_spread,
        p.vibrato_rate_hz,
        p.vibrato_depth,
        p.quackiness,
        p.breath,
        p.speed
    )
}

/// One command. `Ok((line, payload))`: the answer line (without its newline) and the bytes that
/// follow it; `Err`: the reason, said as `err <reason>`.
fn run(voice: &mut Option<Voice>, line: &str) -> Result<(String, Vec<u8>), String> {
    let mut words = line.split_whitespace();
    let command = words.next().unwrap_or("");
    let need = |voice: &mut Option<Voice>| -> Result<(), String> {
        if voice.is_none() {
            return Err("no voice yet (voice <seed> <kind> <variant>)".into());
        }
        Ok(())
    };
    match command {
        "voice" => {
            let seed: u32 = parse(words.next(), "seed")?;
            let kind = words.next().unwrap_or("wheee");
            let variant: u32 = parse(words.next().or(Some("0")), "variant")?;
            let personality = Personality::from_seed(seed);
            let stream = match kind {
                "wheee" => Stream::wheee(&personality, variant),
                "choral" => Stream::choral(&personality, variant),
                "plain" => Stream::new(&personality, "plain", variant),
                other => return Err(format!("unknown voice kind {other:?}")),
            };
            let (lo, hi) = stream.range_hz();
            *voice = Some(Voice { personality, stream });
            Ok((
                format!(
                    "ok {lo} {hi} {} {}",
                    personality.pitch_center_hz, personality.pitch_spread
                ),
                Vec::new(),
            ))
        }
        "hz" => {
            need(voice)?;
            let position: f64 = parse(words.next(), "position")?;
            let v = voice.as_ref().expect("checked");
            let hz = sounds::stream::hz_at(&v.personality, position);
            Ok((format!("ok {hz}"), Vec::new()))
        }
        "set" => {
            need(voice)?;
            let hz: f64 = parse(words.next(), "hz")?;
            let level: f64 = parse(words.next(), "level")?;
            let open: f64 = parse(words.next(), "open")?;
            voice.as_mut().expect("checked").stream.set(hz, level, open);
            Ok(("ok".into(), Vec::new()))
        }
        "setpos" => {
            need(voice)?;
            let position: f64 = parse(words.next(), "position")?;
            let level: f64 = parse(words.next(), "level")?;
            let open: f64 = parse(words.next(), "open")?;
            let v = voice.as_mut().expect("checked");
            let hz = v.stream.hz_at(position);
            v.stream.set(hz, level, open);
            Ok((format!("ok {hz}"), Vec::new()))
        }
        "level" => {
            need(voice)?;
            let level: f64 = parse(words.next(), "level")?;
            voice.as_mut().expect("checked").stream.set_level(level);
            Ok(("ok".into(), Vec::new()))
        }
        "rolloff" => {
            need(voice)?;
            let hz: f64 = parse(words.next(), "hz")?;
            let rolloff = if hz > 0.0 { Some(hz) } else { None };
            voice
                .as_mut()
                .expect("checked")
                .stream
                .set_speaker_rolloff(rolloff);
            Ok(("ok".into(), Vec::new()))
        }
        "block" => {
            need(voice)?;
            let n: usize = parse(words.next(), "sample count")?;
            if n == 0 || n > 480_000 {
                return Err("a block is 1..480000 samples".into());
            }
            let format = words.next().unwrap_or("i16");
            let mut block = vec![0.0f32; n];
            voice.as_mut().expect("checked").stream.block(&mut block);
            let bytes = match format {
                "i16" => samples_i16(&block),
                "f32" => samples_f32(&block),
                other => return Err(format!("unknown sample format {other:?}")),
            };
            Ok((format!("ok {}", bytes.len()), bytes))
        }
        "silent" => {
            need(voice)?;
            let silent = voice.as_ref().expect("checked").stream.is_silent();
            Ok((format!("ok {}", u8::from(silent)), Vec::new()))
        }
        "segments" => {
            let seed: u32 = parse(words.next(), "seed")?;
            let variant: u32 = parse(words.next(), "variant")?;
            let personality = Personality::from_seed(seed);
            let (start, loop_seg, end) = sounds::voices::wheee_segments(&personality, variant);
            let (a, b, c) = (samples_i16(&start), samples_i16(&loop_seg), samples_i16(&end));
            let line = format!("ok {} {} {}", a.len(), b.len(), c.len());
            let mut bytes = a;
            bytes.extend(b);
            bytes.extend(c);
            Ok((line, bytes))
        }
        "render" => {
            let seed: u32 = parse(words.next(), "seed")?;
            let tag = words.next().ok_or("missing tag")?.to_owned();
            let variant: u32 = parse(words.next(), "variant")?;
            let personality = Personality::from_seed(seed);
            let buffer = sounds::render(&tag, &personality, variant).map_err(|e| e.to_string())?;
            let bytes = samples_i16(&buffer);
            Ok((format!("ok {}", bytes.len()), bytes))
        }
        "seed_from_id" => {
            let rest: Vec<&str> = words.collect();
            let text = rest.join(" ");
            Ok((format!("ok {}", sounds::seed_from_id(&text)), Vec::new()))
        }
        "show" => {
            let seed: u32 = parse(words.next(), "seed")?;
            Ok((format!("ok {}", show(&Personality::from_seed(seed))), Vec::new()))
        }
        "" => Err("empty command".into()),
        other => Err(format!("unknown command {other:?}")),
    }
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());
    let mut voice: Option<Voice> = None;
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim();
        if line == "quit" {
            break;
        }
        let answer = run(&mut voice, line);
        let written = match answer {
            Ok((head, payload)) => out
                .write_all(head.as_bytes())
                .and_then(|()| out.write_all(b"\n"))
                .and_then(|()| out.write_all(&payload)),
            Err(why) => {
                let why = why.replace('\n', " ");
                out.write_all(format!("err {why}\n").as_bytes())
            }
        };
        if written.and_then(|()| out.flush()).is_err() {
            break; // the reader went away
        }
    }
}
