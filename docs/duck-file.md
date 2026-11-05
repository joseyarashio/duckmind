# The duck file: one duck, one file

[繁體中文](duck-file.zh-Hant.md)

A duck you raise in the simulator lives in one folder on your computer (its *life folder*). A **duck
file** packs what makes that duck itself into a single file, so you can move it to another computer
and it is still the same duck: the same name and personality, the same growth day, stage and branch,
the same people it knows, the same tricks and how well it does each one, and the same calls.

The home does **not** move with it. The map of the home, the furniture, the spots it learnt and its
nest spot stay behind: in a new home (or a new body) it gets to know the place again, which is what
a real pet does too.

## Using it

**On the page.** Open the 我的小鴨 (my duck) tab. The *My duck file* card has two buttons:

* **Export duck file** downloads `<name>-<date>.duck` of the duck this simulator raises. Its mind is
  saved first, so the file is the duck as it is right now. (The button shows only when the
  simulator was started with `--life FOLDER`.)
* **Import duck file** lets you pick a `.duck` file. The page checks it first and tells you who is in
  it (name, growth day, how many people it knows, how many tricks it has practised). Press **Bring it
  to this computer** to make a new life folder for it. Then start the simulator with that folder
  (`--life FOLDER`) and it is the same duck. The page never switches ducks by itself: the card's
  fold-out *After importing: how do I switch to it?* says so, with the command.

**On the command line.**

```
python -m duckmind.life.duckfile export LIFE_FOLDER [-o FILE.duck] [--no-voice]
python -m duckmind.life.duckfile inspect FILE.duck
python -m duckmind.life.duckfile import FILE.duck [--into NEW_FOLDER] [--replace]
```

`import` makes a new folder in `<DUCKMIND_HOME>/lives/` (by default `~/.duckmind/lives/`) named
after the duck. Export a duck while no server is living it, or use the page's button: the page saves
the duck's mind first, the command line reads the files as they are on disk.

## What is inside

A duck file is a zip archive:

| Member | What it holds |
|---|---|
| `manifest.json` | format `duckmind.duckfile`, version, when it was made, the build that made it, the duck in numbers (id, name, seed, birth, nature, growth day, stage, branch, how many people, each trick's proficiency), the voice library, what stayed behind, and the SHA-256 and size of every other member |
| `duck/life.json` | the life record: name, seed, birth, nature, the hardware it hatched with |
| `duck/core.json` | Duck Core's save: its whole growth record, every input, byte for byte |
| `duck/neural.json` | what the neural duck learnt: the people it knows (the names the family gave them and their looks), the cues, the trick book, habituation and drives; **without** places, things, nest spot and map |
| `voice/<key>/…` | the voice library its lineage sounds like now: the learned calls as wav clips and their plan |

Left out on purpose: the home map and the furniture (places, things, nest spot, the lasting map),
the eye's model of one home's floor (`eye/`), the layouts the editor applied (`layouts/`), the mind's
previous snapshot, and the species voice bank (it is rendered again from the duck's seed on the
other computer). A duck file is about 1–2 MB.

**Who the duck is.** A duck's id is made from its seed and its birth moment. Two ducks of the same
seed hatched at different times are different ducks; every copy of one duck has the same id.

**The calls.** The voice library is named by a key computed from the duck's lineage, so the same
lineage gives the same key and the same clips. The file carries the clips, and import puts them where
the server looks for them; the other computer's server then uses them as they are. When the duck
grows into a new voice later, the new library is made from that computer's own voice bank.

## Safety rules

* **Never overwritten silently.** Import always makes a new life folder. If this computer already
  has the same duck (same id), import is refused unless you ask to replace it (`--replace`, or the
  page's **Replace (keep a backup)** button); the folder it replaces is first moved aside as
  `<folder>.replaced-<time>`. The duck a running simulator lives is never replaced. A folder name
  that is taken by anything else gets a new name; nothing is written over.
* **Checked before anything is written.** Every member must match the checksum and size in the
  manifest, and no other member may be there. A file from a newer build (a newer duck-file,
  life-record or snapshot version) is refused; older versions are read through their migrations.
  Duck Core's save must replay on this build: a duck grown on another version of the growth engine
  cannot be raised on by this one. The parts must all be of the same duck.
* **Private.** The file holds people's names and looks (as numbers, not photos). The command line
  writes it readable by you only (a download from the page is saved by your browser), the program
  never sends it anywhere, and the simulator's export and import answer only a browser on the same
  computer. The checksums find damage and accidental changes; they are not a signature, so only
  import files you made or trust.
