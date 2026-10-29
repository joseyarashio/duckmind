[English](README.md) | [繁體中文](README.zh-Hant.md) | [简体中文](README.zh-Hans.md) | [Español](README.es.md) | [Français](README.fr.md) | **日本語** | [한국어](README.ko.md)

# duckmind

duckmind は、小さなロボットのアヒルに自分の心を持たせます。自分が誰か、気分、記憶、
知っている人、やりたいこと、そして学んだこと。Microduck 互換のロボット向けに書かれていて、
ブラウザで開けるシミュレーターが付いています。

> **非公式プロジェクトです。** duckmind は Pollen Robotics とは関係がなく、Pollen Robotics の
> 承認や支援も受けていません。Microduck は Pollen Robotics の名称です。duckmind は Pollen の
> オープンソースのコード、ロボットモデル、ポリシーを使います。これらはセットアップのときに
> 上流の配布元からダウンロードします。

## 現状

- **シミュレーター：動きます。** アヒルはあなたのパソコンの中で暮らし、成長し、
  シミュレーションの家を歩き回り、人を覚えます。
- **実機：未検証です。** duckmind は本物のロボットではテストしていません。ボード向けの版は
  候補にすぎません。まず [SAFETY.md](SAFETY.md) を読んでください。

## これは何か

duckmind はアヒルが「何をしたいか」を決めます。モーターは直接動かしません。体を動かすのは
Pollen の `robotd` と学習済みのポリシーで、その安全ルールが常に優先されます。

```text
cognition -> behavior intent -> embodied skill -> robotd -> policy -> safety -> body
                                      ^                              |
                                      +-------- robot.state --------+
```

シミュレーターでは `robotd` より下の体が MuJoCo の物理シミュレーションに置き換わるので、
上の層はシミュレーターでも実機でも同じです。

## シミュレーターのページは 6 か国語

シミュレーターのページは 繁體中文（既定）、English、日本語、Español、简体中文、한국어 に
対応しています。ページで言語を切り替えると、ページ、サーバー、アヒルの頭脳が出す文字が
すべて切り替わります。家族が付けた名前（アヒルの名前、人の名前）はそのまま表示します。

## はじめかた

Python 3.12 以降、`git`、[`uv`](https://docs.astral.sh/uv/) が必要です。

```bash
git clone https://github.com/joseyarashio/duckmind.git
cd duckmind
uv run --python 3.12 --extra sim python -m duckmind.setup_cmd
```

`python -m duckmind.setup_cmd` は、duckmind が使うけれど同梱していないものをダウンロードします。

- Pollen の `microduck` のコード（robotd とアヒルの鳴き声）。固定したコミットです。
- Pollen の `microduck_rl` のコードと 3D ロボットモデル。固定したコミットです。3D メッシュの
  ライセンスは CC BY-NC-SA 4.0（非営利）です。setup はこのライセンスを表示し、あなたが
  「yes」と入力したときだけダウンロードします。
- Hugging Face にある Pollen の v5 ポリシー。ファイルごとに固定した sha256 と照合します。

ファイルは既定で `~/.duckmind` の下に置かれます（`DUCKMIND_HOME` で変えられます）。
すでにあるものを setup が書き換えることはありません。

次にシミュレーターのページを起動します。setup があなたのパスでこのコマンドを表示します。

```bash
uv run --python 3.12 --extra sim python -m duckmind.simweb --port 8765 \
  --microduck-mjcf ~/.duckmind/upstream/microduck_rl/src/mjlab_microduck/robot/microduck
# http://127.0.0.1:8765/ を開く
```

サーバーはあなた自身のパソコンからの接続だけを受け付けます。詳しくは
[docs/validation/run-duck-simulator.md](docs/validation/run-duck-simulator.md)、robotd を使った
リアルタイム物理は [docs/validation/run-live-sim.md](docs/validation/run-live-sim.md) を見てください。

クラウド音声（Google Gemini Live）は既定でオフです。オンにする前に [PRIVACY.md](PRIVACY.md)
を読んでください。

## ドキュメント

ドキュメントはほとんど英語です。

- [アーキテクチャ](architecture.md)と[ソースコードのレビュー](docs/architecture-review.md)
- 取り決め：[身体スキル](docs/contracts/embodied-skill.md)、
  [System 1 / System 2](docs/contracts/system1-system2.md)、
  [記憶](docs/contracts/memory.md)、[来歴](docs/contracts/provenance.md)、
  [シミュレーター](docs/contracts/simulator.md)
- [上流との検証](docs/upstream-validation.md)
- 実行：[シミュレーターのページ](docs/validation/run-duck-simulator.md)、
  [リアルタイム物理](docs/validation/run-live-sim.md)、
  [ペットのシミュレーション](docs/validation/run-pet-sim.md)
- 検査：[リプレイと敵対的検証](docs/validation/replay-adversarial.md)、
  [学習のクラッシュからの復旧](docs/validation/learning-crash-recovery.md)、
  [robotd 候補の比較](docs/validation/robotd-candidate.md)
- 学習：[ロボット外での強化学習](docs/training/official-rl.md)、
  [ローカル CPU での学習](docs/training/local-cpu.md)
- ボード：[導入](docs/deploy/board.md)、[オフラインリリース](docs/deploy/release.md)、
  [書き込み前の関門](docs/ready-to-burn.md)
- ページ：[6 か国語](docs/i18n.md)、[デザイン基準](docs/simulator-style.md)

## 開発

```bash
uv run --python 3.12 --extra dev pytest
uv run --python 3.12 --extra dev ruff check src tests
uv run --python 3.12 --extra dev ruff format --check src tests
uv run --python 3.12 --extra dev mypy
```

追加の依存グループ：`sim`（MuJoCo の物理）、`train`（mjlab でのポリシー学習）、
`kws`（名前の検出器）。変更を送る前に [CONTRIBUTING.md](CONTRIBUTING.md) を読んでください。
コードを書く AI エージェントは [AGENTS.md](AGENTS.md) も読んでください。
セキュリティの問題は [SECURITY.md](SECURITY.md) へ。

## ライセンス

duckmind は Apache License 2.0 で提供しています。[LICENSE](LICENSE)、[NOTICE](NOTICE)、
[THIRD_PARTY.md](THIRD_PARTY.md) を見てください。

第三者の素材の一部は Apache-2.0 ではなく、このリポジトリには入っていません。いちばん大事なのは、
Microduck の 3D メッシュが CC BY-NC-SA 4.0（非営利のみ）だということです。あなたがこの
ライセンスに同意したあとでだけ、setup が Pollen のリポジトリからダウンロードします。
`tests/fixtures/voice/` のテスト音声は Qwen3-TTS（Apache-2.0）で作りました。
`tests/fixtures/live/` の録音は Gemini モデルの出力で、テストにだけ使います。
