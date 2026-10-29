[English](README.md) | [繁體中文](README.zh-Hant.md) | [简体中文](README.zh-Hans.md) | [Español](README.es.md) | [Français](README.fr.md) | **日本語** | [한국어](README.ko.md)

# duckmind

duckmind は、小さなロボットのアヒルに自分の心を持たせます。自分が誰か、気分、記憶、
知っている人、やりたいこと、そして学んだこと。Microduck 互換のロボット向けに書かれていて、
ブラウザで開けるシミュレーターが付いています。

> **非公式プロジェクトです。** duckmind は Pollen Robotics とは関係がなく、Pollen Robotics の
> 承認や支援も受けていません。Microduck は Pollen Robotics の名称です。duckmind は Pollen の
> オープンソースのコード、ロボットモデル、ポリシーを使います。これらはセットアップのときに
> 上流の配布元からダウンロードします。

## 見てみる

<p align="center"><img src="docs/media/screens/main-desktop.en.webp" alt="duckmind シミュレーターの画面" width="880"></p>

| <img src="docs/media/clips/walk.webp" width="220" alt="歩く"> | <img src="docs/media/clips/bow.webp" width="220" alt="おじぎ"> | <img src="docs/media/clips/lie-low.webp" width="220" alt="伏せる"> |
|:---:|:---:|:---:|
| 歩く | おじぎ | 伏せる |
| <img src="docs/media/clips/spin.webp" width="220" alt="くるっと回る"> | <img src="docs/media/clips/flip.webp" width="220" alt="前転"> | <img src="docs/media/clips/getup.webp" width="220" alt="転んで起き上がる"> |
| くるっと回る | 前転 | 転んで起き上がる |

スクリーンショットと動画の一覧：[docs/media](docs/media/README.md)（Pollen の Microduck 3D モデルが写っているため、ライセンスは CC BY-NC-SA 4.0）。アヒルができる動きの分類一覧（英語）：[docs/motions.md](docs/motions.md)。

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

## このプロジェクトがやりたいこと

ひとことで言うと：**ロボットの小さなアヒルが、少しずつ、本当にあなたの家のペットに育っていくこと。**

- ラジコンのおもちゃでも、台本どおりに演じるロボットでもありません。ひとつひとつの動きは、アヒル自身が「何かを感じて、何かをしたくなって」から決めています。
- 使えるのは自分の感覚だけです。目（カメラ）、耳、距離センサー、体の感覚。シミュレーターの中の「正解」（物が本当はどこにあるか、地図がどうなっているか）は一切見えません。本物のアヒルと同じです。
- **ニューラル・ダックは生きたインターフェース、成長こそが核。** その瞬間ごとの反応（びっくりする、気になる、眠い、遊びたい）は「ニューラル・ダック」の脳の担当。一日一日どんなアヒルに育っていくかは「Duck Core」の成長のしくみの担当。どちらが欠けてもだめです。

### 一羽ずつ違う：ランダムシード

- 孵化するときに、「シード」という数をひとつもらいます。同じシードからは同じ遺伝子が育ち、シードを変えれば別のアヒルになります。
- シードで決まるのは、最初の性格の傾向（甘えんぼ寄り、探検好き寄り、のんびり夢見がち…）、動きの速さとランダムさ、そして自分だけの鳴き声のセットです。
- シードはスタート地点にすぎません。そのあと何に出会うか、家族にどう接してもらうかで、いろいろな方向に「分かれて」いきます。同じシードを二つの家族に渡すと、しばらく育てたあとには、違う二羽のアヒルになっています。
- 同じシードで同じ経験をすれば、同じ結果を再現できます。だから研究やデバッグができるし、「どうしてこうなったのか」をたどることもできます。

### 育てる：本当に大きくなる

- 年齢は「成長日」で数えます。電源を入れていた時間ではありません。一日のうち十分長く起きていて、はじめて一日ぶん育ちます。
- ヒナ、子アヒル、若アヒル、おとなアヒルと進むにつれて、できることが増えていきます。たとえば宙返りができるのは若アヒル（だいたい 45 成長日前後）になってから。実際に何日目かは、そのアヒルの育つ速さしだいです。
- どうお世話するかで、どのタイプに育つかが変わります：好奇心タイプ、のんびりタイプ、甘えんぼタイプ、歌好きタイプ。
- ちゃんと覚えています：知っている人、好きなもの、転んだことがあって避けたい場所、家の様子。
- 鳴き声は世代を重ねて変わっていきます（鳴き声の系譜）。最初は片言から始まり、家族が「いい子」とほめた鳴き声が残って、だんだんその子だけの歌になります。
- 芸を教えられます。やってと頼んで、できたらすぐにほめるかおやつをあげると、練習するほど上手になります。長く練習しないと腕がなまりますが、一度覚えたことを全部忘れはしません。

### シミュレーターから実物へ：いつか本物のアヒルに「焼き込める」

- 今はすべてパソコンの中の物理シミュレーターで動いています（MuJoCo と Pollen の robotd のシミュレーション）。体の物理、モーターの遅れ、バッテリーは、できるだけ本物の Microduck に合わせてシミュレーションしています。
- **一羽のアヒルは一つのファイル**：シード、遺伝子、成長の記録、覚えたこと、鳴き声。設計の目標は、シミュレーターで育てたアヒルを本物の Microduck につなげば、同じアヒルが本物の体に移るだけ、ということ。名前も性格も記憶も芸もそのままです。
- 脳は家のパソコンに置いたまま、体だけ本物の小さなアヒルにできます。脳のプログラムは同じで、違うのは体と感覚のキャリブレーションだけです。
- **正直に言うと：** 実物のアヒルはまだ実機で検証していません。「アヒルのファイルのエクスポート／インポート」と「リモートの体」は計画中の機能です。上の「現状」の節と [SAFETY.md](SAFETY.md) を見てください。

### いますぐ遊べること、できること

- **ふれあう：** 手をたたく、名前を呼ぶ、頭をなでる、ボールを投げる、抱き上げる、電気を消す。反射や気持ちがどう反応するか見てみましょう。
- **指示を出す（6 か国語）：** おすわり、立つ、おいで、見て、止まれ、ついてきて、お家へ。床をタップしてそこへ行かせる（試験中）。リモートモードに切り替えることもできます。
- **芸を教える：** 伏せ、お辞儀、ぶるぶる、くるくる、宙返り。「しつけ」タブで、それぞれの芸がどれくらい身についたか見られます。
- **心の中をのぞく：** いま何をしたいのか、なぜなのか、怖がっていないか、どこまで育ったか。
- **家を模様替えする：** 家具や敷居、ラグを自分で置いて、アヒルが敷居を越えたり、壁をよけたり、家への帰り道を探したりする様子を見られます。
- **新しい動きを学習させる：** 同じ道具で新しい動きのネットワークを学習させ（mjlab、A100 か手元のパソコンで）、動きの一覧に入れて使わせられます。
- 動きの全一覧は [docs/motions.md](docs/motions.md)（繁中：[docs/motions.zh-Hant.md](docs/motions.zh-Hant.md)）にあります。

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

## 使う・引用する・改変するとき：作者と出典を明記してください

このプロジェクトの作者は **joseyarashio** です。使う、引用する、書き換える、あるいはこれをもとに派生物を作るときは、作者と出典を明記してください。たとえば：

> duckmind、作者 joseyarashio — https://github.com/joseyarashio/duckmind

- コードは Apache-2.0 です。ライセンスの第 4 条により、元の版や改変した版を配布するときは、[LICENSE](LICENSE) を添え、[NOTICE](NOTICE) にあるクレジットを残し、変更したファイルには変更したことを明記しなければなりません。
- 記事や論文で引用するときは、リポジトリの [CITATION.cff](CITATION.cff) を使えます。
- スクリーンショットと動画（[docs/media](docs/media/README.md)）は別に CC BY-NC-SA 4.0 です。クレジット表記が必要、商用利用は不可、改変したものは同じライセンスで。画面に Pollen の 3D モデルが写っているためです。

## ライセンス

duckmind は Apache License 2.0 で提供しています。[LICENSE](LICENSE)、[NOTICE](NOTICE)、
[THIRD_PARTY.md](THIRD_PARTY.md) を見てください。

第三者の素材の一部は Apache-2.0 ではなく、このリポジトリには入っていません。いちばん大事なのは、
Microduck の 3D メッシュが CC BY-NC-SA 4.0（非営利のみ）だということです。あなたがこの
ライセンスに同意したあとでだけ、setup が Pollen のリポジトリからダウンロードします。
`tests/fixtures/voice/` のテスト音声は Qwen3-TTS（Apache-2.0）で作りました。
`tests/fixtures/live/` の録音は Gemini モデルの出力で、テストにだけ使います。
