[English](README.md) | **繁體中文** | [简体中文](README.zh-Hans.md) | [Español](README.es.md) | [Français](README.fr.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

# duckmind

duckmind 讓一隻小機器鴨有自己的心：牠是誰、牠的心情、牠的記憶、牠認得的人、牠想做什麼，
還有牠學到的東西。它是為相容 Microduck 的機器人寫的，附一個在瀏覽器裡就能打開的模擬器。

> **非官方專案。** duckmind 與 Pollen Robotics 無關，沒有得到 Pollen Robotics 的認可或贊助。
> Microduck 是 Pollen Robotics 的名稱。duckmind 會用到 Pollen 的開源程式、機器人模型和政策，
> 這些都是安裝時從上游來源下載的。

## 現況

- **模擬器：可以用。** 小鴨會在你的電腦裡生活、長大、在模擬的家裡走動，也會記得人。
- **真機：還沒驗證。** duckmind 沒有在真的機器人上測試過。板子上的版本只是候選版。
  請先讀 [SAFETY.md](SAFETY.md)。

## 這是什麼

duckmind 決定小鴨「想做什麼」，但不直接控制馬達。身體由 Pollen 的 `robotd` 和它訓練好的政策來動，
它們的安全規則一直在上面把關：

```text
cognition -> behavior intent -> embodied skill -> robotd -> policy -> safety -> body
                                      ^                              |
                                      +-------- robot.state --------+
```

在模擬器裡，`robotd` 下面的身體換成 MuJoCo 物理模擬，所以上面幾層在模擬器和真機上是一樣的。

## 模擬器頁面有六種語言

模擬器頁面有繁體中文（預設）、English、日本語、Español、简体中文和 한국어。在頁面上切換語言，
頁面、伺服器和小鴨的腦送出的文字都會跟著換。家人取的名字（小鴨的名字、人的名字）照原樣顯示，不翻譯。

## 快速開始

需要 Python 3.12 以上、`git` 和 [`uv`](https://docs.astral.sh/uv/)。

```bash
git clone https://github.com/joseyarashio/duckmind.git
cd duckmind
uv run --python 3.12 --extra sim python -m duckmind.setup_cmd
```

`python -m duckmind.setup_cmd` 會下載 duckmind 要用、但沒有放在 repo 裡的東西：

- Pollen 的 `microduck` 程式（robotd 和小鴨的叫聲），固定在一個釘住的版本；
- Pollen 的 `microduck_rl` 程式和 3D 機器人模型，固定在一個釘住的版本。3D 網格的授權是
  CC BY-NC-SA 4.0（非商業）。setup 會先顯示這份授權，你輸入「yes」才會下載；
- Hugging Face 上 Pollen 的 v5 政策，每個檔案都會和釘住的 sha256 比對。

檔案預設放在 `~/.duckmind` 底下（用 `DUCKMIND_HOME` 可以換地方）。已經存在的東西，setup 一律不改。

接著啟動模擬器頁面。setup 會印出這行指令，裡面是你自己的路徑：

```bash
uv run --python 3.12 --extra sim python -m duckmind.simweb --port 8765 \
  --microduck-mjcf ~/.duckmind/upstream/microduck_rl/src/mjlab_microduck/robot/microduck
# 打開 http://127.0.0.1:8765/
```

伺服器只在你自己的電腦上接受連線。更多說明見
[docs/validation/run-duck-simulator.md](docs/validation/run-duck-simulator.md)；
接上 robotd 的即時物理見 [docs/validation/run-live-sim.md](docs/validation/run-live-sim.md)。

雲端語音（Google Gemini Live）預設是關的。打開之前請先讀 [PRIVACY.md](PRIVACY.md)。

## 文件

- [架構](architecture.md)和[原始碼審查](docs/architecture-review.md)
- 介面約定：[具身技能](docs/contracts/embodied-skill.md)、
  [System 1 / System 2](docs/contracts/system1-system2.md)、
  [記憶](docs/contracts/memory.md)、[來源紀錄](docs/contracts/provenance.md)、
  [模擬器](docs/contracts/simulator.md)
- [上游驗證](docs/upstream-validation.md)
- 執行：[模擬器頁面](docs/validation/run-duck-simulator.md)、
  [即時物理](docs/validation/run-live-sim.md)、
  [寵物模擬](docs/validation/run-pet-sim.md)
- 檢查：[重播與對抗驗證](docs/validation/replay-adversarial.md)、
  [學習當機復原](docs/validation/learning-crash-recovery.md)、
  [robotd 候選比較](docs/validation/robotd-candidate.md)
- 訓練：[離機強化學習](docs/training/official-rl.md)、
  [本機 CPU 訓練](docs/training/local-cpu.md)
- 板子：[部署](docs/deploy/board.md)、[離線發行](docs/deploy/release.md)、
  [燒錄前關卡](docs/ready-to-burn.md)
- 頁面：[六種語言](docs/i18n.md)、[設計規範](docs/simulator-style.md)

文件大多是英文。

## 開發

```bash
uv run --python 3.12 --extra dev pytest
uv run --python 3.12 --extra dev ruff check src tests
uv run --python 3.12 --extra dev ruff format --check src tests
uv run --python 3.12 --extra dev mypy
```

額外套件組：`sim`（MuJoCo 物理）、`train`（用 mjlab 訓練政策）、`kws`（名字偵測器）。
送出修改之前請讀 [CONTRIBUTING.md](CONTRIBUTING.md)；寫程式的 AI 代理另外讀 [AGENTS.md](AGENTS.md)。
安全問題見 [SECURITY.md](SECURITY.md)。

## 授權

duckmind 採用 Apache License 2.0 授權。見 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和
[THIRD_PARTY.md](THIRD_PARTY.md)。

有些第三方素材不是 Apache-2.0，也不在這個 repo 裡。最重要的是：Microduck 的 3D 網格是
CC BY-NC-SA 4.0（只能非商業使用）；你同意這份授權之後，setup 才會從 Pollen 的 repo 下載。
`tests/fixtures/voice/` 裡的測試語音是用 Qwen3-TTS（Apache-2.0）做的。
`tests/fixtures/live/` 裡的錄音是 Gemini 模型的輸出，只拿來測試。
