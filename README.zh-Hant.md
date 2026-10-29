[English](README.md) | **繁體中文** | [简体中文](README.zh-Hans.md) | [Español](README.es.md) | [Français](README.fr.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

# duckmind

duckmind 讓一隻小機器鴨有自己的心：牠是誰、牠的心情、牠的記憶、牠認得的人、牠想做什麼，
還有牠學到的東西。它是為相容 Microduck 的機器人寫的，附一個在瀏覽器裡就能打開的模擬器。

> **非官方專案。** duckmind 與 Pollen Robotics 無關，沒有得到 Pollen Robotics 的認可或贊助。
> Microduck 是 Pollen Robotics 的名稱。duckmind 會用到 Pollen 的開源程式、機器人模型和政策，
> 這些都是安裝時從上游來源下載的。

## 看看牠

<p align="center"><img src="docs/media/screens/main-desktop.zh-Hant.webp" alt="duckmind 模擬器畫面" width="880"></p>

| <img src="docs/media/clips/walk.webp" width="220" alt="走路"> | <img src="docs/media/clips/bow.webp" width="220" alt="鞠躬"> | <img src="docs/media/clips/lie-low.webp" width="220" alt="趴低"> |
|:---:|:---:|:---:|
| 走路 | 鞠躬 | 趴低 |
| <img src="docs/media/clips/spin.webp" width="220" alt="轉圈"> | <img src="docs/media/clips/flip.webp" width="220" alt="翻跟斗"> | <img src="docs/media/clips/getup.webp" width="220" alt="跌倒後站起來"> |
| 轉圈 | 翻跟斗 | 跌倒後站起來 |

全部截圖和短片：[docs/media](docs/media/README.md)（授權是 CC BY-NC-SA 4.0，因為畫面裡有 Pollen 的 Microduck 3D 模型）。小鴨會做的所有動作，分類列表：[docs/motions.zh-Hant.md](docs/motions.zh-Hant.md)。

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

## 這個專案想做什麼

一句話：**讓一隻機器小鴨，慢慢長成一隻真正屬於你家的寵物。**

- 牠不是遙控玩具，也不是照劇本演的機器人。牠每一個動作，都是自己「感覺到什麼、想要什麼」之後決定的。
- 牠只能用自己的感官：眼睛（鏡頭）、耳朵、距離感測器、身體的感覺。模擬器裡的「標準答案」（東西真正在哪、地圖長怎樣），牠一律看不到，跟真的小鴨一樣。
- **神經鴨是活的介面，成長才是核心。** 每一刻的反應（被嚇到、好奇、想睡、想玩）由「神經鴨」的大腦負責；一天一天會長成什麼樣子，由「Duck Core」的成長機制負責。兩個缺一不可。

### 每一隻都不一樣：隨機種子

- 孵化的時候給牠一個數字，叫「種子」。同一個種子長出同樣的基因；換一個種子，就是另一隻鴨。
- 種子決定牠一開始的個性傾向（比較黏人、比較愛探索、比較慢吞吞愛做夢…）、動作的快慢和隨機程度，還有牠自己的一套叫聲。
- 種子只是起點。之後牠遇到什麼、家人怎麼對牠，會讓牠往不同方向「發散」：同一個種子交給兩個家庭，養一陣子以後，會變成兩隻不一樣的鴨。
- 同樣的種子、同樣的經歷，結果可以重現。這讓研究和除錯變得可能，也讓「牠為什麼變成這樣」查得到原因。

### 養成：牠真的會長大

- 用「成長日」算年紀，不是用開機時數：一天裡醒著夠久才算長大一天。
- 從雛鴨、幼鴨、少年鴨到成鴨，會做的事跟著解鎖。例如翻跟斗大約要到少年鴨（45 個成長日上下）才會，實際哪一天看每隻鴨長得快慢。
- 你怎麼照顧牠，會影響牠長成哪一型：好奇型、悠哉型、依偎型、愛唱歌型。
- 牠會記得：認得的人、喜歡什麼、哪裡跌倒過要避開、家裡長什麼樣子。
- 叫聲會一代一代演變（叫聲系譜）：從牙牙學語開始，家人說「乖」的那幾聲會被留下來，慢慢變成牠自己的歌。
- 可以教把戲：叫牠做，做到了馬上稱讚或給點心，牠就越練越會；太久沒練會生疏，但學會過的不會全忘。

### 從模擬到實體：以後可以「燒進」真的小鴨

- 現在全部在電腦裡的物理模擬器跑（MuJoCo 加上 Pollen 的 robotd 模擬）。身體的物理、馬達的延遲、電量，都盡量照真的 Microduck 來模擬。
- **一隻鴨就是一個檔案**：種子、基因、成長紀錄、學到的東西、叫聲。設計的目標是：在模擬器裡養大的鴨，連上實體 Microduck 之後，就是同一隻鴨換了真的身體——名字、個性、記憶、把戲都還在。
- 大腦可以留在家裡的電腦，身體換成真的小鴨；同一套大腦程式，差別只在身體和感官的校正。
- **老實說：** 實體小鴨還沒在真機上驗證過；「小鴨檔案匯出／匯入」和「遠端身體」是規劃中的功能。詳見上面的「現況」一節和 [SAFETY.md](SAFETY.md)。

### 現在就能玩、能做的事

- **跟牠互動：** 拍手、叫名字、摸頭、丟球、抱起來、關燈，看牠的反射和情緒怎麼反應。
- **下指令（六種語言）：** 坐下、站起來、過來、看我、停、跟著我、回家；點地板叫牠過去（試驗中）；也可以切到遙控模式。
- **教牠把戲：** 趴下、鞠躬、甩一甩、轉圈、翻跟斗，在「訓練」分頁看每個把戲有多熟。
- **看牠的內心：** 牠現在想做什麼、為什麼、怕不怕、長到哪裡了。
- **布置家裡：** 自己擺家具、放門檻和地毯，看牠怎麼過門檻、繞牆、找路回家。
- **訓練新動作：** 用同一套工具訓練新的動作網路（mjlab，A100 或本機），裝進動作表給牠用。
- 完整的動作清單在 [docs/motions.md](docs/motions.md)（繁中：[docs/motions.zh-Hant.md](docs/motions.zh-Hant.md)）。

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

## 使用、引用、修改：請註明作者出處

這個專案的作者是 **joseyarashio**。如果你使用、引用、改寫，或拿它做衍生作品，請註明作者和出處，例如：

> duckmind，作者 joseyarashio — https://github.com/joseyarashio/duckmind

- 程式碼採用 Apache-2.0。依照授權第 4 條，散布原版或修改版時，必須附上 [LICENSE](LICENSE)，保留 [NOTICE](NOTICE) 裡的署名，並在改過的檔案上註明你改了什麼。
- 寫文章或論文引用時，可以用 repo 裡的 [CITATION.cff](CITATION.cff)。
- 截圖和影片（[docs/media](docs/media/README.md)）另外採用 CC BY-NC-SA 4.0：要署名、不能商用、改作要用相同授權，因為畫面裡有 Pollen 的 3D 模型。

## 授權

duckmind 採用 Apache License 2.0 授權。見 [LICENSE](LICENSE)、[NOTICE](NOTICE) 和
[THIRD_PARTY.md](THIRD_PARTY.md)。

有些第三方素材不是 Apache-2.0，也不在這個 repo 裡。最重要的是：Microduck 的 3D 網格是
CC BY-NC-SA 4.0（只能非商業使用）；你同意這份授權之後，setup 才會從 Pollen 的 repo 下載。
`tests/fixtures/voice/` 裡的測試語音是用 Qwen3-TTS（Apache-2.0）做的。
`tests/fixtures/live/` 裡的錄音是 Gemini 模型的輸出，只拿來測試。
