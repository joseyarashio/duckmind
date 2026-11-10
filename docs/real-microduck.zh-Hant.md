# 接上真的 Microduck（遠端身體）

[English](real-microduck.md)

Duck Mind 的大腦要住在家裡的電腦，身體是那隻小機器鴨。機器人自己的程式 `robotd` 留在小鴨身上；小鴨上再跑一個小小的 **bridge**，讓家裡網路上的**一個**大腦連得到它。這一頁說明：哪些已經好了、怎麼接小鴨、哪些還要在真的機器人上確認。

> **現況（2026-11-08）。** 連線、bridge 和安全停車都做好了，用第二個程序裡的**模擬**小鴨，隔著一個會延遲、抖動、掉包、斷線的網路墊片演練過。還沒接過真的 Microduck。下面標「**待驗**」的步驟，都是照官方軟體推出來的，要等有人在機器人上跑過才算數。出貨的小鴨用 Pollen 的量產主板（「beta」）加 V1.0 臉部板、工廠燒好的 Yocto 映像檔：robotd 0.18.x（API 49）、策略組 v8（下一版會換成 v9，也就是 v7 的走路網路）、兩層的 `robotd.toml`、直立裝的 GC2093 相機。模擬器跑的是 robotd 0.18.5 ＋ v8。

## 整體怎麼接

```text
家裡電腦                                         小鴨（Pollen 的板子）
────────                                         ────────────────────
大腦（模擬器頁面／神經鴨）                        robotd   /run/robotd.sock
  │ robotd.sock、tof.sock、media.sock（本機）     tofd     /run/tofd/tof.sock
  ▼                                              mediad   /run/mediad/media.sock
RemoteLink ══ 區網上的一條 TCP 連線 ══════════► bridge ──► 上面那些 unix socket
            密鑰握手、每一框都簽章、               │
            每 100 ms 打一次招呼                   └─ deadman：1.0 秒沒聽到大腦 →
                                                      robot.stop，然後坐下
```

- 官方 `robotd` **只開 unix socket**（沒有 TCP、沒有認證；socket 的 `0660 root:robot` 權限就是全部的門檻）。bridge 不改 robotd：它把大腦的連線轉給 robotd 的 socket（跟 `ssh -L` 一樣），再加上認證和 deadman。
- 電腦這邊，`RemoteLink` 在一個私人資料夾裡把小鴨的程式變成本機 socket，大腦原本的 robotd 程式照平常的方式連上去就好。

## 安全設計

| | |
|---|---|
| 聽在哪裡 | 預設只聽本機，別台機器連不到；你要給小鴨自己的區網位址（第 4 步說怎麼找）。`0.0.0.0` 也可以，但小鴨連著的每個網路都會聽到——不要用。兩端都拒絕不在本機、私有、link-local 範圍的位址。 |
| 誰能開 | 證明自己有共用密鑰的那**一個**大腦（雙方互相出題的 HMAC；密鑰從不上網路）。另一個有密鑰的大腦連進來會接手。 |
| 防竄改 | 握手之後每一框都有序號和簽章（每次連線自己的金鑰，HMAC-SHA256）；壞掉或順序不對的框會斷線並停車。連線**沒有加密**——只在區網用。 |
| 大腦可以送什麼 | robotd：`hello`、`robot.subscribe`、`move`、`head`、`mouth`、`pose`、`look`、`do`、`stop`、`health`、`model`、`modelApi`、`policies`、`safeToRestart`。其他的（`robot.relax`、`loadPolicy`、`shutdown`、`setMode`、`rebootMotors`、`system.*`、`net.*`…）一律回錯誤，不會到 robotd。 |
| 把小鴨開起來 | **Duck Mind 絕不做這件事；要由人用小鴨自己的控制來開。** 操作員的 `robot.enable`、`robot.init`、`robot.poseJoints` 只有在 bridge 後面是模擬身體（`sim-body`）時才會轉過去。真的小鴨上的 bridge（`--backend real`，也是預設）和 `sim-body --as-real` 一律回錯誤（`… is never sent to a real duck; a person turns the duck on with its own control`）、在事件檔記一筆 `refused_operator`，不會送到 robotd。大腦這邊也不會送：接真的小鴨時，它根本不開操作員的連線。 |
| 斷線 | 大腦每 100 ms 打一次招呼。**1.0 秒**沒收到任何東西（或連線斷了、或收到壞框），bridge 關掉所有串流——大腦之前送的東西不會晚到——送 `robot.stop`，小鴨還站著就送 `robot.do sit_toggle`（一直重試到坐下；坐著或跌倒時絕不送）。robotd 自己在最後一個 `robot.move` 之後 0.5 秒就已經把走路歸零。 |
| 連回來 | 大腦重新撥號（0.1 → 1 秒漸慢），在小鴨現在的樣子（坐著）接手。 |

為什麼是 1.0 秒、不更短：Wi-Fi 上的 TCP 偶爾會卡住（掉一個封包要等 200 ms 重傳，連掉三次就是 1.4 秒）。演練時 0.6 秒的 deadman 在普通的 1 % 掉包下就會誤觸發；每次誤觸發都會讓小鴨坐下。

## 演練結果（模擬小鴨，一台 Mac，兩個程序）

同一套大腦端的身體程式，分別開本機身體和墊片後面的遠端身體（單程 50 ± 15 ms、1 % 掉包，照 TCP 重傳處理）。每一輪：20 次指令來回、站起來、走 3 秒、轉彎、踢、坐下。用 `tools/measure_remote_body.py` 量的。

| 身體 | 連線（單程）| 指令來回 p50／p95 | 走路 | 轉彎 | 踢 | 跌倒 | 斷線 |
|---|---|---|---|---|---|---|---|
| 本機 | – | 19／38 ms | 10/10 | 10/10 | 10/10 | 0 | – |
| 遠端 | 20 ± 5 ms、掉包 0.2 % | 65／93 ms | 10/10 | 10/10 | 10/10 | 0 | 0 |
| 遠端 | 50 ± 15 ms、掉包 1 % | 131／285 ms | 10/10 | 9/10 | 10/10 | 0 | 1 |
| 遠端 | 80 ± 20 ms、掉包 3 % | 207／408 ms | 10/10 | 8/10 | 9/10 | 0 | 4 |

*指令來回*：大腦送出 `robot.head` → robotd 的狀態框顯示出來、回到大腦。*斷線*：墊片把大腦的封包卡住超過 1.0 秒（TCP 連續重傳三次）；每次小鴨都停下、坐下，大腦再接回來。沒做成的轉彎和踢，都發生在有斷線的那一輪。

**走路中切斷連線**（共 16 次：50 ms 10 次、20 ms 3 次、80 ms 3 次；墊片把所有東西卡住，就像 Wi-Fi 斷掉）：

| 從切斷到… | 中位數 | 最慢 |
|---|---|---|
| 小鴨停下（真實速度 < 每秒 3 公分、持續 0.3 秒；robotd 自己的 0.5 秒 deadman） | 0.94 秒 | 1.04 秒 |
| bridge 判定斷線、送 `robot.stop` | 1.00 秒 | 1.02 秒 |
| bridge 送坐下 | 1.05 秒 | 1.13 秒 |
| 小鴨坐好（真實軀幹高度低於 8.5 公分） | 1.81 秒 | 2.00 秒 |
| 大腦發現 | 1.17 秒 | 1.25 秒 |

連線恢復後 0.06–0.21 秒就接回來，大腦的身體 0.22–0.95 秒重新收到 robotd 的狀態；16 次切斷後小鴨都能再走；沒有跌倒。坐著掛線 10 分鐘（掉包 1 %），以及 0.2 %、3 % 各 5 分鐘，斷線 0 次。

**神經鴨走這條連線**（模擬器頁面 `--brain neuro --body remote://…`，用頁面遙控以每秒 0.3 公尺走，切斷 4 秒共 10 次，50 ± 15 ms／掉包 1 %）：每次切斷前都有走（5 秒 0.23–0.50 公尺，10/10）；每次切斷後 bridge 0.97 秒（最慢 1.01 秒）停車，robotd 的策略 1.16 秒（最慢 1.22 秒）變成坐下；連線回來後 1.58 秒（最慢 2.17 秒）大腦又開始跑，小鴨每次都能再走（6 秒 0.33–0.64 公尺，10/10）。模擬身體要跟大腦的小鴨一致：大腦的小鴨沒有 ToF、身體卻有（或反過來），行為會不一樣（`sim-body --tof on|off`）。

## 接小鴨——目前知道的步驟

1. **先用 Pollen 自己的工具把小鴨設好**：Wi-Fi（在電腦上 `duckctl wifi connect`，或在小鴨上 `sudo robotctl net connect <ssid> --psk <密碼>`），再用 `duckctl ip` 找位址、在小鴨上跑 `robotctl health`。出貨的小鴨已經燒好 Pollen 的映像檔；自己組的 Radxa Zero 3 照官方的開發板說明。
2. **在小鴨上裝 Duck Mind**，照 [docs/deploy/board.md](deploy/board.md)（venv 裡裝 wheel；Python ≥ 3.12）。bridge 只需要標準函式庫和 Duck Mind 本身。
3. **在電腦上做共用密鑰**，複製到小鴨（兩份都要權限 600）：

   ```sh
   python -m duckmind.remotebody token ~/.duckmind/remote-body.token
   duckctl scp ~/.duckmind/remote-body.token :.duckmind/remote-body.token
   ```

   （`duckctl scp` 自己會找小鴨；開頭的 `:` 是小鴨上的路徑，在 `--user` 的家目錄，沒給就用 `DUCK_BOARD_USER`，再沒有就是 `microduck`。資料夾先用 `duckctl ssh` 建好。）
4. **在小鴨上啟動 bridge**，用 `robot` 群組裡的使用者（這樣才打得開 `/run/robotd.sock`），聽在小鴨自己在家裡網路上的位址：

   ```sh
   python -m duckmind.remotebody bridge --listen <小鴨位址>:8797 \
       --token-file ~/.duckmind/remote-body.token --backend real \
       --robotd /run/robotd.sock --tofd /run/tofd/tof.sock --media /run/mediad/media.sock \
       --events ~/remote-body-events.jsonl
   ```

   `<小鴨位址>` 就是第 1 步的位址：在電腦上跑 `duckctl ip`，或在小鴨上跑 `ip -4 addr`（看 Wi-Fi 那個介面的 `inet` 那一行，不要後面的 `/24`）。不加 `--listen` 的話 bridge 只聽本機（`127.0.0.1`），電腦連不到。請給小鴨自己的位址，不要用 `0.0.0.0`：那樣小鴨連著的每個網路都會聽到。路由器換了小鴨的位址，就用新的位址重開 bridge。

   接的是真的小鴨時，bridge 一啟動就會把 **Pollen 的匿名使用資料**（遙測）關掉一次（主人決定的；0.18.3 起小鴨預設會送）。這個開關在 `updaterd` 手上：bridge 對 `/run/updaterd.sock` 送 `telemetry.set {"enabled": false}`（權限 `0660 root:robot`，bridge 的使用者就能送）。關掉之後重開機、更新都會保持關著，隨機的機器人編號也會刪掉。bridge 會在事件檔記一筆 `telemetry`，寫它看到什麼、做了什麼。加 `--keep-pollen-telemetry` 就不碰這個開關。模擬器沒有跑 `updaterd`，所以這台電腦什麼都不會送出去。

5. **在電腦上檢查連線**——只讀 robotd 的健康和策略，什麼都不會動：

   ```sh
   python -m duckmind.remotebody check remote://<小鴨位址>:8797
   ```

   會印出有哪些通道、deadman 多久、robotd 的健康、大腦拿到的電量（見下面）、連線的來回時間和最長的安靜時間，還有：**時鐘差**（小鴨的時鐘減電腦的時鐘，誤差是來回時間的一半；把它加到電腦的時間上，就能跟 bridge 的事件檔和錄影對上）、**相機走連線每秒幾張**（連續拿 3 秒的 `media.frame`；`--camera-seconds` 可以改）、**mediad 在區網上的門**（8443、8080）從這台電腦敲不敲得開。接的是真的小鴨時，還會印出：小鴨跑的走路和坐站網路（sha256 和策略組版本）、大腦在模擬器裡有沒有量過它們（v5、v8 有；沒量過的走路網路，大腦照 v5 的數字開）、板子是哪一種、`robotd.toml`（`/etc/robot/robotd.toml` 上面再疊主人的 `/var/lib/robot/robotd.toml`）開了什麼、小鴨上要改什麼、duck-eye 要用哪一顆鏡頭。請把 robotd 自己的「閒著東張西望」關掉：`sudo robotctl configure`，把 `[idle]` 的 `enabled` 關掉，再重開 robotd（它會搶走頭，打哈欠會張開嘴）。

   還會印出 bridge 處理後的使用資料狀態：`usage data (Pollen telemetry): off (enabled False, active False; …)` 就是關好了。看到 `ON`（小鴨還在送）或 `UNKNOWN`（bridge 問不到 `updaterd`）：在官方控制台的抽屜裡把「share anonymous usage data」關掉，或在 `updaterd` 的環境變數加 `MICRODUCK_TELEMETRY=0`。

   **家裡 Wi-Fi 的品質**：`check remote://<小鴨位址>:8797 --wifi-minutes 10` 會接著把連線掛 10 分鐘，每分鐘和最後各印一行：來回時間 p50／p95／最長、**小鴨那一邊**最長的安靜時間（兩個心跳到 bridge 之間隔多久，用小鴨自己的時鐘量：就是它的 deadman 在看的東西）、斷線次數，以及**每小時誤觸發幾次 deadman**（電腦一直在送心跳還斷線，就是讓小鴨白白坐下一次），還有差一點就觸發的次數（安靜超過 deadman 的一半）。
6. **沒有機器人也能隨時演練**：`python -m duckmind.remotebody sim-body` 在 bridge 後面跑一隻模擬小鴨，`python -m duckmind.simweb --live --body remote://127.0.0.1:8797` 開牠（中間加 `python -m duckmind.remotebody shim` 就能加延遲、掉包、斷線：輸入 `cut`／`restore`）。

### 在真的小鴨上開頁面

小鴨上的 bridge 開好（第 4 步）、`check` 都沒問題（第 5 步）之後：

```sh
python -m duckmind.simweb --live --brain neuro --life ~/.duckmind/lives/<你的小鴨> --no-nest \
    --microduck <microduck 的資料夾> --microduck-rl <microduck_rl 的資料夾> \
    --policies <策略那一套> --robotd-bin <這台電腦用的 robotd> \
    --body remote://<小鴨位址>:8797 --body-token-file ~/.duckmind/remote-body.token
```

四個路徑跟在這台電腦上跑模擬身體時用的一樣（`python` 也要裝好 `onnxruntime`）。頁面從這裡讀小鴨的模型和策略；用 `--body remote://…` 時，這台電腦不會開 robotd，也不跑物理。少了它們，頁面一開就停，說 `--live needs --microduck, --microduck-rl, --policies and --robotd-bin`。`--no-nest` 是給現在才孵化的小鴨（新的 `--life` 資料夾）：神經鴨的腦不接有充電窩的小鴨，不加的話頁面啟動到一半會停。2026-11-09 用這一行（新的一生、加 `--port 8786`）接 `sim-body --as-real` 實際開過：頁面進入真的小鴨模式（演練），有小鴨的相機畫面和 robotd 的電量。

頁面看到 bridge 說「這是真的小鴨」（backend real），就換成**真的小鴨模式**：大腦只用小鴨自己量到、經過連線送來的東西開牠，沒有任何東西來自模擬世界——因為根本沒有模擬世界。

**小鴨要由你來開，Duck Mind 絕不替你開**（[SAFETY.md](../SAFETY.md)）。頁面啟動時如果小鴨的走路策略還沒開（robotd 回報的不是 walk、stand 或 sit），大腦就等：除了訂閱小鴨的狀態，什麼都不送；3D 畫面那一行會寫「小鴨還沒開：請用牠自己的控制把牠開起來（這個程式不會替你開）」（照頁面的語言），「我的小鴨」的「真的小鴨」卡片也寫一樣的話，`/api/body` 的 `real.turn_on` 帶著這句話。請用小鴨自己的控制把牠開起來（Pollen 自己的工具；出貨的小鴨用哪一個，**待驗**）。等 robotd 回報 walk、stand 或 sit，大腦就自己接手，頁面上不用按任何東西。終端機也會印出它在等（`live body: the real duck is not on …`）。

| | 真的小鴨模式 |
|---|---|
| 3D 畫面 | 小鴨自己的 15 個關節（`robot.state`），位置用 robotd 的接觸里程計（x、y 和高度），傾斜用 IMU、方向用里程計的方向。牠站在一片空的地板格子上：沒有房間、沒有家具、沒有平面圖。里程計會慢慢偏，所以格子上的位置是小鴨「以為」自己在的地方。畫面的標籤寫「小鴨自己的關節和里程計」，不說是物理真相。 |
| 鴨眼小窗 | 小鴨的相機，經過連線送來（mediad 的 `media.frame`，走 bridge 的 `media` 通道）。bridge 沒開相機的話，小窗會說相機沒接上連線。 |
| 狀態 | 跑馬燈上有「真的小鴨」（演練時是「演練」）、連線（來回時間、斷線次數、最久沒消息多久；連著或斷了）、電量（robotd 的 `robot.health`；robotd 還沒回答前是「還不知道」；有沒有在充電不知道）。「我的小鴨」的「真的小鴨」卡片也一樣。 |
| 可以用 | 遙控（腳）、指令列（說話、稱讚、點心、不對）、小鴨的 robotd 會做的動作和把戲、情緒、一天的時間、說話面板。大腦從 robotd 來的感覺也都在：姿勢、跌倒、被抱起來（robotd 的 `safety.picked_up`）、bridge 有開 `tofd` 時的 ToF。「裝死」只有坐著的那一種：躺下去的那一種要用 `robot.poseJoints`、`robot.init`、`robot.enable`，這三個絕不會送給真的小鴨。 |
| 不能用，而且頁面會說為什麼 | 所有模擬世界做的事：逗逗牠的情境（聲音、玩具、人、燈）、世界的手（扶牠起來、抱到地墊上）、家人來訪和換電池、實驗室的世界按鈕（放東西、滾球、收拾、示範）、示範走路、「扶正」和「重設身體」、把擺設套用到家裡、點地板叫牠過去。這些按鈕會變灰，寫著「這是真的小鴨：沒有模擬的世界，這件事做不到」（照頁面的語言），下面還有一行說明；伺服器收到這些要求也用同一句話拒絕。 |
| 還沒從機器人來 | 麥克風和觸覺（見下面）。 |

**沒有機器人也能演練真的小鴨模式**：`python -m duckmind.remotebody sim-body --as-real …` 在 bridge 後面跑模擬小鴨，但讓牠裝成真的小鴨：bridge 說 `backend: real`、不開世界通道（只有 robotd、tofd、相機）。頁面就在這台電腦上用真的小鴨模式開牠，跑馬燈寫「演練」。模擬世界還是在身體那一邊跑（相機的畫面和 ToF 是它做的），但大腦和頁面都看不到它。

**電量**在每個遠端身體上都從身體來：大腦每 2 秒透過連線問 robotd 的 `robot.health` 電量（`LiveBody.battery()`、`duckmind/remotebody/battery.py`），不用模擬世界的電池。robotd 報的是伏特（馬達的供電，平滑過）和它自己換算的百分比（6.6 V 是 0 %、帶負載 8.2 V 是 100 %），所以走路時會往下掉、休息時回來；robotd **沒有「正在充電」的旗子**（Pollen #470），真的小鴨上「有沒有在充電」不知道，大腦當成沒在充。bridge 後面的模擬小鴨（`sim-body`）會把模擬世界的電池放進它的 robotd 的 `robot.health`（帶負載的伏特、robotd 換算的百分比、有沒有在充電），所以演練走的是同一條路。一般的 `sim-body`（模擬的遠端身體，沒有 `--as-real`）頁面還是透過那個身體的模擬世界畫和感覺，跟這台電腦上的身體一樣。頁面「我的小鴨」分頁有一張「真的小鴨」卡片：說現在大腦開的是哪個身體（這台電腦裡的模擬身體、演練，或真的小鴨，加上連線的來回時間、斷線次數、最久沒消息多久），並列出上面這些步驟。

### 量產小鴨的麥克風和觸覺

robotd 0.18.5 裡**沒有任何一個程式把麥克風的聲音給別的程式**（查的是 Pollen 的原始碼，`microduck` #526）：

- `mediad` 管相機、WebRTC 和遠端連線；麥克風只出現在它的說明裡（`mediad/src/lib.rs:1-18` 列了做好的部分，影像管線只有畫面）。`docs/design/architecture.md:454` 的聲音軌是設計，還沒有程式。
- 唯一在聽麥克風的是 **robotd 自己的麥克風工作**：一個 `arecord`，16 kHz、單聲道、16 位元（`pet-detect/src/worker.rs:366-374`），給「摸頭」分類器和環境聲音偵測用。錄音裝置一次只能給一個程式用（`worker.rs:5`）。量產板上是 ALSA 的 `default`，映像檔把它接到臉部板的麥克風（`robotd-params/src/lib.rs:1449-1458`）。要 `[audio] pet_detect = true` 才會跑（預設關，`lib.rs:1426-1433`）。
- 聽到的東西不會離開 robotd：被摸頭時小鴨會咕咕叫（`robotd/src/main.rs:2795-2809`）；環境聲音只寫在除錯紀錄裡，註解說「等自主大腦來了才有人用」（`main.rs:2812-2816`）。沒有任何 IPC 方法或通知帶這些（`duck-ipc-proto/src/lib.rs` 裡沒有）。

要怎麼送過來：bridge 可以在 robotd 的麥克風工作關著時，自己在小鴨上跑 `arecord`（16 kHz 單聲道 16 位元是 256 kbit/s，比相機小很多），當成新的通道送到大腦的聽覺；或是 Pollen 把麥克風（或摸頭、聲音事件）開成 IPC。兩個今天都沒有，所以遠端身體時大腦的聽覺還是模擬世界的。

**觸覺**：量產小鴨**沒有觸覺感測器**（Pollen 的原始碼裡沒有任何觸控或電容元件）。牠能感覺到有人在碰牠的只有：（1）被抱起來——robotd 的分類器，`robot.state` 裡的 `safety.picked_up`，大腦已經在讀；（2）摸頭——上面麥克風的摸頭分類器（「幾乎就是摸頭的接觸式麥克風」，`pet-detect/src/worker.rs:52`），只在 robotd 裡面聽得到；（3）被推、腳被抓住——`robot.state` 裡關節自己的位置、目標和電流，今天已經透過連線送到大腦，但大腦還沒有把它們當成觸覺來讀。要送（2）需要跟麥克風一樣的新路。

## 第一隻真小鴨的檢查清單（都還沒驗）

- [ ] **socket 和權限**：`/run/robotd.sock` 在、權限 `0660 root:robot`；bridge 的使用者在 `robot` 群組；`hello` 有回答（0.18.x 回 API 49；大腦用 37 打招呼，robotd 不會拒絕任何版本）。
- [ ] **策略**：`robot.subscribe` 回 `walk: velstand.onnx`；`check` 會印出走路和坐站網路的 sha256 和策略組版本（照 `docs/deploy/board.md` 記下來），以及大腦有沒有量過它們（v5、v8 有；v9 也就是 v7 的走路網路沒有）。
- [ ] **robotd.toml**：`check` 會印出板子和合併後的 `[idle]`／`[pickup]`／`limp_fall`；在主人那一層把 `[idle]` 關掉（`sudo robotctl configure`）。`[pickup]` 開著時把小鴨抱起來：robotd 會暫停（`safety.picked_up`），大腦要馬上說「被抱起來」（坐著被抱起來時 robotd 不會暫停，這時靠大腦自己的感覺）。看一次跌倒後的起身：在模擬器裡，robotd 的分類器會把側躺時的起身動作當成被抱起來而暫停。
- [ ] **使用資料關掉**：`check` 印出 `usage data (Pollen telemetry): off`（`enabled False, active False`）；小鴨重開機後再 `check` 一次，還是關著。
- [ ] **真機上的 `robot.stop`**：速度歸零、不會軟掉（官方說「stop 不是 limp」）。
- [ ] **真機上的 `sit_toggle`**：從走路和站著坐下要多久、在你用的桌面上安不安全；坐著或跌倒時絕不送。
- [ ] **robotd 自己的 500 ms deadman**：走路中停止送 `robot.move`。
- [ ] **拔網路測試**：走路中把家裡 Wi-Fi 關掉（或把電腦帶出範圍）：從斷線到停下、到坐好要多久，看 bridge 的事件檔和錄影（`check` 會印小鴨的時鐘差，用來把事件檔對上電腦的時間）。
- [ ] **家裡 Wi-Fi 的品質**：`check --wifi-minutes 10`：來回時間、小鴨那邊最長的安靜時間、斷線次數，以及 1.0 秒 deadman 一小時誤觸發幾次。
- [ ] **重連**：Wi-Fi 回來 → 連線回來、大腦接回小鴨。
- [ ] **頭的設定**：從 HOME 算的頭部差值（`robot.head`）在真的頭上（板子上的服務要加 `--approve-real-v5-head-profile` 才會送；這個參數名字還留著 v5）。
- [ ] **電量**：程式好了（robotd 的 `robot.health` 電量透過連線送到大腦；用 sim-body 演練過）；真機上：`check` 會印 `battery (what the brain gets)`，再看走路時百分比會不會往下掉。
- [ ] **相機走連線**：`media.frame` 經過 bridge——一張 720p UYVY 是 1.8 MB；`check` 會印出透過連線每秒拿到幾張（抽 3 秒；bridge 後面的模擬小鴨、中間墊片 20 ± 5 ms、掉包 0.2 %：每秒 5.2 張 720p、77 Mbit/s），或要求小一點的畫質。量產的相機是直立裝的 GC2093，大約 85° × 55°：duck-eye 要加 `--board beta`（或 `face-board-v1`）。
- [ ] **麥克風、觸覺**：沒有程式把麥克風給別人，小鴨也沒有觸覺感測器（見上面「量產小鴨的麥克風和觸覺」）；今天大腦的聽覺和觸覺是模擬世界的。
- [ ] **頁面的真的小鴨模式**：3D 畫面跟著真的關節、里程計和傾斜（程式好了，用 `sim-body --as-real` 演練過）；在機器人上確認畫出來的姿勢跟小鴨一樣（坐、站、走、被抱起來），鴨眼小窗有牠的相機畫面，走幾分鐘里程計偏多少。
- [ ] **由人把牠開起來**：小鴨關著的時候開頁面：3D 畫面那一行和「真的小鴨」卡片會請你把牠開起來，小鴨不會動，bridge 的事件檔裡沒有 `refused_operator`（大腦什麼都沒要求）。用牠自己的控制把牠開起來：頁面會自己接下去。把用的是 Pollen 的哪一個控制記下來。
- [ ] **小鴨上的其他門**：官方 `mediad` 聽在 `0.0.0.0:8443`／`8080`、沒有認證，區網上的人都能開；藍牙配對（`btd`）預設關著。`check` 會說 8443／8080 從電腦敲不敲得開；`docs/deploy/board.md` 第 5 節有 bridge 使用時把它們關起來的 nftables 設定（只在記憶體裡；預設不會改小鴨上的任何東西）。
- [ ] **小鴨板子的 CPU**：在電腦上量過（Apple M5 Pro，一顆核心的百分比；`tools/measure_bridge_cpu.py`：bridge 自己一個程序，假的 robotd／tofd／mediad 送照 robotd 0.18.5 格式、每行 5.4 kB 的 `robot.state`，tofd 的 8×8 畫面，mediad 的 720p UYVY 畫面）：只有連線 0.4 %，加上 50 Hz 的 `robot.state` 0.6 %，再加 15 Hz 的 ToF 0.6 %，再加相機每秒 2 張 1.2 %、每秒 5 張 2.2 %；不限速時每秒轉了 205 張（3 Gbit/s）用掉 60 %，一張 1.8 MB 的畫面大約 3 ms CPU（其中 0.6 ms 是簽章）。板子上的 RK3566（四顆 1.8 GHz 的 Cortex-A55）**是估的**、沒量過：這種 Python 工作每顆核心慢 15–30 倍（依公開的單核跑分，循序執行的 A55 對 Apple 的效能核心）：狀態加 ToF 大約一顆核心的 10–20 %（整塊板子的 3–5 %），相機每秒多一張再多一顆核心的 5–9 %。相機真正的限制是 Wi-Fi：沒壓縮的 720p UYVY 每秒 2 張就是 29.5 Mbit/s、5 張是 74 Mbit/s。到小鴨上：在小鴨上跑 `tools/measure_bridge_cpu.py`（只要裝了 Duck Mind 就能跑），大腦在開的時候用 `top` 看 bridge。
- [ ] **時鐘**：大腦用 robotd 自己的 `t_ns`（不需要對時）；`check` 會印小鴨的時鐘跟電腦差多少（誤差是來回時間的一半）。
