# 接上真的 Microduck（遠端身體）

[English](real-microduck.md)

Duck Mind 的大腦要住在家裡的電腦，身體是那隻小機器鴨。機器人自己的程式 `robotd` 留在小鴨身上；小鴨上再跑一個小小的 **bridge**，讓家裡網路上的**一個**大腦連得到它。這一頁說明：哪些已經好了、怎麼接小鴨、哪些還要在真的機器人上確認。

> **現況（2026-11-04）。** 連線、bridge 和安全停車都做好了，用第二個程序裡的**模擬**小鴨，隔著一個會延遲、抖動、掉包、斷線的網路墊片演練過。還沒接過真的 Microduck。下面標「**待驗**」的步驟，都是照官方軟體推出來的，要等有人在機器人上跑過才算數。

## 整體怎麼接

```text
家裡電腦                                         小鴨（Radxa Zero 3）
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
| 聽在哪裡 | 預設只聽本機；你要給小鴨的區網位址（或 `0.0.0.0`）。兩端都拒絕不在本機、私有、link-local 範圍的位址。 |
| 誰能開 | 證明自己有共用密鑰的那**一個**大腦（雙方互相出題的 HMAC；密鑰從不上網路）。另一個有密鑰的大腦連進來會接手。 |
| 防竄改 | 握手之後每一框都有序號和簽章（每次連線自己的金鑰，HMAC-SHA256）；壞掉或順序不對的框會斷線並停車。連線**沒有加密**——只在區網用。 |
| 大腦可以送什麼 | robotd：`hello`、`robot.subscribe`、`move`、`head`、`mouth`、`pose`、`look`、`do`、`stop`、`health`、`model`、`modelApi`、`policies`、`safeToRestart`，以及操作員的 `enable`、`init`、`poseJoints`。其他的（`robot.relax`、`loadPolicy`、`shutdown`、`setMode`、`rebootMotors`、`system.*`、`net.*`…）一律回錯誤，不會到 robotd。 |
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

1. **先用 Pollen 自己的工具把小鴨設好**：Wi-Fi（在電腦上 `duckctl wifi connect`，或在小鴨上 `sudo robotctl net connect <ssid> --psk <密碼>`），再用 `duckctl ip` 找位址、在小鴨上跑 `robotctl health`。板子本身照官方的開發板說明。
2. **在小鴨上裝 Duck Mind**，照 [docs/deploy/board.md](deploy/board.md)（venv 裡裝 wheel；Python ≥ 3.12）。bridge 只需要標準函式庫和 Duck Mind 本身。
3. **在電腦上做共用密鑰**，複製到小鴨（兩份都要權限 600）：

   ```sh
   python -m duckmind.remotebody token ~/.duckmind/remote-body.token
   duckctl scp ~/.duckmind/remote-body.token :.duckmind/remote-body.token
   ```

   （`duckctl scp` 自己會找小鴨；開頭的 `:` 是小鴨上的路徑，在 `--user` 的家目錄，沒給就用 `DUCK_BOARD_USER`，再沒有就是 `radxa`。資料夾先用 `duckctl ssh` 建好。）
4. **在小鴨上啟動 bridge**，用 `robot` 群組裡的使用者（這樣才打得開 `/run/robotd.sock`）：

   ```sh
   python -m duckmind.remotebody bridge --listen 0.0.0.0:8797 \
       --token-file ~/.duckmind/remote-body.token --backend real \
       --robotd /run/robotd.sock --tofd /run/tofd/tof.sock --media /run/mediad/media.sock \
       --events ~/remote-body-events.jsonl
   ```

5. **在電腦上檢查連線**——只讀 robotd 的健康和策略，什麼都不會動：

   ```sh
   python -m duckmind.remotebody check remote://<小鴨位址>:8797
   ```

   會印出有哪些通道、deadman 多久、robotd 的健康（電量），以及連線的來回時間和最長的安靜時間。
6. **沒有機器人也能隨時演練**：`python -m duckmind.remotebody sim-body` 在 bridge 後面跑一隻模擬小鴨，`python -m duckmind.simweb --live --body remote://127.0.0.1:8797` 開牠（中間加 `python -m duckmind.remotebody shim` 就能加延遲、掉包、斷線：輸入 `cut`／`restore`）。

模擬器頁面目前只開得動**模擬的**遠端身體：牠的感官（相機、聽覺、觸覺、電量）和 3D 畫面還是從模擬世界來的。真的小鴨要先讓這些感官從機器人來（見下面清單）。

## 第一隻真小鴨的檢查清單（都還沒驗）

- [ ] **socket 和權限**：`/run/robotd.sock` 在、權限 `0660 root:robot`；bridge 的使用者在 `robot` 群組；`hello` 回 API 37。
- [ ] **策略**：`robot.subscribe` 回 `walk: velstand.onnx`；照 `docs/deploy/board.md` 記下策略的雜湊。
- [ ] **真機上的 `robot.stop`**：速度歸零、不會軟掉（官方說「stop 不是 limp」）。
- [ ] **真機上的 `sit_toggle`**：從走路和站著坐下要多久、在你用的桌面上安不安全；坐著或跌倒時絕不送。
- [ ] **robotd 自己的 500 ms deadman**：走路中停止送 `robot.move`。
- [ ] **拔網路測試**：走路中把家裡 Wi-Fi 關掉（或把電腦帶出範圍）：從斷線到停下、到坐好要多久，看 bridge 的事件檔和錄影。
- [ ] **家裡 Wi-Fi 的品質**：用 `check` 看 10 分鐘的來回時間、最長安靜時間、斷線次數；1.0 秒 deadman 一小時誤觸發幾次。
- [ ] **重連**：Wi-Fi 回來 → 連線回來、大腦接回小鴨。
- [ ] **頭的設定**：v5 的頭部差值（`robot.head`）在真的頭上（板子上的服務要加 `--approve-real-v5-head-profile` 才會送）。
- [ ] **電量**：`robot.health.battery` 要送到大腦（現在大腦的電量是模擬世界的）。
- [ ] **相機走連線**：`media.frame` 經過 bridge——一張 720p UYVY 是 1.8 MB；量 Wi-Fi 上每秒幾張，或要求小一點的畫質。
- [ ] **麥克風、觸覺**：還沒有從機器人來的路（現在是模擬世界的模型）。
- [ ] **頁面的 3D 畫面**：改用 `robot.state` 的關節畫，不用模擬器的真相。
- [ ] **小鴨上的其他門**：官方 `mediad` 聽在 `0.0.0.0:8443`／`8080`、沒有認證，區網上的人都能開；藍牙配對（`btd`）預設關著。決定 bridge 使用時要不要用防火牆擋起來。
- [ ] **Radxa Zero 3 的 CPU**：bridge 在 50 Hz 狀態加上 ToF 時的負載。
- [ ] **時鐘**：大腦用 robotd 自己的 `t_ns`（不需要對時）；`check` 會報來回時間，連線也會算時鐘差。
