# 小鴨模擬器設計規範（頁面風格標準）

> 2026-09-26 起生效。模擬器的頁面風格受 Pollen Robotics 的 Microduck 產品頁（pollen-robotics.com/microduck）啟發，**以後新增的頁面功能也以這份為標準**。duckmind 是非官方專案，與 Pollen Robotics 無關，這不是 Pollen 的設計。
> 實作在 `src/duckmind/simweb/static/style.css`（檔頭註明本檔）。新增或修改頁面前先讀這份。

## 1. 一句話

深色電視雜訊舞台 ＋ 米白點點紙面；**直角、2–3 px 墨黑粗框、往右下錯開的實心陰影**；黃色大標題帶青／洋紅錯位殘影；Anton 顯示字（中文用 Noto Sans TC 900）；橘色主按鈕加洋紅陰影；斜的黃色跑馬燈。

只借用那個產品頁的**視覺語言**（顏色、字型、形狀、節奏），不複製它的商標、圖片、標語或文字。

## 2. 設計標記（tokens，全部定義在 `:root`）

| 用途 | 變數 | 值 |
|---|---|---|
| 墨黑（字、框、深色區） | `--ink` / `--ink-deep` / `--ink-2` | `#101018` / `#08080c` / `#1c1c28` |
| 紙（淺色區底） | `--paper` / `--cream` / `--white` | `#faf8f2` / `#f2ecdd` / `#ffffff` |
| 黃（標題、跑馬燈、選中） | `--yellow` | `#ffd23f` |
| 橘（主按鈕、小標題） | `--orange` / `--orange-ink` | `#ff7a2f` / `#c2500a`（淺底上的橘字用 `--orange-ink`） |
| 洋紅（強調、錯位殘影） | `--magenta` | `#ff2fa8` |
| 青（錯位殘影、資訊） | `--cyan` | `#2ff0e6` |
| 輔助色 | `--blue` / `--purple` / `--green` | `#3b6fd8` / `#8a5cf5` / `#1f9d57` |
| 次要字 | `--muted` | `#5e5e68`（深底上用 `--on-dark-2` / `--on-dark-3`） |
| 細線 | `--hair` | `rgba(16,16,24,.16)` |
| 即時物理：物理真相／小鴨的計畫 | `--truth` / `--plan` | `var(--green)` / `var(--blue)`（地上的真相線、影子鴨與誤差圈；duck3d.js 讀這兩個） |

舊名稱（`--teal --reed --lilac --rose --duck --orange-soft …`）**保留**，因為 app.js 用它們畫量表、標籤和記憶圖；值已對到新色盤。**不要新增色盤以外的顏色**；需要新語意色時先加 token 再用。

## 3. 字型

- 顯示字：`--font-display` = Anton ＋ Noto Sans TC（900）。一律 `font-synthesis: none`（Anton 只有 400，不准假粗）、`text-transform: uppercase`。
- 內文：`--font-body` = DM Sans ＋ Noto Sans TC。數字、時間、事件 ID：`--font-mono` = JetBrains Mono，`tabular-nums`。
- 小標題 `.eyebrow`：12 px、700、字距 .2em、大寫、`--orange-ink`（深底用 `.on-dark` → `--orange`）。
- 區塊大標 `.display`：墨黑字＋5 px 黃色錯開陰影；第二行 `.outline` 是空心字（`-webkit-text-stroke` ＋ `paint-order: stroke fill`，避免中文筆畫重疊處露出內框）。深色區的大標用白字＋青／洋紅左右錯位，第二行 `.pink`。
- 小鴨名字 `h1`：黃字＋6 px 墨黑陰影＋青／洋紅錯位（和產品頁主標同一種手法）。

## 4. 元件（照抄這些 class，不要自己發明新樣式）

| 元件 | class | 規則 |
|---|---|---|
| 按鈕 | `.btn` | 白底、2 px 墨框、直角、`3px 3px 0` 墨黑陰影；hover 變黃；按下時位移到陰影位置（陰影歸零） |
| 主按鈕 | `.btn.primary` | 橘底、墨字、`4px 4px 0` 洋紅陰影；一個畫面最多一到兩個 |
| 次要 | `.btn.accent`（黃）、`.btn.ghost`（無陰影）、`.btn.small` | 深底上的 small／ghost 用白框白字（見 `.topbar .btn.small`、`.duck3d-live .btn.ghost`） |
| 卡片 | `.card` ＋ `.accent-yellow/orange/magenta/cyan/blue/purple` | 白底、2.5 px 墨框、直角、`8px 8px 0` 彩色陰影（`--accent`）；標題 `.card-title`，編號 `.num`（Anton、用 `--accent-ink`） |
| 區塊 | `.section.paper` / `.section.dark` | 淺色區：米白＋點點格；深色區：墨黑＋放射光芒＋白點，上下斜切（`clip-path`） |
| 跑馬燈 | `.ticker` > `.ticker-inner` > `.ticker-track` | 黃底、上下 3 px 墨框、旋轉 −1.2°；內容由 app.js 複製一份做無縫循環；滑過暫停；減少動態時不動 |
| 狀態列 | `.chip`（放在跑馬燈裡） | 前面一個方塊色標：成長日橘、階段青、醒著綠、小睡紫、關機灰、警示洋紅 |
| 狀態卡 | `.duck-banner`（`data-tone`） | 白卡、墨框、9 px 陰影色依語氣：info 青、warn 橘、sleep／night 紫、rest 黃、off 灰；頂端時間列暫停時黃底、連不上伺服器時洋紅底（`data-state="offline"`） |
| 分頁 | `.tabs` | 一整條墨框分段；選中黃底 |
| 輸入 | `select` `input` | 2 px 墨框、直角；focus 時 `3px 3px 0` 橘色陰影 |
| 開關 | `.toggle` | 墨框方塊；勾選時黃底 |
| 人物選擇 | `.persona` | 大字方塊；選中墨底黃字、洋紅陰影 |
| 舞台 | `.stage-frame` > `.stage-area`（`#duck3d` 或 2D `#duckStage`）＋下方資訊列 | 白框＋黃色偏移陰影；時鐘、字幕、⛶ 放大放在 `.stage-area`（不放進 `#duck3d`），讀數／位置／叫聲／物理列放在它下面——這樣沒有 3D 模型時（2D 小鴨）也都看得到、按得到 |
| 叫聲列 | `.duck3d-voice`（`#callMute` `#callVolume` `#callNote`，voice.js） | 跟位置列同一種墨黑列：靜音鈕是 `.btn.ghost.small`，靜音時洋紅底；音量滑桿是直角墨色軌道＋黃色方塊把手、洋紅實心陰影；提示（先點一下頁面、聲音庫準備中、沒有叫聲程式）用黃字，沒事時藏起來。觸控時靜音鈕與滑桿 44 px 高 |
| 畫面工具 | `.stage-tools` > `.expand-btn`（`#btnCloseup`「近看小鴨」、`#btnExpand`「⛶ 放大」） | 右上一排、手機上下疊（2026-09-28 fid:page）；按下的（`aria-pressed="true"`）黃底＋洋紅陰影；手機每顆 44 px 高，鏡頭小窗往下讓，時鐘不被擋 |
| 舞台下快捷列 | `.stage-stim`（`#neuroQuick` 逗逗牠、`#neuroSound` 聲音情境，神經鴨原型） | 舞台正下方的墨黑列，跟位置列同一種樣子：黃色 Anton 標籤、`.btn.small` 橘色偏移陰影、按下後黃底一下；在 `[data-scenarios]` 容器裡；手機每顆 44 px 高；滑桿同叫聲滑桿（直角軌道、方塊把手） |
| 身體真相 | `.neuro-body`、`.neuro-reflexes .rx-body[data-verdict]` | 「身體真的在做」：米白底、綠色（`--truth`）左邊線，和腦說的不一樣時改洋紅左邊線＋洋紅陰影；反射列下面一行量到的身體動作：做到綠方塊、只做到一部分橘方塊、做不到洋紅方塊＋整列洋紅左邊線 |
| 3D 覆蓋 | `.duck3d-clock` `.duck3d-caption` `.duck3d-bubble` `.expand-btn` | 時鐘墨底黃字洋紅陰影（第三行 `i` 青字＝一天的階段；連不上伺服器時第二行洋紅）；字幕白底墨框青陰影，領頭行 `.lead` 粗體＝牠在做的事，機智／呆萌時刻整塊改洋紅陰影（`.moment`）；泡泡黃底（跟著小鴨頭，留在 `#duck3d` 裡）；資訊列墨黑 |
| 即時物理覆蓋 | `.stage-mode`（**2026-10-11 起不畫在畫面上**，元素與 id 保留、`display: none`；它的兩行字由 shell.js 放進 HUD 的提示文字，左欄讓給平面圖）`.stage-side` > `.trust-badge` `.eye-window` | 沿用字幕的樣子（白底、2 px 墨框、4 px 實心陰影）。時鐘下方的 `.stage-mode` 說畫的是哪一隻（`.chip.truth` 綠「物理真相」／`.chip.plan` 藍「這是小鴨的計畫，不是物理」，陰影同色；手機只留標題）；右上 `.stage-side` 一欄：`.trust-badge`（`<details>`，`.chip.mode-warn`「物理不可信」＋原因，洋紅陰影）與 `.eye-window`（鴨眼小窗，青陰影，`.chip.light-*` 光等級）。都放在 `.stage-area`，所以 2D 小鴨也看得到（2D 時標籤說「這是小鴨的計畫，不是物理」，放在 2D 徽章下方、不壓到小鴨）；右欄不超出畫面（畫面縮小） |
| 平面圖 | `.minimap`（`#miniMap` > `.mm-head`〔`.mm-title` `#miniMapFold`〕＋ `#miniMapBody` > `#miniMapSvg`，minimap.js） | 舞台**左邊緣**的小地圖（2026-10-11，主人要求靠左，不往中間、不貼控制面板）：在左欄「時鐘」下面、「現在在做」（HUD）上面。薄紙面（`--paper` 九成不透明）、2 px 墨框、4 px 墨黑實心陰影、直角；標題列墨底，收起鈕白底方塊。牆墨黑、窗青、傢俱用家的資料色；小鴨是橘點＋洋紅外圈＋墨黑朝向短線，家人是洋紅點。收起狀態每位檢視者自己記（localStorage），點地圖暫時放大 12 秒（往右長，最多到畫面中線）。minimap.js 量周圍的卡片：往上抬到 HUD 與小鴨的話上面（`--mm-lift`）、只用上方卡片與下方卡片之間的空間（`--mm-fit`），放大模式面板打開時寬度只用面板左邊的空間；沒空間時縮成標題列、再縮成只剩圖示，仍然在左邊；手機把面板拉起來、放大模式手機打開面板時不畫。沒有物理身體或沒有家的資料時藏起來。頂視圖的畫法與版面編輯器共用 `plan2d.js` |
| 成長紀錄 | `.growth-view`（`#growthView`，growth.js / growth.css） | 資料頁第一個區塊（2026-10-11）：一個元件畫「一隻鴨子怎麼長大」——`.gv-card`（`.card` ＋ `.accent-*`，6 px 彩色實心陰影）排成自動換欄的格子：身分（墨底黃字的成長日大數字、`.chip` 階段與分支、階段列表，目前的一行黃底墨框）、天性、學到什麼、親近度、經歷（`.gv-tally` 小格）、夜裡長大、叫聲（`.btn.small` 播放）、日記。量表用 `.bar`（`.gv-meter` 三欄：名稱、條、數字）。下面的「快轉測試」是 `.card.accent-orange`：數字／下拉輸入、橘色主按鈕「開始快轉」、`.bar[role=progressbar]` 進度；結果同一個元件兩欄並排（1100 px 以下上下排），上面一張 `.gv-compare` 比較表，兩種照顧不一樣的列黃底。只用 token，沒有新顏色、圓角、柔和陰影 |
| 回饋與訓練 | `.cmd-feedback`（`#cmdFeedback`，指令列）、`.trick-book`（`#trickBook`，「訓練」分頁）＋ `.trick-feedback` `.trick-row` `.trick-note`（tricks.js） | 2026-10-26：指令列的回饋排是指令鍵同款 `.btn.small`（橘色偏移陰影），稱讚是 `.primary`（洋紅陰影），答案行同 `.cmd-out`（左邊線：記住了綠、沒算橘、不對／失敗洋紅）。把戲本是 `.card.accent-cyan`：每列白底、2 px 墨框、3 px 墨黑實心陰影，「下一個練這個」那列改洋紅陰影，標記墨底黃字；熟練度用 `.bar`（學會了綠、其他橘），狀態用 `.chip`（方塊色：學了一半橘、學會了綠、還沒開放灰）。手機每顆按鈕 44 px。只用 token |
| 提示 | `.toast` | 墨底、左側 8 px 黃條、洋紅陰影 |
| 對話框 | `dialog` | 3 px 墨框、直角、`12px 12px 0` 黃陰影；標題 `.dialog-title` |

**不要**：圓角、柔和的模糊陰影、漸層按鈕、半透明卡片（3D 覆蓋除外）、細灰框。

## 5. 版面（2026-10-10 重做，詳見 `docs/design/UI-REBUILD-2026-10-10.md`）

原則：**看小鴨和按按鈕要能同時做**，放大時也一樣。

1. **頂欄**（墨黑，桌面黏在最上面）：品牌、小鴨名字和個性（一行）、世界時鐘、（Duck Core 模式）時間控制、模擬／內心／資料。
2. **應用區 `.hero.app`**（正好一個螢幕高）：左邊舞台（`.stage-col`：3D 小鴨、HUD、逗逗牠快捷列＋叫聲列），右邊**控制欄** `#ctlPanel`（分頁：逗逗牠、動作、情緒、訓練、說話、換鴨子、腦、物理；Duck Core 模式：狀態、互動、工具），自己捲動。放大時整個 `#duckView` 全螢幕，控制欄變右下角的半透明浮動面板；手機（< 1024 px）控制欄是舞台下方可拉起的面板。
3. **跑馬燈**（黃）→ **內心 `#mind`**（深色）→ 頁尾。
4. **資料頁 `/data`**（同一份頁面的另一個檢視）：成長紀錄與快轉測試（最上面）、九種本領、家的擺設、技能測試證據。

新功能放進**意義相符**的分頁（或資料頁）；新的控制項放在 `[data-scenarios]` 容器裡，放進控制欄的卡片要用 `.card` 與 `.accent-*`，不另外發明樣式。控制欄裡的卡片不編號（編號只留給 Duck Core 的 01–09）。

## 6. 程式掛鉤（改樣式不能弄壞功能）

- app.js／duck3d.js 只靠 **id**、`data-*` 屬性和少數 class（`.persona` `.tabs button` `.tab` `.chip` `.duck-stage` 的狀態 class）運作。改版面時保留它們。
- 情境按鈕（`data-act`、`data-party`、`#btnLive`）和設定勾選框用事件委派：放在任何一個 **`[data-scenarios]`** 容器裡都會動。
- app.js 會用舊色名 `var(--reed)` `var(--orange)` `var(--rose)` 等上色：新增色彩一律走 token。
- 跑馬燈 `#tickerTrack` 的內容由 app.js 複製一份（無 id）做循環：狀態 chip 只更新原本那一份。
- 手機（觸控）上每個控制項至少 40–44 px 高；不要把按鈕藏在沒有捲軸的橫向捲動列裡（改成換行）。

## 7. 響應式與無障礙

- 1180 px 以下舞台改成上下排；980 px 以下互動、內心改單欄；760 px 以下是手機版（頂欄不黏、時間控制可橫向滑、3D 高度依螢幕）。
- 任何寬度都**不能左右捲動**（`main { overflow-x: clip }`；傾斜的跑馬燈和硬陰影會超出邊界）。
- 焦點框：3 px 橘色 outline。對比：深底用白字，淺底用墨字。`prefers-reduced-motion` 時雜訊、跑馬燈、卡片傾斜全部停。
- 單一視覺（不跟著系統深色模式切換）；`html` 底色是墨黑，捲動到邊界不露白。

## 8. 新增頁面功能的檢查清單

1. 用上面的 class，不寫新的顏色或圓角。
2. 放在正確的區塊、正確的卡片，編號接續。
3. 在 3D 與 2D、**1440×900 與 390×844** 各檢查一次（用瀏覽器或 Playwright 之類的工具）：所有控制項有反應、沒有 console 錯誤、沒有左右捲動。
4. 3D 覆蓋層不要擋住小鴨；新的覆蓋層沿用 `.duck3d-caption` 的樣式。
5. 頁面文字六種語言一起出（繁體中文是來源，另有 English、日本語、Español、简体中文、한국어），白話；走 i18n 目錄，見 `docs/i18n.md`。
