# ADR-0001：以單一目標來源協調寵物鴨行為

**狀態：PROPOSED／未實作**
**日期：2026-09-30**
**相關設計：** `docs/design/pet-behavior-v1/ARCHITECTURE.md`（研究文件，尚未公開）

## 背景

目標是讓寵物鴨具備快速、連續、帶個性的感知與互動，同時維持可觀測的安全邊界與可成長但可追溯的記憶。現有 Neuro selector 是唯一目標選擇來源，但 Play、Sound、IntegratedMotor 等位置各自處理技能合併與 gating。Neuro prototype 的 motor-shaped `MotorCommand` 與 main repo 正式 `BodyPort` 契約尚未收斂。主 repo 的事件／outbox 基礎也尚未接到 Neuro learner。

此外，上游沒有 robotd request ID／exclusive lease；one-shot bridge 依 policy 狀態推斷本地執行生命週期，已啟動的不可取消 policy 不能因反射或 `robot.stop` 就保證立即停止。故命令 ACK 不能代表身體完成，未知結果不可當成功。

## 決策提案

在既有 Neuro selector 輸出與技能／IntegratedMotor 之間，逐步收斂一個**行為協調權責**：管理 semantic Intent、技能 lifecycle、資源 ownership、能力檢查、中斷類別與 outcome evidence。它不另建 selector 或第二個主腦。慢速 LLM／情境 proposal 先通過 grounding 與新鮮度檢查，再交由既有 selector 決定是否成為目標；協調權責不直接選擇 LLM 目標。本機 reflex 一旦觸發即送入同一資源仲裁，不等待慢評估或 selector tick 的其他工作；同時到達時按 event sequence、reflex priority、resource ownership 的固定順序合併，已派發動作仍受其取消能力限制。

快速本機 reflex 和慢速 appraisal／可選 System 2 使用不同時程，但回到同一仲裁權責；反射不等待 LLM、網路或記憶 I/O。技能將語意 Intent 編譯為受能力 profile 限制的身體命令，經 motor／embodied adapter、robotd 到身體。腦與 LLM 不輸出關節或 servo 命令。取消能力依 `interruptible`、`finish-or-unwind`、`unknown` 分類；不對後兩者承諾立即搶占，timeout 不自動重播 one-shot。`hold_settle` 驗證該技能目標和所持資源，不要求整個身體停止；頭部追蹤完成時，腿可由另一個合法 owner 繼續行走。

所有結果區分命令接受、執行中、觀察完成與 unknown。意圖目標需有 target ID、信心與有效期限；完成與學習需連到可追溯證據。模擬器可由 world truth 產生有限、帶噪音的 observations，並只經正常感測通道進入決策；線上決策不直接讀 oracle truth。人格以穩定基調、具證據的慢記憶、短暫有界狀態組合；記憶不直接改 motor 權重。持久 execution ledger 保留 restart 前 pending／dispatched 紀錄；啟動時 reconciliation 將未決項標為 unknown／quarantine，確認身體現況後再規劃，不恢復或重播命令。現有兩個 repo 的契約仍分開，本文不宣稱 Neuro 已接上 main repo BodyPort 或 event store。

## 考慮過的替代方案

| 選項 | 為何未採為方向 |
|---|---|
| 只調整反應頻率／gain | 可能改善單一指標，無法解決多技能資源衝突、完成證據、跨段 lifecycle 與個性持久化；也可能破壞已驗安全包絡。 |
| 固定劇本 | 可穩定呈現指定段落，但遇到不同目標、姿勢或中斷時難以即時重規劃，不能代表自主能力。 |
| 讓 LLM 逐步控制身體 | 延遲與可用性不可預期，擴大安全與執行追蹤問題，且違反本專案不讓認知直控 servo 的邊界。 |
| 新增第二個完整行為腦 | 會與 Neuro selector 爭奪目標，造成重複仲裁與人格／狀態分裂。 |
| **單一 selector + 漸進協調權責（本提案）** | 保留現有目標權威與技能，讓 lifecycle、ownership、能力與結果證據逐步有單一責任點；代價是需做 adapter 收斂與分階段驗證。 |

## 後果與代價

**預期好處：** 技能完成可由觀測判定；reflex 不被慢服務阻塞；多資源疊加和降級規則可檢查；記憶的改變可追溯；硬體差異可透過 profile 處理。

**成本／限制：** 要補 trace 時鐘契約、execution correlation、能力與取消宣告、持久化 migration 和真實硬體測試。robotd 缺少 request ID／exclusive lease 前，部分執行歸屬仍只能保守標為推斷或 unknown；one-shot 的安全處置不等於立即停止或 torque-off。150 ms 僅是快速本機 reflex 的 sensor receipt→第一個適用 dispatch 設計目標，不套用社交停頓或所有技能，也不是實機承諾。

## 漸進導入與回復

先新增不影響輸出的 trace 與技能宣告，再以影子協調比較既有 merge；之後只接一項低風險、可取消且可觀測的技能。每階段保留舊路徑與版本化切換設定；安全延遲、unknown 比例或身體穩定性退化時，停用新輸出並回到已驗路徑。觸覺、one-shot、持久記憶及 System 2 分別評估，不一次全接。

本 ADR 本輪只記錄提案方向，狀態維持 PROPOSED／未實作。未來工作依當時任務範圍與授權安排；實作前應把契約相容性、硬體取消能力、資料政策、基線與驗收門檻納入設計及驗證，不要求每個例行導入步驟另行等待批准。
