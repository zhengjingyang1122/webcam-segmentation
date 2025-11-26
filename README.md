# Webcam Snapper - Modular Edition

以 PySide6 建構的桌面相機工具，提供相機裝置選擇、單張拍照、連拍、錄影及基於 目標分割（Segment Anything，簡稱 SAM）的影像自動分割與透明背景輸出。本專案採用模組化架構（將各功能拆分成類別與檔案），方便後續擴充與維護。

---

## 目錄

- [Webcam Snapper - Modular Edition](#webcam-snapper---modular-edition)
  - [目錄](#目錄)
  - [特色功能](#特色功能)
  - [畫面概覽](#畫面概覽)
  - [安裝](#安裝)
    - [系統需求](#系統需求)
    - [依賴套件](#依賴套件)
    - [安裝步驟](#安裝步驟)
  - [執行方式](#執行方式)
  - [使用流程](#使用流程)
  - [SAM 自動分割詳解](#sam自動分割詳解)
  - [鍵盤快捷鍵](#鍵盤快捷鍵)
  - [日誌與 UI 提示](#日誌與ui提示)
  - [檔案結構](#檔案結構)
  - [開發與擴充](#開發與擴充)
  - [常見問題 FAQ](#常見問題faq)
  - [授權 License](#授權license)
  - [貢獻與提交訊息](#貢獻與提交訊息)

---

## 特色功能

- **即時相機預覽與裝置選擇**：以 QtMultimedia 建立相機 session，將影像輸出至 `QVideoWidget`，並搭配 `QImageCapture` 拍照與 `QMediaRecorder` 錄影。預設嘗試 MPEG‑4/H.264 視訊與 AAC 音訊。
- **單張拍照與具重試機制的安全寫檔**：當相機尚未就緒時會以 `QTimer` 進行延遲重試，失敗會寫入日誌。檔名使用時間戳自動生成。
- **連拍（Burst）**：可設定張數與間隔毫秒，首張立即拍攝後以計時器節奏進行，完成時寫入結束資訊。
- **錄影**：支援開始或繼續、暫停、停止，自動給定輸出路徑與檔名。錯誤會記錄並通知。
- **檔案瀏覽器（Media Explorer）**：左側 Dock 可收放或浮動，顯示圖片與影片，支援重新命名與刪除，右鍵選單可對圖片觸發自動分割。另提供取得最近一張影像或最近一段影片的 API。
- **影像自動分割與透明輸出**：以 目標分割（SAM）模型產生多個遮罩候選，支援滑鼠游標指向預覽、左鍵加入選取、右鍵移除，可個別輸出或取聯集，並以 最小外接矩形 或 原圖尺寸 輸出 PNG 含透明度通道。聯集輸出附形態學開閉運算與最大連通區篩選。
- **模型快取與釋放**：自動分割支援遮罩快取與影像 embedding 快取，下次開啟同圖時可快速載入；模型可卸載釋放 GPU 記憶體。
- **統一狀態列與科幻進度彈窗**：底部狀態列顯示訊息與進度，並提供霓虹風彈窗與模擬進度，另可顯示影像尺寸與滑鼠座標。
- **鍵盤快捷鍵與客製化**：內建 Space 拍照、R 錄影開始或繼續、Shift+R 停止、F9 切換 Dock 等。支援從 JSON 讀取自訂快捷鍵，亦可於 UI 顯示快捷鍵一覽。
- **首次啟動導覽與說明選單**：提供快速導覽精靈，說明選單包含 快速導覽、鍵盤快捷鍵、開啟日誌資料夾等。
- **統一日誌 (logging) 與 UI 提示**：含 PII 去識別化過濾器，JSON 及文字檔案輪替，將 Python 與 Qt 訊息導入日誌。UI 會在錯誤時顯示狀態與節流後的彈窗。

---

## 畫面概覽

- **右側控制區**：輸出路徑、相機裝置與啟停、拍照與連拍、錄影、模型預載與自動分割等控制項，中間為相機預覽。
- **左側 Dock**：檔案瀏覽器，可顯示與管理輸出資料夾中的媒體，支援刪除、重新命名與啟動自動分割。

---

## 安裝

### 系統需求

- Python 3.9 至 3.12（建議 3.11）。
- Windows 10/11（其他平台可自行嘗試）。
- 若要使用 GPU 推論，請安裝對應的 CUDA 版 PyTorch。

### 依賴套件

- PySide6
- opencv-python
- numpy
- torch
- segment-anything

### 安裝步驟

```bash
# 建議使用虛擬環境
python -m venv .venv
.\.venv\Scripts\activate

pip install --upgrade pip
pip install PySide6 opencv-python numpy torch
# 安裝 Segment Anything
pip install git+https://github.com/facebookresearch/segment-anything.git
```

> 若不使用自動分割功能，可不安裝 torch 與 segment-anything。

---

## 執行方式

```bash
python main.py
```

第一次啟動會出現快速導覽精靈與裝置掃描，下次可從選單「說明 → 快速導覽」再次開啟。

---

## 使用流程

1. **指定輸出資料夾**：於右側「輸出路徑」區塊輸入或點選「瀏覽」，偏好會自動保存，檔名使用時間戳自動生成。
2. **選擇相機並啟動**：在「相機設備」下拉選單中選擇後按「啟動相機」，預覽即時顯示。
3. **拍照**：點「拍一張」或按 Space 鍵，檔案會出現在輸出資料夾，左側瀏覽器可立即查看。
4. **連拍**：設定張數與間隔毫秒後點「開始連拍」，可隨時停止。
5. **錄影**：點「開始或繼續」開始錄影，可暫停或停止後自動儲存。錄影格式嘗試 MPEG‑4/H.264/AAC。
6. **管理檔案**：左側瀏覽器支援刪除與重新命名，亦可右鍵啟動自動分割。
7. **影像自動分割**：於右側點「自動分割」開啟選單，可選擇單一影像或資料夾批次。首次使用需載入 SAM 權重；檢視器中以滑鼠選取欲輸出目標，再按「儲存已選目標」。
8. **模型預載與下載**：勾選「預先載入 SAM 模型」可互動式載入模型，預設檔案路徑為 `models/sam_vit_h_4b8939.pth`。若不存在，會詢問是否下載約 2.5 GB 權重並顯示百分比進度。

---

## SAM 自動分割詳解

- **引擎 `SamEngine`**：以 `sam_model_registry` 建立模型並移動至 cuda 或 cpu，提供影像與影片第一幀的自動遮罩。回傳遮罩與分數。新版 `SamEngine` 移除了未使用的暫存屬性，結構更加清晰。
- **快取機制**：為每張影像寫入 `.sam_masks.npz` 與 `.sam_embed.npz` 快取，後續開啟可快速載入遮罩與 embedding。
- **檢視器 `SegmentationViewer`**：
  - **互動**：滑鼠移動顯示 hover 遮罩，左鍵加入選取，右鍵移除。狀態列同步顯示游標座標與影像尺寸。
  - **儲存**：可個別輸出或聯集輸出。聯集輸出附開閉運算與最大連通區過濾，並支援最小外接矩形或原圖大小輸出含透明度 PNG。
  - **熱鍵**：支援上一張與下一張、儲存快捷鍵以及 F9 切換 Dock。

---

## 鍵盤快捷鍵

預設快捷鍵如下，可由 `shortcuts.json` 覆寫。

**Main 範圍：**

- Space：拍照 `capture.photo`
- R：錄影開始或繼續 `record.start_resume`
- Shift+R：停止錄影 `record.stop_save`
- F9：切換檔案 Dock `dock.toggle`

**Viewer 範圍：**

- Left 或 PageUp：前一張 `nav.prev`
- Right 或 PageDown：下一張 `nav.next`
- S 或 Ctrl+S：儲存已選 `save.selected`
- U：聯集輸出 `save.union`
- F9：切換 Dock
- Esc：關閉視窗

專案也附一份 `shortcuts.json` 範例供參考。

---

## 日誌與 UI 提示

- **檔案輸出**：日誌輸出於 `logs/app.log` 與 `logs/app.jsonl` 輪替，支援 JSON 格式與去識別化（包含 email、電話、統一編號等防誤曝）。
- **UI 串接**：任何 `logging.WARNING` 以上訊息會顯示於狀態列，`ERROR` 以上可彈窗，並具節流與去重。可於偏好設定調整 popup level 與節流毫秒。
- **Qt 訊息代理**：將 Qt 的警告與錯誤導入 Python logging。
- **主程式初始化**時讀取偏好並安裝日誌與 UI handler。說明選單提供「開啟日誌資料夾」以利除錯。

---

## 檔案結構

以下為主要檔案與目錄，已重構為階層式模組以方便管理。

```text
.
├─ main.py                               # 入口點與主視窗組裝
├─ modules/
│   ├─ app/
│   │   ├─ actions.py                    # 業務行為 facade，調用控制器管理攝影與分割
│   │   ├─ camera_controller.py          # 相機行為控制（啟動、拍照、連拍、錄影）
│   │   └─ segmentation_controller.py    # SAM 分割流程控制與模型管理
│   ├─ infrastructure/
│   │   ├─ devices/
│   │   │   └─ camera_manager.py         # 相機與錄影 session 封裝與焦距品質計算
│   │   ├─ io/
│   │   │   ├─ photo.py                  # 單張拍照與重試
│   │   │   ├─ burst.py                  # 連拍控制 BurstShooter
│   │   │   ├─ recorder.py               # 錄影控制 VideoRecorder
│   │   │   └─ path_manager.py           # 管理輸出檔案路徑
│   │   ├─ logging/
│   │   │   └─ logging_setup.py          # 日誌設定與 UI 橋接
│   │   └─ vision/
│   │       └─ sam_engine.py            # SAM 推論、快取與模型卸載（已移除未使用暫存屬性）
│   └─ presentation/
│       ├─ qt/
│       │   ├─ explorer/
│       │   │   ├─ explorer.py           # 檔案瀏覽器 Dock
│       │   │   └─ explorer_controller.py# Dock 切換與路徑同步控制
│       │   ├─ segmentation/
│       │   │   └─ segmentation_viewer.py# 分割檢視器與互動輸出
│       │   ├─ onboarding.py             # 快速導覽精靈
│       │   ├─ shortcuts.py              # 快捷鍵管理與對話框
│       │   ├─ status_footer.py          # 統一狀態列與科幻進度彈窗
│       │   ├─ ui_main.py                # 主視窗 UI 佈局與連接
│       │   └─ ui_state.py               # 控制項可用狀態切換
│       └─ ...（其他視覺元件）
├─ utils/
│   └─ utils.py                          # 路徑管理工具，移除未使用的 ts_ms 函式
├─ shortcuts.json                        # 快捷鍵覆寫檔
└─ README.md
```

---

## 開發與擴充

- **熱鍵註冊**：於主視窗建立時註冊 Main 範圍，在分割檢視器註冊 Viewer 範圍，可透過全域管理器集中管理。
- **Dock 控制**：`ExplorerController` 封裝顯示與可視狀態同步，例外時記錄 warning 或 error。
- **UI 狀態**：依相機與連拍狀態切換按鈕可用性。
- **分割流程**：`Actions` 整合自動分割選單、權重載入/下載與開啟 `SegmentationViewer` 視窗。模組化的 `segmentation_controller.py` 負責模型存取與快取管理。
- **減少不必要的函式**：專案中移除未被調用的 `ts_ms()` 函式，以及修正 `SamEngine.unload()` 中未使用的暫存屬性，讓程式碼更精簡。後續新增功能建議以控制器及 service 類別拆分，以便測試與維護。

---

## 常見問題 FAQ

- **啟動相機失敗**：請確認相機未被其他程式占用，嘗試更換裝置或檢查權限。失敗訊息會彈窗並寫入日誌，可於「說明 → 開啟日誌資料夾」查看詳情。
- **錄影沒有 H.264 或 AAC**：某些平台或編碼器不可用，程式會盡力設定；若失敗會記錄 warning 並使用可用設定。
- **SAM 權重過大無法下載**：可手動將 `sam_vit_h_4b8939.pth` 放入 `models` 目錄，在 UI 中勾選「預先載入 SAM 模型」。若仍失敗可於對話框選擇已有的 .pth 檔。
- **記憶體不足**：可關閉預載或在分割後卸載模型以釋放 GPU 記憶體。

---

## 授權 License

本專案採用 MIT License 授權。您可以自由使用、複製、修改和散布本程式碼，只需在程式碼或文件中保留原始版權宣告及本授權條款。詳見根目錄的 `LICENSE` 檔案。

---

## 貢獻與提交訊息

- **歡迎貢獻**：建議以 模組化方式 擴充功能，並在 `main.py` 掛載或注入依賴。提交 Pull Request 時建議提供動機、變更內容、測試方式、風險評估以及截圖/錄影。
- **提交訊息**：建議使用約定式提交（Conventional Commits）格式，亦可搭配 Gitmoji 標示。
