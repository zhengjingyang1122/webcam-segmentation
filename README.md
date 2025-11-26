# Webcam Segmentation Tool

基於 Segment Anything Model (SAM) 的影像分割工具，使用 PySide6 建構的桌面應用程式。支援單一影像或批次資料夾的自動分割，並提供互動式介面選擇與輸出透明背景物件。

---

## 目錄

- [特色功能](#特色功能)
- [安裝](#安裝)
  - [系統需求](#系統需求)
  - [依賴套件](#依賴套件)
  - [安裝步驟](#安裝步驟)
- [執行方式](#執行方式)
- [使用流程](#使用流程)
- [SAM 模型](#sam-模型)
- [鍵盤快捷鍵](#鍵盤快捷鍵)
- [檔案結構](#檔案結構)
- [常見問題](#常見問題)
- [授權](#授權)

---

## 特色功能

- **多模型支援**：支援 SAM-B (Fast)、SAM-L (Balanced)、SAM-H (Best Quality) 三種模型
- **GPU 加速**：自動偵測並使用 CUDA 進行 GPU 加速推論
- **批次處理**：支援單一影像或整個資料夾的批次分割
- **互動式選擇**：滑鼠游標預覽遮罩，左鍵加入選取，右鍵移除
- **快取機制**：自動快取分割結果與 embedding，加速重複開啟
- **透明輸出**：支援個別物件或聯集輸出為含透明度的 PNG 檔案
- **深色/淺色主題**：內建深色與淺色主題切換

---

## 安裝

### 系統需求

- Python 3.9 至 3.12（建議 3.11）
- Windows 10/11（其他平台可自行嘗試）
- 若要使用 GPU 推論，需安裝 CUDA 版 PyTorch

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
.\\.venv\\Scripts\\activate

# 安裝依賴套件
pip install --upgrade pip
pip install PySide6 opencv-python numpy torch

# 安裝 Segment Anything
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 下載 SAM 模型

將 SAM 模型權重檔案放置於 `models/` 目錄下：

- **SAM-B**: `sam_vit_b_01ec64.pth` (~375 MB)
- **SAM-L**: `sam_vit_l_0b3195.pth` (~1.2 GB)
- **SAM-H**: `sam_vit_h_4b8939.pth` (~2.5 GB)

模型下載連結：[Segment Anything Model Checkpoints](https://github.com/facebookresearch/segment-anything#model-checkpoints)

---

## 執行方式

```bash
python main.py
```

首次啟動時，應用程式會顯示歡迎畫面並提示選擇 SAM 模型。

---

## 使用流程

1. **選擇模型**：在主視窗選擇要使用的 SAM 模型（SAM-B/L/H）
2. **開啟影像**：
   - **單一影像**：檔案 → 開啟影像，選擇要分割的影像檔
   - **批次處理**：檔案 → 開啟資料夾，選擇包含影像的資料夾
3. **互動式選擇**：
   - 滑鼠移動：預覽當前位置的遮罩
   - 左鍵點擊：加入遮罩到選取集合
   - 右鍵點擊：從選取集合移除遮罩
4. **儲存結果**：
   - **儲存已選目標**：個別輸出每個選取的物件
   - **聯集輸出**：將所有選取的遮罩合併為單一物件輸出
5. **瀏覽影像**：使用左右方向鍵或 PageUp/PageDown 切換影像

---

## SAM 模型

### 模型選擇

- **SAM-B (vit_b)**：最快速，適合即時預覽或大量批次處理
- **SAM-L (vit_l)**：平衡速度與品質
- **SAM-H (vit_h)**：最高品質，適合精細分割需求

### 快取機制

分割結果會自動快取於影像所在目錄：
- `<image_name>.sam_masks.npz`：遮罩資料
- `<image_name>.sam_embed.npz`：影像 embedding

重新開啟相同影像時會自動載入快取，大幅加速處理速度。

---

## 鍵盤快捷鍵

### 分割檢視器

- **Left / PageUp**：前一張影像
- **Right / PageDown**：下一張影像
- **S / Ctrl+S**：儲存已選目標
- **U**：聯集輸出
- **Esc**：關閉視窗

---

## 檔案結構

```text
.
├── main.py                                    # 應用程式入口點
├── models/                                    # SAM 模型權重檔案目錄
│   ├── sam_vit_b_01ec64.pth
│   ├── sam_vit_l_0b3195.pth
│   └── sam_vit_h_4b8939.pth
├── modules/
│   ├── infrastructure/
│   │   ├── logging/
│   │   │   └── logging_setup.py              # 日誌設定
│   │   └── vision/
│   │       ├── sam_engine.py                 # SAM 推論引擎
│   │       └── segment_anything/             # SAM 模型實作
│   └── presentation/
│       └── qt/
│           ├── progress_dialog.py            # 進度對話框
│           ├── segmentation/
│           │   └── segmentation_viewer.py    # 分割檢視器
│           └── theme_manager.py              # 主題管理
├── utils/
│   ├── get_base_path.py                      # 取得應用程式根目錄
│   └── utils.py                              # 工具函數
└── README.md
```

---

## 常見問題

### Q: 找不到 SAM 模型檔案

**A**: 請確認模型檔案已下載並放置於 `models/` 目錄下，檔名需與預設名稱一致。

### Q: GPU 推論失敗

**A**: 請確認已安裝 CUDA 版本的 PyTorch。可透過以下指令檢查：
```python
import torch
print(torch.cuda.is_available())  # 應回傳 True
```

### Q: 記憶體不足

**A**: 
- 使用較小的模型（SAM-B）
- 處理較小尺寸的影像
- 關閉其他佔用記憶體的應用程式

### Q: 分割速度很慢

**A**:
- 確認使用 GPU 模式（CUDA）
- 使用較小的模型（SAM-B）
- 第二次開啟相同影像時會自動載入快取，速度會大幅提升

---

## 授權

本專案採用 MIT License 授權。詳見 `LICENSE` 檔案。

---

## 貢獻

歡迎提交 Issue 或 Pull Request！

---

**開發者**: 使用 PySide6 與 Segment Anything Model 建構
