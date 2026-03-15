# openpilot_runner

openpilot の SuperCombo モデル (v0.8.10) を単体で動かすスタンドアロンモジュールです。  
ウェブカメラのリアルタイム映像、または保存済み画像フォルダに対して推論を行い、車線・道路端・走行経路・先行車のオーバーレイを描画します。

## 機能

- **リアルタイム推論** (`openpilot_on_webcam.py`) — ウェブカメラ映像にオーバーレイをリアルタイム表示、MP4 保存も可能
- **バッチ処理** (`test/test_images.py`) — 画像フォルダを一括処理して結果画像を書き出し
- **SuperCombo v0.8.10** — ONNX Runtime でモデルを実行。M1/M2 Mac、Linux CPU どちらでも動作
- **左右分割表示** — 左パネル: カメラ原寸（1280×720）、右パネル: モデル入力（512×256 × scale）

## 動作要件

- Python 3.11 以上
- `numpy >= 1.24`
- `opencv-python >= 4.8`
- `onnxruntime >= 1.18`（推論を行う場合）

```bash
pip install -r openpilot_runner/requirements.txt
```

## モデルファイルの配置

`supercombo.onnx`（openpilot v0.8.10）を以下に置きます:

```
openpilot_runner/models/supercombo.onnx
```

モデルがない場合はプレビューモード（推論なし）で起動します。

---

## 使い方

### 1. リアルタイムウェブカメラ

```bash
# 基本（カメラデバイス 0、右側通行）
python -m openpilot_runner.openpilot_on_webcam

# デバイス指定・焦点距離調整
python -m openpilot_runner.openpilot_on_webcam --camera 1 --focal-length 820

# 左側通行（日本・英国）
python -m openpilot_runner.openpilot_on_webcam --rhd

# 動画を保存（タイムスタンプ付き自動ファイル名）
python -m openpilot_runner.openpilot_on_webcam --save-video

# 動画を保存（ファイル名指定）
python -m openpilot_runner.openpilot_on_webcam --save-video output.mp4

# カメラが上下逆に取り付けられている場合（デフォルトで有効）
# --no-flip で無効化
python -m openpilot_runner.openpilot_on_webcam --no-flip
```

| オプション | デフォルト | 説明 |
|---|---|---|
| `--camera INT` | `0` | ウェブカメラのデバイス番号 |
| `--width INT` | `1280` | キャプチャ幅 (px) |
| `--height INT` | `720` | キャプチャ高さ (px) |
| `--focal-length FLOAT` | `908.0` | 焦点距離 (px)。C920 は約 908、広角 78° では約 820 |
| `--display-scale FLOAT` | `2.0` | モデル 512×256 の表示倍率 |
| `--fps-cap INT` | `20` | 最大フレームレート (Hz) |
| `--rhd` | — | 左側通行モード（日本・英国） |
| `--no-flip` | — | 垂直反転を無効化（デフォルト: 反転あり） |
| `--save-video [FILE]` | — | MP4 保存。FILE 省略でタイムスタンプ自動命名 |

**終了**: `q` または `ESC`

---

### 2. 保存済み画像フォルダのバッチ処理

```bash
# 基本
python -m openpilot_runner.test.test_images --input ~/fun/openpilot-runner/c920

# 出力フォルダ・焦点距離指定
python -m openpilot_runner.test.test_images \
    --input  ~/fun/openpilot-runner/c920 \
    --output ~/fun/openpilot-runner/c920_out \
    --focal-length 820

# 左側通行、最初の 100 枚のみ処理
python -m openpilot_runner.test.test_images \
    --input ~/fun/openpilot-runner/c920 \
    --rhd --limit 100
```

| オプション | デフォルト | 説明 |
|---|---|---|
| `--input DIR` | 必須 | 入力画像フォルダ |
| `--output DIR` | `<input>_out` | 出力画像フォルダ |
| `--focal-length FLOAT` | `908.0` | 焦点距離 (px) |
| `--display-scale FLOAT` | `2.0` | モデルビューの表示倍率 |
| `--rhd` | — | 左側通行モード |
| `--flip` | — | 180° 反転を有効化（デフォルト: 無効） |
| `--ext EXT` | `png` | 入力画像の拡張子 |
| `--limit N` | — | 処理する最大フレーム数 |

---

## 座標系と出力の解釈

SuperCombo モデルの出力はデバイスフレーム座標系です:

| 軸 | 方向 |
|---|---|
| x | 前方 |
| y | 右方向 |
| z | 下方向（z=1.22 がカメラ真下の道路面） |

### オーバーレイの色

| 色 | 意味 |
|---|---|
| 緑 | 自車線（left_near / right_near） |
| 橙 | 隣接車線（left_far / right_far） |
| 紫 | 道路端 (road edges) |
| 水色 | 走行経路 (path / plan) |
| 赤 | 先行車 (lead car) |

### オーバーレイのずれについて

モデルはカメラ高さ 1.22m・水平取り付けを前提として学習されています。  
実際の取り付け高さや道路勾配が異なると、描画位置が数十 px ずれることがあります。これはコードのバグではなくカメラ設置条件の差異によるものです。

---

## モジュール構成

```
openpilot_runner/
├── constants.py          カメラ内部パラメータ・座標変換定数
├── preprocess.py         ワープ行列・YUV 変換・モデル入力バッファ生成
├── visualize.py          3D→画像投影・オーバーレイ描画
├── openpilot_on_webcam.py  リアルタイム推論メインスクリプト
├── camera/
│   └── __init__.py       CameraThread, AsyncVideoWriter
├── runner/
│   ├── __init__.py       ModelRunner (ONNX 実行)
│   ├── constants.py      モデル定数（入出力サイズ・インデックス等）
│   └── parser.py         sigmoid, parse_outputs
├── test/
│   ├── test_images.py    バッチ推論スクリプト
│   └── debug_projection.py  投影座標のデバッグ出力ツール
└── models/
    └── supercombo.onnx   (要配置)
```

---

## 焦点距離のチューニング

C920 などの Market ウェブカメラは公称スペックと実測値が異なる場合があります。

| カメラ / 条件 | 推奨焦点距離 (px) |
|---|---|
| C920 / Brio、1280×720、水平視野角 ~70° | `908` |
| 広角ウェブカメラ、水平視野角 ~78° | `820` |

計算式（水平視野角 HFOV から）:
```
focal_length = (width / 2) / tan(HFOV_deg / 2 * π / 180)
```
