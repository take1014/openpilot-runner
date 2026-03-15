# openpilot-runner

A standalone module that runs the openpilot SuperCombo model (v0.8.10) independently.
Performs inference on real-time webcam footage or a folder of saved images, and draws lane lines, road edges, path, and lead-car overlays.

## Features

- **Real-time inference** (`openpilot_on_webcam.py`) — live overlay display on webcam footage, with optional MP4 recording
- **Batch processing** (`test/test_images.py`) — process an image folder and write annotated output images
- **SuperCombo v0.8.10** — model executed via ONNX Runtime; works on M1/M2 Mac and Linux CPU
- **Split view** — left panel: full camera frame (1280×720), right panel: model input view (512×256 × scale)
- **Tested camera** — Logitech C920 (1280×720, HFOV ~70°, focal length ~908 px)

## Requirements

- Python 3.11 or later (below 3.13)
- `numpy >= 1.24`
- `opencv-python >= 4.8`
- `onnxruntime >= 1.14` (required for inference)

### pip (standard virtual environment)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Anaconda / Miniconda

```bash
conda create -n openpilot-runner python=3.11
conda activate openpilot-runner
pip install -r requirements.txt
```

## Model file

Place `supercombo.onnx` (openpilot v0.8.10) at:

```
openpilot-runner/models/supercombo.onnx
```

If the model is absent, the script starts in preview-only mode (no inference).

---

## Usage

### 1. Real-time webcam

```bash
# Basic (camera device 0, left-hand traffic)
python -m openpilot-runner.openpilot_on_webcam

# Specify device and focal length
python -m openpilot-runner.openpilot_on_webcam --camera 1 --focal-length 820

# Right-hand drive traffic (Japan / UK)
python -m openpilot-runner.openpilot_on_webcam --rhd

# Save to video (auto-generated timestamped filename)
python -m openpilot-runner.openpilot_on_webcam --save-video

# Save to video (explicit filename)
python -m openpilot-runner.openpilot_on_webcam --save-video output.mp4

# Camera mounted upside-down (vertical flip is ON by default)
# Use --no-flip to disable
python -m openpilot-runner.openpilot_on_webcam --no-flip
```

| Option | Default | Description |
|---|---|---|
| `--camera INT` | `0` | Webcam device index |
| `--width INT` | `1280` | Capture width (px) |
| `--height INT` | `720` | Capture height (px) |
| `--focal-length FLOAT` | `908.0` | Focal length (px). ~908 for C920, ~820 for 78° HFOV webcam |
| `--display-scale FLOAT` | `2.0` | Display scale factor for the 512×256 model view |
| `--cam-fps INT` | `20` | Camera capture frame rate (Hz) |
| `--fps-cap INT` | `20` | Maximum display frame rate (Hz) |
| `--rhd` | — | Right-hand drive traffic (Japan / UK: drive on left) |
| `--no-flip` | — | Disable vertical flip (default: flip is ON for upside-down mounting) |
| `--save-video [FILE]` | — | Save to MP4. Omit FILE for an auto-generated timestamped name |

**Quit**: press `q` or `ESC`

---

### 2. Batch processing of saved images

```bash
# Basic
python -m openpilot-runner.test.test_images --input ~/fun/openpilot-runner/c920

# Specify output folder and focal length
python -m openpilot-runner.test.test_images \
    --input  ~/fun/openpilot-runner/c920 \
    --output ~/fun/openpilot-runner/c920_out \
    --focal-length 820

# Right-hand drive, process only the first 100 frames
python -m openpilot-runner.test.test_images \
    --input ~/fun/openpilot-runner/c920 \
    --rhd --limit 100
```

| Option | Default | Description |
|---|---|---|
| `--input DIR` | required | Input image folder |
| `--output DIR` | `<input>_out` | Output image folder |
| `--focal-length FLOAT` | `908.0` | Focal length (px) |
| `--display-scale FLOAT` | `2.0` | Display scale factor for the model view |
| `--rhd` | — | Right-hand drive traffic mode |
| `--flip` | — | Enable 180° vertical flip (default: disabled) |
| `--ext EXT` | `png` | Input image file extension |
| `--limit N` | — | Maximum number of frames to process |

---

## Coordinate system and output interpretation

SuperCombo model outputs use the device frame coordinate system:

| Axis | Direction |
|---|---|
| x | Forward |
| y | Right |
| z | Downward (z=1.22 is the road surface directly below a 1.22 m mounted camera) |

### Overlay colours

| Colour | Meaning |
|---|---|
| Green | Ego lane lines (left_near / right_near) |
| Blue | Adjacent lane lines (left_far / right_far) |
| Purple | Road edges |
| Orange | Predicted path / plan |
| Red | Lead car |

### Note on overlay alignment

The model was trained assuming a camera mounted at 1.22 m height, horizontally level.
Differences in actual mounting height or road gradient may cause overlays to shift by tens of pixels. This is not a bug — it reflects the difference between training conditions and real-world camera setup.

---

## Module structure

```
openpilot-runner/
├── constants.py            Camera intrinsics and coordinate transform constants
├── preprocess.py           Warp matrix, YUV conversion, model input buffer construction
├── visualize.py            3D-to-image projection and OpenCV overlay drawing
├── openpilot_on_webcam.py  Real-time inference main script
├── camera/
│   └── __init__.py         CameraThread, AsyncVideoWriter
├── runner/
│   ├── __init__.py         ModelRunner (ONNX execution)
│   ├── constants.py        Model constants (I/O sizes, indices, etc.)
│   └── parser.py           sigmoid, parse_outputs
├── test/
│   ├── test_images.py      Batch inference script
│   └── debug_projection.py Projection coordinate debug tool
└── models/
    └── supercombo.onnx     (place model file here)
```

---

## Coordinate transforms

### Pipeline overview

```
┌─────────────────────┐
│  Webcam frame       │  BGR, e.g. 1280×720
│  (camera space)     │
└────────┬────────────┘
         │  build_warp_matrix() + cv2.warpPerspective()
         │  homography:  M = webcam_K · inv(model_K)
         ▼
┌─────────────────────┐
│  Model input frame  │  YUV 512×256  (+ previous frame stacked)
│  (medmodel space)   │
└────────┬────────────┘
         │  supercombo.onnx (ONNX Runtime)
         ▼
┌─────────────────────┐
│  Model outputs      │  lane_lines, road_edges, plan, lead
│  (device frame)     │  x=forward  y=right  z=down  [metres]
└────────┬────────────┘
         │  road_to_img():  pinhole back-projection
         │  u = fx·(y/x) + cx
         │  v = fy·(z/x) + cy
         ▼
┌─────────────────────┐
│  Overlay pixels     │  drawn on the 512×256 model view
│  (model image px)   │  (scaled by display_scale for the window)
└─────────────────────┘
```

### Device frame axes

```
              x (forward)
              ▲
              │
              │        ← top-down view →
   y ◄────────┤  (camera)
  (right)     │
              │
         ─────┼──────────────────────  road surface (z = 1.22 m)
```

```
   camera ●
          │  z (down)
          │
          ▼
   ───────────────────  road surface  z = MEDMODEL_HEIGHT = 1.22 m
```

The model outputs all distances in **metres** in device frame.
Road surface is at approximately **z = 1.22** (camera height above road).

### Warp homography

The webcam frame is re-projected into the 512×256 model input space using a
homography derived from both camera intrinsic matrices:

```
M = webcam_K · inv(medmodel_K)
```

| Matrix | fx | fy | cx | cy |
|---|---|---|---|---|
| `medmodel_K` | 910.0 | 910.0 | 256.0 | 47.6 |
| `webcam_K` (C920) | 908.0 | 908.0 | 640.0 | 360.0 |

If the camera is mounted upside-down (`--flip`), a 180° rotation is baked
into M so no separate `cv2.flip()` call is needed.

### Back-projection (device frame → image pixels)

`road_to_img()` in [visualize.py](visualize.py) uses a simple pinhole model:

$$u = f_x \cdot \frac{y}{x} + c_x \qquad v = f_y \cdot \frac{z}{x} + c_y$$

Points with $x \le 0.5$ m (behind or very close to the camera) are discarded.

### Plan z-offset

The `plan` output stores z as **road-surface-relative** (z=0 at road surface).
To convert to device frame for projection, `MEDMODEL_HEIGHT` is added:

```python
z_device = plan_z + MEDMODEL_HEIGHT   # 0 + 1.22 = 1.22 m at road surface
```

---

## Tuning the focal length

Market webcams such as the C920 often differ between their nominal specification and actual measured value.

| Camera / condition | Recommended focal length (px) |
|---|---|
| C920 / Brio, 1280×720, HFOV ~70° | `908` |
| Wide-angle webcam, HFOV ~78° | `820` |

Formula (from horizontal field of view HFOV):
```
focal_length = (width / 2) / tan(HFOV_deg / 2 * π / 180)
```
