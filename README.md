# COMP6001 Assignment 1: Image Restoration and Object Detection

> A multimodal pipeline for motion deblurring and object detection using the GoPro dataset.
> Combines classical filtering, deep learning (Residual U-Net and NAFNet), YOLOv8-based detection analysis, and fine-tuning on pseudo-labeled deblurred data.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Pipeline Summary](#pipeline-summary)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [Dataset](#dataset)
- [Running the Notebook](#running-the-notebook)
- [Tasks Breakdown](#tasks-breakdown)
- [Results](#results)
- [AI Tool Attribution](#ai-tool-attribution)
- [Ethical Considerations](#ethical-considerations)
- [License](#license)

---

## Project Overview

This project builds an end-to-end pipeline that:
1. Deblurs motion-blurred images using classical methods (Wiener, Richardson–Lucy) and two deep learning models (Residual U-Net from scratch, and NAFNet).
2. Evaluates the effect of deblurring on object detection performance using YOLOv8.
3. Prepares a stratified, pseudo-labeled dataset from deblurred images and fine-tunes a YOLOv8 detector.
4. Conducts a comprehensive performance comparison across blurred, deblurred, sharp, and fine-tuned conditions.

---

## Pipeline Summary

```
GoPro Dataset (Blurred + Sharp pairs)
        │
        ▼
Task 2: Image Deblurring
  ├── Classical: Wiener Filter, Richardson–Lucy
  ├── Deep Learning: Residual U-Net (AdvancedUNet, from scratch)
  └── Deep Learning: NAFNet (width=64, GoPro pretrained)
        │
        ▼
Task 3: Object Detection Analysis
  └── YOLOv8-m inference on Blurred vs Deblurred (Sharp as GT proxy)
      Metrics: Precision, Recall, F1, IoU boxplot,
               Confidence shift scatter, Per-class F1
        │
        ▼
Task 4: Dataset Preparation + Fine-Tuning
  ├── Blur-level stratified split (Low/Mid/High via Laplacian variance)
  ├── 80/20 train/val split within each stratum
  ├── Pseudo-label generation via YOLOv8-m (conf ≥ 0.40)
  └── YOLOv8 fine-tuned on pseudo-labeled deblurred dataset
        │
        ▼
Task 5: Comprehensive Performance Comparison
  └── Blurred | Deblurred | Sharp (GT proxy) | Pseudo-label FT
      Metrics: mAP@50, per-class AP, precision–recall curves,
               detection gap heatmap, blur-level mAP breakdown
```

---

## Repository Structure

The `main` branch contains the full integrated pipeline across four progressive versions, along with an AI prompt log covering all stages of development:

```
comp6001-assignment-1/          ← main branch
│
├── README.md
├── ai_logs.md                  # AI prompt and output log for all versions (v1–v4)
├── full_pipeline-v1.ipynb
├── full_pipeline-v2.ipynb
├── full_pipeline-v3.ipynb
└── full_pipeline-v4.ipynb      # Final submission version
```

In addition, each task has its own dedicated branch containing the isolated development history for that task:

| Branch | Contents |
|--------|----------|
| `main` | Full integrated pipeline — all 4 versions (v1–v4) + AI logs |
| `task2-deblurring` | Image deblurring development (classical methods + U-Net + NAFNet) |
| `task3-detection` | Object detection analysis with YOLOv8 |
| `task4-finetuning` | Pseudo-label dataset preparation and YOLOv8 fine-tuning |
| `task5-analysis` | Performance comparison and critical analysis |

---

---

## Setup and Installation

### Requirements

- Python 3.12+
- CUDA-enabled GPU (recommended: NVIDIA Tesla T4 or equivalent)
- [Kaggle](https://www.kaggle.com) account (for dataset access)

### Install Dependencies

```bash
pip install torch torchvision opencv-python scikit-image matplotlib tqdm
pip install ultralytics
pip install Pillow pandas numpy
```

Or on Kaggle, dependencies are installed inline within the notebook cells.

### NAFNet Weights

The notebook auto-downloads pretrained NAFNet weights via `gdown` on first run and saves them to:

```
weights/NAFNet-GoPro-width64.pth
```

NAFNet GitHub: https://github.com/megvii-research/NAFNet

---

## Dataset

This project uses the **GoPro Motion Blur Dataset**.

| Source | Link |
|--------|------|
| Curated deblurring datasets | https://www.kaggle.com/datasets/jishnuparayilshibu/a-curated-list-of-image-deblurring-datasets |



---

## Running the Notebook

The final notebook (`full_pipeline-v4.ipynb`) is structured as four sequential cells, one per task:

```
Cell 0 → Task 2: Image Deblurring
Cell 1 → Task 3: Object Detection and Analysis
Cell 2 → Task 4: Dataset Preparation and Fine-Tuning
Cell 3 → Task 5: Performance Comparison
```

**Run in order** — each task depends on outputs from the previous one.

### On Kaggle

1. Upload or fork the notebook to Kaggle.
2. Enable GPU accelerator (Tesla T4).
3. Add the dataset as a data source.
4. Run all cells sequentially.



---

## Tasks Breakdown

### Task 1 — Version Control and AI-Assisted Coding
- Git workflow with task-based branches and version-based commits on main — each commit on main corresponds to a pipeline version (v1–v4), with development history tracked per task in dedicated branches.
- All AI-generated code documented in `ai_logs.md` with prompts, outputs, and attribution.
- Ethical reflection and licensing compliance noted throughout.

### Task 2 — Image Deblurring
- **Classical methods:** Wiener Filter and Richardson–Lucy deconvolution via `skimage.restoration`, using a 5×5 horizontal PSF.
- **Deep learning — Residual U-Net (AdvancedUNet):** Built from scratch with residual conv blocks, MaxPool downsampling, and bilinear upsampling. Trained for 10 epochs with Adam (lr=3e-4), CosineAnnealingLR, MSE loss, and an 85/15 train/val split on crop size 192×192.
- **Deep learning — NAFNet:** Pretrained on GoPro (width=64), auto-downloaded via `gdown`. Used for inference only.
- **Augmentations:** RandomCrop (256×256), RandomHorizontalFlip (p=0.5), RandomVerticalFlip (p=0.2).
- **Evaluation:** PSNR and SSIM computed across all four methods on the full test set (1111 images):

  | Method | Avg PSNR | Avg SSIM |
  |--------|----------|----------|
  | Wiener Filter | 23.43 dB | 0.6905 |
  | Richardson–Lucy | 18.86 dB | 0.6536 |
  | Residual U-Net | 24.52 dB | 0.7441 |
  | **NAFNet** | **28.62 dB** | **0.8714** |
- **Visualisation:** 6×6 reference grid — 6 images × 6 columns (Blurred, Wiener, Lucy, ResUNet, NAFNet, Sharp GT).
- Reproducibility fixed with `set_seed(67)`.

### Task 3 — Object Detection and Analysis
- **Model:** YOLOv8-m (`yolov8m.pt`), pretrained on COCO. Inference at `conf=0.25, imgsz=640`.
- **Conditions:** Sharp (GT proxy), Blurred, Deblurred (NAFNet). Sharp used as ground truth for matching.
- **Metrics:** Precision, Recall, F1 (IoU-matched at 0.5), average detections, average confidence, latency.
- **Visualisations:**
  - 6×3 reference grid — bounding boxes on Blurred / Deblurred / Sharp for 6 reference images.
  - IoU boxplot — detection boundary accuracy (Blurred vs Sharp, Deblurred vs Sharp).
  - Confidence shift scatter — per-object confidence change after deblurring.
  - Per-class F1 bar chart — top 10 classes by GT count.
  - Failure case analysis — 4 worst images by missed detections after deblurring.
- Reference images stratified across blur intensity percentiles (0th, 33rd, 67th) and motion types (3 types).

### Task 4 — Dataset Preparation and Fine-Tuning
- **Stratification:** Images binned into Low/Mid/High blur levels using Laplacian variance, ensuring balanced representation across blur intensities.
- **Split:** 80% train / 20% validation within each blur stratum.
- **Crop size:** 512×512 (center crop).
- **Pseudo-labeling:** YOLOv8-m run on deblurred training images at `conf=0.40` to generate YOLO-format labels.
- **Fine-tuning hyperparameters:** 30 epochs, `lr0=0.0001`, first 10 layers frozen (`freeze=10`), `patience=20`, `batch=8`, `imgsz=512`.
- **Augmentations:** Mosaic (1.0), HSV jitter, horizontal flip (0.5), scale (0.5), translate (0.1).
- Visual comparison on 6 reference images: Sharp (GT) vs Baseline YOLO vs Pseudo-label FT.

### Task 5 — Performance Comparison and Critical Analysis
- **Four conditions evaluated:** Blurred, Deblurred, Sharp (GT proxy), Pseudo-label FT.
- Sharp used as GT proxy; mAP computed for the other three conditions against it.
- **Metrics and outputs:**
  - mAP@50 bar chart and average detections per image.
  - Per-class AP@50 for Pseudo-label FT (top 20 classes).
  - Precision–Recall curves for Blurred, Deblurred, and Pseudo-label FT.
  - Detection gap heatmap — mean detection gap vs Sharp GT, broken down by blur level (Low/Mid/High).
  - Blur-level mAP breakdown — mAP@50 computed separately per blur stratum for each condition.
  - Failure case analysis — per-image detection gap summary.
- Visual reference comparison: 6 reference images × 4 conditions with bounding boxes.

---

## Results

| Condition | mAP@50 | Avg Detections | Avg Confidence | Notes |
|-----------|--------|---------------|----------------|-------|
| Blurred | 0.1873 | 3.71 | 0.3573 | Baseline, degraded performance |
| Deblurred (NAFNet) | 0.2504 | 4.69 | 0.4304 | Improved boundaries and confidence |
| Sharp (GT proxy) | — | 5.03 | 0.4438 | Upper bound |
| Pseudo-label FT | 0.2804 | 3.97 | 0.4241 | Fine-tuned on deblurred distribution |

---

## AI Tool Attribution

All AI-assisted code generation is documented in [`ai_logs.md`](./ai_logs.md), covering prompts, outputs, and attribution across the project.

Each entry includes:
- The prompt provided to the AI tool
- A summary of the output received
- Notes on modifications made and why
- Attribution of the tool used (e.g., Claude, ChatGPT, Gemini)

---

## Ethical Considerations

- **Dataset licensing:** GoPro dataset is used for academic research purposes only.
- **Model weights:** NAFNet and YOLOv8 pretrained weights are used within their respective licenses (MIT / AGPL-3.0).
- **Bias awareness:** Pseudo-labeling introduces confirmation bias — labels are only as good as the teacher model. Performance on underrepresented classes may be weaker. Blur-level stratification partially mitigates sampling bias.
- **Responsible AI:** AI-generated code was reviewed, tested, and adapted; not used verbatim without understanding.

---

## License

This project is submitted as academic coursework for COMP6001 at the University of Adelaide.
