# FDCS-Net V4: AI vs Real Image Classifier

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A production-grade, hybrid deep learning architecture that distinguishes real photographs from AI-generated images by fusing spatial, frequency-domain, and color-stability forensic cues through attention-based fusion.

## Overview

With generative models like Midjourney, DALL·E, and Stable Diffusion producing increasingly convincing synthetic imagery, traditional CNNs trained only on raw pixels often fail to generalize across generators. FDCS-Net V4 addresses this by combining three complementary signal domains — spatial features, FFT magnitude artifacts, and color-perturbation stability — and dynamically weighing their contribution per image via an attention mechanism.

## Architecture

                         Input Image (256x256x3)
                                   |
              +--------------------+--------------------+
              |                    |                     |
      +---------------+   +-----------------+   +-----------------+
      | Spatial Branch |   |  Color Branch   |   | Frequency Branch|
      | EfficientNet-B0|   | Color-Stability |   |  FFT Magnitude  |
      | (pretrained)   |   | Map -> CNN      |   |  Map -> CNN     |
      +-------+--------+   +--------+--------+   +--------+--------+
              |                    |                     |
              +----------> L2-Normalize each branch <-----+
                                   |
                        +----------+----------+
                        |  Attention Fusion    |
                        |  (softmax weights)   |
                        +----------+----------+
                                   |
                        +----------+----------+
                        |   Classifier Head    |
                        |  (128 -> 1, sigmoid) |
                        +----------+----------+
                                   |
                          Real / AI-Generated

- **Spatial branch:** EfficientNet-B0 (ImageNet-pretrained), staged unfreezing across training stages.
- **Color branch:** Perturbation-based color-difference map → lightweight CNN.
- **Frequency branch:** Log-scaled FFT magnitude spectrum → lightweight CNN.
- **Fusion:** L2-normalized branch features, concatenated and passed through a softmax attention head to produce a weighted sum before the final classifier.

## Features

- **Hybrid Multi-Branch Architecture** — EfficientNet-B0 spatial backbone + custom CNN branches for FFT magnitude and color stability maps.
- **Attention-Based Fusion** — Learns per-image weights across the three branches instead of naive concatenation.
- **Auxiliary Supervision** — Dedicated classifier heads on the color and frequency branches during training to prevent branch collapse.
- **Three-Stage Training Pipeline** — Staged freezing/unfreezing of the backbone for stable convergence and reduced overfitting.
- **Forensic Transparency** — Exposes intermediate FFT and color-stability maps for interpretability.
- **Reproducible, Leak-Free Evaluation** — Classification threshold is selected on the validation set only and applied to the test set for final metrics.

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.9+ |
| Deep Learning | TensorFlow, Keras |
| Data Processing | NumPy, Scikit-learn |
| Visualization | Matplotlib |

## Project Structure

```text
ai-vs-real-image-classifier/
├── configs/
│   └── config.py              # Hyperparameters, paths, classification threshold
├── notebooks/
│   └── fdcsnet-v4-training.ipynb   # Original end-to-end training notebook (Colab)
├── reports/                   
│   └── fdcsnet-v4-report.pdf      # Complete report
├── src/
│   ├── data_preprocessing.py  # Dataset loading, augmentation, tf.data pipelines
│   ├── model.py                # FDCS-Net V4 architecture
│   ├── train.py                 # 3-stage training script
│   ├── predict.py               # CLI inference
│   └── utils.py                  # Compilation, callbacks, evaluation, plotting
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/Rishijain411/ai-vs-real-image-classifier.git
cd ai-vs-real-image-classifier

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

### 1. Prepare the Dataset

```text
data/
├── train/
│   ├── real/
│   └── ai/
└── test/
    ├── real/
    └── ai/
```

Supported formats: `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`.

### 2. Train

Runs the full 3-stage pipeline (spatial branch → auxiliary branches → end-to-end fine-tuning). Hyperparameters live in `configs/config.py`.

```bash
python src/train.py
# or save checkpoints elsewhere:
python src/train.py --output-dir models/
```

### 3. Predict

```bash
# Single image
python src/predict.py --model models/fdcsnet_v4_final.keras --image path/to/image.jpg

# Directory of images
python src/predict.py --model models/fdcsnet_v4_final.keras --dir path/to/images/
```

The classification threshold defaults to the value in `configs/config.py` (`CLASSIFICATION_THRESHOLD`); override per run with `--threshold`.

## Results

Evaluated on a held-out test set, with the classification threshold selected on the validation set (not the test set) to avoid leakage.

| Metric | Real Images | AI Images | Overall |
|---|---|---|---|
| Precision | 0.93 | 0.95 | **0.94** |
| Recall | 0.96 | 0.91 | **0.94** |
| F1-Score | 0.94 | 0.93 | **0.94** |

- **Optimal Threshold (validation-selected):** 0.8
- **AUC-ROC:** 0.9748
- **Test Accuracy:** 93.78%


## License

Distributed under the MIT License. See `LICENSE` for details.

## Authors

- **Dhruv Rathi** — [@dhruv-rathi-tech](https://github.com/dhruv-rathi-tech)