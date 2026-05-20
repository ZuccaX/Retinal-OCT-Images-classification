# Retinal OCT Disease Classification

Four-class retinal optical coherence tomography (OCT) disease classification using both traditional computer vision features and transfer-learning convolutional neural networks.

The project classifies OCT scans into:

- `CNV`
- `DME`
- `DRUSEN`
- `NORMAL`

It includes a reproducible data audit pipeline, patient-aware validation splitting, traditional feature baselines, deep learning models, processed-data export, and report-ready evaluation artifacts.

## Highlights

- Audits OCT image files and builds a manifest-driven experiment protocol.
- Uses patient-aware splitting for model selection to reduce validation leakage.
- Compares traditional features: HOG, LBP, and SIFT Bag-of-Visual-Words.
- Trains classical classifiers: Linear SVM, RBF-SVM, and Random Forest.
- Benchmarks transfer-learning CNNs: ResNet18, ResNet34, ResNet50, VGG16, and MobileNetV2.
- Reports both official test performance and a stricter patient-disjoint robustness subset.
- Generates confusion matrices, prediction tables, per-class metrics, and analysis notebooks.

## Results Summary

Primary model selection uses `train_final -> val_final` with Macro-F1 as the main metric.

| Model | Protocol | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| HOG + RBF-SVM | official test | 0.9597 | 0.9598 |
| HOG + RBF-SVM | strict patient-disjoint subset | 0.8557 | 0.8353 |
| VGG16 | official test | 0.9638 | 0.9637 |
| VGG16 | strict patient-disjoint subset | 0.9701 | 0.9655 |

The official test split is useful for comparability with the dataset convention, but the data audit found patient overlap between training and official test images. The strict subset removes overlapping patients and is reported as a more conservative robustness check.

## Dataset

This project was developed with the Kermany OCT2017 retinal OCT dataset, available from Kaggle:

https://www.kaggle.com/datasets/paultimothymooney/kermany2018

The dataset is not included in this repository because of size and licensing constraints. After downloading, place the raw dataset under:

```text
data/raw/
```

The code supports the common Kaggle layout:

```text
data/raw/
  train/
    CNV/
    DME/
    DRUSEN/
    NORMAL/
  test/
    CNV/
    DME/
    DRUSEN/
    NORMAL/
  val/
    CNV/
    DME/
    DRUSEN/
    NORMAL/
```

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Conda or Miniconda:

```bash
conda create -n oct-classification python=3.11
conda activate oct-classification
pip install -r requirements.txt
```

## Project Structure

```text
.
├── main.py
├── requirements.txt
├── src/
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── processed.py
│   ├── evaluate.py
│   └── deeplearning/
│       ├── dataset_manifest.py
│       ├── models.py
│       ├── trainer.py
│       ├── run_experiment.py
│       └── run_all_deep_models.py
├── notebooks/
│   ├── 01_data_visualisation.ipynb
│   ├── 02_feature_demo.ipynb
│   ├── 03_results_analysis.ipynb
│   ├── interpretation.ipynb
│   ├── deeplearningexplain.ipynb
│   └── all_5_results_dashboard.ipynb
├── data/          # not tracked
└── outputs/       # not tracked
```

## Reproducible Workflow

### 1. Build the dataset audit and split manifest

```bash
python main.py audit --val-fraction 0.15 --seed 42
```

This scans the dataset, checks image metadata, parses patient identifiers, creates `split_manifest.csv`, and writes audit reports under `outputs/tables/`.

### 2. Export processed images

```bash
python main.py export-processed \
  --profile oct224_gray_png_v2 \
  --image-size 224 \
  --output-format png \
  --splits train_final,val_final,test_final
```

Processed files are written to `data/processed/` and indexed by `outputs/tables/processed_manifest_*.csv`.

### 3. Run a traditional baseline

```bash
python main.py train-trad \
  --feature hog \
  --classifier rbf_svm \
  --eval-split val_final \
  --image-size 224
```

Supported features:

- `hog`
- `lbp`
- `sift_bovw`

Supported classifiers:

- `linear_svm`
- `rbf_svm`
- `random_forest`

### 4. Run the full traditional matrix

```bash
python main.py train-matrix --image-size 224 --seed 42
```

This evaluates HOG, LBP, and SIFT-BoVW with Linear SVM, RBF-SVM, and Random Forest.

### 5. Train a deep learning model

```bash
python main.py train-dl \
  --model vgg16 \
  --eval-split val_final \
  --epochs 20 \
  --batch-size 32 \
  --image-size 224 \
  --data-mode processed \
  --processed-profile oct224_gray_png_v2 \
  --freeze-backbone
```

Supported models:

- `resnet18`
- `resnet34`
- `resnet50`
- `vgg16`
- `mobilenet_v2`

### 6. Run final test evaluation

Traditional final test:

```bash
python main.py final-test \
  --feature hog \
  --classifier rbf_svm \
  --image-size 224 \
  --seed 42 \
  --confirm-final-report
```

Deep learning final test:

```bash
python main.py train-dl \
  --model vgg16 \
  --eval-split test_final \
  --final-test \
  --epochs 20 \
  --batch-size 32 \
  --image-size 224 \
  --data-mode processed \
  --processed-profile oct224_gray_png_v2 \
  --freeze-backbone
```

The `--final-test` flag is required for deep learning evaluation on `test_final` to make final-test usage explicit.

### 7. Collect results

```bash
python main.py collect-results
```

This merges traditional and deep learning results into report-ready summary tables.

## Evaluation Protocol

The project uses the following split policy:

- `train_final`: training data.
- `val_final`: patient-aware validation data for model selection.
- `test_final`: official final test split.
- `val_raw_holdout`: original tiny validation folder, used only as a sanity check.
- `strict_test_subset`: patient-disjoint subset of `test_final`, used for robustness analysis.

The primary metric is Macro-F1 because the dataset is imbalanced. Accuracy is reported as a secondary metric.

## Notebooks

- `01_data_visualisation.ipynb`: dataset inspection and class distribution.
- `02_feature_demo.ipynb`: traditional feature demonstration.
- `03_results_analysis.ipynb`: result tables and plots.
- `interpretation.ipynb`: traditional model interpretation with occlusion sensitivity.
- `deeplearningexplain.ipynb`: CNN confusion matrix, ROC curves, Grad-CAM, and Integrated Gradients.
- `all_5_results_dashboard.ipynb`: comparison dashboard for official and strict results.

## Notes on Data and Artifacts

This repository intentionally excludes:

- raw OCT images
- processed images
- model checkpoints
- cached feature matrices
- generated output tables and figures

These files are large and should be regenerated locally. The `.gitignore` should exclude `data/`, `outputs/`, `*.pt`, `*.joblib`, `*.npz`, and archives.


