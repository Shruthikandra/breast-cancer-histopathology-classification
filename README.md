# Breast Cancer Histopathology Classification (BreakHis)

Deep learning models that classify breast tissue histopathology images as **benign** or **malignant**, and a study of how **class-balancing strategy** (oversampling vs undersampling) interacts with model choice.

> MSc Data Science & Analytics dissertation — University of Hertfordshire.
> *A research classifier to support diagnosis, never to replace a specialist's judgement.*

---

## 📌 Overview

The [BreakHis dataset](https://www.kaggle.com/datasets/ambarish/breakhis) contains **7,909** histopathology images, but the classes are imbalanced (~69% malignant / ~31% benign). This project builds a full image-classification pipeline and compares three architectures under two balancing strategies to find what actually works best — and shows that there is **no single best setup**: the balancing method and the model interact.

## 🧪 Methodology

- **Preprocessing:** image validation, resizing to 224×224, normalisation, augmentation (rotation, horizontal flip, zoom), and a **stratified 70/15/15** train/validation/test split.
- **Class balancing:** applied **only to the training set** (to avoid leakage) — compared **oversampling** vs **undersampling**.
- **Models:** a **custom CNN** (baseline) vs **DenseNet121** and **MobileNetV2** (transfer learning, ImageNet weights).
- **Training:** Adam (lr 1e-4), dropout 0.5, early stopping on validation loss.
- **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, and confusion-matrix analysis.

## 📊 Results

| Model | Balancing | Accuracy | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Custom CNN** | **Undersampling** | **90.9%** | **95.5%** | **93.2%** | **97.2%** |
| DenseNet121 | Oversampling | 89.6% | 93.8% | 92.3% | 95.2% |
| MobileNetV2 | Oversampling | lower | — | — | — |

**Best model:** custom CNN with undersampling.
**Key insight:** a larger pretrained network isn't automatically better — the right data strategy depends on the model it's paired with.

## 📁 Repository structure

```
├── breast_cancer_UNDERSAMPLING_COLAB.ipynb   # pipeline + models, undersampling
├── breast_cancer_OVERSAMPLING.ipynb           # pipeline + models, oversampling
├── figures/                                   # confusion matrices, comparison charts
└── README.md
```

## ▶️ How to run

1. Open either notebook in **Google Colab** (recommended — it was built there) or Jupyter.
2. Download the BreakHis dataset from Kaggle and update the dataset path in the notebook.
3. Run the cells top to bottom. A GPU runtime is recommended for training.

## 🛠️ Tech stack

Python · TensorFlow · Keras · scikit-learn · pandas · NumPy · Matplotlib · Seaborn

## 📄 Related publications

Two peer-reviewed NLP papers (IARJSET, 2024) — see my [LinkedIn](https://www.linkedin.com/in/shruthi-kandra).

## 👤 Author

**Shruthi Kandra** — Data Analyst · Luton, UK
[LinkedIn](https://www.linkedin.com/in/shruthi-kandra) · shruthikandra123@gmail.com
