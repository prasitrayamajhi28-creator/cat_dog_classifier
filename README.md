# 🐱🐶 Cat vs Dog Image Classifier

A deep learning image classification project that classifies images as either **Cat** or **Dog** using **PyTorch and a pretrained ResNet18 model**.

The project uses transfer learning and includes image prediction, a Flask web application, and real-time webcam classification.

---

## 📊 Results

The trained model achieved a **98.51% test accuracy**.

| Metric | Result |
|---|---:|
| Total Images | 24,998 |
| Training Images | 17,498 |
| Validation Images | 3,749 |
| Testing Images | 3,751 |
| Test Accuracy | **98.51%** |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Dog | 0.98 | 0.99 | 0.99 | 1,888 |
| Cat | 0.99 | 0.98 | 0.98 | 1,863 |
| **Overall Accuracy** | | | **0.99** | **3,751** |

### Confusion Matrix

```text
              Predicted
              Dog    Cat

Actual Dog    1831    32
Actual Cat      24  1864