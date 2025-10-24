# Assignment 7: Multi-Class Model Selection using ROC and Precision-Recall Curves

| Field       | Detail                          |
|-------------|---------------------------------|
| Name        | Harish B                        |
| Roll Number | DA24S024                        |
| Course      | DA5401 – Data Analytics Lab     |
| Assignment  | 7                               |

---

## Overview

Multi-class classification model evaluation on Statlog Landsat Satellite dataset using ROC-AUC and Precision-Recall curves with One-vs-Rest (OvR) approach.

**Dataset:** Statlog Landsat Satellite  
**Samples:** 6,435 | **Features:** 36 spectral bands | **Classes:** 6 (1, 2, 3, 4, 5, 7)

---

## Part A: Data Preparation and Baseline

**Models Trained:**
- K-Nearest Neighbors
- Decision Tree  
- Dummy Classifier (Prior)
- Logistic Regression
- Gaussian Naive Bayes
- Support Vector Machine (probability=True)

**Results:**
- Best: K-Nearest Neighbors (Accuracy: 0.898, F1: 0.897)
- Worst: Dummy Classifier (Accuracy: 0.230, F1: 0.086)

---

## Part B: ROC Analysis

**1. Multi-Class ROC Calculation (OvR):**  
One-vs-Rest converts 6-class problem into 6 binary problems. For each class, treat it as positive and all others as negative. Compute TPR/FPR at various thresholds, then macro-average across classes.

**2. ROC Results (Macro-Average AUC):**
1. Support Vector Machine: 0.9850
2. K-Nearest Neighbors: 0.9805
3. Logistic Regression: 0.9769
4. Naive Bayes: 0.9591
5. Decision Tree: 0.9039
6. Dummy Classifier: 0.5000

**3. Interpretation:**
- Highest AUC: Support Vector Machine (0.9850)
- No model has AUC < 0.5
- **AUC < 0.5 means:** Model ranks true positives below false positives (worse than random). Causes: inverted predictions, wrong feature-target correlation, or implementation errors.

---

## Part C: Precision-Recall Curve Analysis

**1. Why PRC for Imbalanced Classes:**  
PRC focuses on positive class (Precision = TP/(TP+FP)), while ROC uses FPR which is dominated by large TN in imbalanced data. PRC immediately shows minority class performance degradation.

**2. PRC Results (Average Precision):**
1. Support Vector Machine: 0.9252
2. K-Nearest Neighbors: 0.9222
3. Logistic Regression: 0.9207
4. Naive Bayes: 0.9168
5. Decision Tree: 0.8603
6. Dummy Classifier: 0.4862

**3. Interpretation:**
- Highest AP: Support Vector Machine (0.9252)
- Dummy Classifier's curve drops sharply because as recall increases (lower threshold), false positives accumulate rapidly while precision drops to class prevalence.

---

## Part D: Synthesis and Final Recommendation

**Rankings Comparison:**

| Model       | F1 Rank | AUC Rank | AP Rank |
|-------------|---------|----------|---------|
| KNN         | 1       | 2        | 2       |
| SVM         | 2       | 1        | 1       |
| Log Reg     | 4       | 3        | 3       |

**Trade-offs:** KNN best at final predictions (F1), SVM best at ranking by confidence (AUC/AP).

**Final Recommendation: Support Vector Machine**
- Top ROC-AUC (0.9850) and PRC-AP (0.9252)
- Best precision-recall balance across thresholds
- Robust to class imbalance

---

## Brownie Points
![Brownie](https://tenor.com/view/brownie-points-gif-8254590584654661830.gif)

**Additional Models:**
- Random Forest: AUC 0.9903, AP 0.9531
- XGBoost: AUC 0.9911, AP 0.9573 **(Best Overall)**

**AUC < 0.5 Model:**
- Inverted GNB: AUC 0.489, AP 0.511, F1 0.103
- Uses `np.flip(proba)` to systematically invert predictions
- Demonstrates worse-than-random performance

**Updated Recommendation: XGBoost** (highest AUC and AP)

---

## How to Run

1. Open `solution.ipynb`
2. Run cells sequentially
3. Dataset auto-downloads from UCI ML Repository (ID: 146) via `ucimlrepo`
4. No manual dataset download needed

**Requirements:** pandas, numpy, scikit-learn, matplotlib, xgboost

---

## Files

- `solution.ipynb`: Complete analysis with visualizations