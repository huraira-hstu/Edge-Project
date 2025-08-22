# Gamma vs Hadron Classification with Machine Learning

This project implements a **machine learning pipeline** to classify events from the MAGIC Gamma Telescope dataset into two categories: **gamma-ray events (signal)** and **hadron events (background)**. The dataset is sourced from the UCI Machine Learning Repository.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Problem Formulation](#problem-formulation)
4. [Methods](#methods)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Results Visualization](#results-visualization)
7. [Usage](#usage)
8. [References](#references)

---

## Project Overview

Given a dataset $X \in \mathbb{R}^{m \times n}$ of $m$ samples and $n$ features, and corresponding labels $Y = [y_1, ..., y_m]^T$ with $y_i \in \{0,1\}$, the objective is to learn a function:

$$
f: \mathbb{R}^n \rightarrow \{0,1\}, \quad f(x_i) \approx y_i
$$

by minimizing a suitable loss function $\mathcal{L}$, e.g., **binary cross-entropy**:

$$
\mathcal{L} = - \frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \right], \quad \hat{y}_i = P(y_i=1 \mid x_i)
$$

Multiple models $\mathcal{M} = \{ \text{LogReg}, \text{NB}, \text{KNN}, \text{SVM}, \text{DT}, \text{RF}, \text{GB}, \text{NN} \}$ are evaluated to find the optimal predictor.

---

## Dataset

- **Source:** [UCI MAGIC Gamma Telescope Dataset](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)  
- **Features:** 10 numerical features derived from telescope observations  
- **Labels:**  
  - $y=1$ → gamma-ray (signal)  
  - $y=0$ → hadron (background)

---

## Problem Formulation

1. **Standardization (Z-score normalization):**

$$
z_i = \frac{x_i - \mu}{\sigma}
$$

where $(\mu, \sigma)$ are the mean and standard deviation of each feature.

2. **Optional Oversampling:**  
If `oversample=True`, the minority class is oversampled using `RandomOverSampler`.

3. **Model Training:**  
Each model $f_\theta^{(k)}$ is trained to minimize $\mathcal{L}$ over the training set and evaluated on validation/test data.

---

## Methods

- **Logistic Regression:**  

$$
P(y=1 \mid X) = \sigma\Big((\beta_0) + \sum_i (\beta_i)x_i \Big), \quad \hat{y} = \mathbf{1}\{P(y=1\mid X) \ge 0.5\}
$$

- **Naive Bayes:**  

$$
P(y \mid X) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)
$$

- **K-Nearest Neighbors:**  

$$
\hat{y} = \arg\max_{c \in C} \sum_{i \in \text{NN}(X)} \mathbf{1}(y_i = c)
$$

- **Support Vector Classifier (with probability=True):**  

$$
f(X) = \sum_i \alpha_i y_i K(X_i, X) + (\beta_0), \quad P(y=1\mid X) = \frac{1}{1 + e^{-f(X)}}
$$

- **Decision Tree:** Recursive partitioning to minimize impurity (Gini or Entropy):

$$
G = 1 - \sum_c p_c^2, \quad H = - \sum_c p_c \log_2(p_c)
$$

- **Random Forest:** Ensemble of $T$ trees with majority vote:

$$
\hat{y} = \text{mode}\{\hat{y}_1, \hat{y}_2, ..., \hat{y}_T\}
$$

- **Gradient Boosting:** Sequential additive trees:

$$
F_t(x) = F_{t-1}(x) + \eta h_t(x), \quad \hat{y} = \sigma(F_T(x))
$$

- **Neural Network (MLP):**  

$$
h_j = \text{ReLU}\Big(\sum_i x_i (\beta_{ij}) + (\beta_{0j})\Big), \quad
\hat{y} = \sigma\Big(\sum_j h_j (\beta_j) + (\beta_0)\Big)
$$

---

## Evaluation Metrics

- **Accuracy:**  

$$
\text{Accuracy} = \frac{1}{m} \sum_i \mathbf{1}(\hat{y}_i = y_i)
$$

- **Precision, Recall, F1-score:**

$$
\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad
\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \quad
F1_c = 2 \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}
$$

- **ROC-AUC:**

$$
\text{TPR} = \frac{TP}{TP+FN}, \quad
\text{FPR} = \frac{FP}{FP+TN}, \quad
\text{ROC-AUC} = \int_0^1 \text{TPR(FPR)}\, d(\text{FPR})
$$

---

## Results Visualization

- **Confusion Matrix**  
- **ROC and Precision-Recall Curves**  
- **Training Loss and Accuracy Curves** (for Neural Network)  
- **Comparison of all models** via bar plots of Accuracy, F1-score, and ROC-AUC

---

## Usage

```bash
# Clone the repository
git clone https://github.com/huraira-hstu/Edge-Project.git
cd Edge-Project

# Install requirements
pip install -r requirements.txt

# Run the pipeline in Colab or local Jupyter
