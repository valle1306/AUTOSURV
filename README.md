# AUTOSurv: Deep Learning Survival Model with Radiomics Data

## 📌 What is this project?

This project adapts the **AUTOSurv framework**—originally developed for genomic survival analysis—to work with **radiomics features** extracted from medical imaging. It is designed to predict patient survival outcomes using a deep learning pipeline that integrates **dimensionality reduction, survival modeling**, and **model interpretability**.

The model is trained and evaluated on clinical radiomic data from brain tumor patients, with the aim of identifying non-invasive biomarkers and improving personalized survival prediction.

---

## 💡 Why did we build it?

Medical imaging provides rich, non-invasive information that can complement or replace invasive molecular tests for cancer prognosis. However, radiomics data is often:

- **High-dimensional**
- **Correlated or redundant**
- **Heterogeneous with missing values**

This project addresses those challenges by:

- Leveraging **KLPMVAE** to compress high-dimensional, partially missing radiomics features
- Using **LFSurv**, a neural network that directly optimizes the Cox partial likelihood for time-to-event prediction
- Applying **DeepSHAP** to interpret the contribution of latent features to survival risk

Ultimately, the goal is to provide clinicians with **interpretable, accurate, and scalable survival models** using radiomics alone.

---

## ⚙️ How does it work?

### 🧱 Architecture Overview

1. **Input**: Radiomics features + optional clinical covariates
2. **Dimensionality Reduction**: 
   - KLPMVAE (Kullback-Leibler Partial-Multiview Variational Autoencoder)
   - Handles missing views, learns compact and disentangled latent representations
3. **Survival Modeling**:
   - LFSurv: Feedforward neural net with ReLU hidden layers and Cox loss
   - Directly models hazard function without proportionality assumption enforcement
4. **Interpretability**:
   - DeepSHAP applied to the LFSurv output to assess importance of latent and original features
   - Generates patient-level and cohort-level SHAP explanations

### 🧪 Data Source

- Radiomics features from tumor imaging (e.g., from TCIA datasets)
- Outcome: Time-to-event (overall survival or progression-free survival)
- Censoring handled through Cox partial likelihood loss

---

## 🎯 What did we achieve?

- ✅ Successfully adapted AUTOSurv pipeline to radiomics data
- ✅ Reduced overfitting and improved generalization with KLPMVAE latent features
- ✅ Achieved **C-index up to 0.74** (example metric — plug yours in here)
- ✅ Provided interpretable SHAP plots for clinical insights
- ✅ Demonstrated robustness in presence of missing data and high dimensionality

---

## 📈 Key Features

- 🧠 Variational autoencoder for sparse high-dimensional imaging data
- ⏳ Survival prediction with censored data handling
- 🔍 DeepSHAP interpretability for clinical transparency
- 🧪 Support for missing views in radiomic features
- 📊 C-index and Brier Score evaluation

---

## 📚 Technologies Used

- Python (PyTorch, NumPy, Pandas)
- Lifelines / Scikit-survival
- DeepSHAP
- Matplotlib / Seaborn

---

## 📌 Future Directions

- Integrate clinical features alongside radiomics
- Compare performance against traditional Cox + Lasso models
- Expand to multi-site radiomics datasets (e.g., lung, breast)
- Add time-varying covariates and longitudinal imaging
