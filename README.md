# Classification of Alzheimer's Disease from Tabular Clinical Data

**A Comparison of Machine Learning, Deep Learning, Pretrained Tabular Models and Explainability**

PHYS 540 - Big Data Physics, Fall 2026
Author: Hazem Abdelhakim

---

## Overview

This project benchmarks 18 models from five families on the public Alzheimer's Disease Dataset (2,149 patients, 32 demographic, lifestyle, medical-history, clinical, cognitive and symptom features) for binary classification of Alzheimer's disease (AD vs. No AD). All models are evaluated under one leakage-free protocol: stratified 10-fold cross-validation, with scaling and hyperparameter tuning performed inside each training fold, and no resampling or feature selection.

The study also applies a corrected paired significance test across folds, explains the predictions with SHAP and LIME, and compares the results with four published studies that use the same dataset.

**Important note:** the dataset is synthetic and its dominant predictors (MMSE, ADL, functional assessment) overlap with the diagnostic criteria themselves. The results are a methodological comparison and are not evidence for clinical deployment.

![Project workflow](Mindmap.png)

---

## Models

| Family | Models |
|---|---|
| Classical ML | KNN, Logistic Regression, Gaussian NB, Bernoulli NB, Linear SVC, Decision Tree |
| Ensembles | Random Forest, AdaBoost, Gradient Boosting, XGBoost, Hard Voting, Soft Voting |
| Proposed ensemble | LightGBM + Random Forest soft voting (hyperparameters from Mahamud et al. [3]) |
| Neural networks (from scratch) | ANN, ALTARN (attention residual network), SNN (spiking neural network) |
| Pretrained tabular foundation models | TabPFN v2, TabICL |

---

## Main Results

Stratified 10-fold cross-validation, mean over folds (precision, recall and F1 refer to the AD class).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| TabICL | 0.9535 | 0.9466 | 0.9211 | 0.9333 | 0.9556 |
| Proposed (LGBM + RF) | 0.9530 | 0.9436 | 0.9224 | 0.9327 | 0.9528 |
| TabPFN v2 | 0.9530 | 0.9474 | 0.9184 | 0.9324 | 0.9521 |
| XGBoost | 0.9516 | 0.9448 | 0.9171 | 0.9306 | 0.9504 |
| Gradient Boosting | 0.9488 | 0.9368 | 0.9184 | 0.9270 | 0.9508 |
| Decision Tree | 0.9479 | 0.9383 | 0.9132 | 0.9251 | 0.9441 |
| ALTARN | 0.8576 | 0.8471 | 0.7316 | 0.7843 | 0.9131 |
| ANN | 0.8492 | 0.8322 | 0.7197 | 0.7711 | 0.9042 |
| SNN | 0.8283 | 0.7725 | 0.7303 | 0.7501 | 0.8894 |
| KNN | 0.7678 | 0.7432 | 0.5250 | 0.6135 | 0.7798 |

The full table with all 18 models is in the report and in the saved result files.

### Key findings

- The top group (TabICL, the proposed LGBM + RF ensemble, TabPFN v2, XGBoost, Gradient Boosting, Decision Tree) reaches about 0.95 accuracy and 0.93 F1. A Nadeau-Bengio corrected paired t-test with Holm adjustment shows that this group is statistically tied.
- Linear models, Naive Bayes, KNN and all three neural networks trained from scratch are significantly worse than the best model.
- TabPFN v2 and TabICL reach the top group without any hyperparameter tuning.
- SHAP and LIME show that predictions rest almost entirely on five features: FunctionalAssessment, ADL, MemoryComplaints, MMSE and BehavioralProblems. These agree with the four published studies.
- The ALTARN re-implementation reproduces the original paper's F1 (0.784 vs. 0.783). The 97% Random Forest result reported by Khalid et al. could not be reproduced.

---

## Repository Structure

```
.
|-- AD.ipynb                      # Full pipeline: all code in one notebook
|-- AD_I_EDA.ipynb                # Step I:   exploratory data analysis
|-- AD_II_ML.ipynb                # Step II:  machine learning models
|-- AD_III_DL.ipynb               # Step III: deep learning and pretrained models
|-- AD_IV_Statistical.ipynb       # Step IV:  comparison, significance tests, explainability
|-- alzheimers_disease_data.csv   # Dataset
|-- ml_results.pkl                # Saved ML results (metrics, fold scores, predictions)
|-- dl_results.pkl                # Saved DL results (metrics, fold scores, predictions)
|-- AD_latex.zip                  # LaTeX source of the report and figures
|-- Report.pdf                    # Final report
|-- Mindmap.png                   # Project workflow diagram
|-- README.md
```

Adjust the file names above to match the exact names of the files you upload.

---

## Notebooks

| Notebook | Content |
|---|---|
| `AD.ipynb` | Complete pipeline containing all code from data loading to explainability. |
| `AD_I` (EDA) | Dataset description, class balance, categorical and numerical distributions, outliers, correlation analysis, cleaned CSV export. |
| `AD_II` (ML) | 13 machine learning models with nested cross-validation, classification reports, confusion matrices, ROC curves, paired t-test, saves `ml_results.pkl`. |
| `AD_III` (DL) | ANN, ALTARN, TabPFN v2, TabICL and SNN with 10-fold CV, training curves, SNN spiking-activity animation, saves `dl_results.pkl`. |
| `AD_IV` (Statistical) | Merges ML and DL results, full comparison table, corrected paired t-test with Holm adjustment, ROC and precision-recall curves, SHAP, LIME and cross-model importance agreement. |

Run order when using the separate notebooks: I, II, III, IV. Notebook IV loads `ml_results.pkl` and `dl_results.pkl`.

---

## Methodology Summary

- **Data:** 2,149 patients, 32 input features (15 numerical, 17 categorical or binary), no missing values, 64.6% No AD and 35.4% AD. `PatientID` and `DoctorInCharge` are removed.
- **Preprocessing:** z-score standardisation of the 15 numerical features, fitted inside each training fold. No SMOTE and no feature selection.
- **Evaluation:** stratified 10-fold cross-validation (`random_state=42`), identical for all 18 models. Accuracy, precision, recall, F1 and ROC-AUC per fold. Hyperparameters of KNN, Linear SVC and Decision Tree are tuned with 5-fold `GridSearchCV` (F1) inside each training fold.
- **Neural networks:** Adam (1e-3), early stopping (patience 10), ReduceLROnPlateau, 15% validation split per training fold.
- **Significance testing:** Nadeau-Bengio corrected resampled t-test on per-fold F1, with Holm correction for multiple comparisons.
- **Explainability:** SHAP (permutation explainer on 300 patients) and LIME for the proposed model; Tree SHAP for Random Forest and XGBoost.

---

## Installation

Python 3.10 or later is recommended. A GPU is optional; it is used for the SNN and the pretrained tabular models when available.

```bash
git clone https://github.com/Hazem-355/Alzheimers-Tabular-Benchmark/
cd Alzheimers-Tabular-Benchmark

pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels \
            xgboost lightgbm tensorflow torch snntorch tabpfn tabicl \
            shap lime pydot
```

Graphviz is required for the model architecture plots:

```bash
# Ubuntu / Debian
sudo apt-get install graphviz
```

Note: the notebooks were developed in Google Colab. Newer `tabpfn` releases default to a later model version, so TabPFN v2 is requested explicitly in the code.

---

## Usage

1. Place `alzheimers_disease_data.csv` in the same folder as the notebooks.
2. Open the notebooks in Jupyter or Google Colab.
3. Either run `AD.ipynb` end to end, or run the four step notebooks in order (I, II, III, IV).
4. The step notebooks write `alzheimers_disease_cleaned.csv`, `ml_results.pkl` and `dl_results.pkl`, which later steps read.

The full run (10-fold CV for 18 models plus SHAP) takes a considerable time; the SHAP permutation explainer on 300 patients alone takes several minutes.

---

## Report

The full written report is in `Report.pdf`, and its LaTeX source is in `AD_latex.zip`. It covers the dataset, methodology, results for all models, statistical tests, explainability, a comparison with the published studies, discussion and limitations.

---

## Limitations

- The dataset is synthetic. For example, Age, an established real-world risk factor, has almost no importance here, which likely reflects how the data were generated.
- MMSE, ADL and the functional assessment are part of the clinical work-up used to diagnose AD, so the task is closer to confirming a diagnosis than to detecting the disease early.
- SHAP and LIME describe what the models learned, not the causes of the disease. SHAP values for the proposed model are an approximation on 300 patients, and LIME was inspected for only two patients.
- Neural network hyperparameters were not tuned, and a single dataset with a single random seed was used.
- Comparison with the literature is indicative only, since protocols (hold-out vs. cross-validation, SMOTE, feature selection) differ between studies.

---

## References

[1] R. El Kharoua, "Alzheimer's Disease Dataset," Kaggle, 2024. DOI: [10.34740/KAGGLE/DSV/8668279](https://doi.org/10.34740/KAGGLE/DSV/8668279).

[2] A. A. Soladoye, N. Aderinto, D. Osho, and D. B. Olawade, "Explainable machine learning models for early Alzheimer's disease detection using multimodal clinical data," *International Journal of Medical Informatics*, vol. 204, 106093, 2025. DOI: [10.1016/j.ijmedinf.2025.106093](https://doi.org/10.1016/j.ijmedinf.2025.106093).

[3] E. Mahamud, M. Assaduzzaman, J. Islam, N. Fahad, M. J. Hossen, and T. T. Ramanathan, "Enhancing Alzheimer's disease detection: An explainable machine learning approach with ensemble techniques," *Intelligence-Based Medicine*, vol. 11, 100240, 2025. DOI: [10.1016/j.ibmed.2025.100240](https://doi.org/10.1016/j.ibmed.2025.100240).

[4] M. Z. Khalid, N. Iqbal, B. Ali, J. S. Ur Rahman, S. Iqbal, L. Almudaimeegh, Z. Y. Hamd, and A. Gareeballah, "Detection and Classification of Alzheimer's Disease Using Deep and Machine Learning," *Tomography*, vol. 12, no. 1, p. 4, 2026. DOI: [10.3390/tomography12010004](https://doi.org/10.3390/tomography12010004).

[5] K. Balakrishna, A. Hammond, and A. El Idrissi, "ALTARN: A Tabular Residual Neural Network for Alzheimer's Disease Classification and Prediction," *medRxiv*, 2025. DOI: [10.1101/2025.10.21.25338451](https://doi.org/10.1101/2025.10.21.25338451).

[6] N. Hollmann, S. Muller, L. Purucker, A. Krishnakumar, M. Korfer, S. B. Hoo, R. T. Schirrmeister, and F. Hutter, "Accurate predictions on small data with a tabular foundation model," *Nature*, vol. 637, pp. 319-326, 2025. DOI: [10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6).

[7] J. Qu, D. Holzmuller, G. Varoquaux, and M. Le Morvan, "TabICL: A Tabular Foundation Model for In-Context Learning on Large Data," in *Proc. 42nd International Conference on Machine Learning*, vol. 267, pp. 50817-50847, 2025. [Paper](https://proceedings.mlr.press/v267/qu25d.html).

[8] J. K. Eshraghian, M. Ward, E. Neftci, X. Wang, G. Lenz, G. Dwivedi, M. Bennamoun, D. S. Jeong, and W. D. Lu, "Training Spiking Neural Networks Using Lessons From Deep Learning," *Proceedings of the IEEE*, vol. 111, no. 9, pp. 1016-1054, 2023. DOI: [10.1109/JPROC.2023.3308088](https://doi.org/10.1109/JPROC.2023.3308088).

[9] C. Nadeau and Y. Bengio, "Inference for the generalization error," *Machine Learning*, vol. 52, pp. 239-281, 2003.

[10] S. Holm, "A simple sequentially rejective multiple test procedure," *Scandinavian Journal of Statistics*, vol. 6, no. 2, pp. 65-70, 1979.

[11] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems 30 (NeurIPS)*, 2017.

[12] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?': Explaining the predictions of any classifier," in *Proc. 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016, pp. 1135-1144.

---

## Author

Hazem Abdelhakim
