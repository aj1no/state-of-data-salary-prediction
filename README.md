# Salary Range Prediction of Data Professionals in Brazil using Machine Learning

*[Ler em Português](README.pt-br.md)*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/aj1no/state-of-data-salary-prediction/ci.yml?branch=main&label=CI&style=flat-square)](https://github.com/aj1no/state-of-data-salary-prediction/actions)

* **Institution:** Faculty of Technology of Jundiaí (FATEC Jundiaí)  
* **Course:** Computational Intelligence  
* **Professor:** Mateus Guilherme Fuini, M.Sc.  
* **Authors:**
    * Rodolfo Vinicius Cima Takemoto
    * Tiago Galhardo Avelar

---

## 1. Problem Description

In the contemporary corporate landscape, Information Technology — with emphasis on Data Science, Data Engineering, and Business Intelligence — has experienced unprecedented expansion. However, labor pricing and the definition of roles and salaries in Brazil face distortions due to geographic variations, contract types (CLT vs. PJ), organizational sizes, and the diversity of tech stacks required by companies. 

This project addresses the problem of predicting the salary range of data professionals in Brazil from the perspective of Supervised Machine Learning (Classification). Understanding which structural career factors (such as seniority, years of experience, and number of mastered technologies) most heavily influence salary ranges allows professionals to better guide their careers and HR departments to calibrate their commercial hiring offers.

---

## 2. Dataset Description

The project uses the **State of Data Brazil 2024-2025** dataset, the most comprehensive survey on the data job market in the country, promoted by the Data Hackers community in partnership with Bain & Company. 
* **Data Volume:** The raw file originally has 5,217 respondents and 403 variables/columns.
* **Content:** The survey encompasses demographic info (gender, age, location), academic background (education level, major), professional situation (role, seniority, work model, sector, and company size), and boolean questions about the daily use of specific technologies (databases, programming languages, and BI tools).
* **Base File:** The dataset is loaded from the CSV extracted and stored in `data/raw/`.

---

## 3. Objective

The general objective of this project is to build, optimize, and evaluate a complete supervised Machine Learning pipeline to classify the salary range of data professionals in Brazil into three categories: Low salary range, Medium salary range, and High salary range.

---

## 4. Methodology

Development followed the classic approach for Data Science projects:
1. **Feature Selection:** Due to the high number of columns, we filtered and kept the most explanatory and structural career variables, reducing the predictor columns to a set of 12 core variables.
2. **Encoding and Null Treatment:** Text columns were cleaned of encoding artifacts (such as corrupted accents on load) to ensure categorical mappings function properly. Records with missing target variables were discarded.
3. **Feature Engineering:** Creation of new derived variables to enrich the context provided to the model (including `Trabalha_Remoto`, `Experiencia_Categoria`, and `Perfil_Tecnico_Qtd`).
4. **Outlier Analysis and Visualizations:** Exploratory identification of outliers using the IQR method and generation of descriptive visualizations (stack histogram, seniority and experience countplots, boxplot of technology count per range, and exploratory Pearson correlation heatmap with temporarily ordinalized binary/ordinal variables).
5. **Structured Partitioning:** Stratified train/test split to maintain class balance.
6. **Preprocessing and Capping Pipelines:** Use of `ColumnTransformer` and `Pipeline` to apply imputation, robust outlier treatment by capping (via the custom `IQRCapper` transformer), numerical scaling (`StandardScaler`), and categorical variable encoding (`OneHotEncoder`). All preprocessing is encapsulated in the pipeline to avoid data leakage.
7. **Optimization and Validation:** Hyperparameter tuning of a Logistic Regression classifier via `GridSearchCV` (testing different `C` regularization values and solvers like `lbfgs` and `newton-cg` with 1000 max iterations and `error_score="raise"`) associated with a `StratifiedKFold` cross-validation scheme.

---

## 5. Target Variable Mapping and Data Leakage

The original `2.h_faixa_salarial` column has 13 declared salary ranges. To enable robust and balanced classification, we created a simplified target variable mapped as follows:

| Original Monthly Income Range | Simplified Target Class |
| :--- | :--- |
| *Less than R$ 1.000/month* to *from R$ 4.001/month to R$ 6.000/month* | **Low salary range** (up to R$ 6,000) |
| *from R$ 6.001/month to R$ 8.000/month* to *from R$ 8.001/month to R$ 12.000/month* | **Medium salary range** (from R$ 6,001 to R$ 12,000) |
| *from R$ 12.001/month to R$ 16.000/month* to *Above R$ 40.001/month* | **High salary range** (above R$ 12,000) |

### Preventing Data Leakage
To prevent information leakage that causes false optimism in model accuracy:
1. **Discarding Direct Leakers:** Removed columns of secondary gross compensation, additional bonuses, benefits, and salary expectations prior to modeling.
2. **Transformer Encapsulation:** Numerical scaling (`StandardScaler`) and categorical encoding (`OneHotEncoder`) calculations were encapsulated in the `Pipeline`. The fit is restricted to training data, and the test set is treated strictly blindly with inference transformations (`transform`).

---

## 6. Feature Engineering

We created three derived variables to train the classifier:

1. **`Experiencia_Categoria`:** Logical grouping of years of experience in the data field into three levels (`Up to 2 years`, `3 to 6 years`, `More than 6 years`).
2. **`Trabalha_Remoto`:** Binary attribute (`Yes`/`No`) derived from the current work model. Indicates if the professional works 100% remotely.
3. **`Perfil_Tecnico_Qtd`:** Sum of boolean markers of daily used technologies (section 4 of the survey). It is a continuous variable that measures the breadth of tools mastered by the professional.

---

## 7. Model Results

The selected model was **Logistic Regression** optimized by grid search.

* **Best Hyperparameters Found:** `{'model__C': 1.0, 'model__solver': 'lbfgs', 'model__max_iter': 1000}`
* **Mean Cross-Validation Accuracy (5-Fold):** 73.47% (Standard Deviation: 1.69%)
* **Final Accuracy on Test Set:** 72.46%

### Final Classification Report (Test)

| Salary Class | Precision | Recall | F1-Score | Support (Instances) |
| :--- | :---: | :---: | :---: | :---: |
| **High salary range** | 0.76 | 0.82 | 0.79 | 368 |
| **Low salary range** | 0.81 | 0.73 | 0.77 | 258 |
| **Medium salary range** | 0.63 | 0.62 | 0.62 | 347 |
| *Overall Accuracy* | | | **0.72** | **973** |

### Confusion Matrix

```
                     Predicted Class
                    Low     Medium  High
Actual   Low     [   189,     63,     6 ]
Class    Medium  [    43,    216,    88 ]
         High    [     2,     66,   300 ]
```

---

## 8. Conclusions and Limitations

### Academic Conclusions
* The classifier presented solid performance with an accuracy of 72.46% on test data.
* Extreme ranges (High and Low) obtained the best classification indexes (F1-score of 79% and 77% respectively), demonstrating that early-career and senior/managerial data professionals have distinct tech stacks and profiles.
* The Medium salary range proved to be the most complex to classify (F1-score of 62%), which is consistent with the Brazilian market, where mid-level professionals in small companies often have compensation close to junior professionals in large multinationals.

### Study Limitations
* The dataset is based on voluntary self-declared answers, which may present self-reporting and sampling biases.
* Logistic regression imposes linear and additive relationships on the coefficients, preventing direct representation of complex second-degree interactions (for example, the combined impact of living in the Southeast AND mastering cloud computing).

### Future Improvements
1. Experiment with tree-based models (e.g., Random Forest, Gradient Boosting, or XGBoost) that naturally capture non-linear relationships and interactions.
2. Attribute qualitative weights to technologies rather than just summing them (e.g., mastering Spark/Kubernetes has a larger salary impact than mastering Excel alone).
3. Evaluate class balancing techniques if data variance increases in future editions.

---

## Running on Google Colab with KaggleHub

The project was adapted to allow automated execution in both local environments and Google Colab. Data import occurs robustly and dynamically following the flow below:

1. **Automatic Download:** The notebook attempts to download the dataset directly from Kaggle using the official slug `datahackers/state-of-data-brazil-20242025` via the `kagglehub` library.
2. **Authentication:** If Kaggle requests authentication, the user can configure their credentials (Kaggle API token). Otherwise, the library will download anonymously if allowed, or present a controlled error.
3. **Manual Upload Fallback:** If the automatic download by `kagglehub` fails for any reason (connectivity loss, lack of credentials, or pending terms), the notebook triggers a secondary manual upload routine in Google Colab via `files.upload()`. The user will need to upload the corresponding survey CSV file.
4. **GitHub Storage:** Thanks to dynamic loading by API or upload on demand, the raw dataset CSV file does not need to (and should not) be stored in the GitHub repository.
