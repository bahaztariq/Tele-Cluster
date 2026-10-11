# Tele-Cluster

Customer segmentation and churn prediction for a telecommunications company.

From a historical customer dataset, the project builds two things: distinct customer
profiles (clustering) and a model that predicts which customers are likely to cancel their
subscription (classification). It then checks whether the profiles help the prediction,
tunes the models, tracks every experiment in MLflow, and serves the result in a Streamlit
app packaged with Docker.

## Project goals

1. **Data analysis and preparation**: EDA, missing values, duplicates, outliers,
   correlations, encoding, scaling, feature engineering.
2. **Clustering**: K-Means with Elbow and Silhouette to choose *k*; comparison with
   Agglomerative and DBSCAN (Silhouette, Davies-Bouldin, Calinski-Harabasz); PCA
   visualisation; profile interpretation.
3. **Classification**: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost,
   evaluated with precision, recall, F1, ROC-AUC and the confusion matrix; class imbalance
   handled with `class_weight`, Random Oversampling and SMOTE.
4. **Clustering + classification**: add the `cluster` as a feature and compare
   *initial variables* vs *initial variables + cluster*.
5. **Optimization**: cross-validation, `GridSearchCV` / `RandomizedSearchCV`, before vs
   after, contribution of feature engineering and of the cluster.
6. **MLflow**: hyperparameters, metrics, model, preprocessing parameters and artifacts for
   every experiment, and the final model in the Model Registry.
7. **Deployment**: Streamlit interface (cluster, churn prediction, churn probability),
   containerized with Docker.

## Dataset

IBM Telco Customer Churn (Kaggle): 7043 customers, 21 columns, in
`data/raw/wa-fn-usec-telco-customer.csv`.

| Group | Columns |
| --- | --- |
| Identification | `customerID` |
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Services | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| Billing | `MonthlyCharges`, `TotalCharges` |
| Target | `Churn` (26.5% Yes) |

The brief also mentions age, usage, support calls, complaints and satisfaction. These
columns do not exist in this dataset, so they are not modelled.

## Repository layout

```
app/
  config.py              Paths and constants
  extractor.py           DataExtractor: load the CSV, quality report
  transformer.py         DataTransformer: feature engineering, imputation, IQR clipping,
                         one-hot encoding, scaling, optional cluster feature
  eda.py                 Step 1: data quality and EDA figures
  clustering.py          Step 2: choice of k, K-Means vs Agglomerative vs DBSCAN, PCA, profiles
  classification.py      Steps 3-6: models x imbalance strategies, feature sets (with and
                         without cluster), tuning, final model, MLflow
  main.py                Runs eda -> clustering -> classification
  streamlit_app.py       Step 7: Streamlit interface
data/raw/                Source CSV
models/
  segmentation.joblib / .json    Clustering pipeline and segment profiles
  churn.joblib / .json           Final churn model, its parameters and test metrics
Dockerfile, docker-compose.yml   Streamlit app + MLflow UI
```

Generated figures and tables go to `reports/` (`eda/`, `clustering/`, `classification/`),
and MLflow runs to `mlflow.db` and `mlruns/`. None of these are committed.

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Pipeline:** retrains every model in `models/` and logs all runs to MLflow (about 25-30
minutes; the churn classification alone takes about 23, most of it in the hyperparameter
search):

```bash
python -m app.main
```

**Streamlit:**

```bash
streamlit run app/streamlit_app.py      # http://localhost:8501
```

**MLflow UI:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db   # http://localhost:5000
```

**Docker:** one command starts the Streamlit app (with the trained models from `models/`)
and the MLflow UI (official `ghcr.io/mlflow/mlflow` image, reading `mlflow.db` and
`mlruns/`). Run it from the project root:

```bash
docker compose up --build    # Streamlit http://localhost:8501, MLflow http://localhost:5000
docker compose down          # stop both
```

## Methodology

- **No target in the clustering.** `Churn` is never an input to the segmentation. It is
  only used afterwards, to describe the churn rate of each profile.
- **No data leakage.** Imputation, outlier clipping, scaling, encoding, the K-Means used for
  the `cluster` feature, and SMOTE / Random Oversampling all live inside one `imblearn`
  pipeline. They are fitted on the training data only (on each training fold during
  cross-validation), and the samplers never touch validation or test rows.
- **Model selection uses the training set only** (5-fold stratified cross-validation, F1).
  The 20% test set is only used to report the final scores.
- **`customerID` is dropped.** It is a unique key with no predictive value.
- **`TotalCharges`** is read as text because 11 rows are blank (customers with tenure 0).
  It is converted to numbers, and the blanks are imputed inside the pipeline.

## Results

### 1. Data analysis

- 7043 customers. 22 duplicated profiles are removed, leaving 7021. 11 `TotalCharges`
  values are missing. No IQR outliers in `tenure`, `MonthlyCharges` or `TotalCharges`.
- The target is imbalanced: 26.5% churn, so accuracy is not used; models are compared on
  precision, recall, F1 and ROC-AUC.
- Main churn drivers:

  | Variable | Churn rate |
  | --- | --- |
  | Contract | month-to-month 43%, one year 11%, two year 3% |
  | Internet | fiber optic 42%, DSL 19%, none 7% |
  | Payment | electronic check 45%, other methods 15-19% |
  | Tech support / online security | without 42%, with 15% |
  | Senior citizen | 42% vs 24% |

  Churners have shorter tenure (18 vs 38 months on average) and higher monthly charges
  (75 vs 61).
- 7 features are engineered: number of services, average monthly spend, charges per
  service, month-to-month flag, automatic payment, family, tenure group.

### 2. Customer segments

Clustering uses all customer features, and *k* is picked automatically by Silhouette
(k = 2…8), which gives **k = 3**:

| Segment | Customers | Churn rate |
| --- | --- | --- |
| Low risk - No internet, Two year | 1513 | 7% |
| Medium risk - Fiber optic, Two year | 2435 | 14% |
| High risk - Fiber optic, Month-to-month | 3073 | 46% |

K-Means beats Agglomerative and DBSCAN on Silhouette (0.227 vs 0.207 / 0.217) and
Calinski-Harabasz. Churn was not used to build the segments; it is only used here to
describe them.

- **Low risk: phone-only customers** (22%). None have internet, so they average 1.2
  services and pay about 21/month. Two-thirds are on a one- or two-year contract, and few
  pay by electronic check (8%). They are cheap to serve and rarely leave (7% churn).
  *Action:* keep them stable, and offer internet bundles carefully, since adding services
  is what makes the other segments more expensive and more volatile.
- **Medium risk: loyal, fully-equipped customers** (35%). They have been customers for
  about 4.5 years and have the most services (6.5 on average, 60% with tech support or
  online security, 71% streaming). They also pay the most (about 87/month) and are mostly
  on one- or two-year contracts with automatic payment (65%) and families (74%). Churn is
  low (14%), and most of it comes from the fiber customers in this group (20% vs 5% on DSL).
  *Action:* these are the most valuable customers. Retain them with loyalty rewards and
  contract renewals, and keep an eye on fiber service quality.
- **High risk: new month-to-month customers** (44%). 93% are month-to-month and 54%
  joined in the last 12 months (average tenure about 16 months). Few have tech support or
  security (about 20%), and over half pay by electronic check. Their bill is high for the
  few services they use (about 69/month for 3.7 services). Nearly half leave (46%), rising
  to 58% for those on fiber.
  *Action:* this is the priority retention target. Offer discounts for moving to a 1- or
  2-year contract, switch them to automatic payment, bundle tech support and security
  during onboarding, and review fiber pricing.

### 3. Classification and class imbalance

Five models x four imbalance strategies, 5-fold CV on the training set
(`reports/classification/model_comparison.csv`):

| Model | Best strategy | CV recall without rebalancing | CV recall with it | CV F1 |
| --- | --- | --- | --- | --- |
| Random Forest | class_weight | 0.51 | 0.76 | 0.634 |
| XGBoost | Random Oversampling | 0.53 | 0.76 | 0.627 |
| Logistic Regression | SMOTE | 0.53 | 0.79 | 0.627 |
| SVM | class_weight | 0.48 | 0.79 | 0.626 |
| Decision Tree | class_weight | 0.48 | 0.81 | 0.612 |

Without rebalancing, about half of the churners are missed. Every strategy raises recall
(to 0.61-0.81, with SMOTE helping the tree ensembles least) and improves F1; ROC-AUC barely
moves, because rebalancing shifts the decision
threshold rather than ranking customers better. **Rebalancing matters more than the choice
of model.**

### 4. Cluster as a feature

Each model, with its best strategy, on four feature sets (CV F1,
`feature_set_comparison.csv`):

| Model | Raw | Raw + cluster | Raw + FE | Raw + FE + cluster |
| --- | --- | --- | --- | --- |
| Random Forest | **0.637** | 0.633 | 0.634 | 0.633 |
| Logistic Regression | 0.630 | **0.633** | 0.627 | 0.626 |
| XGBoost | 0.632 | **0.632** | 0.627 | 0.632 |
| SVM | 0.624 | 0.624 | 0.626 | **0.626** |
| Decision Tree | 0.612 | 0.612 | 0.612 | **0.615** |

All differences are within ±0.005 F1, i.e. within fold-to-fold noise. The cluster and the
engineered features summarize information the models already get from contract, internet
type, tenure and charges. **The cluster adds interpretability, not predictive power.**

### 5. Optimization

Each model's best configuration is tuned on CV F1: `GridSearchCV` for Logistic Regression,
Decision Tree and SVM (small search spaces), `RandomizedSearchCV` (20 samples) for Random
Forest and XGBoost (`optimisation.csv`, `optimisation.png`):

| Model | Search | CV F1 before | CV F1 after | Best parameters |
| --- | --- | --- | --- | --- |
| Random Forest | random | 0.637 | **0.639** | 370 trees, max_depth 19, min_samples_leaf 16, max_features 0.28 |
| XGBoost | random | 0.632 | 0.637 | 317 trees, max_depth 3, learning_rate 0.019, subsample 0.78 |
| Logistic Regression | grid | 0.633 | 0.635 | C = 3 |
| SVM | grid | 0.626 | 0.626 | C = 1, gamma = scale (the defaults) |
| Decision Tree | grid | 0.615 | 0.625 | max_depth 5, min_samples_leaf 50 |

Tuning helps the Decision Tree most (shallower tree, less overfitting) and brings small
gains elsewhere.

### 6. Final model

The tuned models on the held-out test set (`test_comparison.csv`; confusion matrices and ROC
curves in `test_comparison.png`):

| Model | Imbalance | Features | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| **Random Forest** | class_weight | raw | 0.525 | **0.782** | **0.629** | 0.842 |
| XGBoost | Random Oversampling | raw + cluster | 0.516 | 0.774 | 0.619 | **0.843** |
| Logistic Regression | SMOTE | raw + cluster | 0.513 | 0.761 | 0.613 | 0.839 |
| SVM | class_weight | raw + FE + cluster | **0.611** | 0.616 | 0.613 | 0.827 |
| Decision Tree | class_weight | raw + FE + cluster | 0.512 | 0.753 | 0.609 | 0.827 |

The final model is the one with the best CV F1: the **tuned Random Forest with
`class_weight`** on the raw features. On the test set it reaches **recall 0.78, precision
0.53, F1 0.63, ROC-AUC 0.84**: it catches about 4 out of 5 churners, with roughly one false
alarm per churner caught, which is acceptable for a retention campaign.

SVM's test recall is lower than its CV recall (0.79) because the test metrics use
`predict_proba >= 0.5`, and SVM's probability calibration partly undoes the class weighting.

## MLflow

Two experiments, stored in `mlflow.db` / `mlruns/`:

- **`tele-cluster-segmentation`**: one run per clustering algorithm, with its parameters,
  the three cluster metrics, the elbow/silhouette and PCA figures, the segment profiles and
  the fitted pipeline.
- **`tele-cluster-churn`**: one run per configuration (model x strategy, model x feature
  set) and per tuned model. Each run logs the model hyperparameters, the preprocessing
  parameters (feature engineering, cluster feature, imputation, outlier clipping, scaling,
  encoding), the imbalance strategy, the CV and test metrics, and the fitted pipeline.
  The final run also holds the test comparison, the before/after tuning figure and the
  classification report.

The final model is registered in the Model Registry as **`tele-cluster-churn-app`**.

## Streamlit app

`app/streamlit_app.py` shows a form with the customer's 19 characteristics and returns:

- **the customer's segment** (cluster and profile name),
- **the churn prediction** (will churn / will stay),
- **the churn probability**.

It uses `models/segmentation.joblib` and `models/churn.joblib`.
