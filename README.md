# Tele-Cluster

Customer segmentation and churn prediction for a telecommunications company.

From a historical customer dataset, the project builds two things: distinct customer
profiles (clustering) and a model that predicts which customers are likely to cancel their
subscription (classification). It then checks whether the profiles help the prediction,
tunes the models, tracks every experiment in MLflow, and serves the result in a Streamlit
app packaged with Docker.

The project is implemented twice, with two different approaches:

- **`notebooks/`**: the step-by-step analysis, with explanations, figures and conclusions.
- **`app/`**: the same steps as plain Python scripts, plus the Streamlit app and Docker image.

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
   every experiment.
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
notebooks/
  01_eda_preparation.ipynb       Step 1: EDA, data quality, feature engineering, preprocessing
  02_clustering.ipynb            Step 2: choice of k, K-Means vs Agglomerative vs DBSCAN, PCA, profiles
  03_classification.ipynb        Step 3: 5 models x imbalance strategies, test evaluation
  04_cluster_feature.ipynb       Step 4: initial variables vs initial variables + cluster
  05_optimization_mlflow.ipynb   Steps 5-6: tuning, before/after, final model, MLflow
app/
  config.py                      Paths and constants
  extractor.py                   DataExtractor: load the CSV, quality report
  transformer.py                 DataTransformer: feature engineering, imputation, IQR
                                 clipping, one-hot, scaling, optional cluster feature
  eda.py / clustering.py /       Steps 1-6 as scripts
  classification.py
  main.py                        Runs eda -> clustering -> classification
  streamlit_app.py               Step 7: Streamlit interface
data/
  raw/                           Source CSV
  processed/telco_clean.csv      Cleaned dataset with engineered features (notebook 01)
models/
  cluster_pipeline.pkl           Notebook clustering pipeline (k=4)
  churn_model.pkl                Notebook final churn model
  model_info.json                Notebook final model: params, test metrics, cluster names
  segmentation.joblib / .json    App clustering pipeline and segment profiles
  churn.joblib / .json           App final churn model and its metrics
Dockerfile, docker-compose.yml   Streamlit image
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Notebooks:** run them in order (01 → 05). Each one is self-contained.

```bash
pip install jupyter
jupyter notebook notebooks/
```

**App pipeline:** retrains the app models in `models/` (about 20 minutes):

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

**Docker:** the image includes the trained models from `models/`:

```bash
docker compose up --build               # http://localhost:8501
```

## Methodology

- **No target in the clustering.** `Churn` is never an input to the segmentation. It is
  only used afterwards, to describe the churn rate of each profile.
- **No data leakage.** Imputation, scaling, encoding, the K-Means used for the `cluster`
  feature, and SMOTE / oversampling all live inside one `imblearn` pipeline. They are
  fitted on the training data only (on each training fold during cross-validation), and
  the samplers never touch validation or test rows.
- **`customerID` is dropped.** It is a unique key with no predictive value.
- **`TotalCharges`** is read as text because 11 rows are blank (customers with tenure 0).
  It is converted to numbers, and the blanks are imputed inside the pipeline.

## Results (notebooks)

### 1. Data analysis

- 7043 customers. 22 duplicated profiles were removed, leaving 7021. 11 `TotalCharges`
  values are missing. No abnormal outliers.
- The target is imbalanced: about 26.5% churn.
- Main churn drivers: short tenure, month-to-month contract, fiber optic, no online
  security or tech support, electronic check, high monthly charges.
- 8 features were engineered, e.g. number of services, average monthly spend, charges per
  service, month-to-month flag, auto-payment, family, new customer, tenure group.

### 2. Customer segments

Elbow and Silhouette were computed for k = 2…10. Silhouette is highest at k = 2 (0.35),
but that split only separates customers with internet from those without. **k = 4** is
almost as good (0.32), sits at the elbow, and gives four actionable profiles. Agglomerative
clustering confirms the result; DBSCAN is not suited to this mostly-binary data.

| Profile | Description | Churn | Action |
| --- | --- | --- | --- |
| Basic phone-only | No internet, ~1 service, ~21/month | ~7% | Stable, low value; upsell carefully |
| Loyal premium | ~5 years tenure, many services, ~91/month, 1-2 year contracts | ~14% | Retain and reward |
| Mid-value DSL | DSL, ~20 months, ~52/month, mostly month-to-month | ~25% | Push longer contracts and add-ons |
| New fiber, at risk | Fiber optic, ~15 months, ~85/month, month-to-month, manual payment | ~57% | **Priority retention target** |

### 3. Classification

- Without rebalancing, recall is about 0.5, so half of the churners are missed.
- `class_weight`, oversampling and SMOTE raise recall to 0.75-0.80 and improve F1.
- The best models reach test F1 of about 0.62-0.63 and ROC-AUC of about 0.83-0.84.

### 4. Cluster as a feature

| | Initial variables | Initial + cluster |
| --- | --- | --- |
| Best CV F1 (Logistic Regression) | 0.631 | 0.632 |
| Best test ROC-AUC (Logistic Regression) | 0.841 | 0.841 |

The cluster is informative on its own (ROC-AUC about 0.76 using only `cluster`), but it adds
no significant gain on top of the initial variables. It summarizes information the
classifier already has: contract, internet type, tenure and charges. **The cluster adds
interpretability, not predictive power.**

### 5-6. Optimization and final model

| Step | Result |
| --- | --- |
| Tuning (GridSearchCV / RandomizedSearchCV, 5-fold CV, F1) | Big gain for Decision Tree, +0.02 F1 for Random Forest |
| Feature engineering | Small positive gain |
| Cluster variable | No significant gain |
| **Final model** | Tuned Random Forest (`class_weight="balanced"`, raw + engineered features) |

Final model on the held-out test set: **recall 0.78, precision 0.53, F1 0.63,
ROC-AUC 0.84**. It catches about 4 out of 5 churners. All experiments are tracked in MLflow.

## Results (app)

The app scripts follow the same steps with a different approach. Clustering uses all
customer features, and *k* is picked automatically by Silhouette, which gives **k = 3**:

| Segment | Customers | Churn rate |
| --- | --- | --- |
| Low risk - No internet, Two year | 1513 | 7% |
| Medium risk - Fiber optic, Two year | 2435 | 14% |
| High risk - Fiber optic, Month-to-month | 3073 | 46% |

The conclusions match the notebooks:
- Rebalancing matters more than the choice of model.
- Neither feature engineering nor the cluster improves cross-validated F1 (0.625-0.626 for
  every feature set).
- The final model is a tuned Random Forest with `class_weight`: test **recall 0.78,
  precision 0.53, F1 0.63, ROC-AUC 0.84**.

Figures and tables are written to `reports/` (`eda/`, `clustering/`, `classification/`).

## Streamlit app

`app/streamlit_app.py` shows a form with the customer's 19 characteristics and returns:

- **the customer's segment** (cluster and profile name),
- **the churn prediction** (will churn / will stay),
- **the churn probability**.

It uses the app models (`models/segmentation.joblib` and `models/churn.joblib`).
