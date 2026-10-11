"""Churn classification, in four steps (all model selection is cross-validated on the train set):

1. every model with every imbalance strategy;
2. each model's best strategy on four feature sets (raw / + cluster / + FE / + FE + cluster);
3. hyperparameter search on each model's best configuration, before vs after;
4. the tuned models on the test set; the best one is registered in MLflow and saved for the app.
"""

import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from app.config import CV_FOLDS, MODELS_DIR, N_JOBS, RANDOM_STATE, REPORTS_DIR, TEST_SIZE, setup_mlflow
from app.extractor import DataExtractor
from app.transformer import DataTransformer

OUTPUT_DIR = REPORTS_DIR / "classification"
REGISTERED_MODEL = "tele-cluster-churn-app"
METRICS = ["precision", "recall", "f1", "roc_auc"]

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "DecisionTree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=1, random_state=RANDOM_STATE),
}

STRATEGIES = ["none", "class_weight", "random_oversampling", "smote"]
SAMPLERS = {"random_oversampling": RandomOverSampler, "smote": SMOTE}

# label -> (feature engineering, cluster feature)
FEATURE_SETS = {"raw": (False, False), "raw+cluster": (False, True),
                "raw+FE": (True, False), "raw+FE+cluster": (True, True)}

# GridSearchCV for small search spaces, RandomizedSearchCV (N_ITER samples) for large ones.
N_ITER = 20
SEARCH_SPACES = {
    "LogisticRegression": ("grid", {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]}),
    "DecisionTree": ("grid", {"model__max_depth": [3, 4, 5, 6, 8, 10], "model__min_samples_leaf": [1, 10, 20, 50]}),
    "SVM": ("grid", {"model__C": [0.1, 1, 10], "model__gamma": ["scale", 0.01, 0.03]}),
    "RandomForest": ("random", {"model__n_estimators": randint(100, 500), "model__max_depth": randint(4, 20),
                                "model__min_samples_leaf": randint(1, 30), "model__max_features": uniform(0.1, 0.6)}),
    "XGBoost": ("random", {"model__n_estimators": randint(100, 600), "model__max_depth": randint(2, 8),
                           "model__learning_rate": loguniform(0.01, 0.3), "model__subsample": uniform(0.6, 0.4)}),
}

# Fixed preprocessing choices, logged with every run next to DataTransformer's own parameters.
PREP_PARAMS = {"prep__imputer": "median / 'Missing'", "prep__outliers": "IQR clip",
               "prep__scaler": "StandardScaler", "prep__encoder": "OneHotEncoder"}


@dataclass
class Data:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    pos_weight: float  # negatives / positives, XGBoost's equivalent of class_weight="balanced"
    n_clusters: int  # number of segments found by the clustering step


def load_data() -> Data:
    extractor = DataExtractor()
    extractor.load()
    X, y = extractor.features_and_target()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return Data(
        X_train, X_test, y_train, y_test,
        pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        n_clusters=json.loads((MODELS_DIR / "segmentation.json").read_text())["n_clusters"],
    )


def build_pipeline(data, model, imbalance, feature_set) -> Pipeline:
    estimator = clone(MODELS[model])
    if imbalance == "class_weight":
        estimator.set_params(
            **({"scale_pos_weight": data.pos_weight} if model == "XGBoost" else {"class_weight": "balanced"})
        )
    feature_engineering, cluster = FEATURE_SETS[feature_set]
    steps = [("prep", DataTransformer(feature_engineering, n_clusters=data.n_clusters if cluster else None))]
    if imbalance in SAMPLERS:
        # imblearn runs samplers during fit only, so they never see validation or test rows.
        steps.append(("sampler", SAMPLERS[imbalance](random_state=RANDOM_STATE)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def folds():
    return StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def test_metrics(pipeline, data) -> dict:
    proba = pipeline.predict_proba(data.X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "test_precision": precision_score(data.y_test, pred),
        "test_recall": recall_score(data.y_test, pred),
        "test_f1": f1_score(data.y_test, pred),
        "test_roc_auc": roc_auc_score(data.y_test, proba),
    }


def log_run(run_name, pipeline, metrics, params, data) -> str:
    """Log a fitted pipeline to MLflow: hyperparameters, preprocessing, metrics and the model."""
    hyper = {f"model__{k}": v for k, v in pipeline.named_steps["model"].get_params().items()
             if isinstance(v, (str, int, float, bool))}
    prep = {f"prep__{k}": v for k, v in pipeline.named_steps["prep"].get_params().items()}
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({**hyper, **prep, **PREP_PARAMS, **params})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle",
                                 input_example=data.X_train.head(3))
    return run.info.run_id


def run_config(data, model, imbalance, feature_set, stage) -> tuple[dict, Pipeline]:
    """Cross-validate one configuration, fit it on the train set and log it."""
    pipeline = build_pipeline(data, model, imbalance, feature_set)
    result = cross_validate(pipeline, data.X_train, data.y_train, cv=folds(), scoring=METRICS, n_jobs=N_JOBS)
    metrics = {f"cv_{m}": result[f"test_{m}"].mean() for m in METRICS}
    pipeline.fit(data.X_train, data.y_train)
    config = {"model": model, "imbalance": imbalance, "feature_set": feature_set}
    log_run(f"{model}_{imbalance}_{feature_set}", pipeline, metrics, {**config, "stage": stage}, data)
    print(f"{model:20s} {imbalance:20s} {feature_set:15s} " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    return {**config, **metrics}, pipeline


def best_per_model(table) -> pd.DataFrame:
    return table.sort_values("cv_f1", ascending=False).drop_duplicates("model")


def compare_strategies(data) -> pd.DataFrame:
    print(f"== 1. Models x imbalance strategies ({CV_FOLDS}-fold CV) ==")
    rows = [run_config(data, model, imbalance, "raw+FE", "imbalance")[0]
            for model in MODELS for imbalance in STRATEGIES]
    table = pd.DataFrame(rows).sort_values("cv_f1", ascending=False)
    table.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    return table


def compare_feature_sets(data, strategies) -> tuple[pd.DataFrame, dict]:
    print("\n== 2. Feature sets, each model with its best strategy ==")
    rows, fitted = [], {}
    for r in best_per_model(strategies).itertuples():
        for feature_set in FEATURE_SETS:
            row, fitted[r.model, feature_set] = run_config(data, r.model, r.imbalance, feature_set, "feature_sets")
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT_DIR / "feature_set_comparison.csv", index=False)
    print(table.pivot(index="model", columns="feature_set", values="cv_f1").round(4).to_string())
    return table, fitted


def tune_models(data, best, fitted) -> tuple[pd.DataFrame, dict, dict]:
    """Search each model's best configuration; return before/after metrics, the searches and their MLflow runs."""
    print("\n== 3. Hyperparameter tuning, best configuration per model ==")
    rows, searches, run_ids = [], {}, {}
    for r in best.itertuples():
        kind, space = SEARCH_SPACES[r.model]
        pipeline = build_pipeline(data, r.model, r.imbalance, r.feature_set)
        options = {"scoring": METRICS, "refit": "f1", "cv": folds(), "n_jobs": N_JOBS}
        search = (GridSearchCV(pipeline, space, **options) if kind == "grid"
                  else RandomizedSearchCV(pipeline, space, n_iter=N_ITER, random_state=RANDOM_STATE, **options))
        searches[r.model] = search.fit(data.X_train, data.y_train)

        before = {**{f"cv_{m}": getattr(r, f"cv_{m}") for m in METRICS},
                  **test_metrics(fitted[r.model, r.feature_set], data)}
        after = {**{f"cv_{m}": search.cv_results_[f"mean_test_{m}"][search.best_index_] for m in METRICS},
                 **test_metrics(search.best_estimator_, data)}
        rows += [{"model": r.model, "stage": "before", **before}, {"model": r.model, "stage": "after", **after}]

        params = {"model": r.model, "imbalance": r.imbalance, "feature_set": r.feature_set,
                  "search": kind, "stage": "tuned"}
        run_ids[r.model] = log_run(f"{r.model}_tuned", search.best_estimator_, after, params, data)
        print(f"{r.model:20s} {kind:6s} cv_f1 {before['cv_f1']:.3f} -> {after['cv_f1']:.3f}  {search.best_params_}")

    optimisation = pd.DataFrame(rows).set_index(["model", "stage"])
    optimisation.to_csv(OUTPUT_DIR / "optimisation.csv")
    return optimisation, searches, run_ids


def tuning_figure(optimisation) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    for ax, metric in zip(axes, ["cv_f1", "test_f1"]):
        optimisation[metric].unstack("stage")[["before", "after"]].plot.bar(ax=ax, rot=15)
        ax.set(title=f"{metric}: before vs after tuning", xlabel="", ylim=(0.4, 0.7))
    fig.tight_layout()
    return fig


def test_figure(pipelines, data) -> plt.Figure:
    """One confusion matrix per model, plus all ROC curves on the last panel."""
    fig, axes = plt.subplots(1, len(pipelines) + 1, figsize=(4.5 * (len(pipelines) + 1), 4.5))
    for ax, (name, pipeline) in zip(axes, pipelines.items()):
        ConfusionMatrixDisplay.from_estimator(pipeline, data.X_test, data.y_test, display_labels=["Stay", "Churn"],
                                              ax=ax, colorbar=False)
        ax.set_title(name)
        RocCurveDisplay.from_estimator(pipeline, data.X_test, data.y_test, name=name, ax=axes[-1])
    axes[-1].set_title("ROC curves")
    fig.suptitle("Test set: tuned models (best configuration per model)")
    fig.tight_layout()
    return fig


def save_final(data, best, optimisation, searches, run_ids):
    """Compare the tuned models on the test set; log, register and save the one with the best CV F1."""
    print("\n== 4. Tuned models on the test set ==")
    summary = (best.set_index("model")[["imbalance", "feature_set"]]
               .join(optimisation.xs("after", level="stage"))
               .sort_values("cv_f1", ascending=False))
    summary.to_csv(OUTPUT_DIR / "test_comparison.csv")
    print(summary.round(4).to_string())

    name = summary.index[0]
    final = searches[name].best_estimator_
    report = classification_report(data.y_test, final.predict(data.X_test), target_names=["Stay", "Churn"])
    print(f"\nFinal model: {name}\n{report}")
    (OUTPUT_DIR / "classification_report.txt").write_text(report)

    figures = {"test_comparison": test_figure({m: s.best_estimator_ for m, s in searches.items()}, data),
               "optimisation": tuning_figure(optimisation)}
    with mlflow.start_run(run_id=run_ids[name]):
        mlflow.set_tag("final", "true")
        mlflow.log_text(report, "classification_report.txt")
        for fig_name, fig in figures.items():
            fig.savefig(OUTPUT_DIR / f"{fig_name}.png", dpi=110)
            mlflow.log_figure(fig, f"{fig_name}.png")
    version = mlflow.register_model(f"runs:/{run_ids[name]}/model", REGISTERED_MODEL).version
    print(f"Registered {REGISTERED_MODEL} version {version}")

    joblib.dump(final, MODELS_DIR / "churn.joblib")
    (MODELS_DIR / "churn.json").write_text(json.dumps({
        "model": name,
        "imbalance": summary.loc[name, "imbalance"],
        "feature_set": summary.loc[name, "feature_set"],
        "best_params": searches[name].best_params_,
        "test_metrics": summary.loc[name].filter(like="test_").astype(float).round(4).to_dict(),
        "mlflow_run_id": run_ids[name],
        "registered_model": {"name": REGISTERED_MODEL, "version": int(version)},
    }, indent=2, default=lambda v: v.item()))  # numpy numbers from the search spaces
    print("Saved models/churn.joblib")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_mlflow("tele-cluster-churn")
    data = load_data()

    strategies = compare_strategies(data)
    feature_sets, fitted = compare_feature_sets(data, strategies)
    best = best_per_model(feature_sets)
    optimisation, searches, run_ids = tune_models(data, best, fitted)
    save_final(data, best, optimisation, searches, run_ids)


if __name__ == "__main__":
    main()
