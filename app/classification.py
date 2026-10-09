import json

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from imblearn.over_sampling import SMOTE
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from app.config import CV_FOLDS, MODELS_DIR, N_JOBS, RANDOM_STATE, REPORTS_DIR, TEST_SIZE, setup_mlflow
from app.extractor import DataExtractor
from app.transformer import DataTransformer

OUTPUT_DIR = REPORTS_DIR / "classification"

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "DecisionTree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, n_jobs=1, random_state=RANDOM_STATE),
}

STRATEGIES = ["none", "class_weight", "smote"]

SEARCH_SPACES = {
    "LogisticRegression": {"model__C": loguniform(1e-3, 1e2)},
    "DecisionTree": {"model__max_depth": randint(3, 15), "model__min_samples_leaf": randint(1, 60)},
    "RandomForest": {"model__n_estimators": randint(100, 500), "model__max_depth": randint(4, 20),
                     "model__min_samples_leaf": randint(1, 30), "model__max_features": uniform(0.1, 0.6)},
    "SVM": {"model__C": loguniform(1e-2, 1e2), "model__gamma": loguniform(1e-4, 1e-1)},
    "XGBoost": {"model__n_estimators": randint(100, 600), "model__max_depth": randint(2, 8),
                "model__learning_rate": loguniform(0.01, 0.3), "model__subsample": uniform(0.6, 0.4)},
}

METRICS = ["precision", "recall", "f1", "roc_auc"]


def build_pipeline(name, strategy, pos_weight, feature_engineering=True, n_clusters=None) -> Pipeline:
    model = clone(MODELS[name])
    if strategy == "class_weight":
        # XGBoost has no class_weight; scale_pos_weight is its equivalent.
        model.set_params(**({"scale_pos_weight": pos_weight} if name == "XGBoost" else {"class_weight": "balanced"}))
    steps = [("prep", DataTransformer(feature_engineering=feature_engineering, n_clusters=n_clusters))]
    if strategy == "smote":
        # imblearn runs samplers during fit only, so SMOTE never sees validation or test rows.
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
    steps.append(("model", model))
    return Pipeline(steps)


def cross_validated(pipeline, X, y) -> dict:
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    result = cross_validate(pipeline, X, y, cv=cv, scoring=METRICS, n_jobs=N_JOBS)
    return {f"cv_{m}": result[f"test_{m}"].mean() for m in METRICS}


def test_metrics(pipeline, X_test, y_test) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "test_precision": precision_score(y_test, pred),
        "test_recall": recall_score(y_test, pred),
        "test_f1": f1_score(y_test, pred),
        "test_roc_auc": roc_auc_score(y_test, proba),
    }


def log_run(run_name, pipeline, metrics, params, example, figures=None):
    """Log one experiment; `pipeline` must already be fitted on the training set."""
    with mlflow.start_run(run_name=run_name):
        prep = pipeline.named_steps["prep"]
        model = pipeline.named_steps["model"]
        hyper = {k: v for k, v in model.get_params().items() if isinstance(v, (str, int, float, bool))}
        mlflow.log_params({f"model__{k}": v for k, v in hyper.items()})
        mlflow.log_params({
            "model": type(model).__name__,
            "prep__feature_engineering": prep.feature_engineering,
            "prep__cluster_feature": bool(prep.n_clusters),
            "prep__iqr_factor": prep.iqr_factor,
            "prep__imputer": "median / 'Missing'",
            "prep__scaler": "StandardScaler",
            "prep__encoder": "OneHotEncoder",
            **params,
        })
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        for name, fig in (figures or {}).items():
            mlflow.log_figure(fig, f"{name}.png")
        mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle", input_example=example)


def evaluation_figures(pipeline, X_test, y_test, title) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, display_labels=["Stay", "Churn"], ax=axes[0])
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=axes[1])
    axes[0].set_title("Confusion matrix (test)")
    axes[1].set_title("ROC curve (test)")
    fig.suptitle(title)
    fig.tight_layout()
    return {"evaluation": fig}


def compare_on_test(pipelines: dict, X_test, y_test) -> tuple[pd.DataFrame, plt.Figure]:
    """Test metrics, confusion matrix and ROC curve for each model."""
    fig, axes = plt.subplots(1, len(pipelines) + 1, figsize=(4.5 * (len(pipelines) + 1), 4.5))
    rows = []
    for ax, (name, pipeline) in zip(axes, pipelines.items()):
        ConfusionMatrixDisplay.from_estimator(
            pipeline, X_test, y_test, display_labels=["Stay", "Churn"], ax=ax, colorbar=False
        )
        ax.set_title(name)
        RocCurveDisplay.from_estimator(pipeline, X_test, y_test, name=name, ax=axes[-1])
        rows.append({"model": name, **test_metrics(pipeline, X_test, y_test)})
    axes[-1].set_title("ROC curves")
    fig.suptitle("Test set: confusion matrices and ROC curves (best configuration per model)")
    fig.tight_layout()
    return pd.DataFrame(rows).set_index("model"), fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extractor = DataExtractor()
    extractor.load()
    X, y = extractor.features_and_target()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    n_clusters = json.loads((MODELS_DIR / "segmentation.json").read_text())["n_clusters"]
    example = X_train.head(3)
    setup_mlflow("tele-cluster-churn")

    # 1. Five models x three imbalance strategies (cross-validated on the training set only).
    print(f"== Models x imbalance strategies ({CV_FOLDS}-fold CV, train set) ==")
    rows = []
    for name in MODELS:
        for strategy in STRATEGIES:
            pipeline = build_pipeline(name, strategy, pos_weight)
            metrics = cross_validated(pipeline, X_train, y_train)
            pipeline.fit(X_train, y_train)
            log_run(f"{name}_{strategy}", pipeline, metrics, {"imbalance": strategy, "stage": "comparison"}, example)
            rows.append({"model": name, "imbalance": strategy, **metrics})
            print(f"{name:20s} {strategy:13s} " + "  ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    comparison = pd.DataFrame(rows).sort_values("cv_f1", ascending=False)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    # 2. Value of feature engineering and of the cluster feature, per model (best strategy each).
    print("\n== Feature sets: raw / raw + cluster / raw + FE / raw + FE + cluster ==")
    best_strategy = comparison.drop_duplicates("model").set_index("model")["imbalance"]
    feature_sets = {"raw": (False, None), "raw+cluster": (False, n_clusters),
                    "raw+FE": (True, None), "raw+FE+cluster": (True, n_clusters)}
    rows, fitted = [], {}
    for name, strategy in best_strategy.items():
        for label, (fe, k) in feature_sets.items():
            pipeline = build_pipeline(name, strategy, pos_weight, feature_engineering=fe, n_clusters=k)
            metrics = cross_validated(pipeline, X_train, y_train)
            fitted[name, label] = pipeline.fit(X_train, y_train)
            log_run(f"{name}_{label}", pipeline, metrics,
                    {"imbalance": strategy, "feature_set": label, "stage": "feature_sets"}, example)
            rows.append({"model": name, "imbalance": strategy, "feature_set": label, **metrics})
    features = pd.DataFrame(rows)
    features.to_csv(OUTPUT_DIR / "feature_set_comparison.csv", index=False)
    print(features.pivot(index="model", columns="feature_set", values="cv_f1").round(4).to_string())
    print(features.groupby("feature_set")[["cv_f1", "cv_roc_auc"]].mean().round(4).to_string())

    # 3. Each model's best configuration on the held-out test set.
    print("\n== Test set: best configuration per model ==")
    best_rows = features.sort_values("cv_f1", ascending=False).drop_duplicates("model")
    test_results, test_fig = compare_on_test(
        {r.model: fitted[r.model, r.feature_set] for r in best_rows.itertuples()}, X_test, y_test
    )
    test_results.insert(0, "feature_set", best_rows.set_index("model")["feature_set"])
    test_results.insert(0, "imbalance", best_rows.set_index("model")["imbalance"])
    print(test_results.round(4).to_string())
    test_results.to_csv(OUTPUT_DIR / "test_comparison.csv")
    test_fig.savefig(OUTPUT_DIR / "test_comparison.png", dpi=110)
    with mlflow.start_run(run_name="test_comparison"):
        mlflow.log_param("stage", "test_comparison")
        mlflow.log_metrics({f"{model}_{k}": float(v) for model, row in test_results.iterrows()
                            for k, v in row.items() if k.startswith("test_")})
        mlflow.log_figure(test_fig, "test_comparison.png")
        mlflow.log_text(test_results.round(4).to_csv(), "test_comparison.csv")

    # 4. Hyperparameter search on the best configuration.
    best = features.sort_values("cv_f1", ascending=False).iloc[0]
    fe, k = feature_sets[best["feature_set"]]
    baseline = build_pipeline(best["model"], best["imbalance"], pos_weight, fe, k)
    print(f"\n== RandomizedSearchCV on {best['model']} / {best['imbalance']} / {best['feature_set']} ==")
    search = RandomizedSearchCV(
        baseline, SEARCH_SPACES[best["model"]], n_iter=20, scoring="f1", n_jobs=N_JOBS, random_state=RANDOM_STATE,
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
    ).fit(X_train, y_train)
    tuned = search.best_estimator_
    print("Best params:", search.best_params_)

    # 5. Before vs after optimisation, on CV and on the held-out test set.
    baseline.fit(X_train, y_train)
    before = {**cross_validated(baseline, X_train, y_train), **test_metrics(baseline, X_test, y_test)}
    after = {**cross_validated(tuned, X_train, y_train), **test_metrics(tuned, X_test, y_test)}
    optimisation = pd.DataFrame({"before": before, "after": after})
    optimisation.to_csv(OUTPUT_DIR / "optimisation.csv")
    print(optimisation.round(4).to_string())

    title = f"{best['model']} ({best['imbalance']}, {best['feature_set']}) - tuned"
    figures = evaluation_figures(tuned, X_test, y_test, title)
    figures["evaluation"].savefig(OUTPUT_DIR / "final_evaluation.png", dpi=110)
    report = classification_report(y_test, tuned.predict(X_test), target_names=["Stay", "Churn"])
    print("\n" + report)
    (OUTPUT_DIR / "classification_report.txt").write_text(report)

    params = {"imbalance": best["imbalance"], "feature_set": best["feature_set"], "stage": "final"}
    log_run(f"{best['model']}_baseline", baseline, before, {**params, "stage": "before_tuning"}, example)
    log_run(f"{best['model']}_tuned", tuned, after, params, example, figures)

    joblib.dump(tuned, MODELS_DIR / "churn.joblib")
    (MODELS_DIR / "churn.json").write_text(json.dumps({
        "model": best["model"],
        "imbalance": best["imbalance"],
        "feature_set": best["feature_set"],
        "best_params": {k: getattr(v, "item", lambda: v)() for k, v in search.best_params_.items()},
        "test_metrics": {k: round(float(v), 4) for k, v in after.items() if k.startswith("test_")},
    }, indent=2))
    print("Saved models/churn.joblib")


if __name__ == "__main__":
    main()
