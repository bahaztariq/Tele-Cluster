import json

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline

from app.config import DBSCAN_PARAMS, K_RANGE, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, setup_mlflow
from app.extractor import DataExtractor
from app.transformer import DataTransformer, add_features

OUTPUT_DIR = REPORTS_DIR / "clustering"


def scores(Z, labels) -> dict:
    # DBSCAN noise (-1) is not a cluster, so it is left out of the scores.
    mask = labels != -1
    Z, labels = Z[mask], labels[mask]
    if len(set(labels)) < 2:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"), "calinski_harabasz": float("nan")}
    return {
        "silhouette": silhouette_score(Z, labels),
        "davies_bouldin": davies_bouldin_score(Z, labels),
        "calinski_harabasz": calinski_harabasz_score(Z, labels),
    }


def elbow_silhouette(Z) -> tuple[pd.DataFrame, plt.Figure]:
    rows = []
    for k in K_RANGE:
        km = KMeans(k, n_init=10, random_state=RANDOM_STATE).fit(Z)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Z, km.labels_)})
    table = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(data=table, x="k", y="inertia", marker="o", ax=axes[0]).set_title("Elbow method")
    sns.lineplot(data=table, x="k", y="silhouette", marker="o", ax=axes[1]).set_title("Silhouette score")
    fig.tight_layout()
    return table, fig


def profile(X, y, labels) -> pd.DataFrame:
    df = add_features(X).assign(cluster=labels, churn=y.to_numpy())
    table = df.groupby("cluster").agg(
        size=("churn", "size"),
        churn_rate=("churn", "mean"),
        tenure=("tenure", "mean"),
        monthly_charges=("MonthlyCharges", "mean"),
        num_services=("num_services", "mean"),
        month_to_month=("is_month_to_month", "mean"),
        internet=("InternetService", lambda s: s.mode()[0].replace("No", "No internet")),
        contract=("Contract", lambda s: s.mode()[0]),
    )
    # Churn is only used here, after clustering, to describe the segments.
    rank = table["churn_rate"].rank(method="first")
    risk = rank.map(lambda r: "Low risk" if r == 1 else "High risk" if r == len(table) else "Medium risk")
    table.insert(0, "name", risk + " - " + table["internet"] + ", " + table["contract"])
    return table


def pca_plot(Z, labels, title) -> plt.Figure:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(Z)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, palette="tab10", s=8, linewidth=0, ax=ax)
    ratio = pca.explained_variance_ratio_
    ax.set(title=title, xlabel=f"PC1 ({ratio[0]:.0%})", ylabel=f"PC2 ({ratio[1]:.0%})")
    fig.tight_layout()
    return fig


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    extractor = DataExtractor()
    extractor.load()
    X, y = extractor.features_and_target()  # y is never given to the clustering

    prep = DataTransformer().fit(X)
    Z = prep.transform(X)
    print(f"Clustering {Z.shape[0]} customers on {Z.shape[1]} features (churn excluded)")

    elbow, elbow_fig = elbow_silhouette(Z)
    best_k = int(elbow.loc[elbow["silhouette"].idxmax(), "k"])
    print(elbow.round(4).to_string(index=False))
    print(f"Best k by silhouette: {best_k}\n")

    estimators = {
        f"KMeans_k{best_k}": KMeans(best_k, n_init=10, random_state=RANDOM_STATE),
        f"Agglomerative_k{best_k}": AgglomerativeClustering(best_k, linkage="ward"),
        "DBSCAN": DBSCAN(**DBSCAN_PARAMS),
    }
    results = {name: est.fit_predict(Z) for name, est in estimators.items()}
    comparison = pd.DataFrame(
        {name: {"n_clusters": len(set(l) - {-1}), "noise": int((l == -1).sum()), **scores(Z, l)}
         for name, l in results.items()}
    ).T
    print(comparison.round(4).to_string())

    model = Pipeline([("prep", DataTransformer()), ("kmeans", KMeans(best_k, n_init=10, random_state=RANDOM_STATE))])
    labels = model.fit(X).predict(X)
    profiles = profile(X, y, labels)
    print("\n" + profiles.round(2).to_string())

    figures = {
        "elbow_silhouette": elbow_fig,
        "pca_kmeans": pca_plot(Z, labels, f"K-Means segments (k={best_k}) on PCA"),
        "pca_dbscan": pca_plot(Z, results["DBSCAN"], "DBSCAN on PCA (-1 = noise)"),
    }
    for name, fig in figures.items():
        fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=110)
    elbow.to_csv(OUTPUT_DIR / "elbow_silhouette.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "algorithm_comparison.csv")
    profiles.to_csv(OUTPUT_DIR / "cluster_profiles.csv")

    joblib.dump(model, MODELS_DIR / "segmentation.joblib")
    (MODELS_DIR / "segmentation.json").write_text(json.dumps({
        "n_clusters": best_k,
        "profiles": {str(c): {"name": r["name"], "size": int(r["size"]), "churn_rate": round(r["churn_rate"], 3)}
                     for c, r in profiles.iterrows()},
    }, indent=2))

    setup_mlflow("tele-cluster-segmentation")
    for name, estimator in estimators.items():
        with mlflow.start_run(run_name=name):
            params = DBSCAN_PARAMS if name == "DBSCAN" else {"n_clusters": best_k}
            mlflow.log_params({"algorithm": name.split("_")[0], **params,
                               "prep__feature_engineering": True, "prep__scaler": "StandardScaler",
                               "prep__encoder": "OneHotEncoder", "prep__outliers": "IQR clip"})
            mlflow.log_metrics({k: float(v) for k, v in comparison.loc[name].items()})
            if name.startswith("KMeans"):
                for fig_name, fig in figures.items():
                    mlflow.log_figure(fig, f"{fig_name}.png")
                mlflow.log_text(profiles.round(3).to_csv(), "cluster_profiles.csv")
                mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle", input_example=X.head(3))
            else:
                # Agglomerative and DBSCAN cannot label new customers, so they are logged
                # as fitted on the training data (preprocessing + clustering).
                fitted = Pipeline([("prep", prep), ("model", estimator)])
                mlflow.sklearn.log_model(fitted, name="model", serialization_format="cloudpickle")
    print(f"\nSaved models/segmentation.joblib, reports in {OUTPUT_DIR}, runs in MLflow")


if __name__ == "__main__":
    main()
