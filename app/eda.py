import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from app.config import REPORTS_DIR, TARGET
from app.extractor import DataExtractor
from app.transformer import add_features

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity", "SeniorCitizen"]

OUTPUT_DIR = REPORTS_DIR / "eda"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / name, dpi=110)
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    extractor = DataExtractor()
    df = extractor.load()

    report = extractor.report()
    print(json.dumps(report, indent=2))
    (OUTPUT_DIR / "quality_report.json").write_text(json.dumps(report, indent=2))
    print(df.describe().T.round(2))

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x=TARGET, ax=ax)
    ax.set_title("Churn distribution (imbalanced)")
    save(fig, "churn_distribution.png")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for i, col in enumerate(NUMERIC):
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, ax=axes[0, i])
        sns.boxplot(data=df, x=TARGET, y=col, ax=axes[1, i])
    fig.suptitle("Numeric distributions and outliers by churn")
    save(fig, "numeric_distributions.png")

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, col in zip(axes.flat, CATEGORICAL):
        rate = df.groupby(col)[TARGET].apply(lambda s: (s == "Yes").mean()).sort_values()
        sns.barplot(x=rate.index.astype(str), y=rate.values, ax=ax)
        ax.set_title(f"Churn rate by {col}")
        ax.tick_params(axis="x", rotation=20)
    save(fig, "churn_rate_by_category.png")

    engineered = add_features(df.drop(columns=TARGET)).assign(Churn=(df[TARGET] == "Yes").astype(int))
    corr = engineered.select_dtypes("number").corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlations (raw + engineered numeric features)")
    save(fig, "correlations.png")

    print(f"\nEDA figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
