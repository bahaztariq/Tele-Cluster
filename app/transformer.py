import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import RANDOM_STATE

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["num_services"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1) + (df["InternetService"] != "No")
    df["avg_monthly_spend"] = (df["TotalCharges"] / df["tenure"].replace(0, np.nan)).fillna(df["MonthlyCharges"])
    df["charges_per_service"] = df["MonthlyCharges"] / df["num_services"].clip(lower=1)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["auto_payment"] = df["PaymentMethod"].str.contains("automatic").astype(int)
    df["has_family"] = ((df["Partner"] == "Yes") | (df["Dependents"] == "Yes")).astype(int)
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 60, 72], labels=["0-12", "13-24", "25-48", "49-60", "61-72"]
    ).astype(str)
    return df


class DataTransformer(BaseEstimator, TransformerMixin):
    """Feature engineering, imputation, IQR clipping, one-hot encoding and scaling.

    Every statistic is learned in fit(), so inside a Pipeline it only ever sees
    the training data. With n_clusters set, it also fits K-Means on the prepared
    features and appends the customer's cluster as one-hot columns.
    """

    def __init__(self, feature_engineering=True, iqr_factor=1.5, n_clusters=None):
        self.feature_engineering = feature_engineering
        self.iqr_factor = iqr_factor
        self.n_clusters = n_clusters

    def fit(self, X, y=None):
        X = self._engineer(X)
        self.numeric_ = X.select_dtypes("number").columns.tolist()
        self.categorical_ = [c for c in X.columns if c not in self.numeric_]

        num = X[self.numeric_]
        self.medians_ = num.median()
        q1, q3 = num.quantile(0.25), num.quantile(0.75)
        iqr = q3 - q1
        # Binary columns have a zero IQR; clipping them would erase the minority value.
        binary = num.nunique() <= 2
        self.lower_ = (q1 - self.iqr_factor * iqr).mask(binary, -np.inf)
        self.upper_ = (q3 + self.iqr_factor * iqr).mask(binary, np.inf)

        self.scaler_ = StandardScaler().fit(self._clean_numeric(X))
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder_.fit(X[self.categorical_].fillna("Missing"))

        if self.n_clusters:
            self.kmeans_ = KMeans(self.n_clusters, n_init=10, random_state=RANDOM_STATE)
            self.kmeans_.fit(self._encode(X))
        return self

    def transform(self, X):
        out = self._encode(self._engineer(X))
        if self.n_clusters:
            labels = self.kmeans_.predict(out)
            clusters = pd.DataFrame(
                np.eye(self.n_clusters)[labels],
                columns=[f"cluster_{i}" for i in range(self.n_clusters)],
                index=out.index,
            )
            out = pd.concat([out, clusters], axis=1)
        return out

    def _engineer(self, X):
        return add_features(X) if self.feature_engineering else X.copy()

    def _clean_numeric(self, X):
        return X[self.numeric_].fillna(self.medians_).clip(self.lower_, self.upper_, axis=1)

    def _encode(self, X):
        num = pd.DataFrame(self.scaler_.transform(self._clean_numeric(X)), columns=self.numeric_, index=X.index)
        cat = pd.DataFrame(
            self.encoder_.transform(X[self.categorical_].fillna("Missing")),
            columns=self.encoder_.get_feature_names_out(),
            index=X.index,
        )
        return pd.concat([num, cat], axis=1)
