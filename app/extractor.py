import pandas as pd

from app.config import ID_COLUMN, RAW_DATA, TARGET


class DataExtractor:
    """Load the raw customer file and report its quality."""

    def __init__(self, path=RAW_DATA):
        self.path = path
        self.df = None

    def load(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.path)
        # TotalCharges is blank for brand-new customers, so pandas reads it as text.
        self.df["TotalCharges"] = pd.to_numeric(self.df["TotalCharges"], errors="coerce")
        return self.df

    def features_and_target(self) -> tuple[pd.DataFrame, pd.Series]:
        """Customer features and the 0/1 churn target, without ID and duplicates."""
        df = self.df.drop(columns=ID_COLUMN).drop_duplicates().reset_index(drop=True)
        y = (df.pop(TARGET) == "Yes").astype(int)
        return df, y

    def missing_values(self) -> pd.Series:
        missing = self.df.isna().sum()
        return missing[missing > 0]

    def duplicates(self) -> int:
        # customerID is unique, so duplicates are customers identical on every other column.
        return int(self.df.drop(columns=ID_COLUMN).duplicated().sum())

    def outliers(self, factor: float = 1.5) -> pd.Series:
        num = self.df.select_dtypes("number")
        num = num.loc[:, num.nunique() > 2]  # binary flags like SeniorCitizen are not outliers
        q1, q3 = num.quantile(0.25), num.quantile(0.75)
        iqr = q3 - q1
        return ((num < q1 - factor * iqr) | (num > q3 + factor * iqr)).sum()

    def report(self) -> dict:
        return {
            "rows": len(self.df),
            "columns": self.df.shape[1],
            "missing_values": self.missing_values().to_dict(),
            "duplicate_rows": self.duplicates(),
            "outliers_iqr": self.outliers().to_dict(),
            "churn_rate": round(float((self.df[TARGET] == "Yes").mean()), 3),
        }
