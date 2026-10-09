from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA = ROOT / "data" / "raw" / "wa-fn-usec-telco-customer.csv"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
MLFLOW_URI = f"sqlite:///{ROOT / 'mlflow.db'}"

TARGET = "Churn"
ID_COLUMN = "customerID"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_JOBS = 4  # parallel CV workers; more runs out of memory on a 16 GB machine

# Candidate k values for the elbow / silhouette analysis; the best one is saved
# to models/segmentation.json and reused by the classification step.
K_RANGE = range(2, 9)
DBSCAN_PARAMS = {"eps": 3.0, "min_samples": 10}


def setup_mlflow(experiment: str) -> None:
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(ROOT / "mlruns" / experiment).as_uri())
    mlflow.set_experiment(experiment)
