"""Run the whole pipeline: EDA -> clustering -> churn classification.

Usage: python -m app.main
"""

from app import classification, clustering, eda

if __name__ == "__main__":
    eda.main()
    clustering.main()  # must run first: classification reuses its number of clusters
    classification.main()