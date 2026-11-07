from pathlib import Path

# next to this folder, so that the notebook and the scripts find the cache from any working directory
RESULTS = Path(__file__).parent.parent / "results"

UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases"
URLS = {
    "adult": [f"{UCI}/adult/adult.data", f"{UCI}/adult/adult.test"],
    "german": [f"{UCI}/statlog/german/german.data"],
    "compas": ["https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"],
}
DATASETS = list(URLS)

# sensitive attribute per dataset, z = 1 is the privileged group
SENSITIVE = {
    "adult": "sex (male = 1)",
    "german": "age (older than 25 = 1)",
    "compas": "race (Caucasian = 1, African-American = 0)",
}

SEEDS = range(10)
TEST_SIZE = 0.3
K_NEIGHBORS = 5  # consistency, as in Zemel et al. (2013)

# training settings per dataset. Adult is 45k rows, so larger batches keep a run under a minute.
# The authors use batch 100, alpha 0.005 and Adam with lr 0.0005 on 2000 synthetic rows.
TRAIN = {
    "adult": dict(batch_size=500, epochs=100, lr=0.005, alpha=0.005),
    "german": dict(batch_size=50, epochs=300, lr=0.005, alpha=0.005),
    "compas": dict(batch_size=100, epochs=200, lr=0.005, alpha=0.005),
}

# step sizes for the trade off experiment, fewer seeds
ALPHAS = [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02]
SEEDS_ALPHA = range(5)
