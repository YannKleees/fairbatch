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

# Batch sizes and epochs of the paper's DP experiments (Sec. 4 and Table 2): 1,000 and 600 for Adult,
# 200 and 300 for COMPAS. German is not in the paper; it has 700 training rows, so it gets small batches.
# The paper uses Adam but gives no learning rate.
TRAIN = {
    "adult": dict(batch_size=1000, epochs=600, lr=0.005),
    "german": dict(batch_size=50, epochs=300, lr=0.005),
    "compas": dict(batch_size=200, epochs=300, lr=0.005),
}

# alpha is chosen as in the paper (Sec. 4): candidates within [0.0001, 0.05], picked on the training data.
# We hold out a stratified validation set of VAL_SIZE of the training data for seeds SEEDS_VAL, average
# the validation metrics over these seeds, and take the alpha with the highest balanced accuracy among
# those whose mean |DP_Diff| is at most DP_TOLERANCE (the smallest |DP_Diff| if none qualifies).
ALPHAS = [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02]
VAL_SIZE = 0.2
SEEDS_VAL = range(5)
DP_TOLERANCE = 0.05

# FairBatch's DP results in the paper (Table 2, AdultCensus, sex as sensitive attribute):
# accuracy and DP disparity as max_z |P(y_hat = 1 | z) - P(y_hat = 1)|
PAPER_TABLE2_ADULT = {"baseline": (0.845, 0.125), "fairbatch": (0.823, 0.010)}
