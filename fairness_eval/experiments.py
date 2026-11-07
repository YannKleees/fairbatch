import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from . import config as cfg, data, metrics
from .sampler import FairBatchFixed, FlatLogisticRegression, weights_init_normal

# a logistic regression on tabular data is faster on the CPU than on a GPU, and the authors' sampler
# keeps its tensors on the CPU
torch.set_num_threads(1)
METHODS = ["baseline", "fairbatch"]


def split(dataset, seed):
    """Stratified on label and group together, so each (y, z) cell has the same share in train and test."""
    x, y, z = data.load(dataset)
    train, test = train_test_split(np.arange(len(y)), test_size=cfg.TEST_SIZE, stratify=2 * y + z, random_state=seed)
    scaler = StandardScaler().fit(x.iloc[train])
    return (scaler.transform(x.iloc[train]), y[train], z[train]), (scaler.transform(x.iloc[test]), y[test], z[test])


def train(dataset, seed, method, **overrides):
    """Logistic regression trained with uniform minibatches (baseline) or with FairBatch targeting demographic parity."""
    s = cfg.TRAIN[dataset] | overrides
    (x, y, z), test = split(dataset, seed)
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)

    model = FlatLogisticRegression(x.shape[1], 1)
    model.apply(weights_init_normal)
    opt = torch.optim.Adam(model.parameters(), lr=s["lr"])
    x, z = torch.tensor(x, dtype=torch.float32), torch.tensor(z, dtype=torch.float32)
    y = torch.tensor(2 * y - 1, dtype=torch.float32)  # the sampler expects labels in {-1, 1}

    sampler = None
    if method == "fairbatch":
        sampler = FairBatchFixed(model, x, y, z, batch_size=s["batch_size"], alpha=s["alpha"], target_fairness="dp", seed=seed)

    start = time.time()
    for _ in range(s["epochs"]):
        batches = sampler if sampler else torch.randperm(len(y)).split(s["batch_size"])
        for idx in batches:
            idx = torch.as_tensor(idx, dtype=torch.long)
            opt.zero_grad()
            # the authors' loss, BCE on (tanh(f) + 1) / 2, which equals BCE with logits on 2f
            loss = F.binary_cross_entropy_with_logits(2 * model(x[idx]), (y[idx] + 1) / 2)
            loss.backward()
            opt.step()
    seconds = time.time() - start

    xt, yt, zt = test
    with torch.no_grad():
        pred = (model(torch.tensor(xt, dtype=torch.float32)) > 0).numpy().astype(int)
    row = {"dataset": dataset, "seed": seed, "method": method, **s, "seconds": seconds}
    row |= metrics.evaluate(xt, yt, pred, zt)
    if sampler:
        row |= {"lambda1": sampler.lb1, "lambda2": sampler.lb2}
    return row, sampler


def cached(name, compute):
    """Read results/<name>.csv, or compute it and save it. Delete the file to rerun."""
    path = cfg.RESULTS / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    cfg.RESULTS.mkdir(exist_ok=True)
    df = compute()
    df.to_csv(path, index=False)
    return df


def original_fairness():
    """DP_Diff, DIR and consistency of the original labels on the full dataset, before any split or model."""
    def compute():
        rows = []
        for name in cfg.DATASETS:
            x, y, z = data.load(name)
            rows.append({"dataset": name, **metrics.fairness(x.to_numpy(), y, z)})
        return pd.DataFrame(rows)
    return cached("original", compute)


def main_experiment():
    """Test set metrics per seed: the true labels of the test set, the baseline and FairBatch."""
    def compute():
        rows = []
        for name in cfg.DATASETS:
            for seed in tqdm(cfg.SEEDS, desc=name):
                _, (xt, yt, zt) = split(name, seed)
                rows.append({"dataset": name, "seed": seed, "method": "test_labels", **metrics.evaluate(xt, yt, yt, zt)})
                rows += [train(name, seed, method)[0] for method in METHODS]
        return pd.DataFrame(rows)
    return cached("main", compute)


def alpha_experiment():
    """FairBatch with different step sizes for the lambdas, to show the accuracy and fairness trade off."""
    def compute():
        rows = []
        for name in cfg.DATASETS:
            for alpha in tqdm(cfg.ALPHAS, desc=name):
                rows += [train(name, seed, "fairbatch", alpha=alpha)[0] for seed in cfg.SEEDS_ALPHA]
        return pd.DataFrame(rows)
    return cached("alpha", compute)


def lambda_trace(seed=0):
    """The two lambdas of FairBatch after every epoch, one run per dataset."""
    def compute():
        frames = []
        for name in cfg.DATASETS:
            lambdas = train(name, seed, "fairbatch")[1].lambdas
            frames.append(pd.DataFrame(lambdas, columns=["lambda1", "lambda2"]).rename_axis("epoch").reset_index().assign(dataset=name))
        return pd.concat(frames)
    return cached("lambdas", compute)


def summary(df, columns=("dp_diff", "dir", "consistency", "accuracy", "balanced_accuracy")):
    """Mean and std over seeds, one row per dataset and method."""
    order = {m: i for i, m in enumerate(["test_labels", *METHODS])}
    table = df.groupby(["dataset", "method"])[list(columns)].agg(["mean", "std"])
    return table.sort_index(key=lambda s: s.map(order) if s.name == "method" else s.map(cfg.DATASETS.index))


if __name__ == "__main__":
    original_fairness(), main_experiment(), alpha_experiment(), lambda_trace()
