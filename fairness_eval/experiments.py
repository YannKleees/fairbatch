import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from . import config as cfg, data, metrics
from .sampler import FairBatchFixed, FairBatchRate, FlatLogisticRegression, weights_init_normal

# a logistic regression on tabular data is faster on the CPU than on a GPU, and the authors' sampler
# keeps its tensors on the CPU. One thread per run, runs in parallel.
torch.set_num_threads(1)
N_JOBS = -1
SAMPLERS = {"fairbatch": FairBatchFixed, "fairbatch_rate": FairBatchRate}
METHODS = ["baseline", *SAMPLERS]


def split(dataset, seed, validation=False):
    """Stratified on label and group together, so each (y, z) cell has the same share in every part.
    validation=True splits the training part once more and returns (fit, validation) instead of (train, test)."""
    x, y, z = data.load(dataset)
    strata = 2 * y + z
    train, test = train_test_split(np.arange(len(y)), test_size=cfg.TEST_SIZE, stratify=strata, random_state=seed)
    if validation:
        train, test = train_test_split(train, test_size=cfg.VAL_SIZE, stratify=strata[train], random_state=seed)
    scaler = StandardScaler().fit(x.iloc[train])
    return (scaler.transform(x.iloc[train]), y[train], z[train]), (scaler.transform(x.iloc[test]), y[test], z[test])


def train(dataset, seed, method, alpha=None, validation=False):
    """Logistic regression trained with uniform minibatches (baseline) or with a FairBatch sampler targeting
    demographic parity. Returns the evaluation row and the sampler (None for the baseline)."""
    s = cfg.TRAIN[dataset]
    (x, y, z), test = split(dataset, seed, validation)
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)

    model = FlatLogisticRegression(x.shape[1], 1)
    model.apply(weights_init_normal)
    opt = torch.optim.Adam(model.parameters(), lr=s["lr"])
    x, z = torch.tensor(x, dtype=torch.float32), torch.tensor(z, dtype=torch.float32)
    y = torch.tensor(2 * y - 1, dtype=torch.float32)  # the sampler expects labels in {-1, 1}

    sampler = None
    if method in SAMPLERS:
        sampler = SAMPLERS[method](model, x, y, z, batch_size=s["batch_size"], alpha=alpha, target_fairness="dp", seed=seed)

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
    row = {"dataset": dataset, "seed": seed, "method": method, "alpha": alpha, **s, "seconds": seconds}
    row |= metrics.evaluate(xt, yt, pred, zt)
    if sampler:
        row |= {"lambda1": sampler.lb1, "lambda2": sampler.lb2}
    return row, sampler


def row(*args, **kwargs):
    return train(*args, **kwargs)[0]


def run(jobs):
    return Parallel(n_jobs=N_JOBS)(delayed(row)(*a, **k) for a, k in jobs)


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


def alpha_validation():
    """Both FairBatch variants for every candidate alpha, trained on the fit part and evaluated on the
    validation part of the training data. The test set is not touched."""
    jobs = [((name, seed, method), dict(alpha=alpha, validation=True))
            for name in cfg.DATASETS for method in SAMPLERS for alpha in cfg.ALPHAS for seed in cfg.SEEDS_VAL]
    return cached("alpha_validation", lambda: pd.DataFrame(run(jobs)))


def choose_alpha(validation):
    """Per dataset and variant: the alpha with the highest mean validation balanced accuracy among those with
    mean |DP_Diff| <= DP_TOLERANCE, or the smallest mean |DP_Diff| if none qualifies."""
    mean = validation.assign(abs_dp=validation.dp_diff.abs()).groupby(["dataset", "method", "alpha"])[["abs_dp", "balanced_accuracy"]].mean()
    chosen = {}
    for (name, method), g in mean.groupby(level=[0, 1]):
        g = g.droplevel([0, 1])
        ok = g[g.abs_dp <= cfg.DP_TOLERANCE]
        chosen[name, method] = ok.balanced_accuracy.idxmax() if len(ok) else g.abs_dp.idxmin()
    return chosen


def main_experiment():
    """Test set metrics per seed: the true labels of the test set, the baseline and both FairBatch variants
    with the alpha chosen on validation data."""
    def compute():
        alphas = choose_alpha(alpha_validation())
        jobs = [((name, seed, method), dict(alpha=alphas.get((name, method))))
                for name in cfg.DATASETS for seed in cfg.SEEDS for method in METHODS]
        rows = run(jobs)
        for name in cfg.DATASETS:
            for seed in cfg.SEEDS:
                _, (xt, yt, zt) = split(name, seed)
                rows.append({"dataset": name, "seed": seed, "method": "test_labels", **metrics.evaluate(xt, yt, yt, zt)})
        return pd.DataFrame(rows)
    return cached("main", compute)


def trace(seed=0):
    """Lambdas and the training DP gap after every epoch, both variants, one run per dataset, chosen alpha."""
    def compute():
        alphas = choose_alpha(alpha_validation())
        frames = []
        for name in cfg.DATASETS:
            for method in SAMPLERS:
                sampler = train(name, seed, method, alpha=alphas[name, method])[1]
                df = pd.DataFrame(sampler.trace, columns=["lambda1", "lambda2", "train_dp_diff"])
                frames.append(df.rename_axis("epoch").reset_index().assign(dataset=name, method=method))
        return pd.concat(frames)
    return cached("trace", compute)


def summary(df, columns=("dp_diff", "dir", "consistency", "accuracy", "balanced_accuracy")):
    """Mean and std over seeds, one row per dataset and method."""
    order = {m: i for i, m in enumerate(["test_labels", *METHODS])}
    table = df.groupby(["dataset", "method"])[list(columns)].agg(["mean", "std"])
    return table.sort_index(key=lambda s: s.map(order) if s.name == "method" else s.map(cfg.DATASETS.index))


if __name__ == "__main__":
    original_fairness(), main_experiment(), trace()
