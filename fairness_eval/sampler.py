import random
import sys
from pathlib import Path
import numpy as np
import torch

# the authors' files live in paper_code/, unchanged
sys.path.insert(0, str(Path(__file__).parent.parent / "paper_code"))
from FairBatchSampler import FairBatch  # noqa: E402
from models import LogisticRegression, weights_init_normal  # noqa: E402,F401


class FlatLogisticRegression(LogisticRegression):
    """fix 1: the authors' model returns shape (n, 1), which adjust_lambda compares with targets of shape (n,).
    Older PyTorch only warned; PyTorch 2.x raises a ValueError. Returning shape (n,) fixes it without touching their code."""

    def forward(self, input_data):
        return super().forward(input_data).squeeze(-1)


class FairBatchFixed(FairBatch):
    """The authors' sampler with two fixes (see the README). Also records, once per epoch, the lambdas and the
    demographic parity gap P(y_hat = 1 | z = 0) - P(y_hat = 1 | z = 1) of the model on the training data.
    Expects y in {-1, 1} and z in {0, 1}, both as float tensors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = []

    def train_dp_gap(self):
        with torch.no_grad():
            pred = (self.model(self.x_data) > 0).float()
        return float(pred[self.z_data == 0].mean() - pred[self.z_data == 1].mean())

    def update(self):
        # fix 2: the original runs a full forward pass with autograd on and leaves the model in eval mode
        with torch.no_grad():
            super().adjust_lambda()

    def adjust_lambda(self):
        self.trace.append((self.lb1, self.lb2, self.train_dp_gap()))
        self.update()
        self.model.train()

    def select_batch_replacement(self, batch_size, full_index, batch_num, replacement=False):
        if replacement:
            return super().select_batch_replacement(batch_size, full_index, batch_num, replacement)
        # fix 3: the original sets the next start to len - start after wrapping around a group, so later
        # batches overlap earlier ones. Here each group is shuffled once and read cyclically.
        idx = full_index.cpu().numpy().copy()
        random.shuffle(idx)
        return np.split(np.resize(idx, batch_size * batch_num), batch_num)


class FairBatchRate(FairBatchFixed):
    """Our variant, not from the paper. The authors' DP update (Algorithm 4) compares the surrogate loss against
    label 1 within (y, z) cells and moves only the lambda of the worse cell. This variant keeps the same batch
    composition but moves both lambdas by the sign of the actual DP gap of thresholded predictions on the
    training data: when group z = 1 receives more positive predictions, fewer z = 1 positives and more z = 1
    negatives are sampled, and the other way round."""

    def update(self):
        step = -np.sign(self.train_dp_gap()) * self.alpha  # gap < 0 favours z = 1
        self.lb1 = min(max(self.lb1 - step, 0), 1)  # lb1: share of z = 1 among the sampled positives
        self.lb2 = min(max(self.lb2 + step, 0), 1)  # lb2: share of z = 1 among the sampled negatives
