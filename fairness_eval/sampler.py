import random
import sys
from pathlib import Path
import numpy as np
import torch

# the authors' files live in the repository root
sys.path.insert(0, str(Path(__file__).parent.parent))
from FairBatchSampler import FairBatch  # noqa: E402
from models import LogisticRegression, weights_init_normal  # noqa: E402,F401


class FlatLogisticRegression(LogisticRegression):
    """fix 3: the authors' model returns shape (n, 1), which adjust_lambda compares with targets of shape (n,).
    Older PyTorch only warned; PyTorch 2.x raises a ValueError. Returning shape (n,) fixes it without touching their code."""

    def forward(self, input_data):
        return super().forward(input_data).squeeze(-1)


class FairBatchFixed(FairBatch):
    """The authors' sampler, unchanged except for two fixes (see the README), plus a record of the lambdas.
    Expects y in {-1, 1} and z in {0, 1}, both as float tensors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambdas = [(self.lb1, self.lb2)]

    def adjust_lambda(self):
        # fix 1: the original runs a full forward pass with autograd on and leaves the model in eval mode
        with torch.no_grad():
            super().adjust_lambda()
        self.model.train()
        self.lambdas.append((self.lb1, self.lb2))

    def select_batch_replacement(self, batch_size, full_index, batch_num, replacement=False):
        if replacement:
            return super().select_batch_replacement(batch_size, full_index, batch_num, replacement)
        # fix 2: the original sets the next start to len - start after wrapping around a group, so later
        # batches overlap earlier ones. Here each group is shuffled once and read cyclically.
        idx = full_index.cpu().numpy().copy()
        random.shuffle(idx)
        return np.split(np.resize(idx, batch_size * batch_num), batch_num)
