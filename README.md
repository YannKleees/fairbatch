# FairBatch on Adult, German and COMPAS

This fork evaluates FairBatch (Roh et al., ICLR 2021) as a bias mitigation method on three public datasets. FairBatch is an in processing method: it changes how many examples of each (label, group) cell enter every minibatch, so that the trained model satisfies demographic parity. For each dataset, the code computes DP_Diff, the disparate impact ratio (DIR) and consistency on the original labels, trains a logistic regression with and without FairBatch on a stratified split, and reports the same measures plus accuracy and balanced accuracy on the held out test set.

The repository has two separate parts:

- `paper_code/` is the authors' code, exactly as published in [yuji-roh/fairbatch](https://github.com/yuji-roh/fairbatch), with their README.
- Everything else is ours: the `fairness_eval/` package, the notebook, the results and this README. Our code imports the authors' sampler and model from `paper_code/` and never edits them.

## Where to look

```text
├── evaluation.ipynb          our evaluation: runs every experiment and shows all tables and figures, with saved outputs
├── fairness_eval/            our code behind the notebook
│   ├── config.py             datasets, sensitive attributes, seeds, training settings, alpha candidates
│   ├── data.py               downloads and preprocesses Adult, German and COMPAS
│   ├── metrics.py            DP_Diff, DIR, consistency, the paper's DP disparity, accuracy, balanced accuracy
│   ├── sampler.py            the authors' sampler with three fixes, and our rate based variant
│   └── experiments.py        splits, training loop, alpha selection and the experiments
├── results/                  cached results, one CSV per experiment
│   ├── original.csv          fairness of the original labels, full datasets
│   ├── alpha_validation.csv  both FairBatch variants for six alphas on validation data, 5 seeds
│   ├── main.csv              test set results of all methods, 10 seeds
│   └── trace.csv             lambdas and training DP gap after every epoch, seed 0
├── docs/FairBatch.pdf        the paper
└── paper_code/               the authors' code, unchanged
    ├── README.md             the authors' README
    ├── FairBatchSampler.py   the FairBatch sampler
    ├── models.py             logistic regression and their test function
    ├── FairBatch_Synthetic.ipynb  their demo on synthetic data (run it from inside paper_code/)
    └── synthetic_data/       the data of their demo
```

## Running

You need Python 3.11 or newer and an internet connection, because the datasets are downloaded from the UCI repository and the ProPublica GitHub repository on first use. A GPU is not needed.

```
git clone https://github.com/YannKleees/fairbatch.git
cd fairbatch
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Then open `evaluation.ipynb` with this environment from the repository root and run all cells. The results are cached in `results/`, so this takes under a minute. To rerun an experiment, delete its CSV; to rerun everything, delete the `results/` folder. `main.csv` and `trace.csv` depend on the alphas chosen from `alpha_validation.csv`, so delete them too when you rerun the alpha selection. A full rerun takes about 25 minutes on a two core CPU; runs are parallelized over all cores. Without the notebook, `python -m fairness_eval.experiments` runs all experiments and writes the CSVs.

Tested with Python 3.13, PyTorch 2.15 (CPU), numpy 2.5, pandas 3.0, scikit-learn 1.9.

## Methods

- `baseline`: logistic regression trained with uniform minibatches.
- `fairbatch`: the same model trained with the authors' FairBatch sampler, `target_fairness="dp"` (Algorithm 4 of the paper), with the fixes listed below.
- `fairbatch_rate`: our variant, not from the paper. It keeps FairBatch's batch composition but drives both lambdas by the sign of the actual demographic parity gap of the model's predictions on the training data, instead of the paper's comparison of surrogate losses within (label, group) cells. It tests why the authors' update overshoots on COMPAS (notebook section 7).

## Settings

| | Adult | German | COMPAS |
|---|---|---|---|
| Source | UCI, `adult.data` and `adult.test` | UCI, `german.data` | ProPublica, `compas-scores-two-years.csv` |
| Rows after cleaning | 45,222 | 1,000 | 5,278 |
| Sensitive attribute, z = 1 privileged | sex, male | age, older than 25 | race, Caucasian (African-American = 0) |
| Favorable label y = 1 | income above 50K | good credit | no recidivism within two years |
| Batch size, epochs | 1,000, 600 (paper) | 50, 300 (not in the paper) | 200, 300 (paper) |
| alpha, fairbatch / fairbatch_rate | 0.02 / 0.02 | 0.0025 / 0.0005 | 0.0005 / 0.01 |

All datasets: logistic regression from the authors' `models.py`, Adam with learning rate 0.005 (the paper does not give one), 70/30 train/test split stratified on label and group together, features standardized on the training data, seeds 0 to 9. The baseline uses the same batch size and epochs.

Choice of alpha: as in the paper, alpha is selected on the training data only, from candidates in the paper's range [0.0001, 0.05]: 0.0005, 0.001, 0.0025, 0.005, 0.01 and 0.02. For seeds 0 to 4, 20% of the training part is held out as a validation set, stratified on label and group. Per dataset and variant, we take the alpha with the highest mean validation balanced accuracy among those whose mean validation |DP_Diff| is at most 0.05, or the smallest mean |DP_Diff| if no alpha reaches it. The test set is used only for the final results.

Preprocessing:

- Adult: rows with missing values are dropped, `fnlwgt` (a census sampling weight) and `education` (duplicates `education_num`) are removed, categorical features are one hot encoded.
- German: all 20 attributes, categorical codes one hot encoded.
- COMPAS: the filters of the ProPublica analysis (charge within 30 days of screening, known recidivism, no ordinary traffic offences, a valid score), only African-American and Caucasian defendants. Features are sex, age, age category, race, charge degree, priors count and the three juvenile counts. The COMPAS score itself is not used.
- The sensitive attribute stays in the features, as in the authors' code.

Metrics:

- DP_Diff = P(y=1 | z=0) − P(y=1 | z=1). Zero is parity; negative means the unprivileged group receives fewer favorable outcomes.
- DIR = P(y=1 | z=0) / P(y=1 | z=1). One is parity; below 0.8 fails the four fifths rule.
- Consistency (Zemel et al., 2013) = 1 − mean |y_i − mean label of the 5 nearest neighbours of i|, with neighbours found on standardized features, excluding the point itself.
- "Before" is computed on the original labels of the full dataset; "after" on the predictions for the test set.
- `dp_paper` is the paper's DP disparity, max over groups of |P(ŷ=1 | z) − P(ŷ=1)|. It equals |DP_Diff| times the share of the larger group, so it is smaller than DP_Diff. We report it only to compare with the paper's Table 2 (Adult with sex, the one setting we share with the paper).

## Changes to the authors' sampler

`fairness_eval/sampler.py` subclasses the authors' `FairBatch` and `LogisticRegression` without editing their files:

1. Output shape. The authors' model returns shape (n, 1), and `adjust_lambda` compares it with targets of shape (n,). Older PyTorch only warned; PyTorch 2.x raises a `ValueError`, so the original code no longer runs. `FlatLogisticRegression` returns shape (n,).
2. Lambda update. `adjust_lambda` runs a forward pass over the whole training set with autograd enabled and then leaves the model in eval mode. It now runs under `torch.no_grad()` and switches the model back to training mode. This makes no difference for logistic regression, but it would for models with dropout or batch normalization.
3. Batch indices. When a group runs out of examples within an epoch, `select_batch_replacement` sets the next start index to `len - start` instead of `batch_size - (len - start)`, so later batches overlap earlier ones. Each group is now shuffled once per epoch and read cyclically.

The training loop iterates over the sampler directly instead of wrapping it in a `DataLoader` with batch size 1, and uses `binary_cross_entropy_with_logits(2f, y)`, which equals the authors' loss, BCE on (tanh(f) + 1) / 2, but is numerically stable.
