# FairBatch on Adult, German and COMPAS

This fork evaluates FairBatch (Roh et al., ICLR 2021) as a bias mitigation method on three public datasets. FairBatch is an in processing method: it changes how many examples of each (label, group) cell enter every minibatch, so that the trained model satisfies demographic parity. For each dataset, the code computes DP_Diff, the disparate impact ratio (DIR) and consistency on the original labels, trains a logistic regression with and without FairBatch on a stratified split, and reports the same measures plus accuracy and balanced accuracy on the held out test set.

The authors' files (`FairBatchSampler.py`, `models.py`, `FairBatch_Synthetic.ipynb`, `synthetic_data/`) are unchanged. Their original README is kept at the bottom of this file.

## Where to look

```text
├── evaluation.ipynb          runs every experiment and shows all tables and figures, with saved outputs
├── fairness_eval/            the code behind the notebook
│   ├── config.py             datasets, sensitive attributes, seeds and training settings
│   ├── data.py               downloads and preprocesses Adult, German and COMPAS
│   ├── metrics.py            DP_Diff, DIR, consistency, accuracy and balanced accuracy
│   ├── sampler.py            the authors' FairBatch sampler with three fixes, see below
│   └── experiments.py        split, training loop and the experiments
├── results/                  the cached results, one CSV per experiment
│   ├── original.csv          fairness of the original labels, full datasets
│   ├── main.csv              test set results, baseline and FairBatch, 10 seeds
│   ├── lambdas.csv           FairBatch's two lambdas after every epoch, seed 0
│   └── alpha.csv             FairBatch with six step sizes, 5 seeds
├── FairBatchSampler.py       the authors' sampler (unchanged)
├── models.py                 the authors' logistic regression (unchanged)
└── FairBatch_Synthetic.ipynb the authors' demo on synthetic data (unchanged)
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

Then open `evaluation.ipynb` with this environment from the repository root and run all cells. The results are cached in `results/`, so this takes under a minute. To rerun an experiment, delete its CSV; to rerun everything, delete the `results/` folder. A full rerun takes about ten minutes on one CPU core. Without the notebook, `python -m fairness_eval.experiments` runs all experiments and writes the CSVs.

Tested with Python 3.13, PyTorch 2.15 (CPU), numpy 2.5, pandas 3.0, scikit-learn 1.9.

## Settings

| | Adult | German | COMPAS |
|---|---|---|---|
| Source | UCI, `adult.data` and `adult.test` | UCI, `german.data` | ProPublica, `compas-scores-two-years.csv` |
| Rows after cleaning | 45,222 | 1,000 | 5,278 |
| Sensitive attribute, z = 1 privileged | sex, male | age, older than 25 | race, Caucasian (African-American = 0) |
| Favorable label y = 1 | income above 50K | good credit | no recidivism within two years |
| Batch size, epochs | 500, 100 | 50, 300 | 100, 200 |

All datasets: logistic regression from the authors' `models.py`, Adam with learning rate 0.005, FairBatch step size `alpha` = 0.005 (the authors' value), `target_fairness="dp"`, 70/30 train/test split stratified on label and group together, features standardized on the training set, seeds 0 to 9. The baseline is the same model trained with uniform minibatches of the same size.

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

The authors' `test_model` reports DP as max |P(ŷ=1 | z) − P(ŷ=1)|, the largest deviation of a group from the overall rate. That is smaller than the DP_Diff above, so it is not used.

## Changes to the authors' sampler

`fairness_eval/sampler.py` subclasses the authors' `FairBatch` and `LogisticRegression` without editing their files:

1. Output shape. The authors' model returns shape (n, 1), and `adjust_lambda` compares it with targets of shape (n,). Older PyTorch only warned; PyTorch 2.x raises a `ValueError`, so the original code no longer runs. `FlatLogisticRegression` returns shape (n,).
2. Lambda update. `adjust_lambda` runs a forward pass over the whole training set with autograd enabled and then leaves the model in eval mode. It now runs under `torch.no_grad()` and switches the model back to training mode. This makes no difference for logistic regression, but it would for models with dropout or batch normalization.
3. Batch indices. When a group runs out of examples within an epoch, `select_batch_replacement` sets the next start index to `len - start` instead of `batch_size - (len - start)`, so later batches overlap earlier ones. Each group is now shuffled once per epoch and read cyclically.

The training loop iterates over the sampler directly instead of wrapping it in a `DataLoader` with batch size 1, and uses `binary_cross_entropy_with_logits(2f, y)`, which equals the authors' loss, BCE on (tanh(f) + 1) / 2, but is numerically stable.

---

# Original README of the authors

## FairBatch: Batch Selection for Model Fairness

#### Authors: Yuji Roh, Kangwook Lee, Steven Euijong Whang, and Changho Suh
#### In Proceedings of the 9th International Conference on Learning Representations (ICLR), 2021
----------------------------------------------------------------------

This repo contains codes used in the ICLR 2021 paper: [FairBatch: Batch Selection for Model Fairness](https://arxiv.org/abs/2012.01696)

*Abstract: Training a fair machine learning model is essential to prevent demographic disparity. Existing techniques for improving model fairness require broad changes in either data preprocessing or model training, rendering themselves difficult-to-adopt for potentially already complex machine learning systems. We address this problem via the lens of bilevel optimization. While keeping the standard training algorithm as an inner optimizer, we incorporate an outer optimizer so as to equip the inner problem with an additional functionality: Adaptively selecting minibatch sizes for the purpose of improving model fairness. Our batch selection algorithm, which we call FairBatch, implements this optimization and supports prominent fairness measures: equal opportunity, equalized odds, and demographic parity. FairBatch comes with a significant implementation benefit -- it does not require any modification to data preprocessing or model training. For instance, a single-line change of PyTorch code for replacing batch selection part of model training suffices to employ FairBatch. Our experiments conducted both on synthetic and benchmark real data demonstrate that FairBatch can provide such functionalities while achieving comparable (or even greater) performances against the state of the arts. Furthermore, FairBatch can readily improve fairness of any pre-trained model simply via fine-tuning. It is also compatible with existing batch selection techniques intended for different purposes, such as faster convergence, thus gracefully achieving multiple purposes.*


### Setting
This directory is for simulating FairBatch on the synthetic dataset.
The program needs PyTorch and Jupyter Notebook.

The directory contains a total of 4 files and 1 child directory: 
1 README, 2 python files, 1 jupyter notebook, 
and the child directory containing 6 numpy files for synthetic data.

### Simulation
To simulate FairBatch, please use the **jupyter notebook** in the directory.

The jupyter notebook will load the data and train the models with three 
different fairness metrics: equal opportunity, equalized odds, and demographic parity.

Each training utilizes the FairBatch sampler, which is defined in FairBatchSampler.py.
The pytorch dataloader serves the batches to the model via the FairBatch sampler. 
Experiments are repeated 10 times each.
After the training, the test accuracy and fairness will be shown.

### Other details
The two python files are models.py and FairBatchSampler.py.
The models.py file contains a logistic regression architecture and a test function.
The FairBatchSampler.py file contains two classes: CustomDataset and FairBatch. 
The CustomDataset class defines the dataset, and the FairBatch class implements 
the algorithm of FairBatch as described in the paper.

More detailed explanations of each component can be found in the code as comments.
Thanks!

### Demos using Google Colab
We also release Google Colab notebooks for fast demos.
You can access both the [PyTorch version](https://colab.research.google.com/drive/192tZmf-jXg1uesHW2TSqv0LoDbhAW4X1?usp=sharing) and the [TensorFlow version](https://colab.research.google.com/drive/1VBc7osg-wRKTKav32k1wY2yfKWK-wnDW?usp=sharing).

### Reference
```
@inproceedings{
roh2021fairbatch,
title={FairBatch: Batch Selection for Model Fairness},
author={Yuji Roh and Kangwook Lee and Steven Euijong Whang and Changho Suh},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=YNnpaAKeCfx}
}
```

