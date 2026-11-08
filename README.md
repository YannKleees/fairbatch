# FairBatch on Adult, German and COMPAS

An evaluation of FairBatch (Roh et al., ICLR 2021) as a bias mitigation method for demographic parity on three public datasets. The authors' code is in `paper_code/`, unchanged. Everything else is mine.

## Where things are

```text
├── evaluation.ipynb      the evaluation: all experiments, tables and figures, with saved outputs
├── fairness_eval/        the code behind the notebook
│   ├── config.py         datasets, seeds, training settings, alpha candidates
│   ├── data.py           downloads and preprocesses the datasets
│   ├── metrics.py        fairness and accuracy metrics
│   ├── sampler.py        the authors' sampler with three fixes, and a rate based variant
│   └── experiments.py    training loop, alpha selection and experiments
├── results/              cached results, one CSV per experiment
├── docs/Paper.pdf        the FairBatch paper
└── paper_code/           the authors' code from yuji-roh/fairbatch, with their README
```

## How to run

You need Python 3.11 or newer and an internet connection, since the datasets are downloaded on first use.

```
git clone https://github.com/YannKleees/fairbatch.git
cd fairbatch
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Open `evaluation.ipynb` from the repository root and run all cells. Results are loaded from `results/`, so this takes under a minute. Delete a CSV to rerun that experiment, or the whole folder to rerun everything (about 25 minutes on a two core CPU). You can also run all experiments without the notebook with `python -m fairness_eval.experiments`.
