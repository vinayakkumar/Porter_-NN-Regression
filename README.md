# Porter_-NN-Regression

Predicting food-delivery time for Porter-style orders with a neural network (PyTorch), compared against a Random Forest baseline.

The task is a data-analytics case study: given an order's market, store category, order size, time of day and how busy the market is, predict how many minutes the delivery takes.

## Start here

| File | What it is |
|:--|:--|
| [`problem_statement.html`](problem_statement.html) | The problem brief and the case-study page in one document. Part 1 is the brief: problem statement, data dictionary, process and the 10 questions. Part 2 is the case study: 5 stages with 27 tasks (1.1–1.5, 2.1–2.6, 3.1–3.6, 4.1–4.4, 5.1–5.6), each with an objective and hints. |
| [`solution.ipynb`](solution.ipynb) | **Main deliverable.** Full solution that follows the brief and the case-study page, with all outputs saved. |
| [`solution.pdf`](solution.pdf) | PDF copy of `solution.ipynb`. |
| [`sections/`](sections/) | The same solution split into five notebooks, one per section, for submitting section by section. Each has saved outputs. See [Section notebooks](#section-notebooks) below for how to run them and what they add to `solution.ipynb`. |
| [`scaler_solution/`](scaler_solution/) | The Scaler reference solution (`porter-delivery-time.ipynb`, Keras/TensorFlow) and a PyTorch translation of it (`porter-delivery-time-pytorch.ipynb`). See [Scaler reference solution](#scaler-reference-solution) below. |
| [`feedback/`](feedback/) | Saved copy of the Scaler case-study feedback page (`scaler-feedback-v5.html`). |
| [`solution_report_style.ipynb`](solution_report_style.ipynb) | An earlier, report-style version of the solution. It has no saved outputs and was not updated with the final changes, so treat `solution.ipynb` as the reference. |

Open the HTML file in a browser. Open the notebooks in Jupyter or VS Code.

## How the solution is organised

`solution.ipynb` has five sections. The marks follow the brief (100 in total). Each section opens with the stage objective, its tasks and hints, and ends with how it was addressed.

| Section | Marks | Tasks | Content |
|:--|:--:|:--:|:--|
| 1. Problem understanding and data | 10 | 1.1–1.5 | Business problem, target, features, success measure. |
| 2. Data preparation and features | 30 | 2.1–2.6 | Cleaning, target creation, feature engineering, encoding, split. |
| 3. Exploration and outliers | 20 | 3.1–3.6 | Distributions, relationships, outlier methods and handling. |
| 4. Baseline model | 10 | 4.1–4.4 | Random Forest, metrics, quick comparison with other models. |
| 5. Neural network | 30 | 5.1–5.6 | PyTorch network, 28 experiments, model selection, comparison with the forest, saving the model. |

All 27 tasks and all 10 brief questions are answered in the notebook.

## Section notebooks

The five notebooks in `sections/` continue from each other: Section 2 uses the `df` made in Section 1, and Section 5 uses `X_train` from Section 4. They cannot be run on their own. To re-run them, run all five **in order in one kernel** from the project folder (the data is read from the relative path `data/porter_delivery.csv`):

```bash
venv/bin/python build/run_sections.py "$PWD" sections/section_1_problem_and_data.ipynb sections/section_2_preprocessing.ipynb sections/section_3_visualization_outliers.ipynb sections/section_4_random_forest.ipynb sections/section_5_neural_network.ipynb
```

The run takes about 18 minutes, almost all of it in Section 5. All five ran with 0 errors on 2026-11-25.

After reviewer feedback (`feedback/scaler-feedback-v5.html`, checked against the reviewer's solution in `scaler_solution/`), four section notebooks were edited. The same edits are in `solution.ipynb`, so the two stay in step:

| Notebook | Added |
|:--|:--|
| `section_1_problem_and_data.ipynb` | A "Business context" paragraph at the start of the Task 1.1 answer: the \$40 billion market, 1,50,000+ driver-partners, 5+ million customers, and why the delivery-time estimate matters. |
| `section_2_preprocessing.ipynb` | Wording in 2.1 that `pd.to_datetime` checks the format and works out the order of the date parts by itself, and a "Handling the Null Values" step (`df.dropna()`, run with output: 174,837 rows before and after, 0 removed). |
| `section_4_random_forest.ipynb` | Answer cells for 4.2 (baseline with default hyperparameters, `random_state=42` for reproducibility) and 4.3 (metrics including R² and MAPE, and feature importance). |
| `section_3_visualization_outliers.ipynb` | A "Task 3.5 Answer" cell, and one figure that puts the scatterplots before (top) and after (bottom) the outlier removal side by side. |
| `section_5_neural_network.ipynb` | A labelled "Task 5.x Answer" cell for each of Tasks 5.1 to 5.6 (the 5.2 answer says the input layer is sized to the 24 inputs and points to the three-hidden-layer networks in 5.4); justification comments in the code cells of 5.2, 5.3 and 5.4; a table of the training and validation loss by epoch (the text form of the loss plot); and, in 5.4 and 5.5, text on what "validation loss tracking training loss" means and why both models need the same metrics. |

`build/split_sections.py` makes the section notebooks from `solution.ipynb` and gives the same notebooks as the ones here. `build/sync_solution_from_sections.py` copies edits made in a section notebook back into `solution.ipynb` (dry run by default, `--apply` to write). `solution.pdf` was made before these edits and has not been regenerated.

## Scaler reference solution

`scaler_solution/porter-delivery-time.ipynb` is the Scaler solution written with Keras. `scaler_solution/porter-delivery-time-pytorch.ipynb` has the same steps with the neural network rewritten in PyTorch (same layers, optimizer, learning rate, epochs and validation split; a hand-written training loop). It has saved outputs and runs in about 90 seconds from the `scaler_solution` folder. Like the Keras notebook, it starts with `!gdown 1WFa46c7_uSZ6GSzgYrDk4TKJd3y9zv8f`, which saves the data as `scaler_solution/data_2.csv` (needs `gdown` on the PATH, for example with the project's `venv` activated).

| Model (test set) | RMSE (min) | MAE (min) | R² | MAPE |
|:--|:--:|:--:|:--:|:--:|
| Random Forest | 1.795 | 1.286 | 0.963 | 2.77% |
| Neural network (PyTorch) | 0.932 | 0.827 | 0.990 | 1.85% |

These errors are much lower than the main solution's (RMSE about 4.4 to 4.7) because the Scaler notebook keeps `estimated_store_to_consumer_driving_duration` as a feature. The main solution leaves it out (see Known limitations), so the two sets of numbers are not comparable. The Scaler notebook also uses a different outlier method (Local Outlier Factor instead of Isolation Forest). The PyTorch notebook is made by `build/make_pytorch_version.py`.

## Results (test set, 34,583 orders)

| Model | RMSE (min) | MAE (min) | R² |
|:--|:--:|:--:|:--:|
| Neural network (1 hidden layer of 32 ReLU units, 833 weights) | 4.44 | 3.49 | 0.770 |
| Gradient boosting | 4.57 | – | – |
| Random Forest, tuned | 4.65 | – | – |
| Random Forest, default | 4.69 | 3.68 | 0.744 |
| Linear regression | 5.13 | – | – |

The network is better on every test I ran: the paired bootstrap, five network runs against four forest runs, and every market-busyness group. The lead is modest, about 0.25 minutes (5%). The notebook states this plainly.

## Known limitations

- One train/validation/test split (70/10/20); the multi-seed runs change the model seed, not the split.
- The outlier rules (Isolation Forest and a 99.9th-percentile cap on items) were fitted on all the data before the split. The notebook notes this.
- `estimated_store_to_consumer_driving_duration` is left out because it is not in the brief's data dictionary. Its effect was not measured.
- Store category is an integer code, which suits a neural network less well than an embedding.

## Data

`data/porter_delivery.csv` has 175,777 rows and 14 columns (172,914 rows are left after cleaning). It is the Google Drive file shared in the problem brief (file id `1kkgBGldeswHBgVKEJiqqG7VZvqZNMc-5`, 15.7 MB).

To download it:

```bash
python download_data.py            # skips the download if the file is already there
python download_data.py --force    # download again and replace it
```

This needs `gdown` (in `requirements.txt`). You do not have to run it yourself: `solution.ipynb` and the section 1 notebook download the file with `gdown` when it is missing, and the notebooks in `scaler_solution/` download their own copy (`data_2.csv`) with `!gdown`. The reference notebook's file (`data_2.csv`, id `1WFa46c7_uSZ6GSzgYrDk4TKJd3y9zv8f`) was checked and is byte-for-byte the same file.

## Running it

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab solution.ipynb
```

Running every cell takes about 9 minutes on a laptop. The notebook saves the trained network and its preprocessing to `models/` (`porter_delivery_nn.pt`, `porter_delivery_preprocessing.joblib`).

## Other folders and files

| Path | What it is |
|:--|:--|
| `models/` | Saved models. `porter_nn.pt` is from earlier work; the `porter_delivery_*` files come from `solution.ipynb`. |
| `nn_day_1` … `nn_day_12` | Course exercises from the neural-network course. Not part of the case study. |
| `context.md` | Working log: decisions, changes and open items. |
| `helper.md`, `intro_to_neural_networks.md`, `env_check.ipynb` | Course notes and an environment check. |
| `requirements.txt` | Python packages. |
