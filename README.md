# Porter_-NN-Regression

Predicting food-delivery time for Porter-style orders with a neural network (PyTorch), compared against a Random Forest baseline.

The task is a data-analytics case study: given an order's market, store category, order size, time of day and how busy the market is, predict how many minutes the delivery takes.

## Start here

| File | What it is |
|:--|:--|
| [`problem_statement.html`](problem_statement.html) | The problem brief and the case-study page in one document. Part 1 is the brief: problem statement, data dictionary, process and the 10 questions. Part 2 is the case study: 5 stages with 27 tasks (1.1–1.5, 2.1–2.6, 3.1–3.6, 4.1–4.4, 5.1–5.6), each with an objective and hints. |
| [`solution.ipynb`](solution.ipynb) | **Main deliverable.** Full solution that follows the brief and the case-study page, with all outputs saved. |
| [`solution.pdf`](solution.pdf) | PDF copy of `solution.ipynb`. |
| [`sections/`](sections/) | The same solution split into five notebooks, one per section, for submitting section by section. Outputs are saved from the full run; to re-run the code, use `solution.ipynb`. |
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

`data/porter_delivery.csv` (172,914 rows after cleaning). If it is missing, the notebook downloads it from the Google Drive link in the brief using `gdown`.

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
