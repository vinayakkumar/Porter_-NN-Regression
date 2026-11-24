# Helper

## Project
Porter NN Regression

## Overview
Describe the goal of the project here (e.g. predicting delivery time/price with a neural network regression model).

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure
```
Porter_NN_Regression/
├── helper.md
├── data/          # raw and processed datasets
├── notebooks/     # exploration and experiments
├── src/           # training / preprocessing code
└── models/        # saved models
```

## Workflow
1. Load and clean data
2. Feature engineering
3. Train/validation/test split
4. Build and train the neural network
5. Evaluate (MAE, RMSE, R²)
6. Save the model and results

## Useful Commands
| Task | Command |
|------|---------|
| Activate env | `source venv/bin/activate` |
| Run training | `python src/train.py` |
| Launch notebook | `jupyter notebook` |

## Notes
- Add experiment results, decisions and TODOs here.

### Reference Reading

- [Intro to Neural Networks: Why a straight line can't grade a battery cell](https://scaler-content.github.io/DSML-NN-1/) (Scaler DSML, Neural Networks 1). A local summary is in `intro_to_neural_networks.md`.
