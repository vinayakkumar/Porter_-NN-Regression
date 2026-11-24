# Porter NN Regression: Full Context

Last updated: 2026-11-24 (fourth update, after the strict-evaluator additions and the final run). This file records everything loaded into context while preparing `solution.ipynb`: the task, every file read (and *how* it was read), what the course material teaches, the dataset facts, the work done so far and the plan. Sections 8.3 and 8.5 and section 10 describe the current state.

**How facts are marked**
- **[me]**: read first-hand by the assistant in this session.
- **[agent]**: read by a delegated sub-agent and known only through its written report (not re-checked against the source).
- **[fetch]**: a WebFetch summary written by a small model (not the raw page).
- **[mine-derived]**: a conclusion the assistant drew, not stated by the course.

---

## 1. Task and user instructions

### 1.1 The task
Create `solution.ipynb` that solves the business case in `problem_brief.html` (Porter delivery-time estimation, a regression problem), using the course material in `nn_day_1` … `nn_day_12` and the links in each `note_link.txt` as the **primary reference**.

### 1.2 What the user asked, in order
1. Go through all docs in the folder; create `solution.ipynb` and solve the business case.
2. "Without fail" go through all files and the link inside every `note_link.txt` (primary reference).
3. Read the missing files too (the image-only slide PDFs, notebook images, cheat sheet); read all notes "like a student" to understand full context **before** starting the solution.
4. **Do not build the solution until told.** (The assistant waited.)
5. Then: read `problem_brief.html`, make sections per **Evaluation Criteria (100 points)**, add details, add code and output, and **do only Section 1 (10 points) for now**.
6. Store this `context.md`.
7. (Later) The notebook grew to all five sections. The user then asked to **update `solution.ipynb` as per `gemini-code-1795089251907.html` and `problem_brief.html`**, with the sections as named in the brief.
8. **Before each section, mention the stage Objective and its tasks with hints** (pasted Stage 1 text as the example), taken from the Gemini page.
9. **Re-verify that every task of the Gemini page is present in its section**; add what is missing and align the notebook. The user then went away, so this was done unattended.
10. **Whatever files are created must go in the project folder, not `/tmp`** ("keep everything here only").

### 1.3 Working preferences observed
- Thorough reading before building; wants honest statements of what has and has not been read.
- Build one evaluation section at a time, with explanations, code **and executed output**.
- Only act when told to ("wait once I say then only build").
- **All files I create live in the project folder** (builders, intermediate notebooks, logs, scratch), never in `/tmp`.
- The user asked to see the stage Objective, tasks and hints at the start of each section.

---

## 2. The business case (from `problem_brief.html` and `gemini-code-1795089251907.html`) [me]

### 2.1 Problem statement
Porter is India's largest marketplace for intra-city logistics (150,000+ driver-partners, 5+ million customers). It works with many restaurants for direct delivery. It wants to give customers an **estimated delivery time** based on what they order, from where, and the available delivery partners. The dataset trains a **regression model** for delivery-time estimation.

### 2.2 Dataset
- Drive link: `https://drive.google.com/file/d/1kkgBGldeswHBgVKEJiqqG7VZvqZNMc-5/view?usp=share_link` (file id `1kkgBGldeswHBgVKEJiqqG7VZvqZNMc-5`).
- One row = one unique delivery.

### 2.3 Data dictionary in the brief vs the real file
| Brief | File |
|---|---|
| `market_id` | `market_id` |
| `created_at` | `created_at` |
| `actual_delivery_time` | `actual_delivery_time` |
| `store_primary_category` | `store_primary_category` |
| `order_protocol` | `order_protocol` |
| "total_items subtotal" (typo: two columns joined) | `total_items`, `subtotal` |
| `num_distinct_items` | `num_distinct_items` |
| `min_item_price`, `max_item_price` | same |
| `total_onshift_partners` | `total_onshift_dashers` |
| `total_busy_partners` | `total_busy_dashers` |
| `total_outstanding_orders` | same |
| *(not in brief)* | `estimated_store_to_consumer_driving_duration` (seconds; known at order time, so a valid feature) |

### 2.4 Process in the brief
1. Import data, structure analysis.
2. Preprocessing: clean, create target `time_taken` from `created_at` and `actual_delivery_time` (minutes), hour of day and day of week, pandas datetime understanding, null handling, encode categoricals.
3. Visualisation and cleaning: countplots, scatterplots, outlier check and removal, re-plot.
4. Split train/test, **Random Forest baseline**, metrics (MSE, RMSE, MAE).
5. Scale data for NN, build a simple NN, try configurations (activations, optimizers, hyperparameters), train, plot losses, compare NN vs RF.

### 2.5 Evaluation criteria (100 points)
| # | Block | Points | Items |
|---|---|---|---|
| 1 | Problem statement, import data, data structure analysis | 10 | |
| 2 | Data preprocessing and feature engineering | 30 | cleaning, null handling, target creation, hour and day of week, encode categorical |
| 3 | Data visualisation and cleaning | 20 | visualisations, outlier check, outlier removal, compare plots |
| 4 | Model training with Random Forest | 10 | data splitting, RF regression |
| 5 | Regression with neural networks | 30 | scaling, architecture, different combinations and hyperparameters, training, comparison with RF |

### 2.6 Questions in the brief
1. Define the problem statement and where it (and modifications of it) can be used.
2. List 3 functions pandas datetime provides, with one line each.
3. Short note on datetime, timedelta, timespan (period).
4. Why check for outliers?
5. Name 3 outlier removal methods.
6. Classical ML methods other than random forest for regression?
7. Why is scaling required for neural networks?
8. Briefly explain the choice of optimizer.
9. Which activation function did you use and why?
10. Why does a neural network perform well on a large dataset?

### 2.7 The gemini HTML (task walkthrough) [me]
A "Scaler case study" page with the same brief split into **5 stages / 27 tasks** (1.1–1.5, 2.1–2.6, 3.1–3.6, 4.1–4.4, 5.1–5.6), each with a hint. It is the best checklist for the notebook:
- **1.1** state the business problem plus one other reuse scenario. **1.2** import into pandas. **1.3** inspect shape, columns, dtypes; note columns needing conversion. **1.4** three pandas datetime functions. **1.5** datetime vs timedelta vs Period.
- **2.1** convert both timestamps to datetime. **2.2** target in minutes. **2.3** hour and day of week. **2.4** drop columns no longer needed (leakage). **2.5** nulls. **2.6** encode `store_primary_category` via category codes.
- **3.1** correlation heatmap; scatter of `time_taken_mins` vs `subtotal`, and `total_items` vs `subtotal`. **3.2** why outliers matter. **3.3** three outlier methods. **3.4** apply a method on several columns at once, print rows flagged, remove. **3.5** re-plot the same scatter and compare. **3.6** volume by day of week and hour, effect of time of day on delivery time.
- **4.1** reproducible train/test split. **4.2** RF regressor. **4.3** several metrics plus important features. **4.4** other classical methods.
- **5.1** scale; why NNs need it and trees do not. **5.2** architecture and activations (linear output for a continuous target; strong default for hidden layers). **5.3** loss and optimizer; justify an adaptive optimizer. **5.4** train, plot train vs validation loss, check overfitting. **5.5** evaluate with the same metrics as RF, compare. **5.6** why NNs do well as data grows.
- Page metadata: deadline text "31 June 2026" (not a real date); environment text mentions Colab/BigQuery.

---

## 3. Project folder and file inventory

Root: `/Users/vinayakkumar.udikeri/Documents/Porter_NN_Regression/`

### 3.1 Root files
| File | How read |
|---|---|
| `problem_brief.html` | [me] in full |
| `gemini-code-1795089251907.html` | [me] in full |
| `helper.md` | [me] in full: project template (Setup, structure `data/ notebooks/ src/ models/`, workflow load → features → split → NN → evaluate MAE/RMSE/R² → save) |
| `requirements.txt` | [me]: jupyter, ipykernel, ipywidgets, nbformat, numpy, pandas, scikit-learn, scikit-image, matplotlib, seaborn, tensorflow, torch, torchvision, lime. **`gdown` was added at the user's request** (see 8.1) |
| `intro_to_neural_networks.md` | [me] partly, [agent] in full: summary of NN-1 (Hexa Cell) |
| `env_check.ipynb` | [me] in full: environment smoke test (see 5.4) |
| `venv/` | Not read on purpose (Python 3.13; torch 2.15.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, lime installed, **shap not installed**) |
| `data/porter_delivery.csv` | Added this session (downloaded from the Drive link). Profiled, not read row by row |
| `solution.ipynb` | **Main deliverable. Done and executed (2026-11-24):** 244 cells, 105 code cells all executed with saved outputs, 0 errors, 27/27 task headings (see 8.6) |
| `solution_report_style.ipynb` | Older report-style copy. **Not changed** by the Gemini alignment |
| `build/` | Working folder created this session (see 8.3): builder script, rebuilt notebook, backup of the previous `solution.ipynb`, and `scratch/` with everything that used to live in `/tmp` |
| `context.md` | This file |

### 3.2 Course folders
Each folder has a `note_link.txt` with one link `https://scaler-content.github.io/DSML-NN-<n>/` [me, all 12 fetched]. **Link numbering quirk:** `nn_day_5/note_link.txt` → NN-6 (backprop) and `nn_day_6/note_link.txt` → NN-5 (PyTorch part 1).

Legend: **V** = slide PDF rendered to images and viewed by an agent [agent]; **T** = text PDF read as text [agent]; **NB** = notebook source read by an agent (code and markdown, outputs stripped) [agent].

| Folder | Files (path under the folder) | Read as |
|---|---|---|
| `nn_day_1` | `Day_1_NN.pdf` | V |
| | `Lecture_1_Preread.docx.pdf` | T |
| | `Copy_of_Intro_to_Neural_Networks_Lecture.ipynb` | NB |
| `nn_day_2` | `NN_Day_2_Sept26.pdf` | V |
| | `Forward_Backward_Propagation_Preread.docx.pdf` | T |
| | `Copy_of_NN_Forward_and_Back_Propagation.ipynb` | NB |
| `nn_day_3` | `NN_Day_3.pdf`, `Extra_read_for_Maths.pdf` | V |
| | `N-Layer_Neural_Networks_1_Preread.docx.pdf` | T |
| | `Copy_of_NN_Lecture_N_layer_NN_revamped.ipynb` | NB |
| `nn_day_4` | `N_Layerd_2_Sept26.pdf` | V |
| | `N-Layer_Neural_Networks_2_Preread.docx.pdf` | T |
| | `N_layered_Neural_Network_Sep26.ipynb`, `05_NN_Layer_BackPropagation.ipynb` | NB |
| `nn_day_5` | `NN_Lecture_5_Sept26.pdf` | V |
| | `Backpropagation_Preread.docx.pdf` | T |
| | `Copy_of_Copy_of_05_NN_Layer_BackPropagation_(1).ipynb` | NB |
| `nn_day_6` | `PyTorch_Day_1.pdf` | V |
| | `PyTorch1_Preread.docx.pdf` | T |
| | `Pytorch_1.ipynb` | NB |
| `nn_day_7` | `Multi_Model_Branches.pdf` | V |
| | `PyTorch2_Preread.docx.pdf` | T |
| | `Pytorch_Day_2.ipynb` | NB |
| `nn_day_8` | `Optimisers_Pytorch.pdf`, `Pytorch_End_to_End_.pdf` | V |
| | `Optimizers_Preread.docx.pdf` | T |
| | `PyTorch_Part2_Penguins_Easy.ipynb` | [me] in full (also NB) |
| | `Optimizers_Pytorch.ipynb`, `Training_End_2_End_Pytorch.ipynb`, `Copy_of_NN_Lecture_Optimisers_Revamped.ipynb` | NB |
| `nn_day_9` | `Overfitting_Fixes.pdf` | V |
| | `Hyperparameter_Tuning_Preread.docx.pdf` | T |
| | `Copy_of_Hyperparameter_Tuning_Finding_Better_NN.ipynb` | NB |
| `nn_day_10` | `Autoencoders_Oct26.pdf` | V |
| | `Autoencoders_Preread.docx.pdf` | T |
| | `Autoencoders_Example_Oct26.ipynb`, `Copy_of_Autoencoders_Lecture_Revamped.ipynb` | NB |
| `nn_day_11` | `Designing_to_prototype.pdf` | V |
| | `Designing_Debugging_MLPs_Preread.docx.pdf` | T |
| | `Designing_to_Debegging.ipynb` | NB |
| `nn_day_12` | `NN Cheat Sheet.pdf` | [me] viewed all 4 pages (also T) |
| | `SwiftLend_Case_Study.pdf` | [me] viewed all 20 pages (also T) |
| | `Shapley.pdf` | [me] viewed all 3 pages (it has no extractable text) |
| | `Lime_Oct26.pdf` | V |
| | `Model_Interpretability_Preread.docx.pdf` | T |
| | `SWIFTLEND_BUSINESS_CASE.ipynb` | [me] first ~40 cells, plus NB |
| | `Model_Interpretability_Classwork.ipynb`, `Copy_of_Practical_Aspects_of_NN_Part1.ipynb`, `Copy_of_Practical_Aspects_of_NN_Part2.ipynb`, `lime_fashion_mnist.ipynb` | NB |

### 3.3 Reading gaps and limits
- **Notebook cell outputs** (printed results and plots) were **not read**, because dumps held source cells only. Exception: 25 images embedded in the Practical Aspects (Part 1 and 2) and Day 4/5 backprop notebooks were viewed by an agent (one duplicate skipped, the bottom of one image is cropped).
- Agent reports were **not verified** against source files.
- Slide PDFs are handwritten and image-based. Everything about them comes from agents' viewing, except `Shapley.pdf` and the cheat sheet [me]. A few scribbles are ambiguous (`Lime_Oct26` p04, `Optimisers_Pytorch` p02 bias-correction arithmetic).
- Pages with no lecture content: `Extra_read_for_Maths` p07 (blank), `N_Layerd_2_Sept26` p05 (platform screenshot).
- Lesson pages were read as static HTML text. Interactive widgets, charts and some collapsed "show code" sections (for example NN-7's code) are absent. The notebooks hold the same code.
- Nothing in the course covers: outlier removal, datetime features, regression metrics (MAE/RMSE/R²), Huber/MAE losses, or SHAP code.

---

## 4. The 12 lesson pages (primary reference) [me, raw text]

**NN-1: Intro (Hexa Cell battery grading).** 1,200 cells, 2 features, 3 ring-shaped grades. Logistic regression, KNN, trees and kernel SVM each fail for a specific reason (straight line; stores the whole set; axis-parallel staircase and bad on pixels; m×m kernel matrix). Central lesson: representation vs boundary. Logistic regression went 0.343 → 0.973 after adding r² = x1²+x2². A neuron is `z = wᵀx + b`, `o = f(z)`; weights are signed importances and comparable only after standardising. Activations: step, sigmoid, tanh, ReLU. Without an activation, any depth collapses to one linear map (silent bug). Batch form `Z = XW + b`. Logistic regression = sigmoid neuron; perceptron = step; linear SVM = hinge loss. Softmax for multi-class. Params per layer `n_in×n_out + n_out` (2→5→3 = 33). Four ReLU units give about 0.97 accuracy with 27 parameters. With small data, classical models (boosted trees) often win; NNs keep improving with more data.

**NN-2: Multi-class (GreenSort).** One neuron per class; `Z = XW + b`; independent sigmoids don't form a distribution; softmax (positive, sums to 1, amplifies gaps); categorical cross-entropy with one-hot labels collapses to −log p(true class); k=2 equals log loss. The same stack ends most classifiers, including LLMs. Training loop: init → forward → loss → backward (chain rule) → update.

**NN-3: Multi-layer (hiring pipeline).** `dZ = P − Y` exactly (softmax + cross-entropy); `dW = XᵀdZ`, `db = ΣdZ`. Hidden layers need non-linearity. XOR demo: 64 linear units fail (plateau near 0.5), 4 tanh units hit 100%. Sigmoid/tanh derivatives < 1 → vanishing gradients; ReLU (derivative 1) → dying ReLU → Leaky ReLU (α≈0.01); GELU/SiLU modern; He init for ReLU. Multi-path chain rule sums gradients (autograd does it). Parameter counts: 3→5→2 = 32, 4→8→3 = 67. ResNets/LayerNorm enable very deep nets. Recognising vanishing gradients: training loss barely moves, only the last layer changes.

**NN-4: Any depth, overfitting.** Notation `Z[l] = A[l-1]W[l] + b[l]`, `A[l] = g(Z[l])`; loop over layers; `dZ[l] = (dZ[l+1]W[l+1]ᵀ) ⊙ g′(Z[l])`. Overfitting demo (30 samples, 200 units): train loss → 0.003 while validation loss bottoms ≈0.14 at epoch 200 then rises. Debug order for 99%/61%: rule out leakage, add regularisation, more data, early stopping on validation loss. Gradient checking (centered difference, agree to ~1e-8; > 1e-5 = bug; slow, debugging only). For 500 rows / 12 features use 1–2 small hidden layers, not 6×512.

**NN-5 (read as "NN 6 post read"): backprop in a 2→4→3 network.** Spiral data. Gradients in order: `dZ2 = p − y`, `dW2 = A1ᵀdZ2/m`, `db2`, `dA1 = dZ2·W2ᵀ`, `dZ1` (zero where Z1 ≤ 0), `dW1`, `db1`; shapes decide transposes; memoization. `NN` class (known quirks: `fit` reads a global `y`, standalone loop defines `reg` but never uses it). sklearn `MLPClassifier` equivalent. **Regression: a linear last neuron gives regression; sigmoid/softmax gives classification; hidden layers stay non-linear.**

**NN-6 (read as "NN 5 post read"): PyTorch Part 1 (Healthyfi.me).** Encode (gender binary, class ordinal), heatmap and boxplot, split 90/10 twice (≈81/9/10), `StandardScaler` fit on train only. `nn.Linear(in, out)` needs both sizes; output layer has no activation (logits); params `(in×out)+out`; default init Kaiming-uniform, Xavier alternative. `CrossEntropyLoss` takes raw logits and integer labels. Loop: `zero_grad → forward → loss → backward → step` (without `zero_grad` gradients accumulate). `seed_everything`. Validation loss bottoms near epoch 120 → overfitting.

**NN-7: PyTorch Part 2.** `nn.Sequential` can't branch; custom `nn.Module` (layer order set by `forward`, not `__init__`). `model.eval()` + `torch.no_grad()` do different jobs. **Double-softmax bug** (softmax before `CrossEntropyLoss`): no error, loss floor ≈0.55 for 3 classes. A ReLU on a regression head clips negatives. Early stopping + checkpoint at end of each epoch. TensorBoard `add_scalar`.

**NN-8: Optimisers (NovaMart, Aria's hill).** Vanilla mini-batch GD zig-zags; EMA (biased low at start); momentum (EMA of gradient); RMSprop (scale by squared-gradient average); Adam (both + bias correction); Adam still needs a learning rate (an AI claim that it doesn't is wrong). LR decay `η = η₀/(1+r₀·epoch)` via `LambdaLR`. Course settings: SGD lr 0.05, momentum 0.9, Adam lr 0.01 betas (0.9, 0.999), batch 64, manual `randperm` batching, 16-16 ReLU network, Xavier init.

**NN-9: Hyperparameter tuning (QuickCart).** Baseline 71% train vs 63% validation. Target encoding (smoothed for rare categories), standardisation (round loss bowl, faster convergence). BCE vs MSE gradient. Four fixes: L2 (`L + λ/2m ΣW²`, update `W(1−ηλ/m) − η dW`), dropout (mask, scale 1/(1−r), off at test), BatchNorm (normalise per batch, learnable γ, β), early stopping (patience, keep best). Tune λ, r, BN placement, patience together.

**NN-10: Autoencoders (Nightwatch).** Encoder → bottleneck → decoder; embedding is the useful output; loss follows data type (MSE continuous, BCE binary, CCE one-hot); PCA is linear, autoencoders non-linear; frozen-encoder features (0.89), denoising (noisy input, clean target), item embeddings + cosine similarity; link to semantic search, vector DBs, RAG. **Not needed for Porter.**

**NN-11: Practical aspects (debugging).** Diagnose → tune → verify → measure → judge → analyse errors → extend. Orthogonalization: one knob per symptom, change one thing at a time (high train error = underfit; big train/val gap = overfit; good train/val but poor test = unrepresentative validation set). More data is not the first move. Tune order: learning rate first; then momentum, units, batch size; layers, LR decay, Adam betas last. LR on a **log scale**; random search then grid in the best region. Gradient checking only for custom backward code (float64, dropout off). One optimising metric + satisficing thresholds; macro-F1 exposes majority-class models. Avoidable bias = train − human error; variance = val − train. Error analysis by class/segment, share only aggregates with AI tools. Multi-task: sigmoid + BCE. Splitting rule: fit scaler on train only; keep val and test from the same distribution.

**NN-12: Interpretability.** LIME: local, interpretable surrogate, model-agnostic; perturb, fit surrogate; slow and unstable at low sample counts. Shapley values (game theory, exponential in features); SHAP approximates and is additive (values sum to prediction − baseline). LIME for quick exploration, SHAP when it must survive an audit; both are approximations.

---

## 5. Material read first-hand besides the lessons [me]

### 5.1 NN Cheat Sheet (4 pages, viewed)
- Overview graphic: NN vs traditional methods, perceptron → linear-activation issue → non-linear and classification activations, hyperparameters, backprop/forward prop, training issues → fixes (weight init, regularisation/batchnorm/early stopping/dropout, optimizers/LR/epoch), autoencoder.
- Activation table (sigmoid, tanh, ReLU, Leaky ReLU: equation, derivative, range, problem, advantage). Output-layer table (sigmoid: binary; softmax: multi-class). Tanh works with Glorot init; ReLU with He init.
- NN vs traditional ML: NNs adapt to unstructured data, need more compute, learn features implicitly, hard to interpret; traditional ML suits structured/tabular data, needs less compute, manual feature engineering, easy to interpret.
- Training issues: exploding gradients (derivatives > 1; fix Glorot/Xavier), vanishing (derivatives < 1), dead neurons (zero/constant init; use random init). Formulas for Glorot normal/uniform and He normal/uniform.
- Optimizers table: Batch GD, SGD, mini-batch GD, momentum, RMSprop, Adam with update rules, advantages, disadvantages. Batch norm: normalise then `γ·znorm + β`; "good practice to use before the activation". Learning rate too low / just right / too high; LR decay.
- Regularisation: Frobenius-norm penalty `(λ/2n)Σ‖Wᵏ‖²_F`; dropout (drop rate r, test-time scaling by p = 1−r); early stopping curve. Autoencoders vs PCA. Hyperparameter list: learning rate, momentum, epochs, batch size, weight init, number of hidden layers.

### 5.2 SwiftLend case study (instructor answer key, 20 pages, viewed)
Loan triage, 5,000 rows, 3 classes (Approve 0.45 / Decline 0.28 / Review 0.27), 13 features after one-hot. Structure: Part 0 business problem → 1 data to tensors → 2 baseline to beat → 3 network and forward pass → 4 how the network learns → 5 diagnose overfitting → 6 four fixes (scoreboard) → 7 optimiser race → 8 silent bug → 9 ship it → 10 recap. Key code and results:
- Split 60/20/20 stratified, `random_state=42`; scaler fit on train; seeds `manual_seed(42)`, and `manual_seed(0)` before each comparison run.
- Baseline logistic regression val accuracy **0.784**; net `13→128→64→3` has **10,243** parameters.
- `fit(model, opt, epochs, sched, patience)` with best-state cloning; `show(h, title)` plots train/val.
- Baseline 100 epochs: last train 0.089, val 0.353, best val 0.28 at epoch 42 (clear overfitting).
- **Scoreboard** (train_loss, val_loss, best_val, epochs): Baseline 0.089/0.353/0.280/100; +L2 (wd 1e-2) 0.303/0.316/0.313/100; +Dropout 0.3 0.176/0.263/0.257/100; **+BatchNorm 0.089/0.359/0.287/100 (did not help)**; +Early stopping (patience 15) 0.165/0.309/0.280/58.
- Optimiser race (val loss @ epoch 5 / 30): SGD 0.984/0.521; SGD+Momentum 0.465/0.324; RMSprop 0.404/0.293; Adam 0.424/0.301.
- LR decay (`StepLR(step_size=15, gamma=0.5)`, Adam lr 3e-3, 60 epochs): 0.313 with decay vs 0.427 fixed.
- Silent bug: adding `Softmax` before `CrossEntropyLoss` gives loss 0.695 with accuracy 0.856.
- Final recipe: `dropout 0.3, Adam lr 3e-3, weight_decay 1e-3, StepLR(20, 0.5), patience 15`; test accuracy **0.905** (test used once, after `model.eval()`); crosstab error table; confidence thresholds (≥0.9 auto-decides 64% at 98.7% accuracy).
- The student notebook `SWIFTLEND_BUSINESS_CASE.ipynb` is the same code without explanations.

### 5.3 Shapley.pdf (3 handwritten pages, viewed)
LIME is local, model-independent, supports text/tabular/image, uses Lasso as an interpretable surrogate, relies on locality; disadvantages: sensitive to hyperparameter α and to the neighbourhood. SHAP: bank-loan example (baseline 15%, prediction 20%); marginal contribution `80 − 65 = 15%`; 2³ = 8 subsets, 16 features → exponential; Fashion-MNIST shirt vs pullover heat-map illustration.

### 5.4 `env_check.ipynb` (project's own template, regression-ready)
Synthetic data stand-in for Porter; RF baseline (`n_estimators=100, random_state=42, n_jobs=-1`) with metric dict `MSE, RMSE, MAE, R2`; Keras network `Dense 32 → 16 → 1` with a scaled target and `inverse_transform` of predictions; PyTorch regression net `Linear(6,32) → ReLU → Linear(32,16) → ReLU → Linear(16,1)` with `MSELoss` and Adam lr 1e-2; LIME in `mode='regression'` on the RF; comparison table `pd.DataFrame({'RandomForest': ..., 'NeuralNet': ...})`.

### 5.5 `PyTorch_Part2_Penguins_Easy.ipynb` (read in full)
Palmer penguins; split 70/15/15 (`test_size=0.15`, then `0.176`, `stratify`); scaler fit on train only; **the regression target (body mass) has its own `StandardScaler`**, predictions are inverse-transformed before computing MAE in grams; features `float32`, class labels `long`, float targets shaped `(N,1)`; manual training loop, `evaluate_model` with `eval()` + `no_grad()`; double-softmax demo; early stopping (`patience=20`) with `torch.save`/`load_state_dict`; TensorBoard; multi-output model with `loss = CE + MSE`. **Code-review bug: ReLU on the regression head clips negatives; the standardised target can be negative, so the head must be a plain linear output.**

---

## 6. Reports from the sub-agents [agent]

### 6.1 Days 1–3
- Classification-only material; no regression code, scaling code, RF baseline or optimiser discussion. Regression conventions (linear output, MSE) are inferred, not stated.
- Conventions: shapes `X (m,d), W (d,n), b (1,n)`; params `n_in·n_out + n_out`; non-linearity needed in hidden layers; sigmoid/tanh vanish, ReLU default; He init for ReLU; standardise for NNs, trees don't need scaling; classical models often beat NNs on small tabular data.
- Slides (viewed): hot/cold neuron analogy; feature-scaling example (x1 20–90 with weight 0.04, x2 0.1–0.6 with weight 3.1 → weights not comparable until scaled); GreenSort quizzes (scores 2,1,0 → Paper; cross-entropy −log 0.5 ≈ 0.69); full derivation that the sigmoid/BCE terms cancel to `dJ/dW = (a−y)x`; stacked-linear collapse example `3(2x+1)+1 = 6x+4`; ReLU "used in about 99% of networks"; derivative chart (sigmoid max 0.25, tanh `1−tanh²`, ReLU 1 or 0, Leaky ReLU α≈0.01, GELU).

### 6.2 Days 4–5
- The two big backprop notebooks have byte-identical code. Day 5 adds an agenda, "Ask AI" prompts and a wrap-up. No `MLPRegressor`, gradient-checking code, early stopping, validation split or PyTorch in them (those are promised in pre-reads only).
- Backward formulas, `/m` averaging done once, ReLU mask should use `Z1` (instructor fix), L2 gradient `+= reg·W`, biases not regularised, stable softmax (subtract max) appears only in the Day 4 widget notebook.
- Only regression statement: "if the last neuron is linear the NN does regression". MSE gradient `dZ = ŷ − y` is inferred.
- Slide fragments: 6→4→3 funnel gives `W2` 4×3; exercise 500 examples / 12 features / 4 classes compares an NN with a random forest (50 trees, depth 4–5); train/val curve with "early stopping around this point".
- Hand-drawn diagram in the notebook: ReLU, ReLU, sigmoid → y ∈ {0,1} (classification) vs ReLU, ReLU, **linear** → y ∈ ℝ (regression).

### 6.3 Days 6–9
- No end-to-end PyTorch regression in these days; the Healthyfi, NovaMart and QuickCart examples are classification.
- Splits: Penguins 70/15/15 is the model to follow (test touched once at the end); the optimiser notebook uses test as validation (don't copy).
- Training loop order and `zero_grad` first; per-epoch metrics at epoch level (`Training_End_2_End_Pytorch` wrongly nests evaluation inside the batch loop); `lr_scheduler.step()` once per epoch; running loss via `loss.item()*len(batch)`.
- Early stopping: `best_val_loss = inf`, patience counter, `torch.save` on improvement, **reload the best checkpoint afterwards**.
- Optimisers: Adam lr 0.001 in the PyTorch notebooks, 0.01 in the optimiser lecture; betas "standard"; momentum 0.9; PyTorch's SGD momentum has no `(1−β)` factor (agent's own knowledge); LR decay numbers: `0.1/(1+0.1·50) = 0.0167`, `0.02/(1+0.05·20) = 0.01`; Adam eps 1e-8.
- Regularisation is concepts only (no `weight_decay`, `nn.Dropout`, `nn.BatchNorm1d` code in Days 6–9). Slides: L2 = weight decay with factor `(1 − ηλ/m)`; dropout rates 0.2, 0.5, worked example 0.3, inverted scaling `A·B/(1−r)`; BatchNorm with learnable scale/shift; early stopping on validation peak with patience.
- Day 9 slides: target encoding `TE(c) = mean(y)` for high-cardinality categories (about 40 varieties); ordinal for ordered values (High/Medium/Low → 2/1/0); one-hot for a few unordered classes; standardisation gives a round bowl.
- Multi-output: `loss = CE + 0.1·MSE` (Day 7) vs `CE + MSE` (Penguins).

### 6.4 Days 10–12 and misc
- Autoencoder material is irrelevant to Porter (only the idea that embeddings can feed simpler models).
- Debugging workflow, tuning priority, log-scale random search (`lr = 10 ** rng.uniform(-4, -1)`), gradient checking helper (`numerical_gradient`, `relative_difference`), human-level error, error analysis by class/slice (Practical Aspects Part 1 and 2).
- Data prep in Part 1: null handling (median or 0, otherwise `dropna`, with a warning to check how many rows are lost and whether missing-ness relates to the target), drop ID columns, ordinal map / `get_dummies(dtype=float)`, keep an unscaled copy of the validation frame for error analysis.
- LIME tabular: `LimeTabularExplainer(training_data, feature_names, mode, random_state)`, `predict_fn` takes raw units and scales internally, report `exp.score`; SHAP appears only in the pre-read, no code.
- The SwiftLend student notebook is code-only; there is no explicit rubric text in the case.

### 6.5 Slides (viewed by agents)
- **Day 6 `PyTorch_Day_1`:** quiz "can a network do regression? Yes, linear output"; toy regression `ŷ = w·x`: x=2, y=6, w=1 → MSE 16 → `dL/dw = −16` → with η=0.1, w = 2.6, loss 0.64; He/Glorot init table (`kaiming_normal_` for ReLU, `xavier_uniform_` for sigmoid/tanh); 5-step loop; zero_grad before backward.
- **Day 7 `Multi_Model_Branches`:** classification head + regression head; combined loss `Class + 0.1·Reg`.
- **Day 8:** optimiser slides (momentum/EMA arithmetic, RMSprop, Adam, LR decay formula); `Pytorch_End_to_End_` quizzes (branching needs custom `forward`, `no_grad` saves memory, double-softmax warning, early-stop marker near epoch 50).
- **Day 9 `Overfitting_Fixes`:** target encoding, scaling, BCE, L2 derivation (decay factor), dropout (inverted scaling, 0.2/0.5/0.3), batch norm (shifting histograms), early stopping with patience.
- **Day 10 `Autoencoders_Oct26`:** encoder 100→80→50, MSE, PCA vs autoencoder, denoising.
- **Day 11 `Designing_to_prototype`:** orthogonalisation, symptom → knob lists (A bad training accuracy: epochs, deeper, SGD→Adam, more data; B bad validation: simpler NN, dropout, regularisation, batch norm, diverse data; C bad test: bigger validation set, loss function), hyperparameter priority, "gradient checking rarely needed in PyTorch", F1 quiz.
- **Day 12 `Lime_Oct26`:** LIME steps (pick x_q, ~1000 perturbations within radius α, fit a simple model), global vs local quiz.

### 6.6 Notebook images (Practical Aspects Part 1/2 and backprop)
Orthogonalisation (DJ mixer), bias/variance fix lists, hyperparameter tuning order (notebook: learning rate, momentum β, neurons, batch size, layers, LR decay, Adam betas), grid vs random search then grid in the best region, numerical derivative of x³ at x=2 (12.0001 vs 12), gradient-checking setup, F1 arithmetic (0.462 vs 0.492), human-level and Bayes-error examples, error analysis (A=5, B=25, C=60, D=10), multi-label BCE, classification vs regression network sketch.

### 6.7 Disagreements and subtleties noted
- **Tuning order:** the Day 11 slide puts batch size 3rd, layers 4th, neurons 5th; the Practical Aspects notebook puts neurons 3rd, batch size 4th, layers 5th. Both: learning rate first, momentum next, LR decay late, Adam betas last (not recommended).
- **L2 wording:** the cheat sheet says L2 "cannot be used" with hidden layers (Frobenius norm instead). The SwiftLend code simply uses Adam `weight_decay`.
- **BatchNorm placement:** the cheat sheet says before the activation. The SwiftLend scoreboard shows BN did **not** help there.
- **Log-scale learning-rate search** is stated in NN-11/Practical Aspects (so it **is** course content); the slide images only show random-then-grid search.
- **Bayes/human reference** has no regression analogue in the course. For Porter the natural references are the train-mean predictor and the RF.

---

## 7. Dataset facts (profiled in this session) [me]

`data/porter_delivery.csv`: **175,777 rows × 14 columns**, 15.7 MB, no nulls, no duplicates.

| Column | dtype | Notes |
|---|---|---|
| `market_id` | float64 | 6 distinct (1–6); label only |
| `created_at`, `actual_delivery_time` | text | 2015-01-21 15:22:03 → 2015-02-18 06:00:44 (27 days); no delivery before order |
| `store_primary_category` | int64 | 73 distinct (0–72); top: 4 (18,183), 55, 46, 13, 58 |
| `order_protocol` | float64 | 7 distinct (1–7) |
| `total_items` | int64 | median 3, 99th pct 12, **max 411** |
| `subtotal` | int64 | median 2,224, 99th pct 9,460, max 26,800, min 0 |
| `num_distinct_items` | int64 | 1–20 |
| `min_item_price`, `max_item_price` | int64 | min price down to −86; max price up to 14,700 |
| `total_onshift_dashers`, `total_busy_dashers`, `total_outstanding_orders` | float64 | min −4, −5, −6 |
| `estimated_store_to_consumer_driving_duration` | float64 | seconds; 0–2,088, mean 546 |

- **Target preview** (delivery minutes): 32 to 110, mean 46.2, median 45, std 9.3, quartiles 39/45/52; always whole minutes.
- Correlations with the target (raw): driving duration ≈ 0.46, subtotal ≈ 0.41, outstanding orders ≈ 0.38, total items ≈ 0.22, on-shift dashers ≈ 0.17.
- **Quality flags:** min_item_price < 0: 12 rows; on-shift dashers < 0: 21; busy dashers < 0: 21; outstanding orders < 0: 41; subtotal = 0: 161; min price > max price: 701; **busy > on-shift: 39,290 (22.35%)**; driving duration = 0: 9. **940 rows (0.53%)** have at least one impossible value. Busy > on-shift is too common to be a typo, so those rows are not removed.

---

## 8. Work done so far

### 8.1 Files created or added in the project
- `data/porter_delivery.csv`: the dataset (downloaded from the Drive link).
- `solution.ipynb`: **course style (main file)**. This note was first written when it held Section 1 only (59 cells). It now holds all five sections (160 cells before the Gemini alignment, 167 in the rebuilt version; 83+ code cells) written in the same style. The copy on disk at the time of this update has no saved outputs (it was not executed after the later sections were added).
- `solution_report_style.ipynb`: **earlier report style**, kept as a separate file (159 cells, no saved outputs). Same content in a formal-report layout with `Section N` headings. **Not touched** by the Gemini alignment.
- `requirements.txt` (existing file): one line added, `gdown`, because both notebooks now download the dataset with `gdown` (instead of `urllib`) when `data/porter_delivery.csv` is missing. `gdown` 6.5.2 was also installed into the project `venv`. The call `gdown.download(id=FILE_ID, output=FILE)` was tested separately and returned a file identical to the local copy; inside the notebooks the local file exists, so the download branch is not triggered on a normal run.
- `context.md`: this file.

**Superseded heading scheme.** The "Option A" scheme described next was replaced by the Gemini-aligned scheme in 8.4 (headings now use the brief's section names, point values and `(Task X.Y)` labels). It is kept here only as history.

**Heading scheme (Option A, applied to `solution.ipynb` only):** numbered sections named after the work, with a Contents table (section, points, status) at the top; no "Part N" wording. `# 1. Problem Definition and Data Understanding` with `## 1.1 Problem Statement` (includes the Business Case), `## 1.2 Importing the Data`, `## 1.3 Data Structure Analysis`, `## 1.4 Date and Time in pandas`, `## 1.5 Summary: Objective and How It Was Addressed`; then `# 2. Data Preprocessing and Feature Engineering`, `# 3. Exploratory Visualisation and Outlier Treatment`, `# 4. Baseline Model: Random Forest`, `# 5. Neural Network Regression and Model Comparison` (sections 2–5 currently hold a one-line scope only). Anchors `sec1`..`sec5` back the Contents links. `solution_report_style.ipynb` keeps its original `Section N` headings. The pre-Option-A builder is saved as `/tmp/porter_build/build_solution_v2_before_optionA.py`.

**Additions made to maximise the 10 marks for Section 1 (both notebooks):** explicit "Brief Question 1/2/3" labels; one other reuse scenario in detail (ride-hailing arrival time, with a comparison table); `df.describe(exclude="number")` for the text columns; a "Variable Types and Conversions Needed" table plus `df.nunique()` for all columns; a "What the `.dt` accessor provides by default" cell (42 attributes and functions, with examples; the brief asks about "what function it provides by default"); and a checklist/summary table mapping every task (1.1–1.5) and question (1–3) of the brief to where it is answered. Pre-change builders are saved in `/tmp/porter_build/` as `build_solution_v2_before_checklist.py` and `build_solution_before_checklist.py`.

**Charts added to Section 1 (both notebooks; matplotlib, blue `#2a78d6` with orange `#eb6834` as the accent, no top/right spines, light grid, titles say what is shown):** (1) bar charts of orders per `market_id`, per `order_protocol`, and the 10 most common `store_primary_category` values, with count labels; (2) log-scale histograms of `total_items` and `subtotal` with a dashed 99th-percentile line (1% of orders have more than 12 items, the maximum is 411; 99% of subtotals are below 9,460, the maximum is 26,800); (3) orders per calendar day (29 dates; first and last are partial days, shown in orange; full days range about 5,000–8,100; busiest 7, 8, 14, 15 Feb = weekends); (4) histogram of the target in minutes with mean (46.2) and median (45) lines (right-skewed: half of deliveries take 39–52 min, 95% finish within 63). Each chart has a short reading under it. The charts were rendered and inspected; one label overlap in chart (2) was fixed. Pre-chart builders are saved as `build_solution_v2_before_graphs.py` and `build_solution_before_graphs.py`. Outlier-specific plots, the correlation heatmap and the hour/day analysis are left for Section 3.

**History of the notebook:** the first version was a formal report. The user pointed out it did not follow the style of the `nn_day_*` notebooks, so a course-style version was written, and then the user asked to **keep both files**. In the course-style file the user also asked to **remove all Quiz sections and the Ask AI section**, and to replace "Recap of Part 1" with the section's motto and how it is addressed. Builders (in `/tmp/porter_build/`): `build_solution_v2.py` writes `solution.ipynb`; `build_solution.py` writes `solution_report_style.ipynb`.

### 8.2 `solution.ipynb` structure (current)
- Title, objective line, flow line and a **Contents** table (sections 1–5 with points and status).
- **`# 1. Problem Definition and Data Understanding` (done, 10 points):**
  - *1.1 Problem Statement* (Business Case story, then Defining the Problem Statement): what we predict, why it matters, the ML framing table, **Important** note on leakage, where the idea can be used (inside Porter, outside, modifications). Answers brief question 1.
  - *1.2 Importing the Data*: Setup (pandas, numpy, seed 42), `Id | Features | Description` table of the 14 columns with a **Note** on name differences from the brief, Load the Data (local file, Drive fallback), `head()`, `tail()`.
  - *1.3 Data Structure Analysis*: shape (with the hand-written Records | Features table), `info()`, null values and duplicates (none), `describe().T`, ID-like columns (`nunique`, `value_counts`), data-quality checks (940 impossible rows, 39,290 busy > on-shift kept).
  - *1.3 also contains* a Variable Types and Conversions Needed table, `nunique()` and `describe(exclude="number")`.
  - *1.4 Date and Time in pandas*: timestamps as text and their range, the three pandas datetime functions (brief question 2) with a demo on a copy, datetime vs timedelta vs Period (question 3) with demo, a peek at the target (32–110 min, mean ≈ 46, std ≈ 9).
  - *1.5 Summary: Objective and How It Was Addressed*: a motto line ("Understand the problem and the data before touching either") and a checklist table mapping each task (1.1–1.5) and question (1–3) of the brief to where it is answered and the result. **No quizzes, no Ask AI, no recap** (removed at the user's request).
- **Sections 2–5:** *(outdated: this described the first version)*. They are now fully written, see 8.5.

### 8.4 Style of the course notebooks (to follow in every later part)
Observed first-hand in `Practical_Aspects_Part1`, `Hyperparameter_Tuning`, `Intro_to_Neural_Networks`, `Optimisers_Revamped`, `Penguins_Easy` (and SwiftLend for numbering):
- Opening: title, an **agenda** list, a **flow line** in a code block, a story-style `# Business Case` with the dataset link.
- Headings: `#` for major topics (Business Case, EDA, Baseline Model), `###` for steps (Setup, Data Description, Load the Data, Handling Null Values); SwiftLend uses `Part N ·` and numbered sub-headings.
- Data description as a table with columns `Id | Features | Description`.
- One idea per cell; a short markdown cell **before each code cell** explaining the functions used (for example "`pd.read_csv` reads the file into a DataFrame, and `.head()` previews the first five rows"); short code; raw outputs (`df.head()`, `df.shape`, `df.info()`); one plain sentence after an output ("From the output above, ...").
- A hand-written result table after an output (for example `| Records | Features |`).
- Callouts: **Note:**, **Important:**; analogies; "Common misconception". The course also uses quizzes (`**Answer: X)**`) and `### Ask AI` prompts, but **the user asked to leave Quiz and Ask AI sections out of our notebook**, and to end each part with its motto and how it was addressed instead of a recap.
- Simple language, imports in small cells (pandas, numpy, matplotlib), `random_state=42` / seed 42, `X_val_df`-style copies kept for error analysis.
- Not used by the course: heavy summary tables of "implications", custom print helpers, display-option overrides.

### 8.3 Working files (moved into the project folder at the user's request)
The user asked that everything created stays in the project folder, so the former `/tmp` files were moved:
- `build/build_solution.py`: the **self-contained notebook builder** for the Gemini-aligned version (paths are absolute project paths). It writes `build/pass3.ipynb`; `R("key")` slots insert short readings from `build/readings.json` if that file exists.
- `build/pass3.ipynb`: the rebuilt notebook, 167 cells, **not executed**.
- `build/solution_before_align.ipynb`: backup of `solution.ipynb` as it was before the alignment (160 cells).
- `build/scratch/`: the old `/tmp/porter_*` material: `porter_dump` (text of PDFs and notebooks), `porter_pages`, `porter_pages2` (rendered slide pages), `porter_nbimgs` (embedded notebook images), `porter_lessons` (raw lesson HTML/text), `porter_data` (dataset copy), `porter_charts`, `porter_proto`, helper scripts and a test CSV (about 103 MB).
- `build/pass3_exec.ipynb`: the first full execution (before readings were added); `build/outputs_dump.txt`: text dump of its outputs; `build/make_readings.py` and `build/readings.json`: the 31 readings and section summaries; `build/dump_outputs.py`, `build/verify_momentum.py`: helper scripts; `build/exec_final.log`: log of the final run.
- `build/older_versions/`: every older builder (v2 to v7), the report-style builders, `block_brief.py`, `assemble_both.py`, `align_v7.py`, `align_v8.py`, earlier executed passes and a dataset copy, moved here from `/tmp/porter_build`. `/tmp/porter_build` no longer exists.
- **Still outside the project:** only `/tmp/pdftools/` (271 MB Python venv for reading PDFs; moving a venv breaks it; no longer needed; deletion awaits the user's decision).
- Executing a notebook with `nbconvert` runs it from the notebook's own folder, so a run from `/tmp` downloads the dataset again with `gdown`. Run it from the project folder to use `data/porter_delivery.csv`.

### 8.5 Gemini and brief alignment (current work)

The Gemini page splits the brief into **5 stages / 27 tasks**. The rebuilt notebook (`build/pass3.ipynb`, built by `build/build_solution.py`) is aligned to it as follows.

**Section names, points and numbering (as in the brief)**
| Section | Points | Tasks | Heading in the notebook |
|---|---|---|---|
| 1 | 10 | 1.1 to 1.5 | `Section 1: Problem Statement, Data Import and Data Structure Analysis (10 points)` |
| 2 | 30 | 2.1 to 2.6 | `Section 2: Data Preprocessing and Feature Engineering (30 points)` |
| 3 | 20 | 3.1 to 3.6 | `Section 3: Data Visualization, Outlier Detection and Cleaning (20 points)` |
| 4 | 10 | 4.1 to 4.4 | `Section 4: Model Training with Random Forest (10 points)` |
| 5 | 30 | 5.1 to 5.6 | `Section 5: Regression with Neural Networks (30 points)` |

- **Contents table** at the top lists sections, points and tasks, plus a table that maps **brief questions 1 to 10** to sections: Q1 to 1.1, Q2 to 1.4, Q3 to 1.5, Q4 to 3.2, Q5 to 3.3, Q6 to 4.4, Q7 to 5.1, Q8 to 5.3, Q9 to 5.2, Q10 to 5.6.
- **Every task heading carries its label**, for example `## 2.5 Data Cleaning and Null Values (Task 2.5)`. A check on the rebuilt notebook found each of the 27 task labels exactly once.
- **Each section now opens with "Objective and Tasks of this Section"**: the stage title, the Objective, and a Task / What to do / Hint table, copied word for word from the Gemini page (the user pasted Stage 1 as the model).

**Renumbering done**
- Section 1: `1.4` is now the three pandas datetime functions (with the `.dt` defaults), `1.5` is datetime vs timedelta vs Period (with "A Peek at the Target"), and the old `1.5` summary became `1.6`.
- Section 2: the old `2.5 Data Cleaning: Impossible Values`, `2.6 Null Values` and `2.7 Encoding` became `2.5 Data Cleaning and Null Values` (two sub-headings) and `2.6 Encoding the Categorical Column`.
- Section 5: the experiments, scoreboard and final model are now sub-sections of `5.4` (the brief asks for different configurations; they were folded into Task 5.4). `5.5` is the test evaluation and comparison with the Random Forest, and `5.6` is "why neural networks do well on large datasets". References to "Section 5.5" in the text were changed to 5.4.

**Gaps found against the Gemini tasks and fixed**
| Task | Gap | Fix |
|---|---|---|
| 3.1 | No correlation heatmap | Added a heatmap (`sns.heatmap` of `df.corr`) and the correlations with the target |
| 3.3 | The hint names a plot-based method; the table had only IQR, z-score and Isolation Forest | Added a "Box plot or scatterplot (visual method)" row |
| 3.4 | Only the number of flagged rows was shown | Added a cell that prints the first ten flagged rows |
| 4.3 | No feature importance, no percentage view | Added `feature_importances_` chart (top 15) and **MAPE** in `regression_report` (so it also appears in the Section 5 comparison) |
| 5.6 text | Said the network has "about 13,000" parameters; 24 inputs with 128 and 64 hidden units give **11,521** | To be corrected in the final build (the real count must be checked in the executed output) |

**Decisions kept:** `estimated_store_to_consumer_driving_duration` is described as present in the file but **not used**, because the brief's dictionary does not list it (dropped in 2.4). Rows where busy partners exceed on-shift partners are kept. Outlier method: Isolation Forest on the eight numeric inputs (not the target), `contamination=0.01`. Split 80/20, then 12.5% of train for validation (70/10/20).

**Pipeline (see 8.3):** `build_solution.py` writes the notebook; the notebook is executed with the project `venv`; short "From the output above..." readings and the end-of-section summaries (motto plus how each task was addressed, no quiz, no Ask AI, no recap) are written into `readings.json` from the real outputs; the final `solution.ipynb` is then written with the saved outputs.

### 8.6 Final result and verification (done, 2026-11-24)

`solution.ipynb` was rebuilt by `build/build_solution.py` (readings from `build/readings.json`, written by `build/make_readings_v3.py`), executed in the project folder with the project `venv`, and the outputs were merged in with `build/merge_outputs.py`. `build/final_check.py` re-runs the checks below.

- **Verified:** 244 cells, 105 code cells, all executed, **0 errors**; 27 of 27 task headings; 5 of 5 stage blocks (objective, tasks, hints); 5 of 5 evaluation-criteria blocks; 10 of 10 question quotes; all 96 numbers quoted in the readings found in the outputs; no stale phrases. Chart claims were checked by viewing the charts.
- **Added after the first pass (strict-evaluator review):** the brief's own text (Stages Overview, How You're Evaluated, Process, verbatim Problem Statement and Data Dictionary, the 10 questions repeated above each answer, per-section evaluation criteria with where each is done); histograms of numeric columns; a table of how many rows IQR / z-score / Isolation Forest flag (36,173 / 11,222 / 1,749); a **second outlier rule** (99.9th percentile of `total_items`, 174 rows, removes the 411-item order); a note that outlier rules ran before the split (mild leakage); quick test of Linear Regression, Decision Tree and Gradient Boosting; Experiment 6 (weight decay, dropout, batch norm); six further tests that compare the network with the forest and a verdict; saving the model to `models/` and a new-order demo. The "(as given in the brief)" labels were removed at the user's request.
- **Data after cleaning:** 174,837 rows after impossible values, 173,088 after the Isolation Forest, **172,914** after the percentile rule. Split: 138,331 train and 34,583 test; for the network 121,039 / 17,292 / 34,583.
- **Test results:** Random Forest RMSE 4.692, MAE 3.678, R² 0.744, MAPE 8.135% (train RMSE 1.763, overfits). Tuned forest (5 rows per leaf, half the features, chosen on validation) RMSE 4.654. Gradient Boosting 4.573, Linear Regression 5.134, Decision Tree 5.195. **Neural network RMSE 4.442, MAE 3.494, R² 0.770, MAPE 7.713%** (final model: 1 hidden layer of 32 ReLU units, 833 parameters, Adam lr 0.001, batch 256, best epoch 53, validation RMSE 4.472).
- **Evidence the network is stronger:** paired bootstrap RMSE gain 0.250 min (95% interval 0.229 to 0.271, wins in 1,000 of 1,000 resamples); all 5 network seeds (4.434 to 4.454) beat all 4 forest seeds (4.688 to 4.697); within 5 minutes 74.7% vs 72.4%; late by more than 5 minutes 12.7% vs 13.4%; better in all four market-busyness groups (by 0.21 to 0.31); test minus train error 0.014 vs 2.929; 833 weights vs about 13.7 million tree nodes. The notebook states plainly that the lead is **modest** (about 0.25 minutes, 5%).
- **Experiments:** 28 runs (six experiments), best validation RMSE 4.482 to 4.540 (spread 0.06), so no single hyperparameter matters much; regularisation barely helps (dropout 0.3: 4.485; batch norm slightly worse). SGD with momentum at lr 0.01 gives NaN (verified).
- **Known limitations, stated in the notebook:** one train and test split; outlier rules worked out before the split; `estimated_store_to_consumer_driving_duration` excluded because the brief does not describe it (effect not measured); the large-data claim in 5.6 is general knowledge, not measured; both models tend to under-predict very slow deliveries.
- `solution_report_style.ipynb` was **not** updated. Backups of earlier `solution.ipynb` versions are in `build/` (`solution_before_align`, `solution_before_questions`, `solution_before_brief_text`, `solution_before_final`).

---

## 9. Original plan for Sections 2 to 5 (now written; kept as the design record)

Design decisions to carry into the notebook (course-consistent unless marked):
- **Split:** 70/10/20 (train/validation/test) with `random_state=42`. Regression, so no `stratify`. Test used once, at the end. Scaler fit on train only.
- **Section 2:** `pd.to_datetime`, target `time_taken_mins`, `hour`, `day_of_week`, drop raw timestamps, null report (none), encode `store_primary_category` with category codes as the brief says. [mine-derived options: smoothed target encoding fit on train only, per NN-9; cyclical hour; ratios such as busy/on-shift.]
- **Section 3:** correlation heatmap, scatter plots (time vs subtotal, items vs subtotal), box plots, IQR/z-score/Isolation Forest (multi-column, per task hint), invalid-value rule, before/after plots and comparison table, volume and delivery time by day and hour. Outlier removal and datetime features are **not** in the course (from the brief).
- **Section 4:** `RandomForestRegressor` (≈ `n_estimators=100–200`, `n_jobs=-1`, `random_state=42`), metrics MSE/RMSE/MAE/R² (+ MAPE and "within ±5 / ±10 min"), feature importance, trivial train-mean baseline, other classical methods (Ridge/linear, gradient boosting, KNN, SVR).
- **Section 5:** `StandardScaler` for X **and** y (inverse-transform before reporting minutes); PyTorch MLP, ReLU hidden layers, **single linear output (no activation)**, `MSELoss`, Adam (lr 1e-3 to 1e-2), He/Xavier init, y shaped `(N,1)`; manual training loop with per-epoch evaluation; early stopping on validation loss and reload of the best checkpoint; scoreboard-style experiments one change at a time (depth/width, activations, optimizers, L2/dropout/BatchNorm, LR decay `η₀/(1+r₀·epoch)`, log-scale random search on learning rate); train-vs-validation curves; final test evaluation with the same metrics as the RF; comparison and honest framing (tabular data: RF may win or tie); error analysis by segment; optional LIME (regression mode) / permutation importance. Pitfalls to avoid: ReLU or sigmoid on the output, wrong y shape (silent `(N,N)` broadcast), fitting the scaler on all data, evaluation inside the batch loop, forgetting `zero_grad`, trusting BatchNorm helps.
- **Expected outcome (still to be confirmed from the executed notebook):** a baseline of predicting the mean gives an error of about 9 minutes (std 9.3), so any model must beat that.
- The answers to brief questions 4–10 are to be written in Sections 3–5 (see the map in the notebook).

---

## 10. Open items / things to confirm with the user
- Whether to delete `/tmp/pdftools` (271 MB venv, no longer needed).
- Whether `solution_report_style.ipynb` should also be aligned to the Gemini structure (only `solution.ipynb` was requested).
- Whether to test the driving-duration column as a feature (currently excluded because the brief's dictionary does not list it).
- Whether to keep or delete `build/scratch/` (about 100 MB of reading material), `build/older_versions/` and the older run files in `build/`.
- Whether Section 2 should compare target encoding with the integer category codes the brief asks for.
