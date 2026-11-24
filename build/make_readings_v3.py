import json
P = "/Users/vinayakkumar.udikeri/Documents/Porter_NN_Regression/build/"
R = json.load(open(P + "readings_v1.json"))          # unchanged readings carried over (Sections 2, 3.1, 3.6)

# ---------------------------------------------------------------- Section 3
R["s3_3"] = "From the table, the choice of method changes the number of flagged rows enormously: the IQR rule flags 36,173 rows (20.7% of the data) and the z-score rule flags 11,222 rows (6.4%), while the Isolation Forest flags 1,749 rows (1.0%). The Isolation Forest flags exactly the share we ask for (`contamination=0.01`), whereas the other two decide the number from the data. Almost every column here is skewed to the right, and the IQR rule assumes a more even spread, so it would throw away a fifth of ordinary orders because one column is on the high side. That is why the Isolation Forest is our main method."

R["s3_4b"] = "From the output above, the largest item counts are 411, then 66, 64, 59 and so on: the 411-item order is far away from everything else. The 99.9th percentile of `total_items` is 21.9, and 174 rows (0.10%) are above it, so 172,914 rows remain. These rows are not slower than normal orders (42.7 minutes on average against 46.1 for all rows), so removing them does not throw away the slowest deliveries."

R["s3_5"] = """From the output above, after both steps 172,914 rows remain: 1,923 fewer than before (1,749 from the Isolation Forest and 174 from the percentile rule), which is 1.1% of the data. The largest subtotal fell from 26,800 to 18,920, the largest item count from 411 to 21 and the largest number of waiting orders from 285 to 278. The spread of subtotal fell from 1,830 to 1,703 and the spread of delivery time from 9.33 to 9.23, and the correlation between delivery time and subtotal barely changed (0.41 to 0.40).

So the cleaning removed the extreme rows and left the main relationships alone: the "after" scatterplot of items against subtotal is no longer stretched by a single 411-item order, and the cloud of delivery time against subtotal keeps its shape."""

R["s3_sum"] = """## Section 3 Summary: Objective and How It Was Addressed

**Motto of this section:** *Look at the data, find what does not belong, and check that removing it helped.*

| Task | Where it is addressed | Result |
|:--|:--|:--|
| **Task 3.1:** correlation heatmap and scatterplots | **3.1**: countplots, histograms, correlation heatmap, scatterplots (time vs subtotal, items vs subtotal), box plots | Delivery time relates most to subtotal (0.41), orders waiting (0.39) and hour (-0.35) |
| **Task 3.2 and Question 4:** why check for outliers | **3.2** | Extreme values pull averages, scalers and squared errors |
| **Task 3.3 and Question 5:** three methods | **3.3**: IQR rule, z-score, Isolation Forest, the visual box plot and scatterplot, and a table of how many rows each would flag | IQR would flag 20.7% of rows, z-score 6.4%, Isolation Forest 1.0%; Isolation Forest chosen |
| **Task 3.4:** detect, print and remove | **3.4**: Isolation Forest on eight numeric columns (1,749 rows), then a 99.9th-percentile rule on `total_items` (174 rows) | 172,914 rows left (1.1% removed) |
| **Task 3.5:** re-plot and compare | **3.5**: same scatterplots and box plots, before and after, plus a table | Largest items 411 to 21, largest subtotal 26,800 to 18,920 |
| **Task 3.6:** volume and delivery time by day and hour | **3.6**: countplots and line plots with 95% bands | Orders cluster at hours 0 to 4 and 19 to 23; delivery is slowest at hours 2 to 3 and fastest at hour 21 |

**Limitation to remember:** the outlier rules were worked out on all rows before the split, so they saw rows that later became test rows (only input columns were used, never the target)."""

# ---------------------------------------------------------------- Section 4
R["s4_1"] = "From the output above, the split gives 138,331 training rows and 34,583 test rows, each with 24 feature columns. In the histogram the delivery-time curves of the two sets lie almost on top of each other, so the test set is a fair sample of the same kind of orders."

R["s4_3"] = """From the output above, on the test set the forest is off by 3.68 minutes on average (MAE), its RMSE is 4.69 minutes, its R² is 0.744, and its average error is 8.1% of the true delivery time (MAPE). On the training set the same forest has an RMSE of only 1.76 and an R² of 0.963, so it has partly memorised the training rows. The gap between 1.76 and 4.69 is the price of that overfitting, and the test numbers are the ones to trust.

For scale, a model that always predicts the average delivery time would be off by about 9 minutes (the standard deviation of the target), so the forest removes roughly half of that error. In the chart on the left the forest follows the diagonal, but for slow deliveries (above about 70 minutes) it tends to predict too low."""

R["s4_4"] = "From the table, the ensembles are clearly better than the two simple models: Linear Regression (RMSE 5.13) and the single Decision Tree (5.20) are about 0.4 to 0.5 minutes worse than the Random Forest (4.69), and Gradient Boosting is the best of the four (4.57, R² 0.756). So the link between the features and delivery time is not a straight line, and it needs a flexible model. These are test scores for models with default settings, shown for perspective. Section 5.5 compares the neural network (RMSE 4.44) with all of them."

R["s4_sum"] = """## Section 4 Summary: Objective and How It Was Addressed

**Motto of this section:** *Set a baseline that is strong and honest, so that the neural network has something real to beat.*

| Task | Where it is addressed | Result |
|:--|:--|:--|
| **Task 4.1:** reproducible split | **4.1**: `pd.get_dummies` for `market_id` and `order_protocol`, `train_test_split(test_size=0.2, random_state=42)` | 138,331 train rows, 34,583 test rows, 24 features |
| **Task 4.2:** Random Forest | **4.2**: `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)` | Trained on the training rows only |
| **Task 4.3:** metrics and important features | **4.3**: MSE, RMSE, MAE, R² and MAPE on train and test, charts, `feature_importances_` | Test RMSE 4.69, MAE 3.68, R² 0.744, MAPE 8.1%; most important: orders waiting, subtotal, partners on shift |
| **Task 4.4 and Question 6:** other classical methods | **4.4**: table of five methods, and a quick test of three | Linear 5.13, Decision Tree 5.20, Gradient Boosting 4.57 (test RMSE) |

**Baseline to beat:** test RMSE **4.69** minutes (train RMSE 1.76, so the forest overfits)."""

# ---------------------------------------------------------------- Section 5
R["s5_1"] = "From the output above, the data is split into 121,039 training rows, 17,292 validation rows and 34,583 test rows. The scaler and the target scaler were fitted on the training rows only (target mean 46.09, standard deviation 9.22 minutes). One batch holds features of shape (256, 24) and targets of shape (256, 1), which is the shape of the model output."

R["s5_3"] = "From the output above, the untrained network has a loss of 1.17 in scaled units. This is close to 1, as expected: an untrained network predicts values near 0 for a target that has been scaled to variance 1, and the random starting weights add a little more. The gradient shapes match the weight shapes ((128, 24) and (1, 64)), so backpropagation is connected properly."

R["s5_4"] = "From the output and the chart, the baseline network reaches its best validation RMSE of 4.495 minutes at epoch 23. After that, the training RMSE keeps falling slowly (4.33 at epoch 60) while the validation RMSE stays near 4.5 (4.54 at epoch 60), so a gap of about 0.21 minutes opens up. This is mild overfitting, not a collapse: the validation loss does not climb steeply, and keeping the weights of the best epoch removes the problem."

R["s5_5a"] = "From the scoreboard, all five sizes finish within 0.04 minutes of each other (best validation RMSE 4.482 to 4.513). The smallest network, one hidden layer of 32 units, is the best. The largest, (256, 128, 64), has the lowest training error (4.192) but the highest validation error at its last epoch (4.566), a sign of overfitting. More capacity did not help here."

R["s5_5b"] = "From the scoreboard, the four activations are within 0.007 minutes of each other (4.490 to 4.497). ReLU and Leaky ReLU reach their best epoch at epoch 23, Tanh at epoch 34 and Sigmoid at epoch 38, so Sigmoid trains the most slowly. The gaps are too small to call a winner from one run each, so we keep ReLU for the reasons given in Section 5.2."

R["s5_5c"] = "From the scoreboard, Adam (4.495) and RMSprop (4.509) beat plain SGD (4.540) and SGD with momentum (4.537), which were still improving at epoch 39 of 40 and would need more epochs or a larger learning rate. SGD with momentum at learning rate 0.01 did not train at all: the loss became NaN from the first epoch (we re-ran it for five epochs to confirm), which is why that run uses 0.001. This is the practical reason to choose Adam: it gives a good result quickly without careful tuning of the learning rate."

R["s5_5d"] = "From the scoreboard, learning rates from 0.0003 to 0.01 all land between 4.495 and 4.512, and 0.001 and 0.003 tie for the best (4.495). Only 0.0001 is clearly slower (4.528, best epoch 39, still improving). A wide range works because Adam adapts the step for every weight and the inputs are standardised."

R["s5_5e"] = "From the scoreboard, batch sizes from 64 to 1024 all finish between 4.492 and 4.503, so batch size hardly matters here. The two largest batches reach their best epoch later (epochs 36 and 35 for 512 and 1024, against epochs 17 to 23 for 64 to 256) because they make fewer weight updates per epoch."

R["s5_5g"] = "From the scoreboard, all five regularised networks land within 0.015 minutes of each other (4.485 to 4.500), against 4.495 for the same network without regularisation. Dropout 0.3 is the best of them (4.485) and it shrinks the gap between training and validation error at the last epoch from 0.18 to 0.06 minutes, so it does what it should, but the gain in validation error is only 0.010 minutes. Batch normalisation is slightly worse than no regularisation (4.500), as it also was in the SwiftLend case study. None of them beats the small 32-unit network (4.482): a small network is itself a form of regularisation."

R["s5_5f"] = "From the chart, there are 28 runs (five of them are the same baseline configuration, counted once in each experiment). Their best validation RMSE runs from 4.482 to 4.540 minutes, a spread of 0.06 minutes (about 1.3%). The bars start at zero on purpose, so that the chart does not exaggerate the gaps: no single hyperparameter matters much for this problem, and the order of the top few runs should not be over-read. Choosing the best run by **validation** error is still the right rule, because the test set must not be used for choosing."

R["s5_6"] = "From the output above, the best configuration on the scoreboard is one hidden layer of 32 units with ReLU, Adam (learning rate 0.001), batch size 256 and no regularisation. That network has only 833 parameters. Trained for 60 epochs, its best epoch is 53 with a validation RMSE of 4.472 minutes. The training and validation curves lie on top of each other, so this small network does not overfit: it is limited by what the features can explain, not by memorisation."

R["s5_7"] = """From the output above, on the same 34,583 test rows the neural network is slightly better on every metric: RMSE 4.44 against 4.69 minutes, MAE 3.49 against 3.68, R² 0.770 against 0.744 and MAPE 7.7% against 8.1%. Its MSE is 19.7 against 22.0, about 10% lower.

The improvement is small in minutes (about 0.25 minutes of RMSE, or 5%), and both models make the same kind of mistake: they tend to predict too low for very slow deliveries (above about 70 minutes), as the predicted-against-actual charts show. Both explain about three quarters of the variation in delivery time. The forest fits its training rows far better than its test rows (RMSE 1.76 against 4.69), while the network's training and validation errors are almost equal. The next six tests ask whether this small lead is real."""

R["s5_7a"] = "From the table, the network puts more orders inside every error band: 51.5% against 49.7% within 3 minutes, 74.7% against 72.4% within 5 minutes and 97.6% against 96.5% within 10 minutes. It also breaks its promise by more than 5 minutes less often (12.7% of orders against 13.4%). For a customer the gain is small but it points the same way: about two more orders in a hundred land within 5 minutes of the estimate."

R["s5_7b"] = "From the table and the chart, the network's RMSE is lower by 0.250 minutes on average, with a 95% interval of 0.229 to 0.271, and its MAE is lower by 0.185 minutes (0.168 to 0.203). In all 1,000 resampled test sets the network wins, and the whole interval lies above zero. So the lead is not the luck of which rows ended up in the test set. This test only covers the choice of test rows: it does not cover the choice of training rows or the random start, which the next test looks at."

R["s5_7c"] = "From the table, the network's test RMSE moves very little between seeds (4.434 to 4.454, standard deviation 0.007) and so does the forest's (4.688 to 4.697, standard deviation 0.004). The **worst** of the 5 network runs (4.454) is better than the **best** of the 4 forest runs (4.688), by 0.23 minutes. So the difference is not a lucky seed. All runs use the same train and test split, so this still does not show how much a different split would change the numbers."

R["s5_7d"] = "From the table, the network is far smaller and faster. The forest stores about 13.7 million tree nodes and the network 833 weights (these are different kinds of numbers, so read this as the order of magnitude, not as an exact ratio). Predicting all the test orders took about 0.17 seconds for the forest and about 0.001 seconds for the network on this computer (timings change a little from run to run). The forest's test error is 2.93 minutes above its training error, while the network's test error is only 0.014 minutes above its own, which shows how much more the forest memorises its training rows."

R["s5_7e"] = "From the table and the chart, the network is better in all four groups, by 0.21 minutes when the market is quiet, 0.29 when it is busy and 0.31 when it is busiest, so its lead is largest when accurate estimates matter most. Both models are least accurate when few orders are waiting (RMSE 4.9 to 5.1) and most accurate when the market is busiest (3.6 to 3.9)."

R["s5_7f"] = "From the tables, the best forest on the validation rows has at least 5 rows per leaf and half of the features at each split (validation RMSE 4.691). Scored once on the test rows, it improves a little on the default forest (RMSE 4.654 against 4.692, MAE 3.665 against 3.678) but the network is still ahead (RMSE 4.442, MAE 3.494, R² 0.770 against 0.748). Together with Gradient Boosting (4.573, Section 4.4), this means the network's lead does not come from a poorly set-up baseline."

R["s5_8"] = """### Verdict: Which Model Is Stronger?

**The neural network is the stronger model here**, and the evidence is consistent:

| Evidence | Random Forest | Neural Network |
|:--|:--:|:--:|
| Test RMSE (minutes), lower is better | 4.692 | **4.442** |
| Test R², higher is better | 0.744 | **0.770** |
| Orders predicted within 5 minutes | 72.4% | **74.7%** |
| Paired bootstrap: network wins in | | **1,000 of 1,000 resamples** |
| Worst run over several seeds (RMSE) | 4.697 | **4.454** |
| Tuned forest (RMSE) / Gradient Boosting (RMSE) | 4.654 / 4.573 | **4.442** |
| Test error minus training error (minutes) | 2.929 | **0.014** |
| Stored numbers / time to predict the test orders | about 13.7 million / 0.17 s | **833 / 0.001 s** |
| Better in every market-busyness group | no | **yes (by 0.21 to 0.31 minutes)** |

**How strong is "stronger"?** The network wins on every test, but the gain is modest: about 0.25 minutes (15 seconds) of RMSE, or 5%. Both models explain about three quarters of the variation in delivery time, and both tend to under-predict very slow deliveries. The remaining error comes from information the features do not contain (traffic, weather, kitchen delays), so a bigger network cannot remove it. The network's real advantages are that it generalises cleanly (no overfitting), is far smaller and faster, and wins consistently, which matters when millions of estimates are served.

**What these tests do not show:** all results use one train and test split, and the driving-duration column was left out because the brief does not describe it."""

R["s5_9"] = "From the output above, the weights and the preprocessing objects were saved in the `models` folder, and a freshly built network that loaded the saved weights gives exactly the same predictions as the trained one (`True`). For the example order (Saturday, hour 20, 3 items, subtotal 2400, 40 partners on shift, 50 orders waiting) the model estimates 42 minutes, which sits inside the range of 40 to 50 minutes that we saw for similar hours and days in Section 3.6."

R["s5_sum"] = """## Section 5 Summary: Objective and How It Was Addressed

**Motto of this section:** *Build the network step by step, and judge it only on data it has never seen, against the baselines it has to beat.*

| Task | Where it is addressed | Result |
|:--|:--|:--|
| **Task 5.1 and Question 7:** scaling | **5.1**: `StandardScaler` on the features and the target, fitted on the training rows only; why a network needs it and a forest does not | 121,039 / 17,292 / 34,583 rows (train / validation / test) |
| **Task 5.2 and Question 9:** architecture and activations | **5.2**: ReLU hidden layers, one linear output neuron, Xavier initialisation | Baseline network 24 -> 128 -> 64 -> 1, 11,521 parameters |
| **Task 5.3 and Question 8:** loss and optimizer | **5.3**: `MSELoss` and Adam | Adam compared with SGD, momentum and RMSprop in 5.4 |
| **Task 5.4:** training, loss plots, overfitting, configurations | **5.4**: own training loop, early stopping on validation loss, six experiments (size, activation, optimiser, learning rate, batch size, regularisation), full scoreboard, final model | 28 runs within 4.482 to 4.540 validation RMSE; final model 32 hidden units, validation RMSE 4.472 |
| **Task 5.5:** same metrics, compare with the Random Forest | **5.5**: MSE, RMSE, MAE, R² and MAPE on the same test rows, charts, six further tests, verdict, saving the model | Neural network RMSE 4.44 vs forest 4.69 (tuned forest 4.65, gradient boosting 4.57); R² 0.770 vs 0.744 |
| **Task 5.6 and Question 10:** why networks do well on large data | **5.6** | Explained, and checked against our own results |

**Final result:** the neural network (test RMSE **4.44** minutes, MAE **3.49**) is stronger than the Random Forest baseline (RMSE 4.69, MAE 3.68), than a tuned forest (4.65) and than Gradient Boosting (4.57). Its lead is modest (about 0.25 minutes) but consistent across metrics, seeds, resampled test sets and market conditions. All models are far better than always predicting the average (about 9 minutes).

**Limits of this result:**
- One train and test split; the outlier rules were worked out before the split (Section 3.4).
- `estimated_store_to_consumer_driving_duration` was left out because the brief does not describe it (Section 2.4); its effect was not measured.
- Extra features such as traffic or weather are not in the data, so about a quarter of the variation stays unexplained."""

R["s3_1d"] = "From the histograms, every column has most of its values at the low end and a thin tail to the right, and the log scale makes the single rows in those tails visible. `total_items` has almost all orders below about 20 items, yet one order has 411; `subtotal` stretches to 26,800 and the two item-price columns to about 14,700, each with only a handful of rows far out. `num_distinct_items` thins out steadily up to 20. The three partner columns are spread more evenly and only thin out above about 125 to 150, while the number of orders waiting falls off steadily towards 280. This long-tailed shape is why the outlier check in 3.2 to 3.4 matters."

json.dump(R, open(P + "readings.json", "w"), indent=1, ensure_ascii=False)
print(len(R), "readings written")
