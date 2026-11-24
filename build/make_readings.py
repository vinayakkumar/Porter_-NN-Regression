import json
R = {}

R["s2_1"] = "From the output above, both timestamp columns changed from `str` (text) to `datetime64[us]`, and all 175,777 rows are still non-null. Pandas can now subtract these columns and pull parts out of them."

R["s2_2"] = "From the output above, delivery time runs from 32 to 110 minutes, with a mean of 46.2 and a median of 45. The mean is above the median and the longest delivery is far above the quartiles (39 and 52), so the distribution is stretched to the right: most deliveries take 39 to 52 minutes and a few take much longer."

R["s2_5"] = "From the output above, 940 rows (0.53%) broke at least one rule and were removed, which leaves 174,837 rows. The most common problem was a cheapest item that costs more than the costliest (701 rows), followed by a zero subtotal (161 rows)."

R["s2_6"] = "From the output above, there are no null values in any column, so nothing had to be dropped or filled. Cleaning the impossible values and creating the target did not create any new nulls."

R["s2_7"] = """From the output above, `store_category_code` now holds an integer code for every restaurant category (in this file the code happens to equal the original category number), and `market_id` and `order_protocol` are integers. The table has 174,837 rows and 14 columns, and every column is numeric.

**Note:** These codes are labels, not amounts: category 40 is not "more" than category 4. A Random Forest copes with this because it only compares thresholds, but a neural network reads the code as a number. This is a limitation of the encoding the brief asks for."""

R["s2_sum"] = """## Section 2 Summary: Objective and How It Was Addressed

**Motto of this section:** *Turn raw records into a clean, numeric table the models can learn from.*

| Task | Where it is addressed | Result |
|:--|:--|:--|
| **Task 2.1:** convert both timestamps to datetime | **2.1**: `pd.to_datetime`, checked with `df.info()` | Both columns are `datetime64[us]` |
| **Task 2.2:** target in minutes | **2.2**: subtract the columns, `.dt.total_seconds() / 60` | `time_taken_mins`: 32 to 110 minutes, mean 46.2 |
| **Task 2.3:** hour of day and day of week | **2.3**: `.dt.hour` and `.dt.day_of_week` | `hour` (0 to 23) and `day_of_week` (Monday = 0) |
| **Task 2.4:** drop columns no longer needed | **2.4**: dropped `created_at`, `actual_delivery_time` (the target's source, so it would leak) and `estimated_store_to_consumer_driving_duration` | 14 columns left |
| **Task 2.5:** nulls and cleaning | **2.5**: six rules for impossible values, then `isnull().sum()` | 940 rows (0.53%) removed; no nulls |
| **Task 2.6:** encode the categorical column | **2.6**: `astype("category").cat.codes` | `store_category_code`; 174,837 rows and 14 columns, all numeric |

**Two decisions to remember:**
- Rows where busy partners exceed on-shift partners (39,290 rows in Section 1) were **kept**. There are far too many of them to be typing mistakes.
- `estimated_store_to_consumer_driving_duration` was **not used**, because the brief's data dictionary does not list it. In the raw file it correlates about 0.46 with delivery time (profiled before this notebook was built), so leaving it out probably costs some accuracy. We did not test that here."""

R["s3_1a"] = "From the charts, market 2 has the most orders (53,210) and market 6 almost none (634). Order protocols 1, 3 and 5 carry most of the orders, while protocols 6 and 7 are rare (669 and 19 orders). A few restaurant categories (4, 55 and 46) account for far more orders than the rest."

R["s3_1h"] = "From the heatmap, delivery time is most strongly related to `subtotal` (0.41), `total_outstanding_orders` (0.39) and `hour` (-0.35: orders placed at later hours are faster), followed by `num_distinct_items` (0.31). The three supply columns (partners on shift, busy partners, orders waiting) carry almost the same signal, with correlations of 0.93 to 0.94 between them, and `total_items` and `num_distinct_items` also move together (0.76). `store_category_code` (0.03) and `day_of_week` (0.05) show almost no straight-line relation to delivery time, although a model can still use them in other ways."

R["s3_1b"] = "From the scatterplots, delivery time rises with subtotal, but the cloud is wide: orders of the same size take anywhere from 32 to about 100 minutes, so subtotal alone cannot predict delivery time. A few orders sit far to the right (one near 27,000), and in the second plot almost all orders have fewer than 20 items, with a handful of points far above."

R["s3_1c"] = "From the box plots, every column has orange dots beyond its whiskers on the right side. `total_items` has one order of 411 items, far away from everything else. `subtotal` and the two item-price columns have long tails of expensive orders, and the three partner columns have dots at the busiest moments. These are the values the rest of this section examines."

R["s3_4"] = "From the output above, the Isolation Forest flagged 1,749 rows (1.00%), and 173,088 rows remain after removing them. The flagged orders are much bigger (average subtotal 8,517 against 2,644, and 9.8 items against 3.1) and were placed at busy moments (112 orders waiting against 58). Their average delivery time is 60 minutes against 46, even though delivery time was **not** given to the method, so the flagged rows are genuinely unusual deliveries and not a random 1%."

R["s3_5"] = """From the output above, removing the flagged rows brought the largest subtotal down from 26,800 to 18,920 and the largest number of waiting orders from 285 to 278. The spread of subtotal fell from 1,830 to 1,704 and the spread of delivery time from 9.33 to 9.23, and the correlation between delivery time and subtotal barely changed (0.41 to 0.40). The plots are tidier at the top end, so the cleaning helped a little, but not everywhere.

**One extreme value was not removed.** The order with **411 items** is still in the data: the maximum of `total_items` is 411 both before and after, and it is still visible in the "after" scatterplot of items against subtotal. The next largest order has 66 items, so 411 is almost certainly a data error. The Isolation Forest looks at all eight columns together, and the rest of that row looks ordinary (5 distinct items, a subtotal of 3,115 and normal partner counts), so it was not flagged. A one-column rule such as the IQR rule on `total_items` would catch it. We kept the method as it is, and note the limitation here."""

R["s3_6"] = """From the charts, orders are not spread evenly over the day. Almost all orders are placed between hours 0 and 4 and between hours 19 and 23 (as recorded in the file), and hours 6 to 15 have very few orders (hours 9 to 13 have none), which is why the line for those hours has a wide band. Delivery takes longest for orders placed at hours 2 and 3 (about 50 minutes) and shortest at hour 21 (40.1 minutes), a difference of about 10 minutes.

Saturdays and Sundays have the most orders (about 30,000 each), and Mondays (48.5 minutes) and Saturdays (48.3) are the slowest days, while Wednesdays (43.5) are the fastest. So the time of the order does affect delivery time, and `hour` and `day_of_week` are worth keeping as features."""

R["s3_sum"] = """## Section 3 Summary: Objective and How It Was Addressed

**Motto of this section:** *Look at the data, find what does not belong, and check that removing it helped.*

| Task | Where it is addressed | Result |
|:--|:--|:--|
| **Task 3.1:** correlation heatmap and scatterplots | **3.1**: countplots, correlation heatmap, scatterplots (time vs subtotal, items vs subtotal), box plots | Delivery time relates most to subtotal (0.41), orders waiting (0.39) and hour (-0.35) |
| **Task 3.2 and Question 4:** why check for outliers | **3.2** | Extreme values pull averages, scalers and squared errors |
| **Task 3.3 and Question 5:** three methods | **3.3**: IQR rule, z-score, Isolation Forest, plus the visual box plot and scatterplot | Isolation Forest chosen because it looks at several columns together |
| **Task 3.4:** detect, print and remove | **3.4**: Isolation Forest on eight numeric columns, flagged rows printed | 1,749 rows (1.00%) removed, 173,088 left |
| **Task 3.5:** re-plot and compare | **3.5**: same scatterplots and box plots, before and after, plus a table | Largest subtotal 26,800 to 18,920; **the 411-item order stayed** |
| **Task 3.6:** volume and delivery time by day and hour | **3.6**: countplots and line plots with 95% bands | Orders cluster at hours 0 to 4 and 19 to 23; delivery is slowest at hours 2 to 3 and fastest at hour 21 |

**Limitation to remember:** the Isolation Forest did not remove the 411-item order. If the project continues, add a one-column rule for `total_items` before Section 4."""

R["s4_1"] = "From the output above, the split gives 138,470 training rows and 34,618 test rows, each with 24 feature columns. In the histogram the delivery-time curves of the two sets lie almost on top of each other, so the test set is a fair sample of the same kind of orders."

R["s4_3"] = """From the output above, on the test set the forest is off by 3.70 minutes on average (MAE), its RMSE is 4.71 minutes, its R² is 0.744, and its average error is 8.2% of the true delivery time (MAPE). On the training set the same forest has an RMSE of only 1.76 and an R² of 0.963, so it has partly memorised the training rows. The gap between 1.76 and 4.71 is the price of that overfitting, and the test numbers are the ones to trust.

For scale, a model that always predicts the average delivery time would be off by about 9 minutes (the standard deviation of the target), so the forest removes roughly half of that error. In the chart on the left the forest follows the diagonal, but for slow deliveries (above about 70 minutes) it predicts too low."""

R["s4_3b"] = "From the chart, the forest relies most on `total_outstanding_orders` (about 0.31), then on `subtotal` and `total_onshift_dashers` (about 0.17 each), followed by `hour`, `market_id_1` and `day_of_week`. The partner columns are strongly correlated with each other (Section 3.1), so the forest splits their importance between them: read the ranking as groups of features, not as exact scores."

R["s4_sum"] = """## Section 4 Summary: Objective and How It Was Addressed

**Motto of this section:** *Set a baseline that is strong and honest, so that the neural network has something real to beat.*

| Task | Where it is addressed | Result |
|:--|:--|:--|
| **Task 4.1:** reproducible split | **4.1**: `pd.get_dummies` for `market_id` and `order_protocol`, `train_test_split(test_size=0.2, random_state=42)` | 138,470 train rows, 34,618 test rows, 24 features |
| **Task 4.2:** Random Forest | **4.2**: `RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)` | Trained on the training rows only |
| **Task 4.3:** metrics and important features | **4.3**: MSE, RMSE, MAE, R² and MAPE on train and test, charts, `feature_importances_` | Test RMSE 4.71, MAE 3.70, R² 0.744, MAPE 8.2%; most important: orders waiting, subtotal, partners on shift |
| **Task 4.4 and Question 6:** other classical methods | **4.4**: table of five methods | Named, not trained (the brief only asks to name them) |

**Baseline to beat:** test RMSE **4.71** minutes (train RMSE 1.76, so the forest overfits)."""

R["s5_1"] = "From the output above, the data is split into 121,161 training rows, 17,309 validation rows and 34,618 test rows. The scaler and the target scaler were fitted on the training rows only (target mean 46.07, standard deviation 9.21 minutes). One batch holds features of shape (256, 24) and targets of shape (256, 1), which is the shape of the model output."

R["s5_2"] = "From the output above, the network has 24 inputs, two ReLU hidden layers of 128 and 64 units, and one linear output neuron. That is 11,521 trainable parameters: 24 x 128 + 128 = 3,200, then 128 x 64 + 64 = 8,256, then 64 + 1 = 65."

R["s5_3"] = "From the output above, the untrained network has a loss of 0.94 in scaled units. This is close to 1, as expected: an untrained network predicts values near 0 for a target that has been scaled to variance 1. The gradient shapes match the weight shapes ((128, 24) and (1, 64)), so backpropagation is connected properly."

R["s5_4"] = "From the output and the chart, the baseline network reaches its best validation RMSE of 4.447 minutes at epoch 17. After that, the training RMSE keeps falling slowly (4.31 at epoch 60) while the validation RMSE stays between about 4.45 and 4.50 (4.485 at epoch 60), so a small gap of about 0.17 minutes opens up. This is mild overfitting, not a collapse: the validation loss does not climb, and keeping the weights of the best epoch removes the problem. The spikes near epochs 7 and 11 are normal noise from mini-batch training."

R["s5_5a"] = "From the scoreboard, all five sizes finish within 0.04 minutes of each other (best validation RMSE 4.434 to 4.468). The smallest network, one hidden layer of 32 units, is the best. The largest, (256, 128, 64), has the lowest training error (4.204) but the highest validation error at its last epoch (4.554), a sign of overfitting. More capacity did not help here."

R["s5_5b"] = "From the scoreboard, the four activations are within 0.011 minutes of each other (4.441 to 4.452). ReLU, Leaky ReLU and Tanh reach their best epoch near epoch 17, while Sigmoid trains more slowly and is still improving at epoch 40. The gaps are too small to call a winner from one run each, so we keep ReLU for the reasons given in Section 5.2."

R["s5_5c"] = "From the scoreboard, Adam (4.447) and RMSprop (4.470) beat plain SGD (4.506) and SGD with momentum (4.505), which were still improving at epoch 40 and would need more epochs or a larger learning rate. SGD with momentum at learning rate 0.01 did not train at all: the loss became NaN from the first epoch (we re-ran it for five epochs to confirm), which is why that run uses 0.001. This is the practical reason to choose Adam: it gives a good result quickly without careful tuning of the learning rate."

R["s5_5d"] = "From the scoreboard, learning rates from 0.0003 to 0.01 all land between 4.440 and 4.459, and 0.003 is the best (4.440). Only 0.0001 is clearly slower (4.481, best epoch 40, still improving). A wide range works because Adam adapts the step for every weight and the inputs are standardised."

R["s5_5e"] = "From the scoreboard, batch sizes from 64 to 1024 all finish between 4.440 and 4.450, so batch size hardly matters here. The larger batches reach their best epoch later (epoch 28 for 512 and epoch 39 for 1024) because they make fewer weight updates per epoch."

R["s5_5f"] = "From the chart, there are 23 runs (five of them are the same baseline configuration, counted once in each experiment). Their best validation RMSE runs from 4.434 to 4.506 minutes, a spread of 0.07 minutes (about 1.6%). The bars start at zero on purpose, so that the chart does not exaggerate the gaps: no single hyperparameter matters much for this problem, and the order of the top few runs should not be over-read. Choosing the best run by **validation** error is still the right rule, because the test set must not be used for choosing."

R["s5_6"] = "From the output above, the best configuration on the scoreboard is one hidden layer of 32 units with ReLU, Adam (learning rate 0.001) and batch size 256. That network has only 833 parameters. Trained for 60 epochs, its best epoch is 52 with a validation RMSE of 4.432 minutes. The training and validation curves lie on top of each other, so this small network does not overfit: it is limited by what the features can explain, not by memorisation."

R["s5_7"] = """From the output above, on the same 34,618 test rows the neural network is slightly better on every metric: RMSE 4.47 against 4.71 minutes, MAE 3.53 against 3.70, R² 0.768 against 0.744 and MAPE 7.8% against 8.2%. Its MSE is 20.0 against 22.2, about 10% lower.

The improvement is small (about 0.24 minutes of RMSE), and both models make the same kind of mistake: they predict too low for very slow deliveries (above about 70 minutes), as the predicted-against-actual charts show. Both explain about three quarters of the variation in delivery time. The forest fits its training rows far better than its test rows (RMSE 1.76 against 4.71), while the network's training and validation errors are almost equal. We compared one trained model of each kind on one split, so the fair reading is "the network is at least as good as the forest here", not a proven win."""

R["s5_sum"] = """## Section 5 Summary: Objective and How It Was Addressed

**Motto of this section:** *Build the network step by step, and judge it only on data it has never seen, against the baseline it has to beat.*

| Task | Where it is addressed | Result |
|:--|:--|:--|
| **Task 5.1 and Question 7:** scaling | **5.1**: `StandardScaler` on the features and the target, fitted on the training rows only; why a network needs it and a forest does not | 121,161 / 17,309 / 34,618 rows (train / validation / test) |
| **Task 5.2 and Question 9:** architecture and activations | **5.2**: ReLU hidden layers, one linear output neuron, Xavier initialisation | Baseline network 24 -> 128 -> 64 -> 1, 11,521 parameters |
| **Task 5.3 and Question 8:** loss and optimizer | **5.3**: `MSELoss` and Adam | Adam compared with SGD, momentum and RMSprop in 5.4 |
| **Task 5.4:** training, loss plots, overfitting, configurations | **5.4**: own training loop, early stopping on validation loss, five experiments (size, activation, optimiser, learning rate, batch size), full scoreboard, final model | Baseline best validation RMSE 4.447; all 23 runs within 4.434 to 4.506; final model 32 hidden units, validation RMSE 4.432 |
| **Task 5.5:** same metrics, compare with the Random Forest | **5.5**: MSE, RMSE, MAE, R² and MAPE on the same test rows, charts | Neural network RMSE 4.47 vs forest 4.71; R² 0.768 vs 0.744 |
| **Task 5.6 and Question 10:** why networks do well on large data | **5.6** | Explained, and checked against our own results |

**Final result:** the neural network (test RMSE **4.47** minutes, MAE **3.53**) is slightly better than the Random Forest baseline (RMSE 4.71, MAE 3.70). Both are far better than always predicting the average (about 9 minutes).

**Limits of this result:**
- One split and one trained model of each kind, so the difference is small and not proven to be larger than run-to-run noise.
- The 411-item order was not removed (Section 3.5), and `estimated_store_to_consumer_driving_duration` was left out because the brief does not describe it (Section 2.4). Neither effect was measured."""

json.dump(R, open("/Users/vinayakkumar.udikeri/Documents/Porter_NN_Regression/build/readings.json", "w"), indent=1, ensure_ascii=False)
print(len(R), "readings written")
