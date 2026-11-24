"""Add the brief's own text (stages, marks, problem statement, data dictionary, process, evaluation criteria) to the builder and to solution.ipynb."""
import json, sys

STAGES = ["Stage 1: Problem Understanding, Data Import & Structure Analysis",
          "Stage 2: Data Preprocessing & Feature Engineering",
          "Stage 3: Data Visualization, Outlier Detection & Cleaning",
          "Stage 4: Baseline Model with Random Forest",
          "Stage 5: Regression with Neural Networks"]
MARKS = [("Problem Statement, Data Import & Structure Analysis", 10),
         ("Data Preprocessing & Feature Engineering", 30),
         ("Data Visualization, Outlier Detection & Cleaning", 20),
         ("Model Training with Random Forest", 10),
         ("Regression with Neural Networks", 30)]

top_insert = ("## Stages Overview\n\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(STAGES, 1)) +
  "\n\n## How You're Evaluated (100 Marks)\n\n| Section | Marks |\n|:--|:--:|\n" +
  "\n".join(f"| {n} | {m} |" for n, m in MARKS) + "\n| **Total** | **100** |\n\n")

process = """

## Process (as given in the brief)

1. Import the data and understand the structure of the data.
2. Usual exploratory analysis steps like checking the structure & characteristics of the dataset.
3. Data preprocessing
    - Cleaning of data
    - Feature engineering: creating the target column time taken in each delivery from order timestamp (created_at) and delivery timestamp (actual_delivery_time)
    - Getting hour of day from the order time and also the day of the week
    - Understanding pandas datetime data type and what function it provides by default
    - Get delivery time in minutes
    - Handling null values
    - Encoding categorical columns
4. Data visualization and cleaning
    - Visualize various columns for better understanding Countplots, scatterplots
    - Check if the data contains outliers
    - Removing outliers by any method
    - Plotting the data again to see if anything has improved
5. Split the data in train and test
6. Creating a baseline model using random forest
7. Checking its various metrics like mse, rsme, mae
8. Scaling the data for neural networks.
9. Creating a simple neural network
10. Trying different configurations
11. Understanding different activation functions, optimizers and other hyperparameters.
12. Training the neural network for required amount of epochs
13. Plotting the losses and checking the accuracy of the model
14. Comparing the results of neural network and random forest"""

problem = """### Problem Statement (as given in the brief)

Porter is India's Largest Marketplace for Intra-City Logistics. Leader in the country's \\$40 billion intra-city logistics market, Porter strives to improve the lives of 1,50,000+ driver-partners by providing them with consistent earning & independence. Currently, the company has serviced 5+ million customers

Porter works with a wide range of restaurants for delivering their items directly to the people.

Porter has a number of delivery partners available for delivering the food, from various restaurants and wants to get an estimated delivery time that it can provide the customers on the basis of what they are ordering, from where and also the delivery partners.

This dataset has the required data to train a regression model that will do the delivery time estimation, based on all those features

**Dataset:** https://drive.google.com/file/d/1kkgBGldeswHBgVKEJiqqG7VZvqZNMc-5/view?usp=share_link

"""

dictionary = """### Data Dictionary (as given in the brief)

Each row in this file corresponds to one unique delivery. Each column corresponds to a feature as explained below

| # | Column Name | Description |
|:--:|:--|:--|
| 1 | market_id | integer id for the market where the restaurant lies |
| 2 | created_at | the timestamp at which the order was placed |
| 3 | actual_delivery_time | the timestamp when the order was delivered |
| 4 | store_primary_category | category for the restaurant |
| 5 | order_protocol | integer code value for order protocol(how the order was placed ie: through porter, call to restaurant, pre booked, third part etc) |
| 6 | total_items subtotal | final price of the order |
| 7 | num_distinct_items | the number of distinct items in the order |
| 8 | min_item_price | price of the cheapest item in the order |
| 9 | max_item_price | price of the costliest item in order |
| 10 | total_onshift_partners | number of delivery partners on duty at the time order was placed |
| 11 | total_busy_partners | number of delivery partners attending to other tasks |
| 12 | total_outstanding_orders | total number of orders to be fulfilled at the moment |

The file we actually have uses slightly different names (for example `total_onshift_dashers`) and has two more columns. The next table describes the real columns.

"""

CRIT = {
 1: (10, [("Defining problem statement, importing the data and data structure analysis", "1.1 to 1.5")]),
 2: (30, [("Data cleaning", "2.5"), ("Null value handling", "2.5"),
          ("Creating the target column(time taken for delivery) from order timestamp and delivery timestamp", "2.2"),
          ("Getting hour and day of the week", "2.3"), ("Encoding categorical column", "2.6")]),
 3: (20, [("Visualization for various features", "3.1, 3.6"), ("Check for outliers", "3.1 to 3.3"),
          ("Remove outliers", "3.4"), ("Compare plots and results", "3.5")]),
 4: (10, [("Data splitting", "4.1"), ("Random forest regression", "4.2, 4.3")]),
 5: (30, [("Data scaling", "5.1"), ("Defining nn architecture", "5.2"),
          ("Trying different combinations and hyperparameters", "5.4"), ("Model training", "5.4"),
          ("Comparing results with random forest", "5.5")]),
}
def crit_block(n):
    pts, items = CRIT[n]
    rows = "\n".join(f"| {a} | {b} |" for a, b in items)
    return (f"**Evaluation criteria of this section ({pts} points), as given in the brief:**\n\n"
            f"| Criterion | Done in |\n|:--|:--|\n{rows}\n\n")

PAIRS = [("## Contents\n\nThe sections, points and tasks follow", top_insert + "## Contents\n\nThe sections, points and tasks follow"),
         ("| 10 | Why does a neural network perform well on a large dataset? | 5.6 |", "| 10 | Why does a neural network perform well on a large dataset? | 5.6 |" + process),
         ("### Business Case\n", problem + "### Business Case\n"),
         ("### Data Description\n\n| Id |", dictionary + "### Data Description\n\n| Id |")]
for n in range(1, 6):
    old = f"Each task below is marked in its heading, for example `(Task {n}.1)`."
    PAIRS.append((old, crit_block(n) + old))

target = sys.argv[1]
if target.endswith(".py"):
    s = open(target).read()
    for old, new in PAIRS:
        assert s.count(old) == 1, (s.count(old), old[:60])
        s = s.replace(old, new, 1)
    open(target, "w").write(s)
else:
    nb = json.load(open(target)); used = set()
    for c in nb["cells"]:
        if c["cell_type"] != "markdown": continue
        src = "".join(c["source"])
        for k, (old, new) in enumerate(PAIRS):
            if old in src and k not in used:
                src = src.replace(old, new, 1); used.add(k)
        c["source"] = src.splitlines(keepends=True)
    assert used == set(range(len(PAIRS))), set(range(len(PAIRS))) - used
    json.dump(nb, open(target, "w"), indent=1)
print("added to", target, "(", len(PAIRS), "edits )")
