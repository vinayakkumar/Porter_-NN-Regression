import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

import json, os
_READ_PATH = "/Users/vinayakkumar.udikeri/Documents/Porter_NN_Regression/build/readings.json"
READINGS = json.load(open(_READ_PATH)) if os.path.exists(_READ_PATH) else {}
def R(key):
    if key in READINGS:
        md(READINGS[key])

# ============================================================ TITLE + AGENDA
md(r'''
# Porter: Delivery Time Estimation using Neural Networks

**Objective:** Estimate how many minutes a delivery will take, using what was ordered, where it was ordered from, and how many delivery partners are available. A Random Forest is built as the baseline and a neural network is compared against it.

**Flow of the case study:**

```
Problem  -->  Data  -->  Preprocessing  -->  Visualisation & Outliers  -->  Random Forest  -->  Neural Network  -->  Compare
```

## Stages Overview

1. Stage 1: Problem Understanding, Data Import & Structure Analysis
2. Stage 2: Data Preprocessing & Feature Engineering
3. Stage 3: Data Visualization, Outlier Detection & Cleaning
4. Stage 4: Baseline Model with Random Forest
5. Stage 5: Regression with Neural Networks

## How You're Evaluated (100 Marks)

| Section | Marks |
|:--|:--:|
| Problem Statement, Data Import & Structure Analysis | 10 |
| Data Preprocessing & Feature Engineering | 30 |
| Data Visualization, Outlier Detection & Cleaning | 20 |
| Model Training with Random Forest | 10 |
| Regression with Neural Networks | 30 |
| **Total** | **100** |

## Contents

The sections, points and tasks follow the **Evaluation Criteria (100 points)** of the problem brief and the five stages of the case-study walkthrough (27 tasks, 10 brief questions).

| # | Section | Points | Tasks |
|:--:|:--|:--:|:--|
| 1 | [Problem Statement, Data Import and Data Structure Analysis](#sec1) | 10 | 1.1 to 1.5 |
| 2 | [Data Preprocessing and Feature Engineering](#sec2) | 30 | 2.1 to 2.6 |
| 3 | [Data Visualization, Outlier Detection and Cleaning](#sec3) | 20 | 3.1 to 3.6 |
| 4 | [Model Training with Random Forest](#sec4) | 10 | 4.1 to 4.4 |
| 5 | [Regression with Neural Networks](#sec5) | 30 | 5.1 to 5.6 |
| | **Total** | **100** | **27 tasks** |

**The 10 questions and where each is answered** (each question is also repeated above its answer):

| # | Question | Answered in |
|:--:|:--|:--:|
| 1 | Defining the problem statements and where can this and modifications of this be used? | 1.1 |
| 2 | List 3 functions the pandas datetime provides with one line explanation. | 1.4 |
| 3 | Short note on datetime, timedelta, timespan(period) | 1.5 |
| 4 | Why do we need to check for outliers in our data? | 3.2 |
| 5 | Name 3 outlier removal methods? | 3.3 |
| 6 | What classical machine learning methods can we use other than random forest for regression? | 4.4 |
| 7 | Why is scaling required for neural networks? | 5.1 |
| 8 | Briefly explain your choice of optimizer. | 5.3 |
| 9 | Which activation function did you use and why? | 5.2 |
| 10 | Why does a neural network perform well on a large dataset? | 5.6 |

## Process

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
14. Comparing the results of neural network and random forest
''')

# ============================================================ BUSINESS CASE
md(r'''
<a id="sec1"></a>

# Section 1: Problem Statement, Data Import and Data Structure Analysis (10 points)

### Objective and Tasks of this Section

**Stage 1 of 5 · 5 Tasks: Problem Understanding, Data Import & Structure Analysis**

**Objective:** You will frame the business problem Porter is trying to solve, load the delivery dataset, and get familiar with its structure before making any changes to it.

| Task | What to do | Hint |
|:--|:--|:--|
| **Task 1.1** | State the business problem in your own words — what is Porter trying to predict, and why does it matter to the business? Then note one other scenario (inside or outside Porter) where a similar delivery/ETA prediction model could be reused. | Think about what decision this predicted number feeds into downstream, and who sees it (the customer, the driver-partner, or an internal ops team). |
| **Task 1.2** | Import the dataset into your notebook environment and load it into a pandas DataFrame. | Think about how you'd pull a file shared via a Google Drive link directly into a Colab session, and which pandas function reads a CSV into a DataFrame. |
| **Task 1.3** | Inspect the structure of the data — shape, column names, and data types — and note which columns look like they will need type conversion before any feature engineering. | Think about which columns look like text/object type but actually represent points in time. |
| **Task 1.4** | List three functions or accessors that pandas' datetime handling provides, with a one-line explanation of what each one does. | Think about the different things you'd want to do with a timestamp column — parsing it, pulling out a component of it, and taking a difference between two of them. |
| **Task 1.5** | Write a short note distinguishing a datetime, a timedelta, and a Period/timespan object. | Think about the difference between a single point in time, a duration between two points, and a fixed span like 'March 2024'. |

**Evaluation criteria of this section (10 points):**

| Criterion | Done in |
|:--|:--|
| Defining problem statement, importing the data and data structure analysis | 1.1 to 1.5 |

Each task below is marked in its heading, for example `(Task 1.1)`.

---


## 1.1 Problem Statement (Task 1.1, Brief Question 1)

### Problem Statement

Porter is India's Largest Marketplace for Intra-City Logistics. Leader in the country's \$40 billion intra-city logistics market, Porter strives to improve the lives of 1,50,000+ driver-partners by providing them with consistent earning & independence. Currently, the company has serviced 5+ million customers

Porter works with a wide range of restaurants for delivering their items directly to the people.

Porter has a number of delivery partners available for delivering the food, from various restaurants and wants to get an estimated delivery time that it can provide the customers on the basis of what they are ordering, from where and also the delivery partners.

This dataset has the required data to train a regression model that will do the delivery time estimation, based on all those features

**Dataset:** https://drive.google.com/file/d/1kkgBGldeswHBgVKEJiqqG7VZvqZNMc-5/view?usp=share_link

### Business Case

Porter is India's largest marketplace for intra-city logistics. It helps 1,50,000+ driver-partners earn consistently and has served 5+ million customers.

Porter also works with a wide range of restaurants for delivering their food directly to people. There are many delivery partners available, and the food comes from many different restaurants.

You are a data scientist at Porter. Your job is to give every customer an **estimated delivery time** at the moment they place an order. The estimate should depend on:

- **what** they are ordering (number of items, price),
- **where** they are ordering from (market, type of restaurant, how the order was placed),
- **who can deliver it** right now (delivery partners on duty, partners already busy, orders waiting).

**Dataset link:** https://drive.google.com/file/d/1kkgBGldeswHBgVKEJiqqG7VZvqZNMc-5/view?usp=share_link

Each row of the dataset is one delivery. Let's start by defining the problem properly.
''')

md(r'''
### Defining the Problem Statement (Brief Question 1)

> **Brief Question 1:** *Defining the problem statements and where can this and modifications of this be used?*


**What do we want to predict?** The number of minutes a delivery will take, from the time the order is placed to the time it is delivered.

**Why does it matter?**

- **Customer:** decides whether to order, and whether to trust Porter. A promise that is too optimistic and then missed hurts trust. A promise that is too pessimistic loses orders.
- **Driver-partner:** helps plan trips and earnings fairly.
- **Operations team:** helps decide how many partners to keep on shift, and warns early when a market is short of partners.

**How do we turn it into a machine learning problem?**

| Item | Choice |
|:--|:--|
| Type of problem | **Regression**, because the answer is a number (minutes), not a class |
| Target | `time_taken_mins` = `actual_delivery_time` - `created_at`, in minutes (we create it in Section 2) |
| Features | Only what is known when the order is placed |
| Metrics | MAE (average error in minutes), RMSE, R² |
| Models | Random Forest as the baseline, then a neural network, compared on the same test data |

**Important:** `actual_delivery_time` is only known after the delivery, and it is the source of our target. If we used it as a feature, the model would simply read the answer. This is called **data leakage**.

### Where can this idea (and modifications of it) be used?

**Inside Porter**

- Delivery time for goods and parcel trips, not only food.
- Pickup time: how long until a driver-partner reaches the restaurant or the sender.
- Setting delivery promises and prices when partners are scarce.
- Deciding how many partners to keep on shift in each market and hour.

**Outside Porter**

- Food delivery and quick-commerce apps (the same features: items, price, restaurant, rider supply).
- E-commerce last-mile delivery, courier and freight companies.
- Ride-hailing arrival and trip-duration estimates.
- Field-service appointment windows and ambulance response times.

**Modifications**

- Predict a range (for example "35 to 50 minutes") instead of a single number.
- Turn it into classification: "will this order be late compared to its promise?" (the QuickCart example in the Hyperparameter Tuning class does exactly this).
- Predict delivery time and delivery cost together with one multi-output network.
- Add live features such as traffic and weather, and retrain regularly because conditions change.

### One Other Scenario in Detail: Ride-Hailing Arrival Time

A ride-hailing app has the same problem in a different business. When a customer books a cab, the app must say "your driver arrives in 6 minutes".

| | Porter delivery time | Ride-hailing arrival time |
|:--|:--|:--|
| What is predicted | Minutes from order to delivery | Minutes from booking to pickup |
| Information known at request time | Items, price, restaurant, partners on duty, partners busy, orders waiting | Pickup distance, drivers nearby, drivers busy, requests waiting, time of day |
| Type of problem | Regression | Regression |
| Same reasoning | Supply (partners) versus demand (orders) drives the waiting time | Supply (drivers) versus demand (requests) drives the waiting time |

The data preparation, the Random Forest baseline and the neural network in this notebook would carry over with only the column names changed.
''')

# ============================================================ IMPORT DATA
md(r'''
## 1.2 Importing the Data (Task 1.2)

### Setup: Import Libraries

We start with the libraries we need for this part: `pandas` for tabular data, `numpy` for numerical work and `matplotlib` and `seaborn` for charts. We also fix the random seed to 42 so that every run of the notebook gives the same results. The two colours defined here (blue and orange) are used in every chart.
''')

code(r'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

np.random.seed(42)

BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8a85"
sns.set_theme(style="whitegrid", palette=[BLUE, ORANGE, AQUA, YELLOW])
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
''')

md(r'''
### Data Dictionary

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

### Data Description

| Id | Features | Description |
| :-- | :-- | :-- |
|01| **market_id** | Integer id of the market where the restaurant lies |
|02| **created_at** | The timestamp at which the order was placed |
|03| **actual_delivery_time** | The timestamp at which the order was delivered |
|04| **store_primary_category** | Category of the restaurant |
|05| **order_protocol** | Integer code for how the order was placed (through Porter, call to the restaurant, pre-booked, third party, ...) |
|06| **total_items** | Number of items in the order |
|07| **subtotal** | Final price of the order |
|08| **num_distinct_items** | Number of distinct items in the order |
|09| **min_item_price** | Price of the cheapest item in the order |
|10| **max_item_price** | Price of the costliest item in the order |
|11| **total_onshift_dashers** | Number of delivery partners on duty when the order was placed |
|12| **total_busy_dashers** | Number of delivery partners busy with other tasks |
|13| **total_outstanding_orders** | Number of orders waiting to be fulfilled at that moment |

**Note:** The problem brief calls columns 11 and 12 `total_onshift_partners` and `total_busy_partners`. In the file they are named `total_onshift_dashers` and `total_busy_dashers`. The brief also joins `total_items` and `subtotal` into one phrase, but they are two separate columns. The file has one more column, `estimated_store_to_consumer_driving_duration` (estimated driving time in seconds). The brief does not describe it, so we follow the brief and **do not use it** as a feature (Section 2.4).

### Load the Data

The dataset is stored in the `data` folder next to this notebook. If the file is missing, the cell below downloads it from the Google Drive link given in the brief using `gdown`, a small library for downloading files from Google Drive. `pd.read_csv` reads the file into a DataFrame, and `.head()` previews the first five rows so we can check that it loaded correctly.
''')

code(r'''
import os
import gdown

FILE = "data/porter_delivery.csv"
FILE_ID = "1kkgBGldeswHBgVKEJiqqG7VZvqZNMc-5"      # file id from the Google Drive link in the brief

if not os.path.exists(FILE):
    os.makedirs("data", exist_ok=True)
    gdown.download(id=FILE_ID, output=FILE)

df = pd.read_csv(FILE)
df.head()
''')

md(r'''
`.tail()` shows the last rows, which is a quick way to confirm that the file was read completely.
''')

code(r'''
df.tail(3)
''')

# ============================================================ STRUCTURE ANALYSIS
md(r'''
## 1.3 Data Structure Analysis (Task 1.3)

Before changing anything, let's understand the structure of the data: how big it is, what type each column has, whether anything is missing, and what the values look like.

### Shape of the Data

`df.shape` returns the number of rows and the number of columns.
''')

code(r'''
df.shape
''')

md(r'''
Total number of samples and features of the data:

| Records | Features |
| :-- | :-- |
| 175777 | 14 |

Every record is one delivery. Two of the 14 columns are the timestamps the target is built from, so 12 columns remain as raw input features.

### Column Names, Data Types and Non-Null Counts

`df.info()` lists every column with its non-null count and its data type (`dtype`). Reading it carefully tells us what preparation is still needed.
''')

code(r'''
df.info()
''')

md(r'''
From the output above:

- Six columns are `float64`, six are `int64`, and two are text (`str`, shown as `object` in older pandas versions).
- The two text columns are `created_at` and `actual_delivery_time`. They hold dates and times but pandas treats them as plain text, so they must be converted to datetime before we can calculate with them.
- Every column has 175777 non-null values, so nothing seems to be missing. Let's confirm that next.

### Null Values and Duplicate Rows

`df.isnull().sum()` counts the missing values in every column, and `df.duplicated().sum()` counts rows that are exact copies of another row.
''')

code(r'''
df.isnull().sum()
''')

code(r'''
df.duplicated().sum()
''')

md(r'''
There are **no null values** and **no duplicate rows** in this file.

**Note:** Having no nulls here does not mean we can skip the check later. In Section 2 we run the null check again after cleaning the data and creating the target.

### Statistical Summary

`df.describe()` gives the count, mean, standard deviation, minimum, quartiles and maximum of every numeric column. `.T` flips the table so each column of the data becomes a row, which is easier to read for 12 columns.
''')

code(r'''
df.describe().T
''')

md(r'''
What stands out:

- `total_items` has a median of 3 but a maximum of 411.
- `subtotal` has a median of 2224 but a maximum of 26800, and a minimum of 0.
- `min_item_price` has a minimum of -86, and `total_onshift_dashers`, `total_busy_dashers` and `total_outstanding_orders` have negative minimums. A price or a count of partners cannot be negative.

These are the places where the data looks wrong or extreme. We will deal with them in Sections 2 and 3.

`df.describe()` skips the text columns. Adding `exclude="number"` gives the same kind of summary for them: how many values, how many are unique, the most frequent value and how often it appears.
''')

code(r'''
df.describe(exclude="number")
''')

md(r'''
`created_at` has 162649 unique values and `actual_delivery_time` has 160344, out of 175777 rows. These are timestamps down to the second, so most values appear only once.

### Columns That Look Numeric but Are Really Categories

`market_id`, `order_protocol` and `store_primary_category` are stored as numbers, but the number is only a label. Market 4 is not "twice" market 2. `.nunique()` counts the distinct values and `.value_counts()` shows how often each one appears.
''')

code(r'''
df[["market_id", "order_protocol", "store_primary_category"]].nunique()
''')

code(r'''
df["market_id"].value_counts().sort_index()
''')

code(r'''
df["order_protocol"].value_counts().sort_index()
''')

code(r'''
df["store_primary_category"].value_counts().head(10)
''')

md(r'''
- Market 2 has the most orders (53469) and market 6 the fewest (638). Protocols 1, 3 and 5 carry most of the orders, while protocol 7 has only 19. Restaurant category 4 is the most common (18183 orders).
- `market_id` has 6 values and `order_protocol` has 7 values, so they are small categorical columns. They are also stored as decimals (`1.0`, `2.0`), which is another hint that they are labels.
- `store_primary_category` has 73 different values. We will see how to encode it in Section 2.

### Variable Types and Conversions Needed

Putting the findings together, this is what each column really is and what has to happen to it before modelling.

| Column | Stored as now | What it really is | Conversion needed |
|:--|:--|:--|:--|
| `created_at` | text (`str`) | Date and time the order was placed | Convert to datetime (Section 2) |
| `actual_delivery_time` | text (`str`) | Date and time of delivery | Convert to datetime, build the target from it, then drop it (Section 2) |
| `market_id` | `float64` | Category label (6 values) | Treat as a category |
| `order_protocol` | `float64` | Category label (7 values) | Treat as a category |
| `store_primary_category` | `int64` | Category label (73 values) | Encode as integer codes (Section 2) |
| `total_items`, `num_distinct_items` | `int64` | Counts | None, but check extreme values |
| `subtotal`, `min_item_price`, `max_item_price` | `int64` | Prices | None, but check zeros, negatives and extremes |
| `total_onshift_dashers`, `total_busy_dashers`, `total_outstanding_orders` | `float64` | Counts of partners and orders | None, but check negatives |
| `estimated_store_to_consumer_driving_duration` | `float64` | Duration in seconds (not described in the brief) | Not used (Section 2.4) |

`df.nunique()` counts the distinct values in every column. Columns with only a few distinct values are categories, and columns with thousands of values are continuous measurements.
''')

code(r'''
df.nunique()
''')

md(r'''
### Data Quality Checks

The checks below do not change the data. They count the rows that look wrong, so we can handle them on purpose in the next parts.
''')

code(r'''
print("min_item_price < 0              :", (df["min_item_price"] < 0).sum())
print("total_onshift_dashers < 0       :", (df["total_onshift_dashers"] < 0).sum())
print("total_busy_dashers < 0          :", (df["total_busy_dashers"] < 0).sum())
print("total_outstanding_orders < 0    :", (df["total_outstanding_orders"] < 0).sum())
print("subtotal == 0                   :", (df["subtotal"] == 0).sum())
print("min_item_price > max_item_price :", (df["min_item_price"] > df["max_item_price"]).sum())
print("busy dashers > onshift dashers  :", (df["total_busy_dashers"] > df["total_onshift_dashers"]).sum())
''')

code(r'''
bad_rows = (
    (df["min_item_price"] < 0) | (df["total_onshift_dashers"] < 0) | (df["total_busy_dashers"] < 0)
    | (df["total_outstanding_orders"] < 0) | (df["subtotal"] == 0)
    | (df["min_item_price"] > df["max_item_price"])
)
print("Rows with at least one impossible value:", bad_rows.sum(), f"({bad_rows.mean() * 100:.2f}% of the data)")
''')

md(r'''
**Important:** 940 rows (about 0.5%) contain an impossible value, such as a negative price or a zero subtotal. We will remove or fix them during cleaning.

The last check is different. In about 22% of the rows (39290), the number of busy partners is larger than the number of partners on duty. That is far too common to be a typing mistake, so we will **not** delete these rows. We keep the columns as they are and let the models learn from them.
''')

# ============================================================ DATETIME
md(r'''
## 1.4 Three Functions pandas Provides for Datetime (Task 1.4, Brief Question 2)

> **Brief Question 2:** *List 3 functions the pandas datetime provides with one line explanation.*


Our target is built from two timestamps, so before using them let's understand how pandas handles dates and times.

### The Timestamp Columns

`created_at` and `actual_delivery_time` are plain text right now. `pd.to_datetime` can convert them. In the cell below we convert them into **separate variables** only to look at their range. The DataFrame `df` is not changed. The real conversion is done in Section 2.
''')

code(r'''
df[["created_at", "actual_delivery_time"]].dtypes
''')

code(r'''
created = pd.to_datetime(df["created_at"])
delivered = pd.to_datetime(df["actual_delivery_time"])

print("First order placed :", created.min())
print("Last order placed  :", created.max())
print("Days covered       :", (created.max() - created.min()).days)
print("Any delivery before its order was placed?", (delivered < created).any())
''')

md(r'''
The data covers about 27 days (21 January to 18 February 2015), and no delivery is recorded before its own order. With only four weeks of data, the model can learn weekly and daily patterns, but not monthly or seasonal ones.

### The Three Functions

| Function | One-line explanation |
|:--|:--|
| `pd.to_datetime()` | Converts text (or numbers) into proper datetime values, so dates can be compared, sorted and subtracted. |
| `.dt` accessor (for example `.dt.hour`, `.dt.day_of_week`) | Pulls one part, such as the hour of the day or the weekday, out of a datetime column. |
| Subtracting two datetimes and `.dt.total_seconds()` | The difference of two datetimes is a `Timedelta`, and `total_seconds()` turns it into a plain number (divide by 60 for minutes). |

Let's try all three on the first 5 rows, using a copy so that `df` stays unchanged.
''')

code(r'''
demo = df[["created_at", "actual_delivery_time"]].head(5).copy()

demo["created_at"] = pd.to_datetime(demo["created_at"])                      # 1. pd.to_datetime
demo["actual_delivery_time"] = pd.to_datetime(demo["actual_delivery_time"])

demo["hour"] = demo["created_at"].dt.hour                                    # 2. .dt accessor
demo["day_of_week"] = demo["created_at"].dt.day_of_week                      #    Monday = 0

demo["time_taken"] = demo["actual_delivery_time"] - demo["created_at"]       # 3. subtraction -> Timedelta
demo["minutes"] = demo["time_taken"].dt.total_seconds() / 60                 #    .dt.total_seconds()

demo
''')

code(r'''
demo.dtypes
''')

md(r'''
`created_at` is now `datetime64`, `time_taken` is `timedelta64`, and `minutes` is a plain `float64` number that a model can use. The original `df` is still unchanged:
''')

code(r'''
df[["created_at"]].dtypes
''')

md(r'''
### What the `.dt` Accessor Provides by Default

A datetime column comes with a whole set of ready-made parts and functions under `.dt`. `dir()` lists everything that is available, and the table below shows some of the most useful ones applied to the first order.
''')

code(r'''
dt_tools = [a for a in dir(created.dt) if not a.startswith("_")]
print("Number of attributes and functions in .dt:", len(dt_tools))

first_order = created.iloc[:1]
pd.Series({
    "year": first_order.dt.year.iloc[0],
    "month": first_order.dt.month.iloc[0],
    "day": first_order.dt.day.iloc[0],
    "hour": first_order.dt.hour.iloc[0],
    "minute": first_order.dt.minute.iloc[0],
    "day_of_week (Monday = 0)": first_order.dt.day_of_week.iloc[0],
    "day_name()": first_order.dt.day_name().iloc[0],
    "quarter": first_order.dt.quarter.iloc[0],
    "is_month_end": first_order.dt.is_month_end.iloc[0],
    "floor('h') (round down to the hour)": first_order.dt.floor("h").iloc[0],
    "strftime('%d %b %Y')": first_order.dt.strftime("%d %b %Y").iloc[0],
}, name="value for the first order")
''')

md(r'''
## 1.5 datetime, timedelta and Period (Task 1.5, Brief Question 3)

> **Brief Question 3:** *Short note on datetime, timedelta, timespan(period)*


| Concept | What it is | pandas type | Example from our data |
|:--|:--|:--|:--|
| **datetime** | A single **point in time** | `Timestamp` | The moment an order was placed: `2015-02-06 22:24:17` |
| **timedelta** | A **duration**: the length of time between two points | `Timedelta` | Delivery time = delivered minus placed = `0 days 00:47:00` |
| **Period** (timespan) | A **fixed span on the calendar** with a start and an end | `Period` | The whole month `2015-02`, or the whole day `2015-02-06` |

How they work together:

- `datetime - datetime` gives a `timedelta`.
- `datetime + timedelta` gives a new `datetime`.
- A `datetime` can be placed in a `Period` with `.to_period()`, and a `Period` tells us its `start_time` and `end_time`.

Our target, the delivery time, is a **timedelta**. We will turn it into a number of minutes because machine learning models need numbers.
''')

code(r'''
t_placed = pd.Timestamp(df.loc[0, "created_at"])
t_delivered = pd.Timestamp(df.loc[0, "actual_delivery_time"])

print("datetime  :", t_placed, "->", type(t_placed).__name__)

duration = t_delivered - t_placed
print("timedelta :", duration, "->", type(duration).__name__, "| minutes =", duration.total_seconds() / 60)

print("placed + 30 minutes :", t_placed + pd.Timedelta(minutes=30))

day = t_placed.to_period("D")
month = t_placed.to_period("M")
print("Period (day)   :", day, "from", day.start_time, "to", day.end_time)
print("Period (month) :", month, "from", month.start_time, "to", month.end_time)
''')

md(r'''
### A Peek at the Target (a timedelta in action)

The target is the time between the two timestamps. Let's look at its range now (on a temporary variable) so that we know what a sensible prediction error looks like later.
''')

code(r'''
peek = (delivered - created).dt.total_seconds() / 60
peek.describe()
''')

md(r'''
Deliveries take between 32 and 110 minutes, with a mean of about 46 minutes and a standard deviation of about 9 minutes. A model that always predicts about 46 minutes would be off by roughly 9 minutes on average, so any useful model must do better than that.
''')

# ============================================================ RECAP
md(r'''
## 1.6 Summary: Objective and How It Was Addressed

**Motto of this section:** *Understand the problem and the data before touching either.*

The table lists every task and question of the brief that belongs to this section, where it is answered, and what we found.

| Requirement in the brief | Where it is addressed | Result |
|:--|:--|:--|
| **Task 1.1 and Question 1:** state the business problem, give one other scenario, and say where the idea and its modifications can be used | **1.1 Problem Statement**: Business Case, problem definition, uses inside and outside Porter, modifications, and the ride-hailing scenario in detail | Regression. Target is delivery time in minutes. `actual_delivery_time` is never used as a feature |
| **Task 1.2:** import the dataset into a pandas DataFrame | **1.2 Importing the Data**: libraries and seed, description of the 14 columns, `gdown` and `pd.read_csv`, `head()` and `tail()` | 175777 rows and 14 columns loaded |
| **Task 1.3:** inspect shape, column names and data types, and note which columns need conversion | **1.3 Data Structure Analysis**: shape, `info()`, `describe()`, `nunique()`, and the table of variable types and conversions needed | Two timestamp columns are text. Three numeric columns are really categories |
| **Task 1.3 (structure and characteristics):** usual exploratory checks | **1.3**: null values, duplicates, ID-like columns, data-quality checks | No nulls, no duplicates. 940 rows with impossible values. 39290 rows with busy partners above on-duty partners, kept on purpose |
| **Task 1.4 and Question 2:** three functions pandas datetime provides, with one line each | **1.4 Three Functions pandas Provides for Datetime**: `pd.to_datetime()`, the `.dt` accessor, subtraction with `.dt.total_seconds()`, plus what `.dt` provides by default | Demonstrated on real rows |
| **Task 1.5 and Question 3:** note on datetime, timedelta and Period | **1.5 datetime, timedelta and Period**: comparison table and a worked example | Point in time, duration, and calendar span, shown on the first order |
| Peek at the target | **1.5**: A Peek at the Target | About 27 days of data. Target between 32 and 110 minutes, mean about 46 |
| Prepare for Section 2 | Problems found above are carried forward | Convert the timestamps, create the target, encode the categories, clean the impossible values, and check `total_items` and `subtotal` for outliers |
''')

# ============================================================ SECTION 2
md(r'''
<a id="sec2"></a>

# Section 2: Data Preprocessing and Feature Engineering (30 points)

### Objective and Tasks of this Section

**Stage 2 of 5 · 6 Tasks: Data Preprocessing & Feature Engineering**

**Objective:** You will convert the raw timestamp columns into a usable target variable, derive new time-based features, handle missing values, and encode the categorical column so the data is model-ready.

| Task | What to do | Hint |
|:--|:--|:--|
| **Task 2.1** | Convert created_at and actual_delivery_time from object type to proper datetime type. | Think about the pandas function you identified in Stage 1 that parses strings into datetime objects, and confirm the conversion using .info(). |
| **Task 2.2** | Create the target column by finding the time taken for each delivery, then convert it into a single numeric value in minutes. | Think about what subtracting two datetime columns produces, and how you'd turn that duration into a plain number rather than a timedelta object. |
| **Task 2.3** | Extract the hour of day and day of week the order was placed as two new features. | Think about which .dt accessors let you pull a specific component out of a datetime column. |
| **Task 2.4** | Drop the columns that are no longer needed now that the target and time features have been extracted. | Think about which raw columns would now duplicate information already captured in your new features, and could leak into the model if left in. |
| **Task 2.5** | Check for null values across all columns and handle them appropriately. | Think about which pandas method summarizes missing values per column, and what a reasonable default action is when nulls are a small fraction of the data. |
| **Task 2.6** | Encode the categorical column store_primary_category into a numeric form so it can be used by both models. | Think about how you can convert a column to a category dtype and then pull out its underlying integer codes. |

**Evaluation criteria of this section (30 points):**

| Criterion | Done in |
|:--|:--|
| Data cleaning | 2.5 |
| Null value handling | 2.5 |
| Creating the target column(time taken for delivery) from order timestamp and delivery timestamp | 2.2 |
| Getting hour and day of the week | 2.3 |
| Encoding categorical column | 2.6 |

Each task below is marked in its heading, for example `(Task 2.1)`.

---


In Section 1 we understood the data. Now we prepare it for modelling, following the steps of the problem brief: convert the timestamps, create the target (time taken in minutes), get the hour of the day and the day of the week, clean the data, handle null values and encode the categorical columns.

**Flow of this section:**

```
Convert timestamps --> Target in minutes --> Hour & day --> Drop columns --> Clean & nulls --> Encode
```
''')

md(r'''
## 2.1 Convert Timestamps to Datetime (Task 2.1)

`pd.to_datetime` turns text into proper datetime values. First we keep an untouched copy of the data in `df_raw`, so that we can compare before and after later.
''')
code(r'''
df_raw = df.copy()

df["created_at"] = pd.to_datetime(df["created_at"])
df["actual_delivery_time"] = pd.to_datetime(df["actual_delivery_time"])

df.info()
''')
R("s2_1")

md(r'''
## 2.2 Create the Target: Time Taken in Minutes (Task 2.2)

Subtracting two datetime columns gives a `Timedelta`. `.dt.total_seconds()` turns it into seconds, and dividing by 60 gives the delivery time in minutes, a plain number a model can learn.
''')
code(r'''
df["time_taken_mins"] = (df["actual_delivery_time"] - df["created_at"]).dt.total_seconds() / 60

df[["created_at", "actual_delivery_time", "time_taken_mins"]].head()
''')
code(r'''
df["time_taken_mins"].describe()
''')
md(r'''
A histogram shows the shape of the new target column.
''')
code(r'''
plt.figure(figsize=(9, 4))
sns.histplot(df["time_taken_mins"], bins=range(30, 115, 2), kde=True, color=BLUE)
plt.axvline(df["time_taken_mins"].mean(), color=ORANGE, linestyle="--", linewidth=1.5, label=f"mean = {df['time_taken_mins'].mean():.1f}")
plt.axvline(df["time_taken_mins"].median(), color="#52514e", linestyle=":", linewidth=1.5, label=f"median = {df['time_taken_mins'].median():.0f}")
plt.title("Delivery time in minutes")
plt.xlabel("minutes from order to delivery")
plt.ylabel("number of orders")
plt.legend()
plt.show()
''')
R("s2_2")

md(r'''
## 2.3 Hour of the Day and Day of the Week (Task 2.3)

The `.dt` accessor pulls the hour and the weekday out of the order time. Monday is 0 and Sunday is 6.
''')
code(r'''
df["hour"] = df["created_at"].dt.hour
df["day_of_week"] = df["created_at"].dt.day_of_week

df[["created_at", "hour", "day_of_week"]].head()
''')

md(r'''
## 2.4 Drop Columns That Are No Longer Needed (Task 2.4)

- `created_at` is now represented by `hour` and `day_of_week`.
- `actual_delivery_time` is the source of the target, so keeping it would give the answer to the model (data leakage).
- `estimated_store_to_consumer_driving_duration` is in the file, but it is not part of the data dictionary of the problem brief. We follow the brief and use only the columns it describes.
''')
code(r'''
df = df.drop(columns=["created_at", "actual_delivery_time", "estimated_store_to_consumer_driving_duration"])

df.columns.tolist()
''')

md(r'''
## 2.5 Data Cleaning and Null Values (Task 2.5)

### Data Cleaning: Impossible Values

In Section 1 we found values that cannot be real: negative prices, negative partner counts, a zero subtotal, and a cheapest item that costs more than the costliest. First we count how many rows break each rule.
''')
code(r'''
rules = {
    "min_item_price < 0": df["min_item_price"] < 0,
    "total_onshift_dashers < 0": df["total_onshift_dashers"] < 0,
    "total_busy_dashers < 0": df["total_busy_dashers"] < 0,
    "total_outstanding_orders < 0": df["total_outstanding_orders"] < 0,
    "subtotal == 0": df["subtotal"] == 0,
    "min_item_price > max_item_price": df["min_item_price"] > df["max_item_price"],
}

rule_counts = pd.Series({name: int(mask.sum()) for name, mask in rules.items()})
rule_counts
''')
code(r'''
plt.figure(figsize=(9, 3.5))
ax = sns.barplot(x=rule_counts.values, y=rule_counts.index, color=BLUE)
ax.bar_label(ax.containers[0], fmt="{:,.0f}", fontsize=9)
plt.title("Rows that break each rule (one row can break more than one)")
plt.xlabel("number of rows")
plt.show()
''')
md(r'''
These rows are a tiny share of the data, so we remove them. `pd.concat(...).any(axis=1)` marks a row if it breaks at least one rule.

The rows where busy partners exceed on-duty partners are **not** removed. There are far too many of them to be mistakes (Section 1.3).
''')
code(r'''
bad_rows = pd.concat(rules.values(), axis=1).any(axis=1)
print("Rows removed:", bad_rows.sum(), f"({bad_rows.mean() * 100:.2f}% of the data)")

df = df[~bad_rows].reset_index(drop=True)
print("Shape after cleaning:", df.shape)
''')
R("s2_5")

md(r'''
### Null Values

`df.isnull().sum()` counts the null values in every column. If only a tiny share of rows had nulls we would drop them, and otherwise we would fill them (for example with the median).
''')
code(r'''
print("Total null values:", int(df.isnull().sum().sum()))
df.isnull().sum()
''')
R("s2_6")

md(r'''
## 2.6 Encoding the Categorical Column (Task 2.6)

`store_primary_category` has 73 different values. As in the problem brief, we convert the column to the `category` type and take its integer codes. `market_id` and `order_protocol` are stored as decimals (`1.0`, `2.0`), so we convert them to integers.
''')
code(r'''
df["store_category_code"] = df["store_primary_category"].astype("category").cat.codes
df["market_id"] = df["market_id"].astype(int)
df["order_protocol"] = df["order_protocol"].astype(int)

df[["store_primary_category", "store_category_code"]].drop_duplicates().sort_values("store_category_code").head()
''')
code(r'''
df = df.drop(columns=["store_primary_category"])

df.head()
''')
md(r'''
**Note:** `market_id` and `order_protocol` are also labels, so we turn them into one-hot columns (`pd.get_dummies`) just before modelling in Section 4. That keeps the original columns available for the charts in Section 3.
''')
code(r'''
print("Shape after Section 2:", df.shape)
df.dtypes
''')
R("s2_7")

R("s2_sum")

# ============================================================ SECTION 3
md(r'''
<a id="sec3"></a>

# Section 3: Data Visualization, Outlier Detection and Cleaning (20 points)

### Objective and Tasks of this Section

**Stage 3 of 5 · 6 Tasks: Data Visualization, Outlier Detection & Cleaning**

**Objective:** You will visualize relationships in the cleaned data, understand why outliers matter, detect and remove them, and confirm visually that the cleaning improved the data.

| Task | What to do | Hint |
|:--|:--|:--|
| **Task 3.1** | Visualize relationships between the target and key features — for example a correlation heatmap and scatterplots of time_taken_mins against subtotal, and total_items against subtotal. | Think about which plot type best shows the correlation strength between many numeric columns at once, versus which best shows the relationship between two specific variables. |
| **Task 3.2** | Explain why it's important to check for outliers before training a model on this data. | Think about how a small number of extreme delivery times could disproportionately pull a regression model's error metric or its learned relationships. |
| **Task 3.3** | Name three methods that can be used to detect or remove outliers. | Think about a simple statistical rule based on quartiles, a plot-based visual method, and a model-based method that scores each point relative to its neighbors. |
| **Task 3.4** | Apply an outlier detection method to the data, print how many rows were flagged, and remove those rows. | Think about a method that can flag unusual rows across several columns at once, rather than looking at just one column in isolation. |
| **Task 3.5** | Re-plot the same scatterplot from Task 3.1 after removing outliers, and compare it to the original to confirm the cleaning helped. | Think about which exact plot you produced before cleaning — reusing it after cleaning is the only way to see whether the distribution actually improved. |
| **Task 3.6** | Visualize order volume by day of week and hour of day, and check whether time of day affects delivery time. | Think about which plot best shows a count/frequency by category, versus which best shows a trend across a continuous-like feature such as hour. |

**Evaluation criteria of this section (20 points):**

| Criterion | Done in |
|:--|:--|
| Visualization for various features | 3.1, 3.6 |
| Check for outliers | 3.1 to 3.3 |
| Remove outliers | 3.4 |
| Compare plots and results | 3.5 |

Each task below is marked in its heading, for example `(Task 3.1)`.

---


Now that the data is clean and complete, we look at it with charts (countplots and scatterplots, as in the brief), check for outliers, remove them, and plot the data again to see whether anything improved.

**Flow of this section:**

```
Countplots & scatterplots --> Why outliers matter --> Three methods --> Detect & remove --> Compare --> Day & hour patterns
```
''')

md(r'''
## 3.1 Visualising the Data (Task 3.1)

We import `IsolationForest`, which we use in Section 3.4, and list the eight numeric columns that describe the order and the partner supply.
''')
code(r'''
from sklearn.ensemble import IsolationForest

num_cols = ["total_items", "subtotal", "num_distinct_items", "min_item_price", "max_item_price",
            "total_onshift_dashers", "total_busy_dashers", "total_outstanding_orders"]

day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
''')

md(r'''
### Countplots of the Categorical Columns

A countplot shows how many orders fall into each category.
''')
code(r'''
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.countplot(data=df, x="market_id", color=BLUE, ax=axes[0])
axes[0].bar_label(axes[0].containers[0], fmt="{:,.0f}", fontsize=8)
axes[0].set_title("Orders per market")
axes[0].set_ylabel("number of orders")

sns.countplot(data=df, x="order_protocol", color=BLUE, ax=axes[1])
axes[1].bar_label(axes[1].containers[0], fmt="{:,.0f}", fontsize=8)
axes[1].set_title("Orders per order protocol")
axes[1].set_ylabel("")

top_codes = df["store_category_code"].value_counts().head(10).index
sns.countplot(data=df, y="store_category_code", order=top_codes, color=BLUE, ax=axes[2])
axes[2].bar_label(axes[2].containers[0], fmt="{:,.0f}", fontsize=8)
axes[2].set_title("10 most common restaurant categories")
axes[2].set_xlabel("number of orders")

plt.tight_layout()
plt.show()
''')
R("s3_1a")

md(r'''
### Histograms of the Numeric Columns

A histogram shows how the values of one column are spread. We use a log scale on the vertical axis so that rare, extreme values stay visible next to the very common small ones.
''')
code(r'''
fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for ax, col in zip(axes.ravel(), num_cols):
    ax.hist(df[col], bins=40, color=BLUE, log=True)
    ax.set_title(col, fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
fig.suptitle("Distribution of the numeric columns (log scale on the vertical axis)")
plt.tight_layout()
plt.show()
''')
R("s3_1d")

md(r'''
### Correlation Heatmap

`df.corr()` measures how strongly each pair of numeric columns moves together, from -1 to +1. `sns.heatmap` draws it with the numbers printed inside. The colours run from blue (negative) through grey (zero) to orange (positive).
''')
code(r'''
DIVERGING = sns.blend_palette([BLUE, "#f0efec", ORANGE], as_cmap=True)

plt.figure(figsize=(11, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap=DIVERGING, vmin=-1, vmax=1,
            annot_kws={"size": 8}, linewidths=0.5, linecolor="white")
plt.title("Correlation between the numeric columns")
plt.show()
''')
code(r'''
df.corr(numeric_only=True)["time_taken_mins"].sort_values(ascending=False).round(2)
''')
R("s3_1h")

md(r'''
### Scatterplots (Before Removing Outliers)

We plot delivery time against subtotal, and total items against subtotal. We define the plots as a function, so that exactly the same plot can be drawn again after the outliers are removed. A random sample of 20000 orders keeps the plot readable.
''')
code(r'''
def scatter_pair(data, title):
    sample = data.sample(20000, random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=sample, x="subtotal", y="time_taken_mins", s=8, alpha=0.2, color=BLUE, edgecolor=None, ax=axes[0])
    axes[0].set_title("Delivery time vs subtotal")
    sns.scatterplot(data=sample, x="subtotal", y="total_items", s=8, alpha=0.2, color=BLUE, edgecolor=None, ax=axes[1])
    axes[1].set_title("Total items vs subtotal")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

scatter_pair(df, "Before removing outliers")
''')
R("s3_1b")

md(r'''
### Box Plots to Check for Outliers (Before Removing Outliers)

A box plot shows the middle half of the values as a box and marks the unusual values as dots. Each column has its own scale.
''')
code(r'''
def box_grid(data, title):
    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    for ax, col in zip(axes.ravel(), num_cols):
        sns.boxplot(x=data[col], color=BLUE, fliersize=2, flierprops={"markerfacecolor": ORANGE, "markeredgecolor": ORANGE}, ax=ax)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

box_grid(df, "Before removing outliers")
''')
R("s3_1c")

md(r'''
## 3.2 Why Check for Outliers? (Task 3.2, Brief Question 4)

> **Brief Question 4:** *Why do we need to check for outliers in our data?*


An **outlier** is a value that sits far away from the rest of the data. We check for them before training because:

- **Averages and spreads get pulled.** One order with 411 items moves the mean and the standard deviation of `total_items`, and `StandardScaler` uses both.
- **Squared errors reward big values.** The mean squared error squares every mistake, so a few extreme rows can dominate the loss and drag the weights of a neural network towards them.
- **Models learn from too little evidence.** A very rare kind of order has so few examples that the model cannot learn it, but it can still distort the rest.
- **They may be mistakes.** Some outliers are typing or system errors that should not be learned at all.

**Note:** An outlier is not automatically wrong. A very large order can be real. So we look at the outliers first, and then decide what to do with them.
''')

md(r'''
## 3.3 Three Methods to Find and Remove Outliers (Task 3.3, Brief Question 5)

> **Brief Question 5:** *Name 3 outlier removal methods?*


| Method | How it works | Looks at |
|:--|:--|:--|
| **IQR rule** (the rule behind a box plot) | A value is an outlier if it is more than 1.5 times the IQR (the width of the middle half) below the first quartile or above the third quartile | One column at a time |
| **Z-score** | A value is an outlier if it is more than 3 standard deviations from the mean | One column at a time |
| **Box plot or scatterplot** (visual method) | Points beyond the whiskers of a box plot, or far away from the main cloud of a scatterplot, are outlier candidates. We drew both in Section 3.1 | One or two columns at a time |
| **Isolation Forest** (also Local Outlier Factor, DBSCAN) | Random trees try to isolate each row. Rows that are isolated with very few splits are unusual | Several columns at once |

The brief allows any method. We use **Isolation Forest** as the main method, because it looks at several columns together instead of one column in isolation. A multi-column method can miss a value that is extreme in only one column, so we add a simple one-column percentile rule after it (Section 3.4).
''')

md(r'''
### How Many Rows Would Each Method Flag?

We apply all three methods to the cleaned data (nothing is removed yet) and count the rows that each one flags. The IQR rule and the z-score flag a row if **any** of the eight numeric columns is extreme. The Isolation Forest looks at the eight columns together.
''')
code(r'''
q1, q3 = df[num_cols].quantile(0.25), df[num_cols].quantile(0.75)
iqr = q3 - q1
iqr_flag = (df[num_cols] < q1 - 1.5 * iqr) | (df[num_cols] > q3 + 1.5 * iqr)
z_flag = ((df[num_cols] - df[num_cols].mean()) / df[num_cols].std()).abs() > 3
iso_flag = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1).fit(df[num_cols]).predict(df[num_cols]) == -1

flag_table = pd.DataFrame({
    "rows flagged": [iqr_flag.any(axis=1).sum(), z_flag.any(axis=1).sum(), iso_flag.sum()],
    "share of the data (%)": [iqr_flag.any(axis=1).mean() * 100, z_flag.any(axis=1).mean() * 100, iso_flag.mean() * 100],
}, index=["IQR rule (any column)", "Z-score above 3 (any column)", "Isolation Forest (eight columns together)"])
flag_table.round(2)
''')
R("s3_3")

md(r'''
## 3.4 Detect and Remove Outliers (Task 3.4)

`contamination=0.01` tells the Isolation Forest to flag about 1% of the rows as the most unusual ones. We give it only the eight numeric input columns and **not** the delivery time, so the target does not decide which rows are removed.
''')
code(r'''
df_before = df.copy()

iso = IsolationForest(n_estimators=100, contamination=0.01, random_state=42, n_jobs=-1)
iso.fit(df[num_cols])
df["is_outlier"] = iso.predict(df[num_cols]) == -1

print("Rows flagged as outliers:", df["is_outlier"].sum(), f"({df['is_outlier'].mean() * 100:.2f}% of the data)")
''')
md(r'''
The rows the Isolation Forest flagged, so that we can see what it considers unusual (the first ten are printed).
''')
code(r'''
df[df["is_outlier"]][num_cols + ["time_taken_mins"]].head(10)
''')
md(r'''
Let's see what the flagged rows look like. In the charts they are drawn in orange.
''')
code(r'''
sample = df.sample(30000, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=sample, x="subtotal", y="total_items", hue="is_outlier", palette={False: BLUE, True: ORANGE},
                s=10, alpha=0.5, edgecolor=None, ax=axes[0])
axes[0].set_title("Flagged rows: total items vs subtotal")
sns.scatterplot(data=sample, x="total_onshift_dashers", y="total_outstanding_orders", hue="is_outlier", palette={False: BLUE, True: ORANGE},
                s=10, alpha=0.5, edgecolor=None, ax=axes[1])
axes[1].set_title("Flagged rows: orders waiting vs partners on shift")
plt.tight_layout()
plt.show()
''')
code(r'''
df.groupby("is_outlier")[num_cols + ["time_taken_mins"]].mean().T.round(1)
''')
R("s3_4")
md(r'''
Now we remove the flagged rows.
''')
code(r'''
df = df[~df["is_outlier"]].drop(columns="is_outlier").reset_index(drop=True)

print("Rows before:", len(df_before))
print("Rows after :", len(df))
''')

md(r'''
### A Second, Simple Rule for One Extreme Column

The Isolation Forest judges all eight columns together, so an order that is extreme in **one** column but ordinary in the others can slip through. Let's check the largest item counts that are left.
''')
code(r'''
df.sort_values("total_items", ascending=False)[["total_items", "subtotal", "num_distinct_items", "max_item_price", "time_taken_mins"]].head(8)
''')
md(r'''
One order has 411 items, and the next largest has only 66. This order has just 5 different items and a subtotal of about 3,100, which does not fit 411 items, so it is almost certainly a data entry error.

A rule such as the IQR fence (`Q3 + 1.5 x IQR`) would be too harsh here: the middle half of the orders has 2 to 4 items, so the fence is only 7 items and would remove thousands of ordinary larger orders. Instead we remove only the most extreme 0.1%: orders with more items than the **99.9th percentile** of `total_items`.
''')
code(r'''
cap = df["total_items"].quantile(0.999)
extreme = df["total_items"] > cap

print(f"99.9th percentile of total_items: {cap:.1f}")
print(f"Rows above it: {extreme.sum()} ({extreme.mean() * 100:.2f}% of the data)")
print(f"Average delivery time of these rows: {df.loc[extreme, 'time_taken_mins'].mean():.1f} minutes (all rows: {df['time_taken_mins'].mean():.1f})")

df = df[~extreme].reset_index(drop=True)
print("Rows after the second rule:", len(df))
''')
md(r'''
**Note on the order of the steps:** the brief cleans the data before splitting it, so the outlier rules (including the 99.9th percentile) were worked out on all rows, also on rows that later land in the test set. Only input columns were used, never the target, so the effect is small, but it is a mild form of leakage. In a live system the rules would be fitted on the training rows only and then applied unchanged to new orders.
''')
R("s3_4b")

md(r'''
## 3.5 Compare the Plots and Results Before and After (Task 3.5)

We draw exactly the same scatterplots and box plots again and compare them with the earlier ones.
''')
code(r'''
scatter_pair(df, "After removing outliers")
''')
code(r'''
box_grid(df, "After removing outliers")
''')
md(r'''
The numbers tell the same story:
''')
code(r'''
def summary(data):
    return pd.Series({
        "rows": len(data),
        "max total_items": data["total_items"].max(),
        "max subtotal": data["subtotal"].max(),
        "max total_outstanding_orders": data["total_outstanding_orders"].max(),
        "std of subtotal": data["subtotal"].std(),
        "std of delivery time": data["time_taken_mins"].std(),
        "corr(time, subtotal)": data["time_taken_mins"].corr(data["subtotal"]),
    })

pd.DataFrame({"before": summary(df_before), "after": summary(df)}).round(2)
''')
R("s3_5")

md(r'''
## 3.6 Order Volume and Delivery Time by Day and Hour (Task 3.6)

Does the time of the order change how long delivery takes? The charts on the left count the orders, and the charts on the right show the average delivery time with its 95% confidence band.
''')
code(r'''
fig, axes = plt.subplots(2, 2, figsize=(15, 9))

sns.countplot(data=df, x="day_of_week", color=BLUE, ax=axes[0, 0])
axes[0, 0].set_xticks(range(7))
axes[0, 0].set_xticklabels(day_names)
axes[0, 0].set_title("Orders by day of the week")
axes[0, 0].set_xlabel("")

sns.barplot(data=df, x="day_of_week", y="time_taken_mins", color=BLUE, errorbar=("ci", 95), n_boot=200, ax=axes[0, 1])
axes[0, 1].set_xticks(range(7))
axes[0, 1].set_xticklabels(day_names)
axes[0, 1].set_ylim(35, 55)
axes[0, 1].set_title("Average delivery time by day of the week")
axes[0, 1].set_xlabel("")
axes[0, 1].set_ylabel("minutes")

sns.countplot(data=df, x="hour", color=BLUE, ax=axes[1, 0])
axes[1, 0].set_title("Orders by hour of the day")

sns.lineplot(data=df, x="hour", y="time_taken_mins", marker="o", color=BLUE, errorbar=("ci", 95), n_boot=200, ax=axes[1, 1])
axes[1, 1].set_title("Average delivery time by hour of the day")
axes[1, 1].set_ylabel("minutes")

plt.tight_layout()
plt.show()
''')
code(r'''
print("Average delivery time by hour (minutes):")
print(df.groupby("hour")["time_taken_mins"].mean().round(1).to_string())
print("\nAverage delivery time by day of the week (minutes):")
print(df.groupby("day_of_week")["time_taken_mins"].mean().round(1).rename(dict(enumerate(day_names))).to_string())
''')
R("s3_6")

R("s3_sum")

# ============================================================ SECTION 4
md(r'''
<a id="sec4"></a>

# Section 4: Model Training with Random Forest (10 points)

### Objective and Tasks of this Section

**Stage 4 of 5 · 4 Tasks: Baseline Model with Random Forest**

**Objective:** You will split the cleaned data into train and test sets, train a random forest baseline, and evaluate it thoroughly so it can act as a benchmark for the neural network.

| Task | What to do | Hint |
|:--|:--|:--|
| **Task 4.1** | Split the cleaned, encoded data into training and test sets in a reproducible way. | Think about which train_test_split parameter controls the proportion held out for testing, and which one makes the split reproducible across runs. |
| **Task 4.2** | Train a random forest regression model as your baseline. | Think about which scikit-learn class implements an ensemble of decision trees for a regression target. |
| **Task 4.3** | Evaluate the random forest on the test set using multiple error metrics, and identify which features matter most. | Think about why reporting more than one error metric (magnitude in the same unit as the target, plus a normalized/percentage view) gives a fuller picture than a single number. |
| **Task 4.4** | Name other classical machine learning methods (besides random forest) that could be used for this regression problem. | Think about other regression algorithms you know that don't rely on neural networks — both linear and tree/ensemble-based. |

**Evaluation criteria of this section (10 points):**

| Criterion | Done in |
|:--|:--|
| Data splitting | 4.1 |
| Random forest regression | 4.2, 4.3 |

Each task below is marked in its heading, for example `(Task 4.1)`.

---


Before building a neural network we need a baseline: a strong, simple model whose numbers the neural network must beat. As in the brief, we split the data into train and test sets, train a Random Forest, and check its MSE, RMSE and MAE.

**Flow of this section:**

```
Encode & split --> Random Forest --> Metrics --> Other classical methods
```
''')

md(r'''
## 4.1 Preparing the Features and Splitting the Data (Task 4.1)

`market_id` and `order_protocol` are labels, so we turn them into one-hot columns with `pd.get_dummies(..., dtype=float)`. `store_category_code` keeps its integer codes. Then `X` holds the inputs and `y` the target.
''')
code(r'''
X = pd.get_dummies(df.drop(columns=["time_taken_mins"]), columns=["market_id", "order_protocol"], dtype=float)
y = df["time_taken_mins"]

print(X.shape, y.shape)
X.head()
''')
md(r'''
`train_test_split` keeps 20% of the rows for testing. `random_state=42` makes the split reproducible, so every run gives the same train and test sets. We do not use `stratify`, because the target is a continuous number.
''')
code(r'''
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train:", X_train.shape, y_train.shape)
print("Test :", X_test.shape, y_test.shape)
''')
md(r'''
The test set must look like the training set. Let's check the delivery time in both.
''')
code(r'''
split_target = pd.concat([
    pd.DataFrame({"minutes": y_train, "set": "train"}),
    pd.DataFrame({"minutes": y_test, "set": "test"}),
])

plt.figure(figsize=(9, 4))
sns.histplot(data=split_target, x="minutes", hue="set", binwidth=2, stat="density", common_norm=False,
             element="step", fill=False, palette=[BLUE, ORANGE])
plt.title("Delivery time in the train and test sets")
plt.show()
''')
R("s4_1")

md(r'''
## 4.2 Random Forest Regression (Task 4.2)

`RandomForestRegressor` builds many decision trees on random parts of the data and averages their predictions. We use 100 trees, `n_jobs=-1` to use all processor cores, and a fixed `random_state`.

**Note:** A Random Forest splits on thresholds, so it does not need scaled features (see Section 5.1).
''')
code(r'''
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
''')

md(r'''
## 4.3 Evaluate the Random Forest and Find the Important Features (Task 4.3)

We report MSE, RMSE and MAE, as in the brief. We also report two normalised views: R² (the share of the variation in delivery time that the model explains, which tells us how accurate the model is) and MAPE (the average error as a percentage of the true delivery time). MAE is the average miss in minutes, RMSE is also in minutes but punishes big misses more, and MSE is the quantity a neural network minimises. The training scores next to the test scores show whether the forest has memorised the training data.
''')
code(r'''
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def regression_report(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mape = float(np.mean(np.abs((np.asarray(y_true) - y_pred) / np.asarray(y_true))) * 100)
    return dict(MSE=mse, RMSE=mse ** 0.5, MAE=mean_absolute_error(y_true, y_pred), R2=r2_score(y_true, y_pred), MAPE_percent=mape)

rf_train_pred = rf.predict(X_train)
rf_test_pred = rf.predict(X_test)

rf_train_res = regression_report(y_train, rf_train_pred)
rf_test_res = regression_report(y_test, rf_test_pred)

pd.DataFrame({"Random Forest (train)": rf_train_res, "Random Forest (test)": rf_test_res}).round(3)
''')
md(r'''
Two charts show where the Random Forest is right and wrong. On the left, each dot is an order: a perfect model would put every dot on the dashed diagonal. On the right is the distribution of the errors (actual minus predicted).
''')
code(r'''
test_sample = pd.DataFrame({"actual": y_test.values, "predicted": rf_test_pred}).sample(8000, random_state=42)
test_sample["error"] = test_sample["actual"] - test_sample["predicted"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=test_sample, x="actual", y="predicted", s=8, alpha=0.3, color=ORANGE, edgecolor=None, ax=axes[0])
axes[0].plot([30, 110], [30, 110], color="#52514e", linestyle="--", linewidth=1)
axes[0].set_title("Random Forest: predicted vs actual (test)")
axes[0].set_xlabel("actual minutes")
axes[0].set_ylabel("predicted minutes")

sns.histplot(test_sample["error"], bins=60, kde=True, color=ORANGE, ax=axes[1])
axes[1].axvline(0, color="#52514e", linestyle="--", linewidth=1)
axes[1].set_title("Random Forest: errors (actual - predicted)")
axes[1].set_xlabel("minutes")
plt.tight_layout()
plt.show()
''')
R("s4_3")

md(r'''
Which features does the forest rely on most? `feature_importances_` shows how much each feature reduced the error across all the trees.
''')
code(r'''
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)

plt.figure(figsize=(9, 6))
sns.barplot(x=importance.values, y=importance.index, color=ORANGE)
plt.title("Random Forest: 15 most important features")
plt.xlabel("importance")
plt.show()
''')
R("s4_3b")

md(r'''
## 4.4 Other Classical Methods for Regression (Task 4.4, Brief Question 6)

> **Brief Question 6:** *What classical machine learning methods can we use other than random forest for regression?*


Besides Random Forest, these classical machine learning methods can predict a number:

| Method | Idea |
|:--|:--|
| Linear Regression (and Ridge, Lasso) | A weighted sum of the features, with optional penalties on large weights |
| Decision Tree Regressor | One tree of yes/no splits |
| Gradient Boosting (XGBoost, LightGBM, `GradientBoostingRegressor`) | Many small trees, each correcting the errors of the previous ones |
| K-Nearest Neighbours Regressor | The average of the most similar training orders |
| Support Vector Regression | A kernel method that fits a tube around the data |
''')

md(r'''
### A Quick Test of Some of Them

The brief only asks us to name the methods. Since it takes a minute, we also train three of them on the same training rows and score them on the same test rows, so that the Random Forest result can be put in perspective. These scores are not used to choose anything.
''')
code(r'''
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

classical = {"Linear Regression": LinearRegression(),
             "Decision Tree (at least 20 rows per leaf)": DecisionTreeRegressor(min_samples_leaf=20, random_state=42),
             "Gradient Boosting (HistGradientBoosting)": HistGradientBoostingRegressor(random_state=42)}

classical_rows = {}
for name, model in classical.items():
    model.fit(X_train, y_train)
    classical_rows[name] = regression_report(y_test, model.predict(X_test))
classical_rows["Random Forest (Section 4.3)"] = rf_test_res

pd.DataFrame(classical_rows).T[["RMSE", "MAE", "R2", "MAPE_percent"]].round(3)
''')
R("s4_4")
R("s4_sum")

# ============================================================ SECTION 5
md(r'''
<a id="sec5"></a>

# Section 5: Regression with Neural Networks (30 points)

### Objective and Tasks of this Section

**Stage 5 of 5 · 6 Tasks: Regression with Neural Networks**

**Objective:** You will scale the data, design and train a neural network regressor, check it for overfitting, and compare its performance directly against the random forest baseline.

| Task | What to do | Hint |
|:--|:--|:--|
| **Task 5.1** | Scale the feature data before feeding it into a neural network, and explain why scaling matters here but wasn't required for the random forest. | Think about how a neural network's gradient-based training reacts to features that are on very different numeric ranges, compared to how a tree-based model splits on raw values. |
| **Task 5.2** | Define a neural network architecture appropriate for this regression task, and justify your choice of activation functions. | Think about what activation the output layer needs for a continuous target, versus what activation is a strong general-purpose default for the hidden layers. |
| **Task 5.3** | Compile the model with an optimizer and loss function suited to this regression task, and briefly justify your choice of optimizer. | Think about the drawback of a fixed, single learning rate across all weights, and what an adaptive, per-parameter learning-rate optimizer offers instead. |
| **Task 5.4** | Train the network for a reasonable number of epochs, then plot training vs validation loss to check for overfitting. | Think about what it means for the validation loss to track closely with (or stay below) the training loss over the course of training. |
| **Task 5.5** | Evaluate the trained network on the test set using the same set of metrics used for the random forest, and directly compare the two models' results. | Think about why using the exact same metrics as your baseline is necessary in order to make a fair, direct comparison. |
| **Task 5.6** | Explain why neural networks tend to perform well as dataset size grows. | Think about the relationship between the number of learnable parameters in a network and the amount of data needed to estimate them reliably without overfitting. |

**Evaluation criteria of this section (30 points):**

| Criterion | Done in |
|:--|:--|
| Data scaling | 5.1 |
| Defining nn architecture | 5.2 |
| Trying different combinations and hyperparameters | 5.4 |
| Model training | 5.4 |
| Comparing results with random forest | 5.5 |

Each task below is marked in its heading, for example `(Task 5.1)`.

---


Now we build a neural network in PyTorch, following the structure used in the course: scale the data, build the model, choose the loss and the optimizer, write our own training loop, and try different configurations one at a time. At the end we compare the network with the Random Forest.

**Flow of this section:**

```
Scale --> Architecture --> Loss & optimizer --> Train & plot losses --> Try configurations --> Final model --> Compare with Random Forest
```
''')

md(r'''
## 5.1 Scaling the Data (Task 5.1, Brief Question 7)

> **Brief Question 7:** *Why is scaling required for neural networks?*


Neural networks learn with gradient descent, so the size of the numbers matters. `subtotal` runs into thousands while `hour` is between 0 and 23. Unscaled inputs give a stretched, lopsided loss surface, and gradient descent zig-zags across it. A forest does not care, because it only compares each feature with a threshold.

**Standardisation** (`z = (x - mean) / std`) puts every column on the same footing. We fit the scaler on the **training set only** and just transform the other sets, otherwise information from data the model must never see would leak into training.

We also scale the **target**. A network starts with outputs near 0 while delivery times are around 46, so scaling `y` makes the first steps of training sensible. Predictions are converted back to minutes with `inverse_transform` before we calculate any metric.

To plot the training and validation loss we need a validation set. We set aside 12.5% of the training rows for it, so the network trains on 70% of the data, validates on 10%, and the test set (20%) stays untouched.
''')
code(r'''
from sklearn.preprocessing import StandardScaler

X_train_nn, X_val, y_train_nn, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_nn)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

y_scaler = StandardScaler()
y_train_s = y_scaler.fit_transform(y_train_nn.values.reshape(-1, 1))
y_val_s = y_scaler.transform(y_val.values.reshape(-1, 1))

y_std = y_scaler.scale_[0]
print("Train / validation / test rows:", len(X_train_nn), len(X_val), len(X_test))
print("Target mean and std used for scaling:", round(y_scaler.mean_[0], 2), round(y_std, 2))
''')
md(r'''
The box plots show the effect: on the left the raw columns live on completely different scales, on the right they are comparable.
''')
code(r'''
show_cols = ["total_items", "subtotal", "max_item_price", "total_onshift_dashers", "total_outstanding_orders", "hour"]
scaled_df = pd.DataFrame(X_train_s, columns=X.columns)

fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
sns.boxplot(data=X_train_nn[show_cols], orient="h", color=BLUE, fliersize=1, ax=axes[0])
axes[0].set_title("Before scaling")
sns.boxplot(data=scaled_df[show_cols], orient="h", color=BLUE, fliersize=1, ax=axes[1])
axes[1].set_title("After scaling (mean 0, std 1)")
axes[1].set_yticklabels([])
plt.tight_layout()
plt.show()
''')
md(r'''
PyTorch models work with tensors. Features become `float32`. The target is a column of shape `(N, 1)`, which is the shape of the model output. A `DataLoader` hands the model small shuffled batches. `seed_everything` fixes NumPy's and PyTorch's random numbers so that runs can be compared fairly.
''')
code(r'''
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything()

X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

y_train_t = torch.tensor(y_train_s, dtype=torch.float32)
y_val_t = torch.tensor(y_val_s, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)

xb, yb = next(iter(train_loader))
print("PyTorch", torch.__version__)
print("One batch -> features:", xb.shape, " targets:", yb.shape)
''')
R("s5_1")

md(r'''
## 5.2 Defining the Network Architecture and the Activation Functions (Task 5.2, Brief Question 9)

> **Brief Question 9:** *Which activation function did you use and why?*


The baseline network has two hidden layers:

| Layer | Description |
|:--|:--|
| Input | One neuron for each of the 24 input columns |
| Hidden 1 | 128 neurons with ReLU |
| Hidden 2 | 64 neurons with ReLU |
| Output | **1 neuron with no activation** |

**Which activation functions, and why?**

- **Hidden layers: ReLU**, `max(0, z)`. It is cheap, it keeps a gradient of exactly 1 for positive inputs so gradients do not vanish as they do with sigmoid and tanh, and it is the default in most modern networks. Its weakness is that a unit stuck at negative values stops learning (dying ReLU), which Leaky ReLU fixes. We test the alternatives in Section 5.4.
- **Output layer: linear (no activation).** The last neuron decides what the network predicts. A linear last neuron gives regression, and sigmoid or softmax would give classification. A sigmoid would squeeze the answer into 0 to 1, and a ReLU would cut off negative values, which the scaled target can take.

The chart compares the four activation functions we try, and their derivatives (the gradient signal that flows back during training).
''')
code(r'''
z = torch.linspace(-4, 4, 400, requires_grad=True)
activations = {"ReLU": torch.relu, "Leaky ReLU": lambda t: nn.functional.leaky_relu(t, 0.01), "Tanh": torch.tanh, "Sigmoid": torch.sigmoid}

curves = []
for name, f in activations.items():
    out = f(z)
    grad, = torch.autograd.grad(out.sum(), z)
    curves.append(pd.DataFrame({"z": z.detach().numpy(), "value": out.detach().numpy(), "derivative": grad.numpy(), "activation": name}))
curves = pd.concat(curves)

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
sns.lineplot(data=curves, x="z", y="value", hue="activation", palette=[BLUE, ORANGE, AQUA, YELLOW], ax=axes[0])
axes[0].set_title("Activation functions")
sns.lineplot(data=curves, x="z", y="derivative", hue="activation", palette=[BLUE, ORANGE, AQUA, YELLOW], legend=False, ax=axes[1])
axes[1].set_title("Their derivatives")
plt.tight_layout()
plt.show()
''')
md(r'''
`make_net` builds the network from a list of hidden sizes and an activation. As in the course, weights start with Xavier initialisation and biases at zero, and the last layer has **no** activation.
''')
code(r'''
def make_net(hidden=(128, 64), act=nn.ReLU, dropout=0.0, bn=False):
    layers, d = [], X_train_t.shape[1]
    for w in hidden:
        layers.append(nn.Linear(d, w))
        if bn:
            layers.append(nn.BatchNorm1d(w))
        layers.append(act())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = w
    model = nn.Sequential(*layers, nn.Linear(d, 1))        # last layer: no activation
    for layer in model:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return model

torch.manual_seed(0)
model = make_net()
print(model)
print("Trainable parameters:", sum(p.numel() for p in model.parameters()))
''')
R("s5_2")

md(r'''
## 5.3 Loss Function and Optimizer (Task 5.3, Brief Question 8)

> **Brief Question 8:** *Briefly explain your choice of optimizer.*


- **Loss: `nn.MSELoss()`**, the mean squared error. It is the standard loss for a continuous target, it is smooth, and it punishes big misses more than small ones. We report results as RMSE in minutes, which is easier to read.
- **Optimizer: Adam**, with learning rate 0.001 to start. Plain gradient descent uses one fixed learning rate for every weight and zig-zags on a lopsided loss surface. Momentum smooths the direction, RMSprop shrinks the step where gradients have been large, and **Adam combines both**, with a correction for its first steps. It is a strong default that needs little tuning. Adam still needs a learning rate, so we compare learning rates in Section 5.4.
''')
code(r'''
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# one forward and backward pass on one batch, before any training
loss = criterion(model(xb), yb)
loss.backward()
print("Loss before training (scaled units):", round(loss.item(), 3))
print("Gradient shapes -> first layer:", tuple(model[0].weight.grad.shape), " last layer:", tuple(model[-1].weight.grad.shape))
''')
R("s5_3")

md(r'''
## 5.4 Training the Network and Plotting the Losses (Task 5.4)

We write the training loop ourselves, as in the course. For every batch: `zero_grad`, forward pass, loss, `backward`, `step`. At the end of every epoch we switch to `model.eval()`, measure the error on the training and validation sets with `torch.no_grad()`, and record it in `h`. Errors are recorded as RMSE in minutes. After training, the function loads the weights of the epoch with the lowest validation error.
''')
code(r'''
def fit(model, opt, epochs=40, bs=256):
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=bs, shuffle=True)
    h = {"train": [], "val": []}
    best, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); criterion(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            h["train"].append(criterion(model(X_train_t), y_train_t).item() ** 0.5 * y_std)
            v = criterion(model(X_val_t), y_val_t).item()
            h["val"].append(v ** 0.5 * y_std)
        if v < best:
            best, best_state = v, {k: t.clone() for k, t in model.state_dict().items()}
    if best_state is not None:                       # stays None only if the loss became NaN (diverged)
        model.load_state_dict(best_state)
    h["best_epoch"] = int(np.nanargmin(h["val"])) if np.isfinite(h["val"]).any() else 0
    return h

def show(h, title=""):
    curves = pd.DataFrame({"epoch": range(1, len(h["train"]) + 1), "training": h["train"], "validation": h["val"]})
    curves = curves.melt(id_vars="epoch", var_name="set", value_name="RMSE (minutes)")
    plt.figure(figsize=(9, 4.5))
    sns.lineplot(data=curves, x="epoch", y="RMSE (minutes)", hue="set", palette=[BLUE, ORANGE])
    plt.axvline(h["best_epoch"] + 1, color="#52514e", linestyle=":", linewidth=1.5, label="best epoch")
    plt.title(title)
    plt.legend()
    plt.show()
''')
md(r'''
Let's train the baseline for 60 epochs and plot the training and validation loss.
''')
code(r'''
torch.manual_seed(0)
base = make_net()
h_base = fit(base, torch.optim.Adam(base.parameters(), lr=1e-3), epochs=60)
show(h_base, "Baseline network: training and validation loss")
''')
code(r'''
print("Last epoch -> train RMSE", round(h_base["train"][-1], 3), "| validation RMSE", round(h_base["val"][-1], 3), "minutes")
print("Best epoch ->", h_base["best_epoch"] + 1, "with validation RMSE", round(h_base["val"][h_base["best_epoch"]], 3), "minutes")
''')
R("s5_4")

md(r'''
### Trying Different Configurations

The brief also asks us to try different combinations and hyperparameters. This part belongs to Task 5.4: the network is trained several times, and we keep the best configuration.

We now follow the debugging method from the Practical Aspects class: **change one thing at a time** and keep a scoreboard. Every run starts from the same seed, so differences come from the change and not from luck. Each run trains for 40 epochs and keeps the weights of the best epoch. `run` records the training error, the last validation error, the best validation error and the best epoch.
''')
code(r'''
results = {}
histories = {}
configs = {}

def make_optimizer(params, optimizer, lr, wd=0.0):
    if optimizer == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if optimizer == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=wd)
    if optimizer == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd)
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)       # "sgd_momentum"

def run(name, group, hidden=(128, 64), act=nn.ReLU, optimizer="adam", lr=1e-3, bs=256, epochs=40, dropout=0.0, bn=False, wd=0.0):
    torch.manual_seed(0)
    m = make_net(hidden, act, dropout, bn)
    opt = make_optimizer(m.parameters(), optimizer, lr, wd)
    h = fit(m, opt, epochs=epochs, bs=bs)
    histories[name] = h
    configs[name] = dict(hidden=hidden, act=act, optimizer=optimizer, lr=lr, bs=bs, dropout=dropout, bn=bn, wd=wd)
    results[name] = [group, h["train"][-1], h["val"][-1], min(h["val"]), h["best_epoch"] + 1]

def scoreboard(group):
    table = pd.DataFrame(results, index=["group", "train_RMSE", "val_RMSE", "best_val_RMSE", "best_epoch"]).T
    table = table[table["group"] == group].drop(columns="group")
    return table.astype(float).round(3)

def plot_group(group, title):
    table = scoreboard(group).sort_values("best_val_RMSE")
    plt.figure(figsize=(9, 0.6 * len(table) + 1.5))
    ax = sns.barplot(x=table["best_val_RMSE"], y=table.index, color=BLUE)
    ax.bar_label(ax.containers[0], fmt="%.3f", fontsize=9)
    plt.title(title)
    plt.xlabel("best validation RMSE (minutes)")
    plt.ylabel("")
    plt.show()

def plot_curves(names, title, palette):
    long = pd.concat([pd.DataFrame({"epoch": range(1, len(histories[n]["val"]) + 1), "validation RMSE": histories[n]["val"], "run": n}) for n in names])
    plt.figure(figsize=(9, 4.5))
    sns.lineplot(data=long, x="epoch", y="validation RMSE", hue="run", palette=palette, linewidth=2)
    plt.title(title)
    plt.show()
''')

md(r'''
#### Experiment 1: Size of the Network

Wider and deeper networks can learn more, but also need more data and time.
''')
code(r'''
for hidden in [(32,), (64, 32), (128, 64), (256, 128, 64), (128, 64, 32, 16)]:
    run(f"size {hidden}", "size", hidden=hidden)

scoreboard("size")
''')
code(r'''
plot_group("size", "Experiment 1: size of the network")
''')
R("s5_5a")

md(r'''
#### Experiment 2: Activation Function

Same network, only the hidden activation changes.
''')
code(r'''
for name, act in [("ReLU", nn.ReLU), ("Leaky ReLU", nn.LeakyReLU), ("Tanh", nn.Tanh), ("Sigmoid", nn.Sigmoid)]:
    run(f"activation {name}", "activation", act=act)

scoreboard("activation")
''')
code(r'''
plot_group("activation", "Experiment 2: activation function")
''')
R("s5_5b")

md(r'''
#### Experiment 3: The Optimiser Race

The same network with four optimisers, as in the SwiftLend case: plain SGD (learning rate 0.01), SGD with momentum (learning rate 0.001), RMSprop (0.001) and Adam (0.001). Momentum makes the effective step larger. We first tried it with a learning rate of 0.01, and on this target the training diverged (the loss became NaN), so we use 0.001.
''')
code(r'''
run("optimiser SGD", "optimiser", optimizer="sgd", lr=0.01)
run("optimiser SGD + Momentum", "optimiser", optimizer="sgd_momentum", lr=1e-3)
run("optimiser RMSprop", "optimiser", optimizer="rmsprop", lr=1e-3)
run("optimiser Adam", "optimiser", optimizer="adam", lr=1e-3)

scoreboard("optimiser")
''')
code(r'''
plot_curves(["optimiser SGD", "optimiser SGD + Momentum", "optimiser RMSprop", "optimiser Adam"],
            "Experiment 3: validation RMSE for four optimisers", [GRAY, YELLOW, AQUA, BLUE])
''')
R("s5_5c")

md(r'''
#### Experiment 4: Learning Rate

The learning rate is the first hyperparameter to tune. As in the Practical Aspects class we try values that differ by large factors, a log-scale spread, with Adam.
''')
code(r'''
for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]:
    run(f"learning rate {lr:g}", "learning rate", lr=lr)

scoreboard("learning rate")
''')
code(r'''
plot_curves([f"learning rate {lr:g}" for lr in [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]],
            "Experiment 4: validation RMSE for five learning rates", [GRAY, YELLOW, AQUA, BLUE, ORANGE])
''')
R("s5_5d")

md(r'''
#### Experiment 5: Batch Size

A small batch updates the weights more often but with noisier gradients. A large batch gives smoother gradients and a faster epoch, but fewer updates.
''')
code(r'''
for bs in [64, 128, 256, 512, 1024]:
    run(f"batch size {bs}", "batch size", bs=bs)

scoreboard("batch size")
''')
code(r'''
plot_group("batch size", "Experiment 5: batch size")
''')
R("s5_5e")

md(r'''
#### Experiment 6: Regularisation

The baseline network showed a small gap between its training and validation error. Three standard remedies from the Hyperparameter Tuning class can shrink such a gap: **weight decay** (an L2 penalty that keeps the weights small, set with Adam's `weight_decay`), **dropout** (randomly switching off a share of the neurons while training, and off again when predicting) and **batch normalisation** (standardising the values inside the network). We apply them one at a time to the baseline network (128, 64).
''')
code(r'''
run("regularisation weight decay 1e-4", "regularisation", wd=1e-4)
run("regularisation weight decay 1e-3", "regularisation", wd=1e-3)
run("regularisation dropout 0.1", "regularisation", dropout=0.1)
run("regularisation dropout 0.3", "regularisation", dropout=0.3)
run("regularisation batch norm", "regularisation", bn=True)

scoreboard("regularisation")
''')
code(r'''
plot_group("regularisation", "Experiment 6: regularisation")
''')
R("s5_5g")

md(r'''
#### The Full Scoreboard

Every run from the six experiments in one chart, coloured by experiment.
''')
code(r'''
board = pd.DataFrame(results, index=["group", "train_RMSE", "val_RMSE", "best_val_RMSE", "best_epoch"]).T
board[["train_RMSE", "val_RMSE", "best_val_RMSE", "best_epoch"]] = board[["train_RMSE", "val_RMSE", "best_val_RMSE", "best_epoch"]].astype(float)
board = board.sort_values("best_val_RMSE")

plt.figure(figsize=(10, 10.5))
sns.barplot(data=board, x="best_val_RMSE", y=board.index, hue="group", dodge=False,
            palette=[BLUE, ORANGE, AQUA, YELLOW, GRAY, "#8e5ea2"])
plt.title("All experiments: best validation RMSE (lower is better)")
plt.xlabel("minutes")
plt.ylabel("")
plt.legend(title="experiment", loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout()
plt.show()

board.head(5).round(3)
''')
R("s5_5f")

md(r'''
### The Final Model

We take the configuration with the lowest validation error on the scoreboard and train it for 60 epochs. The weights of the best epoch are kept, and the loss curves are plotted. The test set is not used for any of these choices.
''')
code(r'''
best_name = board.index[0]
best_cfg = configs[best_name]
print("Best configuration:", best_name)
print(best_cfg)
''')
code(r'''
torch.manual_seed(0)
final = make_net(best_cfg["hidden"], best_cfg["act"], best_cfg["dropout"], best_cfg["bn"])
final_opt = make_optimizer(final.parameters(), best_cfg["optimizer"], best_cfg["lr"], best_cfg["wd"])

h_final = fit(final, final_opt, epochs=60, bs=best_cfg["bs"])
print("Best epoch:", h_final["best_epoch"] + 1, "| validation RMSE:", round(min(h_final["val"]), 3), "minutes")
show(h_final, "Final model: training and validation loss")
''')
R("s5_6")

md(r'''
## 5.5 Test Evaluation and Comparison with the Random Forest (Task 5.5)

Now, once, we use the test set. The network predicts in scaled units, so `y_scaler.inverse_transform` converts the predictions back to minutes. Both models are scored with exactly the same metrics on exactly the same test rows.
''')
code(r'''
final.eval()
with torch.no_grad():
    nn_test_pred = y_scaler.inverse_transform(final(X_test_t).numpy()).ravel()

nn_test_res = regression_report(y_test, nn_test_pred)

comparison = pd.DataFrame({"Random Forest": rf_test_res, "Neural Network": nn_test_res})
comparison.round(3)
''')
md(r'''
The chart compares the two models on the three error metrics of the brief.
''')
code(r'''
bars = comparison.loc[["MSE", "RMSE", "MAE"]].reset_index().melt(id_vars="index", var_name="model", value_name="value")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, metric in zip(axes, ["MSE", "RMSE", "MAE"]):
    sns.barplot(data=bars[bars["index"] == metric], x="model", y="value", hue="model",
                palette={"Random Forest": ORANGE, "Neural Network": BLUE}, legend=False, ax=ax)
    ax.bar_label(ax.containers[0], fmt="%.2f", fontsize=10)
    ax.bar_label(ax.containers[1], fmt="%.2f", fontsize=10)
    ax.set_title(f"{metric} (lower is better)")
    ax.set_xlabel("")
    ax.set_ylabel("")
plt.tight_layout()
plt.show()
''')
md(r'''
Predicted against actual for each model, with the dashed diagonal as the perfect prediction.
''')
code(r'''
pred_frame = pd.DataFrame({"actual": y_test.values, "Random Forest": rf_test_pred, "Neural Network": nn_test_pred})
sample = pred_frame.sample(8000, random_state=42)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharex=True, sharey=True)
for ax, name, color in zip(axes, ["Random Forest", "Neural Network"], [ORANGE, BLUE]):
    sns.scatterplot(data=sample, x="actual", y=name, s=8, alpha=0.3, color=color, edgecolor=None, ax=ax)
    ax.plot([30, 110], [30, 110], color="#52514e", linestyle="--", linewidth=1)
    ax.set_title(f"{name}: predicted vs actual (test)")
    ax.set_xlabel("actual minutes")
    ax.set_ylabel("predicted minutes")
plt.tight_layout()
plt.show()
''')
md(r'''
The distribution of the errors (actual minus predicted) for both models.
''')
code(r'''
errors = pd.DataFrame({"Random Forest": y_test.values - rf_test_pred, "Neural Network": y_test.values - nn_test_pred})
errors_long = errors.melt(var_name="model", value_name="error (minutes)")

plt.figure(figsize=(9, 4.5))
sns.kdeplot(data=errors_long, x="error (minutes)", hue="model", palette={"Random Forest": ORANGE, "Neural Network": BLUE},
            fill=True, alpha=0.25, common_norm=False, linewidth=2)
plt.axvline(0, color="#52514e", linestyle="--", linewidth=1)
plt.title("Test set: distribution of errors")
plt.show()
''')
R("s5_7")

md(r'''
### Is the Network Really Better? More Evidence

The table above shows that the network scores a little better. One test split cannot tell us whether that gap is real or luck, so we look at the question from six more sides, and we report each result as it comes out.
''')

md(r'''
**Evidence 1: what the customer experiences.** An average error hides how often the promise is good enough. We count the share of test orders predicted within 3, 5 and 10 minutes of the real time, and how often the order arrives **more than 5 minutes later than promised** (the prediction was too low).
''')
code(r'''
y_true = y_test.values
preds = {"Random Forest": rf_test_pred, "Neural Network": nn_test_pred}

def customer_view(pred):
    err = y_true - pred
    return {"within 3 min (%)": np.mean(np.abs(err) <= 3) * 100,
            "within 5 min (%)": np.mean(np.abs(err) <= 5) * 100,
            "within 10 min (%)": np.mean(np.abs(err) <= 10) * 100,
            "arrives > 5 min later than promised (%)": np.mean(err > 5) * 100}

pd.DataFrame({name: customer_view(p) for name, p in preds.items()}).round(2)
''')
R("s5_7a")

md(r'''
**Evidence 2: is the gap bigger than the luck of the test sample?** We draw 1,000 new test sets by sampling the test rows with replacement (a *paired bootstrap*: both models are scored on exactly the same rows each time) and record how much lower the network's error is. If the 95% interval stays above zero, the network's lead is not a lucky draw of test rows.
''')
code(r'''
rng = np.random.default_rng(42)
se_rf, se_nn = (y_true - rf_test_pred) ** 2, (y_true - nn_test_pred) ** 2
ae_rf, ae_nn = np.abs(y_true - rf_test_pred), np.abs(y_true - nn_test_pred)

rmse_gain, mae_gain = [], []
for _ in range(1000):
    i = rng.integers(0, len(y_true), len(y_true))
    rmse_gain.append(np.sqrt(se_rf[i].mean()) - np.sqrt(se_nn[i].mean()))
    mae_gain.append(ae_rf[i].mean() - ae_nn[i].mean())

boot = pd.DataFrame({"RMSE": rmse_gain, "MAE": mae_gain})
summary_boot = pd.DataFrame({"mean gain (min)": boot.mean(),
                             "95% lower": boot.quantile(0.025),
                             "95% upper": boot.quantile(0.975),
                             "share of draws where the network wins (%)": (boot > 0).mean() * 100})
summary_boot.round(3)
''')
code(r'''
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col in zip(axes, ["RMSE", "MAE"]):
    sns.histplot(boot[col], bins=40, color=BLUE, ax=ax)
    ax.axvline(0, color="#52514e", linestyle="--", linewidth=1.2)
    ax.set_title(f"Random Forest {col} minus Neural Network {col}\n(above 0 means the network is better)")
    ax.set_xlabel("minutes")
plt.tight_layout()
plt.show()
''')
R("s5_7b")

md(r'''
**Evidence 3: is it just a lucky seed?** A neural network starts from random weights and a forest from random trees, so each result depends on the seed. We train the final network with 5 different seeds and the forest with 4 different seeds, keep the same split, and look at the spread of the test RMSE.
''')
code(r'''
def final_net_rmse(seed):
    torch.manual_seed(seed)
    m = make_net(best_cfg["hidden"], best_cfg["act"], best_cfg["dropout"], best_cfg["bn"])
    fit(m, make_optimizer(m.parameters(), best_cfg["optimizer"], best_cfg["lr"], best_cfg["wd"]), epochs=60, bs=best_cfg["bs"])
    m.eval()
    with torch.no_grad():
        p = y_scaler.inverse_transform(m(X_test_t).numpy()).ravel()
    return regression_report(y_test, p)["RMSE"]

nn_rmse = [nn_test_res["RMSE"]] + [final_net_rmse(sd) for sd in [1, 2, 3, 4]]
rf_rmse = [rf_test_res["RMSE"]]
for sd in [1, 2, 3]:
    rf_sd = RandomForestRegressor(n_estimators=100, random_state=sd, n_jobs=-1).fit(X_train, y_train)
    rf_rmse.append(regression_report(y_test, rf_sd.predict(X_test))["RMSE"])

seed_table = pd.DataFrame({
    "runs": [len(rf_rmse), len(nn_rmse)],
    "mean test RMSE": [np.mean(rf_rmse), np.mean(nn_rmse)],
    "std between runs": [np.std(rf_rmse, ddof=1), np.std(nn_rmse, ddof=1)],
    "best run": [min(rf_rmse), min(nn_rmse)],
    "worst run": [max(rf_rmse), max(nn_rmse)],
}, index=["Random Forest", "Neural Network"])
seed_table.round(3)
''')
R("s5_7c")

md(r'''
**Evidence 4: how much model is needed, and how fast is it?** A model that reaches the same accuracy with far fewer numbers is cheaper to store and faster to run when thousands of customers ask for an estimate at once. We compare the size, the time to predict the test orders, and the gap between training and test error (a large gap means the model memorised its training data).
''')
code(r'''
import time

def predict_time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best

def nn_predict():
    with torch.no_grad():
        return final(X_test_t)

with torch.no_grad():
    nn_train_pred = y_scaler.inverse_transform(final(X_train_t).numpy()).ravel()
nn_train_rmse = regression_report(y_train_nn, nn_train_pred)["RMSE"]

rf_nodes = sum(tree.tree_.node_count for tree in rf.estimators_)
nn_params = sum(p.numel() for p in final.parameters())

pd.DataFrame({
    "Random Forest": {"stored numbers": rf_nodes, "predict all test orders (seconds)": predict_time(lambda: rf.predict(X_test)),
                      "train RMSE": rf_train_res["RMSE"], "test RMSE": rf_test_res["RMSE"], "gap (test - train)": rf_test_res["RMSE"] - rf_train_res["RMSE"]},
    "Neural Network": {"stored numbers": nn_params, "predict all test orders (seconds)": predict_time(nn_predict),
                       "train RMSE": nn_train_rmse, "test RMSE": nn_test_res["RMSE"], "gap (test - train)": nn_test_res["RMSE"] - nn_train_rmse},
}).round(3)
''')
R("s5_7d")

md(r'''
**Evidence 5: does the lead hold when the market is quiet and when it is busy?** A model that wins on average but fails when the market is busy would be a poor choice. We split the test orders into four equal groups by the number of orders waiting (`total_outstanding_orders`) and compare the RMSE of both models in each group.
''')
code(r'''
busy = pd.qcut(X_test["total_outstanding_orders"], 4, labels=["quietest 25%", "quiet", "busy", "busiest 25%"])

rows = {}
for label in busy.cat.categories:
    mask = (busy == label).values
    rows[label] = {"orders": int(mask.sum()),
                   "Random Forest RMSE": np.sqrt(np.mean((y_true[mask] - rf_test_pred[mask]) ** 2)),
                   "Neural Network RMSE": np.sqrt(np.mean((y_true[mask] - nn_test_pred[mask]) ** 2))}
by_busy = pd.DataFrame(rows).T
by_busy["network better by (min)"] = by_busy["Random Forest RMSE"] - by_busy["Neural Network RMSE"]
by_busy.round(3)
''')
code(r'''
plot_busy = by_busy[["Random Forest RMSE", "Neural Network RMSE"]].reset_index().melt(id_vars="index", var_name="model", value_name="RMSE (minutes)")
plt.figure(figsize=(9, 4.5))
sns.barplot(data=plot_busy, x="index", y="RMSE (minutes)", hue="model", palette=[ORANGE, BLUE])
plt.ylim(0, plot_busy["RMSE (minutes)"].max() * 1.3)
plt.title("Test RMSE by how many orders were waiting")
plt.xlabel("")
plt.show()
''')
R("s5_7e")

md(r'''
**Evidence 6: is the forest being treated fairly?** Our Random Forest used the default settings and overfits (its training error is far below its test error). A fair comparison should give it a chance to do better. We try forests with at least 1, 5, 10 or 20 rows per leaf and with all or half of the features at each split, choose the best by the **validation** error (the same validation rows the network used), and only then score that forest on the test set once.
''')
code(r'''
tune_rows = []
for leaf in [1, 5, 10, 20]:
    for feats in [1.0, 0.5]:
        m = RandomForestRegressor(n_estimators=200, min_samples_leaf=leaf, max_features=feats, random_state=42, n_jobs=-1).fit(X_train_nn, y_train_nn)
        tune_rows.append({"min_samples_leaf": leaf, "max_features": feats,
                          "validation RMSE": regression_report(y_val, m.predict(X_val))["RMSE"], "model": m})
tune = pd.DataFrame(tune_rows)
tuned_rf = tune.loc[tune["validation RMSE"].idxmin(), "model"]
tune.drop(columns="model").round(3)
''')
code(r'''
tuned_res = regression_report(y_test, tuned_rf.predict(X_test))
print("Chosen forest:", tuned_rf.get_params()["min_samples_leaf"], "rows per leaf, max_features =", tuned_rf.get_params()["max_features"])

pd.DataFrame({"Random Forest (default)": rf_test_res, "Random Forest (tuned)": tuned_res, "Neural Network": nn_test_res}).round(3)
''')
R("s5_7f")

R("s5_8")

md(r'''
### Saving the Model and Using It for a New Order

A model is only useful if it can be reused. We save the network's weights, the two scalers and the column order in the `models` folder, load the weights into a fresh network to check that it gives identical predictions, and then ask it for the delivery time of one new order.
''')
code(r'''
import joblib, os

os.makedirs("models", exist_ok=True)
torch.save(final.state_dict(), "models/porter_delivery_nn.pt")
joblib.dump({"scaler": scaler, "y_scaler": y_scaler, "columns": list(X.columns), "hidden": best_cfg["hidden"]},
            "models/porter_delivery_preprocessing.joblib")

reloaded = make_net(best_cfg["hidden"], best_cfg["act"], best_cfg["dropout"], best_cfg["bn"])
reloaded.load_state_dict(torch.load("models/porter_delivery_nn.pt"))
reloaded.eval()
with torch.no_grad():
    same = torch.allclose(reloaded(X_test_t), final(X_test_t))
print("Saved files:", sorted(f for f in os.listdir("models") if f.startswith("porter_delivery")))
print("Reloaded network gives identical predictions:", same)
''')
code(r'''
new_order = pd.DataFrame([{"market_id": 2, "order_protocol": 3, "total_items": 3, "subtotal": 2400, "num_distinct_items": 3,
                           "min_item_price": 400, "max_item_price": 1100, "total_onshift_dashers": 40, "total_busy_dashers": 35,
                           "total_outstanding_orders": 50, "hour": 20, "day_of_week": 5, "store_category_code": 4}])

new_X = pd.get_dummies(new_order, columns=["market_id", "order_protocol"], dtype=float).reindex(columns=X.columns, fill_value=0.0)
with torch.no_grad():
    estimate = y_scaler.inverse_transform(reloaded(torch.tensor(scaler.transform(new_X), dtype=torch.float32)).numpy())[0, 0]

print(f"Estimated delivery time for a new order (Saturday, hour 20, 3 items, subtotal 2400): {estimate:.0f} minutes")
''')
R("s5_9")

md(r'''
## 5.6 Why Neural Networks Perform Well on Large Datasets (Task 5.6, Brief Question 10)

> **Brief Question 10:** *Why does a neural network perform well on a large dataset?*


A neural network has many parameters (our baseline network has 11,521, and the final one has 833), and every parameter has to be estimated from data. With little data there are too few examples to pin them all down, so the network overfits or does not train well. With a lot of data, the same flexibility becomes an advantage: the network can learn complicated, smooth relationships and build its own features, and it keeps improving as more examples arrive. Many classical models flatten out earlier, because their capacity is limited by the features we hand them.

**What our own results show.** We have about 138,000 training rows, far more rows than parameters (even the largest network we tried, (256, 128, 64), has only 47,617), so the networks did not overfit badly: the baseline's training error was only about 0.2 minutes below its validation error. But the size experiment in Section 5.4 also shows that more capacity did **not** help: the smallest network (one layer of 32 units) did as well as the larger ones, and the regularisation experiment barely moved the validation error. For these 24 input columns the limit is the information in the features, not the size of the model. We did not train on growing slices of the data, so our experiment does not show that the network keeps improving with more rows. That statement is the general rule, not something we measured here.
''')

R("s5_sum")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, "/Users/vinayakkumar.udikeri/Documents/Porter_NN_Regression/build/pass3.ipynb")
print("written", len(cells), "cells")
