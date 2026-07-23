# ML Learning Journey

This repo is my day-by-day log as I work through an ML course. Each notebook is whatever I actually coded that day, and this README gets a new section added as I go — it's meant to read like a log, not a polished write-up after the fact.

---

## Day 1 — EDA Project: Getting Started with Exploratory Data Analysis

**File:** `Day1_EDA_project.ipynb`
**Dataset:** Seaborn's built-in `tips` dataset

First real EDA day. Started with the basics — shape, `info()`, checking for nulls and duplicates — then dropped duplicate rows. Looped through the columns to see what unique values each one held, which helped me actually understand what I was working with instead of just guessing.

Engineered my first feature: `tip_percentage = (tip / total_bill) * 100`. Then moved into visualizing — histograms and KDE plots for `total_bill` and `tip`, a boxplot to catch outliers, and countplots for `sex`, `smoker`, `day`, and `size`. Wrapped up with a scatterplot of `total_bill` vs `tip` and a first attempt at a regression plot.

**Takeaway:** the EDA workflow finally clicked — understand the data, clean it, engineer something useful, then visualize, first one variable at a time and then two together.

## Day 2 — Same Dataset, Again (On Purpose)

**File:** `Day2_EDA_project.ipynb`
**Dataset:** Seaborn's built-in `tips` dataset

Deliberately repeated the whole Day 1 process on the same data — shape, info, nulls, duplicates, unique values, feature creation, the same univariate plots, same `total_bill` vs `tip` scatterplot. Nothing new conceptually, just trying to make the workflow automatic before throwing messier data at it.

**Takeaway:** repetition day. Wanted the steps to become muscle memory before Day 3.

## Day 3 — FIFA 19 EDA

**File:** `Fifa19_EDA/` folder
**Dataset:** FIFA 19 player dataset (CSV)

First time with a real, messy dataset. Had to load it with `encoding="Latin-1"` since it wasn't plain UTF-8 — a small thing but a good reminder that real data doesn't always cooperate. Checked missing values, dtypes, and ran `describe()` for a summary.

From there: an age distribution histogram, a preferred-foot countplot, and an age-vs-overall-rating scatterplot, then a full correlation heatmap across the numeric columns. Went a bit deeper too — found the top 10 highest-rated players, grouped by club to get average rating per club, checked for wage outliers with a boxplot, and looked at position and nationality distributions with bar charts. Ended with a short written insights section pulling it all together.

**Takeaway:** first time doing groupby analysis and correlation heatmaps on something real, and actually writing up findings instead of just leaving plots to speak for themselves.

## Day 4 — Netflix EDA

**File:** `Netflix_EDA/` folder
**Dataset:** Netflix titles dataset (CSV)

Started by filtering rows on conditions to look up specific titles, then checked nulls, duplicates, and unique values. Made a copy of the dataframe before touching anything — learned the hard way (mentally, at least) why you don't clean the original.

Dropped nulls, stripped whitespace from `date_added`, then converted it to datetime using `pd.to_datetime(..., format="mixed")` and pulled out day, month, and year as their own columns. Finished with a countplot of content type, a countplot of rating, a look at how rating relates to type, and a pie chart of the top ratings by percentage.

**Takeaway:** datetime parsing and extraction, and the habit of working on a copy instead of the original dataframe.

## Day 5 — Feature Engineering: Feature Types

**File:** `Day5_FeatureEngineering.ipynb`

A more conceptual day — went through the different kinds of features and how each one needs to be handled differently:

- Cleaning inconsistent categorical text (`'M'`, `'male'`, `'Fem'`, `'female'` all becoming one consistent label)
- Numerical features — just identifying and inspecting them
- Categorical features — converting to `category` dtype
- Ordinal features — manually mapping ordered categories to numbers (Junior=1, Mid=2, Senior=3)
- Binary/boolean columns
- Date/time features — pulling out day, month, year, weekday name
- Text features — engineering word count and character count
- A first look at image data with `skimage`, just inspecting shape and dimensions

**Takeaway:** not every column should be treated the same way — the right prep depends on what type of feature it actually is.

## Day 6 — Feature Engineering: Missing Values & Encoding

**File:** `Day6_FeatureEngineering.ipynb`
**Dataset:** Seaborn's built-in `titanic` dataset

Learned the three flavors of missing data — MCAR, MAR, and MNAR — and what each implies about how you're allowed to handle it. Covered dropping (`dropna()` on rows and columns, and when that's actually fine) versus imputing:

- Mean imputation for roughly normal distributions
- Median imputation when outliers are around
- Mode imputation for categorical columns
- Group-based imputation (e.g. average age by sex)
- Forward/backward fill for sequential data
- Adding a missing-indicator column as its own feature

Picked up a rough rule of thumb: under 5% missing → drop or simple impute, 5–30% → statistical imputation, over 30% → drop the feature or use something more advanced. Also covered label encoding vs one-hot encoding and the tradeoff between them (label encoding is simple but can imply a false order).

**Takeaway:** how much data is missing and what kind of feature it is should drive the decision — there's no single "correct" way to fill in gaps.

## Day 7 — Feature Engineering: Target Encoding & Scaling

**File:** `Day7_FeatureEngineering_and_Scaling.ipynb`

Learned target/mean encoding — using `groupby` to get the average of a numeric column per category, then mapping that back onto the categorical column. Useful for high-cardinality categories where one-hot encoding would blow up into too many columns.

Then spent most of the day on scaling, since features on wildly different scales can quietly dominate a model:

- **StandardScaler** — Z-score, centers to mean 0 / std 1
- **MinMaxScaler** — squeezes values into a fixed range, usually 0 to 1
- **RobustScaler** — uses median and IQR instead of mean/std, so outliers don't skew it (tested this by comparing the mean of a normal list against one with an extreme outlier thrown in)
- **MaxAbsScaler** — divides by the max absolute value, keeps the sign, range -1 to 1
- **Log transform** (`np.log1p`) — compresses skewed data like income into something closer to normal
- **Power transform (Yeo-Johnson)** — a more general fix for non-normal data, works with negative values too

**Takeaway:** which scaler to reach for depends on the shape of the data — StandardScaler for roughly normal data, MinMaxScaler for a bounded range, RobustScaler when outliers are a problem, log/power transforms for skewed distributions.

## Day 8 — Time Series Feature Engineering

**File:** `Day8_TimeSeries_FeatureEngineering.ipynb`
**Dataset:** Sales data with dates (CSV)

First time treating time as a first-class feature. Converted the date column to datetime and pulled out year, month, day, day of week, week of year (`isocalendar().week`), and quarter. Added `is_weekend` as a simple binary flag off day of week.

Built lag features (`Lag_1`, `Lag_3`, `Lag_7`) with `.shift()` so the model could see recent history alongside the current row. Added a rolling mean to smooth out short-term noise, a difference feature (`.diff()`) to capture day-over-day change, and percentage change to capture relative rather than absolute movement. Took notes on a few things I hadn't coded yet — rolling min/max, rolling std, seasonal features, cyclical encoding, and trend features — as things to come back to.

**Takeaway:** lag, rolling, and difference features are the core toolkit for letting a model "see" trends instead of just a single point in time.

## Day 9 — Time Series Feature Engineering: Cyclical Encoding, Moving Averages & EMA

**File:** `Day9_TimeSeries_FE.ipynb`
**Dataset:** Sales data with dates (CSV)

Picked up where Day 8 left off. Set `date` as the actual index this time instead of just a column, then re-extracted year, month, quarter, day of week, day, day of year, and week of year straight from the index. Added the weekend flag again the same way.

The new concept was **cyclical encoding** — instead of leaving `month` as a plain number (where December and January end up looking numerically far apart even though they're right next to each other), I encoded it with sine and cosine (`month_sin`, `month_cos`) so the "wraparound" is preserved.

Rebuilt the lag, rolling mean, diff, and percent-change features from Day 8 on the indexed data, plotted a correlation heatmap, and dropped the NaN rows the shifting/rolling left behind. Then moved on to moving averages — computed 3-day, 7-day, and 30-day windows with `shift(1).rolling(n).mean()` — and finally exponential moving averages (EMA) with `.ewm(span=n).mean()` at spans of 7 and 30.

Plotted sales against MA7 and EMA7 side by side to see the difference directly: EMA reacts faster to recent changes because it weights recent points more heavily, while a plain moving average treats every point in the window equally.

**Takeaway:** cyclical encoding fixes the "December vs. January" problem for periodic features, and EMA is the better choice when you care more about recent movement than long-run smoothing.

---
## Day 10 — Handling Imbalanced Data

**File:** `Day10_Imbalanced_Data.ipynb`
**Dataset:** Synthetic (generated with numpy / `make_classification`)

Today was about what happens when one class massively outnumbers another — a 90/10 split instead of a nice even 50/50. Started by building an imbalanced dataset by hand: 900 rows for class 0, 100 for class 1, each drawn from its own normal distribution, then concatenated into one dataframe to see the imbalance show up in `value_counts()`.

From there, covered the two classic ways to fix it:

- **Upsampling** — duplicate rows from the minority class (using `resample()` with `replace=True`) until it matches the majority class in size
- **Downsampling** — randomly drop rows from the majority class (`resample()` with `replace=False`) until it matches the minority class

Both are quick fixes, but upsampling just duplicates existing points rather than creating anything new, which is where **SMOTE** (Synthetic Minority Over-sampling Technique) comes in — it generates new synthetic minority-class points through interpolation between existing ones, instead of copying the same rows over and over. Tried it out on a synthetic dataset from `make_classification` (90/10 split), plotted the classes before and after using `SMOTE().fit_resample()`, and could see the minority class visibly fill out in the scatterplot instead of just stacking duplicate points.

**Takeaway:** upsampling and downsampling are the fast, blunt tools for class imbalance, but SMOTE is the smarter fix when you want the model to see *new* plausible minority examples instead of the same ones repeated.

---

*New day, new entry — this file gets a new section added as I go.*

