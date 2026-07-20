# ML Learning Journey

This repo documents my day-by-day progress through an ML course. Each notebook is the hands-on code I wrote that day, and this README is updated daily with what was covered and what I learned.

---

## Day 1 — EDA Project: Intro to Exploratory Data Analysis
**File:** `Day1_EDA_project.ipynb`
**Dataset:** Seaborn built-in `tips` dataset

- Loaded a dataset and did basic data understanding: `shape`, `info()`, checking for nulls and duplicates
- Removed duplicate rows with `drop_duplicates()`
- Explored unique values across all columns using a loop
- Created a new feature: `tip_percentage = (tip / total_bill) * 100`
- **Univariate analysis:** histograms + KDE for `total_bill` and `tip`, boxplot for outliers, countplots for `sex`, `smoker`, `day`, `size`
- **Bivariate analysis:** scatterplot of `total_bill` vs `tip`, started a regression plot

**Key takeaway:** Learned the standard EDA workflow — understand → clean → engineer a feature → visualize univariate → visualize bivariate.

---

## Day 2 — EDA Project: Practicing the EDA Workflow
**File:** `Day2_EDA_project.ipynb`
**Dataset:** Seaborn built-in `tips` dataset

- Repeated the Day 1 EDA workflow on the same dataset to reinforce the pattern (shape, info, nulls, duplicates, unique values, feature creation)
- Same univariate plots (histograms, boxplot, countplots for `sex`, `smoker`, `day`, `size`)
- Bivariate: `total_bill` vs `tip` scatterplot

**Key takeaway:** Repetition day to build muscle memory for the EDA process before applying it to new, messier datasets.

---

## Day 3 — FIFA 19 EDA
**File:** `Day3_Fifa19_EDA.ipynb`
**Dataset:** FIFA 19 player dataset (CSV)

- Loaded a real-world CSV with `encoding="Latin-1"` (handling non-UTF8 data)
- Checked missing values, data types, and used `describe()` for summary statistics
- **Univariate:** age distribution histogram, preferred foot countplot
- **Bivariate:** age vs overall rating scatterplot, full correlation heatmap across numeric columns
- **Deeper analysis:**
  - Top 10 highest-rated players
  - Groupby analysis: average rating per club
  - Outlier detection on `Wage` using a boxplot
  - Player position value counts (bar chart)
  - Nationality distribution (bar chart)
- Wrote a short **Insights** section summarizing findings (age skew, foot preference, club/wage patterns)

**Key takeaway:** First time doing EDA on a large, real, messy dataset with `groupby`, correlation heatmaps, and writing insights instead of just plots.

---

## Day 4 — Netflix EDA
**File:** `Day4_Netflix_EDA.ipynb`
**Dataset:** Netflix titles dataset (CSV)

- Explored data by filtering rows on conditions (e.g., looking up a specific title)
- Checked nulls, unique values, duplicates, and made a working copy of the dataset to avoid mutating the original
- **Data cleaning:** dropped nulls, stripped whitespace from `date_added`
- **Date handling:** converted `date_added` to datetime with `pd.to_datetime(..., format="mixed")`, then extracted `day`, `month`, and `year` into new columns
- **Visualizations:** countplot of content `type`, countplot of `rating`, rating vs type relationship, and a pie chart of top ratings by percentage

**Key takeaway:** Learned datetime parsing/feature extraction and how to safely work on a copy of a dataframe instead of the original.

---

## Day 5 — Feature Engineering: Feature Types
**File:** `Day5_FeatureEngineering.ipynb`

Covered the different categories of features and how to handle each:

- **Raw vs processed data:** cleaning inconsistent categorical text (e.g., normalizing `'M'`, `'male'`, `'Fem'`, `'female'` → consistent labels)
- **Numerical features:** identifying and inspecting numeric columns
- **Categorical features:** converting to `category` dtype
- **Ordinal features:** manually mapping ordered categories to numbers (`Junior=1, Mid=2, Senior=3`)
- **Binary features:** boolean columns
- **Date/time features:** extracting day, month, year, weekday name from a datetime column
- **Text features:** engineering word count and character count from text
- **Image/signal data:** intro to image data using `skimage`, inspecting shape and dimensions of an image array

**Key takeaway:** Learned to recognize different feature types (numeric, categorical, ordinal, binary, datetime, text, image) and the right way to prepare each one for a model.

---

## Day 6 — Feature Engineering: Missing Values & Encoding
**File:** `Day6_FeatureEngineering.ipynb`
**Dataset:** Seaborn built-in `titanic` dataset

- **Types of missingness:** MCAR (missing completely at random), MAR (missing at random), MNAR (missing not at random)
- **Method 1 — Dropping:** `dropna()` on rows and columns, and when this is acceptable
- **Imputation techniques:**
  - Mean imputation (for roughly normal distributions)
  - Median imputation (when outliers are present)
  - Mode imputation (for categorical features)
  - Group-based imputation (e.g., mean age grouped by sex)
  - Forward fill (`ffill`) and backward fill (`bfill`) for sequential/time data
  - Missing indicator engineering (flagging whether a value was missing as its own feature)
- **Rule of thumb guide:** <5% missing → drop/simple impute, 5–30% → statistical imputation, >30% → drop feature or advanced imputation
- **Categorical encoding:**
  - Label Encoding (simple, but risks implying false order between categories)
  - One-Hot Encoding (separate binary column per category)

**Key takeaway:** Learned a structured decision process for handling missing data based on percentage missing and the type of feature, plus the tradeoffs between label encoding and one-hot encoding.

---



---

## Day 7 — Feature Engineering: Target Encoding & Feature Scaling
**File:** `Day7_FeatureEngineering_and_Scaling.ipynb`

- **Target/mean encoding:** used `groupby` to compute the mean of a numeric column per category, then mapped that back onto the categorical column (e.g., encoding `city` using average `price` per city) — an alternative to one-hot/label encoding for high-cardinality categorical features
- **Feature scaling** — normalizing the range of numeric features so no single feature dominates a model just because of its scale:
  - **Standardization (Z-score)** with `StandardScaler` — centers data to mean 0, std 1
  - **Min-Max Scaling** with `MinMaxScaler` — rescales values into a fixed range (0 to 1)
  - **Robust Scaling** with `RobustScaler` — uses median and IQR instead of mean/std, so it's not thrown off by outliers (compared mean of a normal list vs a list with an extreme outlier to see why this matters)
  - **Max Absolute Scaling** with `MaxAbsScaler` — scales by dividing by the max absolute value, keeping sign, output range -1 to 1
  - **Log Transformation** with `np.log1p` — compresses skewed data (like income) into a more normal-looking distribution
  - **Power Transformation (Yeo-Johnson)** with `PowerTransformer` — a more general transform to make data more Gaussian-like, works with negative values too

**Key takeaway:** Learned when to use which scaler — StandardScaler for roughly normal data, MinMaxScaler when you need a bounded range, RobustScaler when outliers are present, and log/power transforms for fixing skewed distributions.

---


