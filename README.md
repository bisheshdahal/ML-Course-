# ML Course

Documenting my AI/ML learning journey — one day, one notebook at a time.

I'm learning this mostly on my own, working through concepts and actually coding them out instead of just watching lectures. This repo is where I'm keeping track of everything so I can look back and see how much I've figured out (and probably cringe at some of it later).

Pushing code daily keeps me accountable. If it helps someone else self-studying ML, even better.

---

## Progress Log

### Day 1 — EDA Basics (tips dataset)
Shape/null/duplicate checks, unique value exploration, built `tip_percentage` feature, started univariate analysis.
- EDA is less about math, more about actually looking at your data before trusting it.

### Day 2 — Full EDA Pipeline (tips + Titanic)
Univariate → bivariate → multivariate → outlier detection. Correlation heatmaps, pairplots, IQR flagging.
- Bill and tip strongly correlated; dinner/weekends tip more
- Titanic: women, kids, and higher fare = better survival odds; 3rd-class men had it worst

### Day 3 — FIFA19 Player Data
Explored ratings, wages, positions, nationalities.
- Prime age 20–30; right foot dominates; European clubs rate highest; wages wildly uneven

### Day 4 — Netflix Catalog
Cleaned `date_added` (whitespace was breaking `pd.to_datetime`), split into day/month/year, compared movies vs. shows and rating distributions.
- Movies outnumber shows; catalog skews mature (TV-MA/TV-14); rating mix differs by type

### Day 5 — Feature Engineering Basics
Covered feature types (numerical, categorical, ordinal, binary, datetime, text, image) and core prep concepts.
- Cleaned messy categorical data (gender labels), encoded ordinal levels, extracted date/text features
- Explored image data as arrays, feature engineering vs. extraction vs. selection, polynomial features + SelectKBest, data leakage via train/test split

