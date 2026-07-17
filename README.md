# ML Course

Documenting my AI/ML learning journey — one day, one notebook at a time.

I'm learning this mostly on my own, working through concepts and actually coding them out instead of just watching lectures. This repo is where I'm keeping track of everything so I can look back and see how much I've figured out (and probably cringe at some of it later).

## Day 1 — Exploratory Data Analysis

Started with EDA basics using the classic tips dataset from seaborn. Nothing fancy, just getting comfortable with the fundamentals before moving into anything model-related.

What I worked through:

Loading and inspecting the data (shape, info, null checks). Cleaning up duplicates. Poking around unique values in each column to understand what I'm working with. Created a new feature — `tip_percentage` — just to see if I could derive something useful from existing columns. Started univariate analysis to look at individual variables before comparing them to each other.

Honestly a lot of this felt obvious once I did it, but I think that's the point — EDA is less about complex math and more about actually looking at your data before you trust it.

### Why this repo exists

Mostly for myself. I want a record of what I learned and when, and pushing code daily keeps me accountable. If it helps someone else who's self-studying ML too, even better.

### What's next

More EDA, then moving into actual model building — decision trees are next on my list.

## Day 2 — From Numbers to Insight

Went deeper. Two datasets, one goal: make data talk.

**Tips dataset — full EDA pipeline**
Univariate → bivariate → multivariate → outlier detection. Correlation heatmaps, pairplots, IQR-based outlier flagging. No black boxes, just structured thinking.

Findings:

- Bill and tip move together — strong positive correlation
- Dinner beats lunch, weekends beat weekdays
- Small groups dominate, tip % swings by day and time
- Outliers exist — and now I know how to catch them

**Titanic dataset — survival analysis**
Classic dataset, real signal. Age, fare, class, gender — mapped against survival.

Findings:

- Women survived at far higher rates than men
- Money mattered — higher fare, better odds
- Kids had an edge
- Third-class men had it worst

### The shift

Day 1 was about looking at data. Day 2 is about asking it questions.

## Day 3 — FIFA19: Reading the Game Through Data

Left sports fandom aside. Looked at football as a dataset instead.

Explored the FIFA19 player dataset — ratings, wages, positions, nationalities — and let the numbers tell the story.

Findings:

- Prime age is 20–30 — the sweet spot for peak performance
- Right foot dominates the pitch
- European clubs stack the highest average ratings
- Wages are wildly uneven — outliers everywhere
- Certain positions and nations flood the dataset, revealing where the talent pipeline runs deepest

### The pattern so far

Three days, three datasets, one skill compounding: turning raw numbers into a story worth telling.

## Day 4 — Netflix: What's Actually on the Platform

Switched gears from sports to streaming. Took the Netflix titles dataset and started digging into what's actually sitting in the catalog.

What I worked through:

Loaded the data and checked shape, nulls, duplicates, unique value counts — standard first pass. Cleaned up whitespace issues in `date_added` and converted it to a proper datetime, then broke it into day, month, and year columns for time-based analysis. Compared movies vs. TV shows using count plots, then looked at how content ratings (TV-MA, PG-13, etc.) are distributed — and how that distribution shifts between movies and shows. Wrapped up with a pie chart of the top ratings by share.

Findings:

- Movies significantly outnumber TV shows in the catalog
- TV-MA and TV-14 dominate the ratings — Netflix skews mature
- Rating distribution isn't the same across movies and shows; some ratings lean heavily toward one type
- Date fields are messier than they look — stray whitespace silently breaks `pd.to_datetime` if you don't strip first

### The pattern so far

Four days in, and the real lesson isn't the plots — it's that every dataset hides a small landmine (a whitespace, a null, a weird dtype) that breaks things if you don't check first.
Four days in, and the real lesson isn't the plots — it's that every dataset hides a small landmine (a whitespace, a null, a weird dtype) that breaks things if you don't check first.
