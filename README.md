# ML - World Cup 2026 Match Predictor - Hiram Israel Mendoza López 

## Project Description

This project builds a complete machine learning pipeline to predict match outcomes and simulate the full bracket of the 2026 FIFA World Cup, hosted jointly by Canada, Mexico, and the United States.

The pipeline covers every stage from raw data ingestion to a Monte Carlo tournament simulation producing championship probabilities for all 48 qualified teams.

The 2026 World Cup introduces a new format with 48 teams divided into 12 groups of 4. The top 2 from each group advance automatically, plus the 8 best third-place teams, for a total of 32 teams entering a knockout bracket that includes a new Round of 32 stage not present in previous editions.

## Dataset Description

Two datasets are used in this project.

### International Football Results (1872–2026)

Available at [Kaggle — martj42](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

The dataset contains over 50,000 international match results spanning from 1872 to 2026, organized as three CSV files:

| File | Description |
|------|-------------|
| `results.csv` | Match results with final scores, team names, tournament, date, and neutral ground flag |
| `shootouts.csv` | Penalty shootout outcomes for knockout matches |
| `goalscorers.csv` | Individual goalscorer records per match |

### 2026 FIFA World Cup Historical ELO Ratings

Available at [Kaggle — afonsofernandescruz](https://www.kaggle.com/datasets/afonsofernandescruz/2026-fifa-world-cup-historical-elo-ratings)

The dataset contains weekly ELO rating snapshots for all 48 WC 2026 qualified teams up to the start of the tournament.

| Column | Description |
|--------|-------------|
| `team` | Team name |
| `elo` | ELO rating at the snapshot date |
| `snapshot_date` | Date of the rating snapshot |

ELO ratings were chosen as the primary strength indicator over the official FIFA ranking because they update continuously based on match results and weight the quality of the opponent, making them a more accurate real-time measure of team strength.

## Dataset Storage

Due to the size of the dataset files, the data is stored in Google Drive instead of being committed to the repository.

The ZIP files downloaded from Kaggle are placed in the following Drive path:

```
MyDrive/ML/wc2026/
  international-football-results-from-1872-to-2017.zip
  2026-fifa-world-cup-historical-elo-ratings.zip
```

After extraction, the working structure is:

```
MyDrive/ML/wc2026/
  data/
    results.csv
    shootouts.csv
    goalscorers.csv
    elo_ratings_wc2026.csv
  models/
    logistic_regression.joblib
    random_forest.joblib
    xgboost.joblib
    lightgbm.joblib
  phase1_checkpoint.pkl
  phase2_checkpoint.pkl
  phase3_checkpoint.pkl
  phase4_checkpoint.pkl
  wc2026_predicted_schedule.csv
```

## Process

### Phase 1 — Dataset Loading and Exploratory Data Analysis

#### Dataset Selection

Two datasets were selected to cover complementary aspects of team strength: the historical results dataset provides the match-level data needed to compute dynamic features, while the ELO dataset provides a validated external measure of pre-tournament team quality.

#### Dataset Storage

The datasets are downloaded as ZIP files from Kaggle and uploaded directly to Google Drive through the Colab file upload interface. The notebook mounts Google Drive, extracts both ZIP files, and reads the CSVs from the extracted directory.

#### Data Preprocessing

The following transformations were applied to the raw data:

- A `result` column (`W` / `D` / `L`) was added from the perspective of the home team
- A numeric encoding (`2` / `1` / `0`) was added as the classification target
- The dataset was split into subsets: full history, modern era (1990+), World Cup only, and competitive matches only (2010+, excluding friendlies)
- A `tournament_weight` column was added assigning an importance factor to each tournament type

#### Exploratory Data Analysis

Six analyses were performed to validate feature hypotheses before engineering:

| Analysis | Finding |
|----------|---------|
| Result distribution | Home teams win ~47% of all matches; this drops at the World Cup due to neutral ground |
| Goals over time | Modern matches average ~2.5 goals per game |
| Win rate per team | Wide spread among the 48 qualified teams, from elite favorites to clear underdogs |
| ELO distribution | Ratings range from ~1500 for underdogs to 2000+ for top teams |
| ELO vs win rate correlation | Strong positive correlation, confirming ELO as the most predictive single feature |
| Head-to-head availability | Sufficient historical data exists between most WC 2026 team pairs |

### Phase 2 — Feature Engineering

#### Team Name Mapping

Team names in the WC 2026 draw differ in some cases from those used in the historical results dataset. An explicit mapping dictionary normalizes all 48 team names before feature computation to prevent silent NaN values in the feature matrix.

#### Tournament Importance Weights

Each match in the historical dataset is assigned a weight based on the competition type. These weights serve two purposes: they modulate the ELO K-factor and filter the competitive form window.

| Tournament type | Weight | K-factor |
|-----------------|--------|----------|
| FIFA World Cup | 1.00 | 60 |
| Continental championship | 0.85 | 50 |
| World Cup qualifier | 0.75 | 40 |
| Nations League / Confederation Cup | 0.65 | 35 |
| Friendly | 0.30 | 20 |

#### Dynamic ELO Ratings

ELO ratings are computed match by match in strict chronological order over the full 50,000+ match dataset. The rating used to predict any given match only reflects matches played before that date, which prevents data leakage.

The formula used is:

```
Expected score:  Ea = 1 / (1 + 10^((Rb - Ra) / 400))
New rating:      Ra_new = Ra + K * gd_mult * (Sa - Ea)
```

A goal-difference multiplier `gd_mult = log(|GD| + 1) + 1` is applied so that dominant wins update ratings more than narrow wins. All teams start at 1500 and diverge quickly from the initial value as their match history is processed.

#### Recent Form

A sliding window of the last 10 matches before each match date is used to compute form features. Friendlies are tracked separately from competitive matches and weighted accordingly. Only matches played strictly before the prediction date are included.

| Feature | Description |
|---------|-------------|
| `home_form_wr` | Win rate in last 10 matches |
| `home_form_gf` | Average goals scored in last 10 matches |
| `home_form_ga` | Average goals conceded in last 10 matches |
| `home_form_comp_wr` | Win rate in last 10 competitive matches only |

#### Head-to-Head

Historical win rates between specific team pairs are computed using the last 20 direct matchups before the prediction date. More recent matches are weighted more heavily using a linear weight vector from 0.5 to 1.0.

#### Final Feature Set

The training dataset contains one row per match from 1995 onwards. Each row includes the following features for both teams:

| Feature | Description |
|---------|-------------|
| `home_elo` / `away_elo` | Dynamic ELO rating at the time of the match |
| `elo_diff` | Difference in ELO ratings (home minus away) |
| `home_form_wr` / `away_form_wr` | Win rate in last 10 matches |
| `home_form_gf` / `home_form_ga` | Avg goals scored / conceded in last 10 matches |
| `home_form_comp_wr` / `away_form_comp_wr` | Win rate in last 10 competitive matches |
| `h2h_home_wr` / `h2h_away_wr` | Head-to-head win rate between the two teams |
| `is_neutral` | Whether the match is played on neutral ground |
| `tournament_weight` | Importance weight of the competition |

### Phase 3 — Model Training and Evaluation

#### Train / Test Split

A time-based split is used instead of a random split. All matches before January 1, 2022 are used for training. All matches from 2022 onwards, including the Qatar World Cup, form the test set.

This simulates real prediction conditions where the model is trained on historical data and evaluated on future matches it has never seen. A random split would allow information from future matches to contaminate the training process.

#### Models Trained

Four classifiers are trained and compared. The target variable has three classes: Home Win (2), Draw (1), Away Win (0).

| Model | Role |
|-------|------|
| Logistic Regression | Linear baseline |
| Random Forest | Ensemble — bagging |
| XGBoost | Ensemble — gradient boosting |
| LightGBM | Ensemble — gradient boosting |

Cross-validation uses `TimeSeriesSplit` with 5 folds, ensuring the validation set is always more recent than the training set within each fold.

#### Results

| Model | CV Accuracy | Test Accuracy | Test Log-Loss |
|-------|------------|---------------|---------------|
| **XGBoost** | **0.574** | **0.595** | **0.892** |
| Random Forest | 0.551 | 0.565 | 0.914 |
| Logistic Regression | 0.553 | 0.562 | 0.910 |
| LightGBM | 0.534 | 0.550 | 0.922 |

XGBoost was selected as the best model with a test accuracy of **59.5%** and a log-loss of **0.892**, both well below the random baseline of 33.3% accuracy and 1.099 log-loss.

#### Feature Importance

ELO difference is by far the most important feature across both Random Forest (0.382) and XGBoost (0.283), confirming the initial hypothesis from the EDA. Head-to-head win rate and goals conceded are the next most informative features.

#### Calibration

The reliability diagram for the home win probability shows the model is well-calibrated — the predicted probabilities closely follow the diagonal across the full 0–1 range. This is important for the Monte Carlo simulation in Phase 4, which relies on the probabilities being realistic rather than just the binary class prediction.

#### Validation on the 2022 World Cup

Evaluated exclusively on the 136 Qatar 2022 World Cup matches in the test set, XGBoost achieves an accuracy of **41.2%**. This is expected: at a World Cup all teams are elite, matches are more evenly contested, and upsets are more frequent than in regular international football. Published models from academic and industry sources typically score 50–55% on World Cup prediction, placing this result in a competitive range given the available data.

### Phase 4 — Tournament Simulation

#### Match Probability Precomputation

All possible ordered pairs of WC 2026 teams (48 × 47 = 2,256 pairs) are evaluated
in a single batched `predict_proba` call before the simulation loop begins. The
resulting probabilities are stored in a dictionary `prob_cache` indexed by
`(home_team, away_team)`.

This reduces the Monte Carlo loop from over 1,000,000 individual model calls to
simple dictionary lookups, bringing the total simulation time from an estimated
9–10 hours down to approximately 3–5 minutes.

#### Group Stage Simulation

Each group plays a full round-robin of 6 matches. Match results are sampled
stochastically from the model's probability distribution using
`numpy.random.choice`. Goal counts are generated using a Poisson distribution
parameterized by each team's recent average goals scored and conceded, then
aligned to match the sampled result.

Group standings are sorted by points, goal difference, and goals scored,
following FIFA 2026 tiebreaker rules.

#### Advancement Rules

Following the 2026 WC format, 32 teams advance from the group stage:

- Top 2 from each of the 12 groups: 24 teams
- Best 8 third-place teams ranked by points, goal difference, and goals scored: 8 teams

#### Monte Carlo Simulation

10,000 independent full-tournament simulations are run. In each simulation, all
group stage matches and all knockout matches are sampled stochastically from the
model's probability distribution. Draws in knockout matches are resolved through
a simulated extra time and penalty shootout.

Championship probability for each team is computed as the proportion of
simulations in which that team wins the Final.

#### Score Distribution

For each of the 103 tournament matches (72 group stage + 31 knockout), 10,000
simulations are run to determine the most likely scoreline. Goal counts are
generated with a Poisson distribution and the most frequent scoreline is reported
alongside the top 3 most common scores and their frequencies.

For the group stage, all 72 matchups are fixed by the official draw, so each is
simulated 10,000 times directly.

For the knockout stage, the bracket is not fixed in advance — which teams reach
each round depends on prior results. To address this, a separate set of 10,000
full tournament simulations is run while tracking which specific teams appear in
each bracket slot at every stage. For each slot, the most likely matchup is
identified and then simulated 10,000 times independently for the score.

The final display shows four pieces of information per knockout match:

| Field | Description |
|-------|-------------|
| Most likely matchup | The two teams that appeared most often in that bracket slot |
| Reach probability | % of simulations in which each team reached that round |
| Matchup frequency | % of simulations in which those two specific teams met in that slot |
| Most likely score | The scoreline that appeared most often across 10,000 simulations of that matchup |

## Results

| Metric | Value |
|--------|-------|
| Best model | XGBoost |
| Test set accuracy | 59.5% |
| WC 2022 accuracy | 41.2% |
| Random baseline | 33.3% |
| Predicted champion | Spain |
| Predicted finalist | France |
| Predicted semifinalists | Brazil, Spain, France, Morocco |

### Monte Carlo Championship Probabilities — Top 10

| Team | Champion | Final | Semifinal |
|------|----------|-------|-----------|
| Spain | 30.7% | 41.1% | 55.0% |
| Brazil | 10.6% | 18.9% | 34.0% |
| France | 10.0% | 20.5% | 30.5% |
| Argentina | 7.2% | 15.3% | 23.7% |
| Germany | 6.5% | 12.8% | 24.8% |
| Morocco | 5.4% | 12.3% | 28.6% |
| England | 4.2% | 10.7% | 18.3% |
| Turkey | 3.4% | 7.7% | 17.5% |
| Netherlands | 3.3% | 6.8% | 15.8% |
| Switzerland | 3.1% | 7.3% | 19.9% |

## Tools Used

- Python 3
- Google Colab
- Google Drive
- pandas
- NumPy
- Matplotlib / Seaborn
- scikit-learn
- XGBoost
- LightGBM
- tqdm

