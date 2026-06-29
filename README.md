# FIFA World Cup 2026 - Match Outcome Prediction
## Round 2 Submission

### Summary (5 lines)
The best performing model was a **Gradient Boosting Classifier** tuned with GridSearchCV, which outperformed the logistic regression baseline on all four metrics. The most useful custom feature was `Rank_Diff` (away rank minus home rank) which had the highest importance score — basically better ranked teams win more often, which makes sense. I also engineered a `Home_Win_Ratio` feature using rolling historical averages which captured long-term team strength without leaking future data. K-Means clustering grouped countries into 3 tiers (Elite Contenders, Mid-Tier, Emerging Underdogs) and Brazil, France, Germany, Argentina cleanly ended up in the elite cluster as expected. The Streamlit app lets you pick any two 2026 teams and shows the predicted win/draw/loss probabilities on screen.

---

## Project Structure

```
├── FIFA_WC_2026_Prediction.ipynb    ← main notebook with everything
├── deployment/
│   └── app.py                       ← streamlit app (bonus)
└── README.md
```

## Tasks Completed

### Core Tasks
- [x] Data cleaning + country name standardization using explicit NAME_MAP dictionary
- [x] Zero NaN values post-merge (assertion in notebook to verify)
- [x] EDA - summary stats, correlation matrix, top-10 goals bar chart with rotation=45
- [x] Target variable: 0=Home Win, 1=Away Win, 2=Draw
- [x] 5 custom engineered features with documented logic
- [x] Temporal split: train on 1930-2018, test on 2022
- [x] Baseline: Logistic Regression
- [x] Advanced: Gradient Boosting Classifier with full metric comparison
- [x] predict_proba() forecasts for 10 x 2026 group fixtures in tabular format
- [x] Regression: Random Forest Regressor for total goals (MAE + RMSE)

### Bonus Tasks
- [x] K-Means clustering with elbow curve + scatter plot
- [x] Hyperparameter tuning with GridSearchCV (5-fold CV)
- [x] Streamlit app in deployment/app.py

## Custom Features

| Feature | What it is |
|---------|-----------|
| `Rank_Diff` | away_rank - home_rank. positive = home team is ranked better |
| `Home_Goal_Rate` | rolling avg goals scored per home match (no data leakage) |
| `Away_Goal_Rate` | rolling avg goals scored per away match |
| `Home_Win_Ratio` | cumulative % of home games won historically |
| `Pressure_Index` | normalized position of match in tournament (0=early, 1=final) |

## How to Run

```bash
# notebook
jupyter notebook FIFA_WC_2026_Prediction.ipynb

# streamlit app
pip install streamlit scikit-learn pandas numpy
streamlit run deployment/app.py
```

## Notes

- couldn't get XGBoost installed in my environment so used sklearn's GradientBoostingClassifier instead
- used synthetic data based on historical stats since the Kaggle dataset isn't bundled here
- draws are the hardest outcome to predict - the model tends to underpredict them
  
~ APP:https://fifa-worldcup-winner-predictor-nbajphxt3dh4rn5sbn4qzz.streamlit.app/
