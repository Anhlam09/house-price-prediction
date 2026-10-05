# House Price Prediction: Ames, Iowa

Predicts home sale prices from 79 features of each house: size, quality, age, neighborhood and more. The data comes from the Kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

**Result:** 0.109 RMSE on log price (5-fold cross-validation), or about **$13,200 average error** against a median house price of $163,000.

## Project structure

```
House Price Prediction/
├── house_prices.py     # the full pipeline: one script, top to bottom
├── data/
│   ├── train.csv       # 1,460 houses with sale prices
│   └── test.csv        # optional: Kaggle's test set (you download it)
├── requirements.txt
├── README.md           # this file
└── EXPLAINED.md        # plain-English walkthrough of every step
```

## How to run

**1. Install Python 3.9 or newer**, then install the libraries:

```bash
pip install -r requirements.txt
```

**2. Run the script** from the project folder:

```bash
python3 house_prices.py
```

It takes about 30 seconds and prints each step as it goes.

**3. (Optional) Make a Kaggle submission.** Download `test.csv` from the [competition data page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data), put it in `data/`, and run the script again. It writes `submission.csv` in the project folder, ready to upload.

## What the output shows

| Section | What it prints |
|---|---|
| 1. Load | Number of rows and columns |
| 2. Target | How skewed the price is before and after the log transform |
| 3. Outliers | The 2 houses that were removed, and why |
| 4. Missing data | Every feature with missing values and the % missing |
| 5. Feature engineering | How many features exist after adding new ones |
| 6. Multicollinearity | Pairs of features that are nearly duplicates (r > 0.8) |
| 7. Cross-validation | The error for each model and for the blend |
| 8. Final model | The 15 features that move the price most |

## Results

| Model | CV RMSE (log price) |
|---|---|
| Ridge | 0.1141 |
| Lasso | 0.1117 |
| ElasticNet | 0.1119 |
| Gradient Boosting | 0.1141 |
| **Blend** | **0.1090** |

## Approach in brief

- **Target:** modeled `log(SalePrice)`, because prices are right-skewed and Kaggle scores on log error.
- **Outliers:** removed 2 houses over 4,000 sq ft that sold under $300k, as the dataset's author recommends.
- **Missing data:** 19 features have gaps and 7 are more than 15% missing. Most gaps mean "this house doesn't have it" (no pool, no garage), so they became a `None` category or 0. LotFrontage was filled with its neighborhood median.
- **Feature engineering:** total square feet, total bathrooms, house age, quality × size, and yes/no flags (has garage, has pool and so on). Quality ratings were converted to a 0–5 scale.
- **Multicollinearity:** handled with regularized models. Lasso kept 113 of 268 features.
- **No leakage:** all imputation, scaling and encoding happens inside a scikit-learn `Pipeline`, so it is refit on each CV fold.

See [EXPLAINED.md](EXPLAINED.md) for the full plain-English explanation.

## Troubleshooting

- **`ModuleNotFoundError`:** run `pip install -r requirements.txt` again. If you have several Python versions, use `python3 -m pip install -r requirements.txt`.
- **`FileNotFoundError: data/train.csv`:** run the script from inside the project folder, or check that `data/train.csv` exists.
