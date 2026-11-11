# House Price Prediction: Ames, Iowa

Predicts home sale prices from 79 features of each house: size, quality, age, neighborhood and more. It's built for the Kaggle competition [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

**Result on the real Kaggle data:** 0.109 RMSE on log price (5-fold cross-validation), or about **$13,200 average error** against a median house price of $163,000.

## About the data

Kaggle's data isn't included in this repo. Instead, `data/sample_train.csv` holds 300 **made-up houses** with the same 81 columns, so the script runs right away. The sample values are random and the prices come from an invented formula, so results on the sample don't mean anything.

For real results, download `train.csv` (and optionally `test.csv`) from the [competition data page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and put them in `data/`. The script uses `data/train.csv` whenever it's there.

## Project structure

```
house-price-prediction/
├── house_prices.py          # the full pipeline: one script, top to bottom
├── data/
│   ├── sample_train.csv     # 300 made-up houses, so the script runs out of the box
│   ├── train.csv            # optional: Kaggle's real training data (you download it)
│   └── test.csv             # optional: Kaggle's test set (you download it)
├── requirements.txt
├── README.md                # this file
└── EXPLAINED.md             # plain-English walkthrough of every step
```

## How to install

You need Python 3.9 or newer.

```bash
git clone https://github.com/Anhlam09/house-price-prediction.git
cd house-price-prediction
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to run

From the project folder:

```bash
python3 house_prices.py
```

It takes under a minute and prints each step as it goes.

**(Optional) Make a Kaggle submission.** Put Kaggle's `train.csv` and `test.csv` in `data/` and run the script again. It writes `submission.csv` in the project folder, ready to upload.

## Example output

This is the start of a run on the made-up sample data:

```
======================================================================
1. LOAD
======================================================================
No data/train.csv found, so using the made-up sample data (data/sample_train.csv).
Results won't mean much. Download train.csv from Kaggle into data/ for real results.
train: (300, 81) | no test.csv
...
======================================================================
7. CROSS-VALIDATION (5-fold RMSE on log price, lower is better)
======================================================================
Ridge              RMSE 0.1417 ± 0.0094
Lasso              RMSE 0.1020 ± 0.0074
ElasticNet         RMSE 0.1020 ± 0.0075
GradientBoosting   RMSE 0.1583 ± 0.0132
Blend              RMSE 0.1183  (out-of-fold)
```

## What the output shows

| Section | What it prints |
|---|---|
| 1. Load | Which data file it used, and the number of rows and columns |
| 2. Target | How skewed the price is before and after the log transform |
| 3. Outliers | The houses that were removed, and why (2 in the real data) |
| 4. Missing data | Every feature with missing values and the % missing |
| 5. Feature engineering | How many features exist after adding new ones |
| 6. Multicollinearity | Pairs of features that are nearly duplicates (r > 0.8) |
| 7. Cross-validation | The error for each model and for the blend |
| 8. Final model | The 15 features that move the price most |

## Results on the real Kaggle data

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
- **`FileNotFoundError`:** run the script from inside the project folder, and check that `data/sample_train.csv` (or `data/train.csv`) exists.

## License

[MIT](LICENSE)
