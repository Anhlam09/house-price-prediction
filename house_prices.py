"""
House Price Prediction — Ames, Iowa (Kaggle: House Prices - Advanced Regression Techniques)

Run:  python3 house_prices.py
Data: data/train.csv from Kaggle (falls back to the made-up data/sample_train.csv),
      and optionally data/test.csv from Kaggle to produce a submission

Pipeline
  1. Load & inspect
  2. Target analysis        -> log-transform SalePrice (Kaggle scores RMSE on log price)
  3. Outlier removal        -> 2 huge, cheap houses the dataset author flags
  4. Missing data           -> "NA = feature absent" vs. "truly unknown" (the interview answer)
  5. Feature engineering    -> totals, ages, ordinal quality scales, skew fixes
  6. Multicollinearity      -> detect correlated pairs, handle with regularization
  7. Cross-validated models -> Ridge, Lasso, ElasticNet, Gradient Boosting, blend
  8. Final fit, feature importance, optional Kaggle submission
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

RANDOM_STATE = 42
DATA_DIR = Path(__file__).parent / "data"
pd.set_option("display.width", 120)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
section("1. LOAD")
train_path = DATA_DIR / "train.csv"
if not train_path.exists():
    train_path = DATA_DIR / "sample_train.csv"
    print("No data/train.csv found, so using the made-up sample data (data/sample_train.csv).")
    print("Results won't mean much. Download train.csv from Kaggle into data/ for real results.")
train = pd.read_csv(train_path)
test_path = DATA_DIR / "test.csv"
test = pd.read_csv(test_path) if test_path.exists() else None
print(f"train: {train.shape}" + (f" | test: {test.shape}" if test is not None else " | no test.csv"))


# ---------------------------------------------------------------------------
# 2. Target: SalePrice is right-skewed -> model log1p(SalePrice)
#    Why: linear models assume roughly normal residuals, and Kaggle's metric is
#    RMSE on log price, so a 10% error on a $100k house costs the same as on a $500k one.
# ---------------------------------------------------------------------------
section("2. TARGET")
print(f"SalePrice skew raw: {skew(train.SalePrice):.2f}  |  after log1p: {skew(np.log1p(train.SalePrice)):.2f}")


# ---------------------------------------------------------------------------
# 3. Outliers: Dean De Cock (dataset author) recommends dropping houses with
#    GrLivArea > 4000. The two cheap ones are partial sales that distort the fit.
# ---------------------------------------------------------------------------
section("3. OUTLIERS")
outliers = train[(train.GrLivArea > 4000) & (train.SalePrice < 300_000)]
print(f"Dropping {len(outliers)} rows (Id {outliers.Id.tolist()}): >4000 sqft but < $300k")
train = train.drop(outliers.index).reset_index(drop=True)

y = np.log1p(train.SalePrice)
train_ids = train.Id
X = train.drop(columns=["Id", "SalePrice"])
X_test = test.drop(columns=["Id"]) if test is not None else None


# ---------------------------------------------------------------------------
# 4. Missing data
#
#    INTERVIEW ANSWER: "I didn't treat missing values uniformly. I read the data
#    dictionary and found most NaNs aren't missing at all — they mean the house
#    doesn't have that feature (no pool, no alley, no garage). Those get an explicit
#    'None' category, or 0 for the matching numeric columns. Dropping them would throw
#    away signal: having no garage is itself predictive of price.
#    Only a few are truly unknown: LotFrontage I imputed with the neighborhood median,
#    since lots on the same street have similar frontage. The handful left over
#    (Electrical, 1 row) get median/mode. All learned imputation happens inside the CV
#    pipeline so validation folds never leak into the fill values."
# ---------------------------------------------------------------------------
section("4. MISSING DATA")
missing = X.isna().mean().sort_values(ascending=False)
missing = missing[missing > 0]
print(f"{len(missing)} features have missing values; {(missing > 0.15).sum()} are >15% missing:\n")
print((missing * 100).round(1).to_string())

# NaN means "this house doesn't have it" (per data_description.txt)
NONE_CATEGORICAL = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "MasVnrType",
    "GarageType", "GarageFinish", "GarageQual", "GarageCond",
    "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2",
]
# Numeric partners of the above: no garage -> 0 cars, no basement -> 0 sqft, etc.
ZERO_NUMERIC = [
    "MasVnrArea", "GarageCars", "GarageArea",
    "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath",
]


def fill_structural_missing(df):
    """Deterministic, rule-based fills: no statistics learned, so safe before CV."""
    df = df.copy()
    for c in NONE_CATEGORICAL:
        df[c] = df[c].fillna("None")
    for c in ZERO_NUMERIC:
        df[c] = df[c].fillna(0)
    # No garage -> no garage year. Use the house's build year so "age" features stay sane;
    # the HasGarage flag (below) keeps the "no garage" information.
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])
    return df


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Fill a column with the median of its group (e.g. LotFrontage by Neighborhood).
    Learns medians in fit(), so it runs inside CV without leaking."""

    def __init__(self, column="LotFrontage", group="Neighborhood"):
        self.column = column
        self.group = group

    def fit(self, X, y=None):
        self.group_medians_ = X.groupby(self.group)[self.column].median()
        self.global_median_ = X[self.column].median()
        return self

    def transform(self, X):
        X = X.copy()
        fill = X[self.group].map(self.group_medians_).fillna(self.global_median_)
        X[self.column] = X[self.column].fillna(fill)
        return X


# ---------------------------------------------------------------------------
# 5. Feature engineering
# ---------------------------------------------------------------------------
QUALITY = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ORDINAL_MAPS = {
    **{c: QUALITY for c in ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                            "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]},
    "BsmtExposure": {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4},
    "BsmtFinType1": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "BsmtFinType2": {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6},
    "GarageFinish": {"None": 0, "Unf": 1, "RFn": 2, "Fin": 3},
    "Functional": {"Sal": 0, "Sev": 1, "Maj2": 2, "Maj1": 3, "Mod": 4, "Min2": 5, "Min1": 6, "Typ": 7},
}


def engineer_features(df):
    df = df.copy()

    # Quality ratings are ordered (Ex > Gd > TA ...). Integers keep that order and
    # use 1 column instead of 6 one-hot columns.
    for col, mapping in ORDINAL_MAPS.items():
        df[col] = df[col].map(mapping)

    # MSSubClass and month sold are codes, not quantities
    df["MSSubClass"] = df["MSSubClass"].astype(str)
    df["MoSold"] = df["MoSold"].astype(str)

    # Totals: buyers price total living space, not floors separately
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = df["FullBath"] + 0.5 * df["HalfBath"] + df["BsmtFullBath"] + 0.5 * df["BsmtHalfBath"]
    df["TotalPorchSF"] = df[["OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "WoodDeckSF"]].sum(axis=1)

    # Ages are more meaningful than raw years
    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

    # Interaction: quality x size is the single strongest price driver
    df["QualxSF"] = df["OverallQual"] * df["TotalSF"]

    # Presence flags preserve the "doesn't have it" signal from step 4
    df["HasGarage"] = (df["GarageArea"] > 0).astype(int)
    df["HasBsmt"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["Has2ndFlr"] = (df["2ndFlrSF"] > 0).astype(int)
    df["HasPool"] = (df["PoolArea"] > 0).astype(int)

    # Near-constant columns carry ~no information (Utilities: 1459/1460 'AllPub')
    df = df.drop(columns=["Utilities", "Street", "PoolQC"])
    return df


X = engineer_features(fill_structural_missing(X))
if X_test is not None:
    X_test = engineer_features(fill_structural_missing(X_test))

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

# Log-transform skewed numeric features (decided on train only, applied to both).
# Exclude binary flags and ordinal scales — skew means nothing for them.
skip = set(ORDINAL_MAPS) | {c for c in num_cols if X[c].nunique() <= 2}
skewness = X[[c for c in num_cols if c not in skip and c != "LotFrontage"]].apply(skew)
skewed_cols = skewness[skewness.abs() > 0.75].index.tolist()
skewed_cols.append("LotFrontage")  # skewed too; transformed after imputation below

section("5. FEATURE ENGINEERING")
print(f"{X.shape[1]} features after engineering ({len(num_cols)} numeric, {len(cat_cols)} categorical)")
print(f"{len(skewed_cols)} skewed numeric features will get log1p")


class Log1pColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columns] = np.log1p(X[self.columns].clip(lower=0))
        return X


# ---------------------------------------------------------------------------
# 6. Multicollinearity
#    79 features describing one house overlap heavily (GarageCars ~ GarageArea,
#    TotalBsmtSF ~ 1stFlrSF). With plain OLS this makes coefficients unstable and
#    uninterpretable. Approach: detect it, then use regularized models (Ridge shrinks
#    correlated coefficients together; Lasso picks one and zeroes the rest).
#    Tree models don't care about collinearity for prediction.
# ---------------------------------------------------------------------------
section("6. MULTICOLLINEARITY")
corr = X[num_cols].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
pairs = upper.stack().sort_values(ascending=False)
print("Most correlated numeric pairs (|r| > 0.8):")
print(pairs[pairs > 0.8].round(3).to_string())


# ---------------------------------------------------------------------------
# 7. Models + cross-validation
#    Everything that learns from data (imputation, scaling, encoding) is inside the
#    Pipeline, so each CV fold fits it on its own training part only — no leakage.
# ---------------------------------------------------------------------------
preprocess = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), RobustScaler()), num_cols),
    ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                          OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat_cols),
])


def build(model):
    return Pipeline([
        ("lot_frontage", GroupMedianImputer("LotFrontage", "Neighborhood")),
        ("log_skewed", Log1pColumns(skewed_cols)),
        ("preprocess", preprocess),
        ("model", model),
    ])


# Each *CV model tunes its own alpha with an inner CV on the training folds
alphas = np.logspace(-4.5, -2, 30)
models = {
    "Ridge": build(RidgeCV(alphas=np.logspace(-1, 3, 50))),
    "Lasso": build(LassoCV(alphas=alphas, max_iter=20_000, random_state=RANDOM_STATE)),
    "ElasticNet": build(ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.5, 0.9],
                                     max_iter=20_000, random_state=RANDOM_STATE)),
    "GradientBoosting": build(GradientBoostingRegressor(
        n_estimators=1000, learning_rate=0.04, max_depth=3, min_samples_leaf=10,
        subsample=0.8, max_features="sqrt", loss="huber", random_state=RANDOM_STATE)),
}

section("7. CROSS-VALIDATION (5-fold RMSE on log price, lower is better)")
kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
oof = {}
folds = list(kf.split(X))
for name, pipe in models.items():
    # Out-of-fold predictions: every row is predicted by a model that never saw it
    oof[name] = cross_val_predict(pipe, X, y, cv=folds, n_jobs=-1)
    scores = np.array([np.sqrt(np.mean((oof[name][val] - y.iloc[val]) ** 2)) for _, val in folds])
    print(f"{name:<18} RMSE {scores.mean():.4f} ± {scores.std():.4f}")

# Blend: linear and tree models make different mistakes, so averaging helps
blend_oof = 0.35 * oof["Lasso"] + 0.15 * oof["Ridge"] + 0.15 * oof["ElasticNet"] + 0.35 * oof["GradientBoosting"]
blend_rmse = np.sqrt(np.mean((blend_oof - y) ** 2))
print(f"{'Blend':<18} RMSE {blend_rmse:.4f}  (out-of-fold)")
mae_dollars = np.mean(np.abs(np.expm1(blend_oof) - np.expm1(y)))
print(f"\nIn dollars: blend is off by ${mae_dollars:,.0f} on average (median price ${np.expm1(y).median():,.0f})")


# ---------------------------------------------------------------------------
# 8. Final fit on all training data + interpretation
# ---------------------------------------------------------------------------
section("8. FINAL MODEL")
for pipe in models.values():
    pipe.fit(X, y)

lasso_pipe = models["Lasso"]
feature_names = lasso_pipe.named_steps["preprocess"].get_feature_names_out()
coefs = pd.Series(lasso_pipe.named_steps["model"].coef_, index=feature_names)
print(f"Lasso alpha={lasso_pipe.named_steps['model'].alpha_:.5f}: kept {(coefs != 0).sum()} of {len(coefs)} "
      f"features, zeroed {(coefs == 0).sum()} (automatic feature selection / collinearity handling)")
print("\nTop 15 features by |coefficient|:")
top = coefs[coefs.abs().sort_values(ascending=False).index[:15]]
print(top.rename(lambda s: s.split("__", 1)[1]).round(4).to_string())

if X_test is not None:
    preds = {name: pipe.predict(X_test) for name, pipe in models.items()}
    blend = 0.35 * preds["Lasso"] + 0.15 * preds["Ridge"] + 0.15 * preds["ElasticNet"] + 0.35 * preds["GradientBoosting"]
    out = Path(__file__).parent / "submission.csv"
    pd.DataFrame({"Id": test.Id, "SalePrice": np.expm1(blend)}).to_csv(out, index=False)
    print(f"\nWrote {out}")
else:
    print("\nAdd Kaggle's test.csv to data/ to generate submission.csv")
