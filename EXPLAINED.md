# The Project Explained Simply

This walks through `house_prices.py` in plain English. Each section matches a numbered section in the script and in its printed output.

---

## The big picture

We have 1,460 houses that already sold in Ames, Iowa. For each one we know 79 facts (size, number of bathrooms, neighborhood, kitchen quality and so on) **and** the price it sold for.

The goal: teach the computer the pattern between the facts and the price, so it can guess the price of a house it has never seen.

This is called **regression**: predicting a number (a price), rather than a category (like "spam / not spam").

---

## Step 1: Load the data

We read `train.csv` into a table: one row per house, one column per fact. That gives 1,460 rows and 81 columns (79 facts, plus an ID and the price).

---

## Step 2: Fix the shape of the price

**The problem:** most houses cost $100k–$250k, but a few cost $500k or more. Those expensive houses pull the model toward them, like one very tall person skewing a class's average height.

**The fix:** we predict the **log** of the price instead of the price itself. The log squashes big numbers more than small ones, so the prices spread out evenly.

**A bonus:** with logs, the model cares about *percentage* mistakes. Being $20k off on a $100k house (20%) counts as worse than being $20k off on a $500k house (4%). That's fair, and it's also how Kaggle scores this competition.

At the end we undo the log (`expm1`) to get dollars back.

---

## Step 3: Remove two strange houses

Two houses were huge (over 4,000 sq ft) but sold very cheaply. They were unusual sales, not normal market prices. If we kept them, the model would learn "big houses can be cheap", which is wrong for almost every other house.

The person who created this dataset recommends removing them, so we did.

---

## Step 4: Missing data (the most important step)

19 of the columns have blank cells. 7 of them are blank more than 15% of the time.

| Column | % blank |
|---|---|
| PoolQC (pool quality) | 99.6% |
| MiscFeature (shed, tennis court…) | 96.3% |
| Alley | 93.8% |
| Fence | 80.7% |
| MasVnrType (brick/stone trim) | 59.8% |
| FireplaceQu (fireplace quality) | 47.3% |
| LotFrontage (feet of street touching the lot) | 17.8% |

**The key insight:** most of these blanks aren't "unknown". They mean **the house doesn't have that thing.**

- PoolQC is blank because 99.6% of houses **have no pool**.
- GarageType is blank because those houses **have no garage**.

So we didn't throw these columns away, and we didn't guess a value. We:

1. **Filled them with "None"** for text columns (e.g. pool quality = "None").
2. **Filled them with 0** for the matching number columns (e.g. garage area = 0).

Why not just delete them? Because "no garage" tells you something real: houses without garages usually sell for less. Deleting the column would throw that information away.

**The one truly unknown column is LotFrontage.** Every house touches a street, so a blank there really means "nobody recorded it". We filled it with the **typical value for that neighborhood**, because houses on the same streets tend to have similar lots.

**The few leftover blanks** (e.g. one house with no Electrical value) get the most common value.

### How to say it in an interview

> "Most of the missing values weren't really missing. The data dictionary said blank means the house doesn't have that feature, like no pool or no garage, so I filled those with 'None' or 0, because not having a garage is useful information. Only LotFrontage was truly unknown, and I filled it with the neighborhood median. All the filling that learns from the data happens inside cross-validation, so it never peeks at the test data."

---

## Step 5: Create better features

The raw data splits information up in ways that don't match how buyers think. We combine and reshape it.

| New feature | How it's made | Why it helps |
|---|---|---|
| **TotalSF** | basement + 1st floor + 2nd floor | Buyers think "how big is the house", not floor by floor |
| **TotalBath** | full baths + ½ × half baths (incl. basement) | Same idea for bathrooms |
| **TotalPorchSF** | all porch and deck areas added up | |
| **HouseAge** | year sold − year built | "30 years old" means more than "built in 1978" |
| **RemodAge**, **IsRemodeled** | years since remodel, and whether it ever was | Recently updated houses sell for more |
| **QualxSF** | quality rating × total size | A big house is worth more if it's *also* well built |
| **HasGarage, HasPool…** | 1 if it has one, 0 if not | Keeps the "doesn't have it" information from Step 4 |

**Turning ratings into numbers:** quality is written as Po, Fa, TA, Gd, Ex (poor → excellent). We converted them to 1–5 so the model knows Excellent > Good > Average.

**Fixing lopsided columns:** 22 number columns (like lot size) are lopsided in the same way price was. We log-transform them too.

**Dropping useless columns:** "Utilities" is the same for 1,459 of 1,460 houses, so it can't tell houses apart. We removed it.

After this step we have 88 features.

---

## Step 6: Multicollinearity (features that say the same thing)

Some features are almost duplicates:
- **GarageCars** (how many cars fit) and **GarageArea** (square feet). A bigger garage fits more cars.
- **YearBuilt** and **HouseAge**. One is just the other subtracted from the sale year.

**Why it's a problem:** imagine two people pushing a car, and you're asked how much each one pushed. You can't tell them apart. A plain regression has the same trouble: it can't decide how much credit each duplicate deserves, so its answers swing wildly.

**Our fix:** we don't delete features by hand. We use models that handle it automatically:
- **Ridge** shares credit between similar features fairly.
- **Lasso** picks one from each group and sets the others to zero. It kept 113 of 268 features and dropped 155.

The script prints all pairs that are more than 80% correlated so you can see them.

---

## Step 7: Train models and test them fairly

### The models

| Model | Simple description |
|---|---|
| **Ridge** | A straight-line formula (price = a × size + b × quality + …) that's kept from going extreme |
| **Lasso** | Same, but it can switch features off completely |
| **ElasticNet** | A mix of Ridge and Lasso |
| **Gradient Boosting** | Builds hundreds of small decision trees ("if quality > 7 and size > 2000 then…"); each new tree fixes the mistakes of the ones before it |

### Cross-validation: how we test fairly

You can't grade a student with the exact questions they studied, so we hold some houses back.

**5-fold cross-validation:**
1. Split the houses into 5 groups.
2. Train on 4 groups and test on the 5th.
3. Repeat 5 times so every group gets a turn as the test.
4. Average the 5 scores.

Every house gets predicted by a model that **never saw it**. That's an honest estimate of how the model will do on new houses.

**No cheating ("leakage"):** steps that learn from the data, like "what's the typical lot frontage in this neighborhood?", are redone inside each round using only the training groups. Otherwise the test houses would sneak into training. A scikit-learn `Pipeline` handles this automatically.

### Results

The error (RMSE on log price) is roughly the typical percentage the guess is off. 0.11 ≈ 11%.

| Model | Error |
|---|---|
| Ridge | 0.1141 |
| Lasso | 0.1117 |
| ElasticNet | 0.1119 |
| Gradient Boosting | 0.1141 |
| **Blend** | **0.1090** |

### Why blending wins

The straight-line models and the tree model make **different kinds of mistakes**. When one guesses too high, the other is often too low, so averaging them cancels some errors out. It's like asking several friends to guess and taking the average.

In dollars, the blend is off by about **$13,200** on average, for houses with a median price of $163,000.

---

## Step 8: What drives the price?

We train every model one last time on all the houses, then look at which features Lasso relies on most.

| Feature | Effect |
|---|---|
| Commercial zoning (`MSZoning_C`) | Big **drop**: houses in commercial areas sell for much less |
| Living area (`GrLivArea`) | Bigger = pricier |
| Overall quality | Better built = pricier |
| Neighborhoods like Crawford, Stone Brook, Northridge | Location premium |
| House age | Older = cheaper |
| New sale (`SaleType_New`) | Brand-new homes sell higher |
| Abnormal sale (e.g. foreclosure) | Sells lower |

It matches common sense: **location, size, quality, age**.

If `data/test.csv` exists, the script also predicts prices for those houses and saves `submission.csv` for Kaggle.

---

## Glossary

| Term | Meaning |
|---|---|
| **Feature** | One fact about a house (a column) |
| **Target** | What we predict (SalePrice) |
| **Regression** | Predicting a number |
| **Log transform** | Squashes big numbers so values spread out evenly |
| **Imputation** | Filling in blank values |
| **Multicollinearity** | Features that say almost the same thing |
| **Regularization** | Keeping a model from going extreme (what Ridge and Lasso do) |
| **Cross-validation** | Testing on data the model didn't train on, in rotation |
| **Data leakage** | Accidentally letting test data influence training, which makes scores look better than they really are |
| **RMSE** | Root mean squared error: the typical size of the model's mistakes |
| **Blend / ensemble** | Averaging several models' predictions |
