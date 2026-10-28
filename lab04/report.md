# Lab 4: Data pipeline and quality (UCI Adult)
## How to run

```
pip install -r requirements.txt
python lab04_pipeline.py
```

One command runs ingestion, validation, split, cleaning, feature engineering, the baseline,
the label audit, hashing and the data card. All settings (seed, split, clip quantiles) are in
`config.yaml`. Outputs go to `results/` and `data_card.md`.

## 1. Problem definition

See `problem_definition.md`: predict income >50K from the 14 features; success metric =
accuracy and F1 on the positive class; constraints = no leakage and a reproducible pipeline.

## 2. Ingestion

`fetch_openml(name='adult', version=2, as_frame=True)` is read without modification:
**48,842 rows and 15 columns** (14 features + income), as the manual expects.

## 3. Validation (`results/validation_report.md`)

I checked 35 things: columns present, the declared type of every column, the four required
ranges (age 17-90, hours-per-week 1-99, capital-gain 0-99999, capital-loss 0-4356), allowed
categories (with `?` allowed), the income labels, education vs education-num, and two extra
checks (education-num 1-16, fnlwgt > 0). **Result: 0 violations, 0 rows dropped.**

Missing values (NaN in the OpenML copy, which the null policy treats like `?`): workclass 2,799
(5.7%), occupation 2,809 (5.8%), native-country 857 (1.8%). They are allowed by the schema and
handled by the imputer.

## 4. Cleaning (no leakage)

| Step | Detail |
|---|---|
| Split first | 70/30 stratified on income, seed 42: train 34,189 / test 14,653. >50K share is 23.93% in both |
| Imputer (train only) | mode for categorical (e.g. workclass = Private, occupation = Prof-specialty), median for numeric (e.g. age 37, hours 40) |
| Clipping (train only) | 1st/99th percentile, e.g. age 17-74, capital-gain 0-15,024, capital-loss 0-1,980, hours 8-80 |
| Applied to test | the same train statistics; test: 1,026 values clipped in total, none refit |
| Duplicates (train only) | 24 exact duplicates in raw train, 28 after imputation and clipping; 28 dropped, train = 34,161 rows. Test untouched (14,653 rows) |
| Saved transformer | `results/preprocessor.joblib` (`TrainFittedCleaner`) |

**Why this order prevents leakage:** The test set is meant to stand in for data the model has never seen, so nothing in the pipeline should learn from it. The mode and median used for imputation and the 1st and 99th percentiles used for clipping are all learned from data. If I had computed them before splitting, they would have used test rows too, and information from the test set would have leaked into the training pipeline. That would make the test results look better than they really are. To avoid this, I split the data 70/30 first (34,189 train rows and 14,653 test rows), fitted every statistic on the training set only, and then applied the same values to the test set without refitting. For example, the clip bound for age (17 to 74) comes from the training set alone. Exact duplicates were also dropped from train only, so the test set was left untouched. The derived features are computed row by row, so they
cannot leak. The baseline's scaler and encoder are fitted inside the model pipeline on train only.

## 5. Derived features (LogisticRegression baseline)

`capital_net = capital-gain - capital-loss`, `hours_per_age = hours-per-week / (age - 16)`.

| Features | Accuracy | F1 (>50K) |
|---|---|---|
| Without derived features | 0.8542 | 0.6610 |
| With both derived features | 0.8577 | 0.6694 |

The derived features give a small improvement (+0.0035 accuracy, +0.0084 F1). This is a single
split and a single model, so I would not call it significant. The two derived features give the model a bit more information. hours_per_age is a ratio of hours worked to years since age 16. A logistic regression is linear, so it cannot build a ratio from hours-per-week and age on its own, and this is probably where most of the gain comes from. capital_net combines capital gain and capital loss, which are both zero for most people, into one net value. Adding both features raised accuracy from 0.8542 to 0.8577 and F1 from 0.6610 to 0.6694. This is a small improvement from a single split, so it suggests the features help a little and does not prove it. I did not test each feature separately.

## 6. Label-quality audit (`results/label_audit.md`)

- **Duplicate-conflict audit (all 48,842 rows):** 52 exact duplicate rows. With all 14 features,
  **5 groups of identical rows have conflicting income labels**. Ignoring `fnlwgt`, **1,029
  groups** conflict. These show a ceiling on achievable accuracy: some people with the same
  recorded features have different incomes. The 5 strict conflicts are the likely label noise.
- **50 random rows vs the data dictionary** (`results/manual_check_sample_50.csv`, seed 42): the
  automated check flagged **0 of 50** rows. Manual check:   50 rows were checked by hand against the data dictionary (age, hours, capital-gain and capital-loss ranges, education vs education-num, relationship vs sex, and category spellings), and 0 impossible values were found. All 50 rows are marked Y in `manual_check_ok`.

## 7. Hash, config and data card

SHA256 of the processed data (identical on three runs, on Python 3.11 and 3.14):

- `processed_train.csv`: `740a11c1eda9b9ee1a2801875b50961afa8643e4303b109bcd4838b4d4bf40e8`
- `processed_test.csv`: `8ca183f456d028bffa64f8442520f4b88cf1d98afff5c4bde6633877553704f7`
- combined: `2ffbd59d8acef565b682ffcd8ba7d3ab72bd4aa73cea3a884df79832dc383ed4`

The config, library versions, fitted fill values and clip bounds and these hashes are in
`results/processed_config.yaml`. The data card is `data_card.md` (source, license, size, known
biases, privacy concerns).    The dataset is licensed under CC BY 4.0, as confirmed on the UCI repository page.

## Limitations

- The data is from 1994 and the 50K threshold is not comparable with today's incomes.
- The data is imbalanced by sex, race and native-country; see `data_card.md`.
- Imputing `?` with the mode hides the fact that missingness is concentrated in a few columns.
- Results come from one split and one seed; no confidence intervals were computed.
