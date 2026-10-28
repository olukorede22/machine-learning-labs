# Problem definition (Lab 4, task 1)

**Task.** Predict whether a person's annual income exceeds 50K USD using the UCI Adult
(Census Income) dataset, loaded with `fetch_openml(name='adult', version=2, as_frame=True)`.
This is a binary classification problem on 48,842 census records from 1994.

**Inputs.** The 14 features of each record: six numeric (age, fnlwgt, education-num,
capital-gain, capital-loss, hours-per-week) and eight categorical (workclass, education,
marital-status, occupation, relationship, race, sex, native-country). Missing values are coded
as `?`. **Output.** `income`, with classes `<=50K` and `>50K`. About a quarter of the people
are in the positive class `>50K`, so the classes are imbalanced.

**Success metrics.** Accuracy and F1 on the positive class (`>50K`). Accuracy alone is
misleading here because always predicting `<=50K` already scores about 76%, so F1 shows
whether the model actually finds the high earners. Both are measured on a held-out 30% test set
that is never used to fit any step of the pipeline.

**Constraints.**
1. *No leakage.* The 70/30 stratified split is made before any cleaning. The imputer
   (mode for categorical, median for numeric) and the 1st/99th percentile clip bounds are
   fitted on the training set only and then applied unchanged to the test set. Exact duplicates
   are dropped from train only; the test set keeps all its rows. Derived features are computed
   row by row, so they cannot carry information from other rows.
2. *Reproducible pipeline.* One command (`python lab04_pipeline.py`) runs ingestion,
   validation, cleaning, feature engineering, the baseline, the label audit and the data card.
   The random seed and every setting live in `config.yaml`, and the processed data is identified
   by a SHA256 hash.
3. *Data quality is reported, not hidden.* Schema, range, missing-value and label-quality
   violations are reported and rows are never silently dropped during validation.

**Out of scope.** Model tuning and fairness mitigation. The baseline is a plain logistic
regression used only to check that the pipeline and the derived features work.
