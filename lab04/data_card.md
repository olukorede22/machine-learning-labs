# Data card: UCI Adult (Census Income), processed for Lab 4

## Source
- **Dataset:** Adult / Census Income, loaded with `fetch_openml(name='adult', version=2, as_frame=True)` (OpenML copy of the UCI dataset).
- **Original source:** US Census Bureau, 1994 Current Population Survey. Extracted by Barry Becker; donated to UCI by Ronny Kohavi and Barry Becker (1996). Citation: R. Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid", KDD 1996.
- **Task:** predict whether a person earns more than 50K USD per year (income >50K) from 14 demographic and work features.
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the UCI Machine Learning Repository page for the Adult dataset (https://archive.ics.uci.edu/dataset/2/adult). The data may be shared and adapted for any purpose as long as appropriate credit is given.

## Size and composition
- Raw: 48842 rows x 15 columns (14 features + income label).
- Features: 6 numeric (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) and 8 categorical.
- Label balance in raw data: 23.9% earn >50K (the majority class, <=50K, is about 76.1%).
- Missing values (coded `?` or NaN): workclass 5.73%, occupation 5.75%, native-country 1.75%.
- Processed (after the 70/30 stratified split): train 34161 rows (after dropping 28 exact duplicates from train only), test 14653 rows (untouched).
- Derived features: `capital_net` = capital-gain - capital-loss, `hours_per_age` = hours-per-week / (age - 16).

## Collection and labels
Labels come from reported annual income thresholded at 50K USD in 1994 dollars. They are not hand-labelled, so label quality was checked with a duplicate-conflict audit: 5 groups of rows with identical features (all 14) have conflicting labels, and 1029 groups when the census weight `fnlwgt` is ignored. See `results/label_audit.md`.

## Known biases and limitations
- **Representation:** 66.8% of rows are male, 85.5% are White and 89.7% have native-country United-States. Models can perform worse for small groups and are known to produce different error rates by sex and race.
- **Historical bias:** the data reflects 1994 US labour-market inequalities (for example gender and racial income gaps). A model trained on it reproduces them.
- **Outdated and fixed threshold:** 50K USD in 1994 is not comparable with today's incomes.
- **Selection:** records were filtered by Becker (age > 16, adjusted gross income > 100, final weight > 1, hours per week > 0), so this is not the whole population.
- **Missingness is not random:** `?` is concentrated in workclass, occupation and native-country, and imputing it with the mode hides that pattern.
- **Heavy tails:** capital-gain and capital-loss are zero for most people. Clipping at the 1st/99th percentile (learned on train) reduces the influence of extreme values but also changes them.
- **Duplicates and conflicts:** some rows are exact duplicates and some identical feature rows have different labels, which puts a ceiling on achievable accuracy.

## Privacy concerns
- The data is public census microdata with no names or addresses, but it contains sensitive attributes (race, sex, marital status, native-country, income).
- Combinations of age, sex, race, country and occupation can act as quasi-identifiers, so linking with other datasets could re-identify individuals. Do not attempt re-identification.
- Do not use the data to infer sensitive attributes about real people or for decisions about individuals (credit, hiring, housing).

## Reproducibility
- Run `python lab04_pipeline.py` (settings in `config.yaml`, seed 42).
- SHA256 of processed data: train `740a11c1eda9b9ee1a2801875b50961afa8643e4303b109bcd4838b4d4bf40e8`, test `8ca183f456d028bffa64f8442520f4b88cf1d98afff5c4bde6633877553704f7`, combined `2ffbd59d8acef565b682ffcd8ba7d3ab72bd4aa73cea3a884df79832dc383ed4`.
- Libraries, config and fitted cleaning parameters are stored in `results/processed_config.yaml`.
