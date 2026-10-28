# Label-quality audit

## 1. Duplicate-conflict audit (full raw dataset)

- Exact duplicate rows (features + label): **52**
- Using all 14 features: 54 duplicate groups (111 rows); **5 groups with conflicting income labels** (10 rows).
- Without `fnlwgt` (a census sampling weight, 13 features): 3771 duplicate groups; **1029 conflicting groups** (3873 rows).

Conflicting rows are listed in `duplicate_conflicts.csv`.

## 2. 50 random rows vs the data dictionary

- Random sample of 50 rows (seed 42) saved to `manual_check_sample_50.csv`.
- Rows flagged by the automated impossible-value checks: **0** of 50.
- Manual check: 50 rows were checked by hand against the data dictionary and 0 impossible values were found. All 50 rows are marked Y in `manual_check_ok`.