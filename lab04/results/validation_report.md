# Validation report

Rows checked: 48842 (rows dropped by validation: 0). Raw shape: (48842, 15).
Checks with violations: **0** of 35.

Policy: '?' (or NaN) means missing and is an allowed category. Violations are reported, rows are never dropped.

## Checks

| check | column | status | violations | detail |
|---|---|---|---|---|
| columns present | (all) | PASS | 0 | missing=[] extra=[]; 15 columns |
| type int | age | PASS | 0 | observed dtype int64; integer dtype |
| type category | workclass | PASS | 0 | observed dtype category |
| type int | fnlwgt | PASS | 0 | observed dtype int64; integer dtype |
| type category | education | PASS | 0 | observed dtype category |
| type int | education-num | PASS | 0 | observed dtype int64; integer dtype |
| type category | marital-status | PASS | 0 | observed dtype category |
| type category | occupation | PASS | 0 | observed dtype category |
| type category | relationship | PASS | 0 | observed dtype category |
| type category | race | PASS | 0 | observed dtype category |
| type category | sex | PASS | 0 | observed dtype category |
| type int | capital-gain | PASS | 0 | observed dtype int64; integer dtype |
| type int | capital-loss | PASS | 0 | observed dtype int64; integer dtype |
| type int | hours-per-week | PASS | 0 | observed dtype int64; integer dtype |
| type category | native-country | PASS | 0 | observed dtype category |
| type category | income | PASS | 0 | observed dtype category |
| range | age | PASS | 0 | allowed [17, 90], observed [17, 90] |
| range | hours-per-week | PASS | 0 | allowed [1, 99], observed [1, 99] |
| range | capital-gain | PASS | 0 | allowed [0, 99999], observed [0, 99999] |
| range | capital-loss | PASS | 0 | allowed [0, 4356], observed [0, 4356] |
| range (extra) | education-num | PASS | 0 | allowed [1, 16], observed [1, 16] |
| range (extra) | fnlwgt | PASS | 0 | fnlwgt must be > 0 |
| allowed categories | workclass | PASS | 0 | unexpected: none |
| allowed categories | education | PASS | 0 | unexpected: none |
| allowed categories | marital-status | PASS | 0 | unexpected: none |
| allowed categories | occupation | PASS | 0 | unexpected: none |
| allowed categories | relationship | PASS | 0 | unexpected: none |
| allowed categories | race | PASS | 0 | unexpected: none |
| allowed categories | sex | PASS | 0 | unexpected: none |
| allowed categories | native-country | PASS | 0 | unexpected: none |
| allowed categories | income | PASS | 0 | unexpected: none |
| education <-> education-num | education | PASS | 0 | each education level has one number |
| missing values | workclass | ALLOWED | 2799 | 5.73% missing ('?' policy: allowed, imputed later) |
| missing values | occupation | ALLOWED | 2809 | 5.75% missing ('?' policy: allowed, imputed later) |
| missing values | native-country | ALLOWED | 857 | 1.75% missing ('?' policy: allowed, imputed later) |

## Missing values per column

| column | NaN | '?' | missing_total | pct |
|---|---|---|---|---|
| age | 0 | 0 | 0 | 0.0 |
| workclass | 2799 | 0 | 2799 | 5.73 |
| fnlwgt | 0 | 0 | 0 | 0.0 |
| education | 0 | 0 | 0 | 0.0 |
| education-num | 0 | 0 | 0 | 0.0 |
| marital-status | 0 | 0 | 0 | 0.0 |
| occupation | 2809 | 0 | 2809 | 5.75 |
| relationship | 0 | 0 | 0 | 0.0 |
| race | 0 | 0 | 0 | 0.0 |
| sex | 0 | 0 | 0 | 0.0 |
| capital-gain | 0 | 0 | 0 | 0.0 |
| capital-loss | 0 | 0 | 0 | 0.0 |
| hours-per-week | 0 | 0 | 0 | 0.0 |
| native-country | 857 | 0 | 857 | 1.75 |
| income | 0 | 0 | 0 | 0.0 |
