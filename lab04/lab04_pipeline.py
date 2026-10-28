"""
Lab 4: Data pipeline and quality (UCI Adult).  ONE COMMAND:

    python lab04_pipeline.py                # real data via fetch_openml (needs internet once)
    python lab04_pipeline.py --synthetic    # fake Adult-like data, only to test the code

Stages (manual task numbers in brackets)
    [2] ingest     read raw data unmodified, log 48842 x 15
    [3] validate   schema, types, ranges, '?' policy, allowed categories (never drops rows)
    [4] split      70/30 stratified BEFORE cleaning
        clean      imputer + 1st/99th percentile clipping fitted on TRAIN only,
                   exact duplicates dropped from TRAIN only, test rows kept
    [5] features   capital_net, hours_per_age; LogisticRegression with / without
    [6] audit      duplicate-conflict audit + 50 random rows vs the data dictionary
    [7] hash       SHA256 of processed data, config + environment in a YAML, data_card.md

Outputs: ./results/*, ./data_card.md
"""
import argparse
import hashlib
import json
import logging
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from lab04_cleaner import MISSING, TrainFittedCleaner  # noqa: E402
log = logging.getLogger("lab04")

# --------------------------------------------------------------------------
# Schema (task 3) and data dictionary
# --------------------------------------------------------------------------
TARGET_RAW, TARGET, LABEL = "class", "income", "income_gt50k"
SCHEMA = {
    "age": "int", "workclass": "category", "fnlwgt": "int", "education": "category",
    "education-num": "int", "marital-status": "category", "occupation": "category",
    "relationship": "category", "race": "category", "sex": "category",
    "capital-gain": "int", "capital-loss": "int", "hours-per-week": "int",
    "native-country": "category", "income": "category",
}
FEATURES = [c for c in SCHEMA if c != TARGET]
NUMERIC = [c for c in FEATURES if SCHEMA[c] == "int"]
CATEGORICAL = [c for c in FEATURES if SCHEMA[c] == "category"]
RANGES = {"age": (17, 90), "hours-per-week": (1, 99),
          "capital-gain": (0, 99999), "capital-loss": (0, 4356)}
EXTRA_RANGES = {"education-num": (1, 16)}  # extra check from the data dictionary
EDU_NUM = {"Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5, "10th": 6,
           "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10, "Assoc-voc": 11,
           "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14, "Prof-school": 15,
           "Doctorate": 16}
ALLOWED = {
    "workclass": {"Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
                  "State-gov", "Without-pay", "Never-worked"},
    "education": set(EDU_NUM),
    "marital-status": {"Married-civ-spouse", "Divorced", "Never-married", "Separated",
                       "Widowed", "Married-spouse-absent", "Married-AF-spouse"},
    "occupation": {"Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
                   "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
                   "Farming-fishing", "Transport-moving", "Priv-house-serv",
                   "Protective-serv", "Armed-Forces"},
    "relationship": {"Wife", "Own-child", "Husband", "Not-in-family", "Other-relative",
                     "Unmarried"},
    "race": {"White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"},
    "sex": {"Female", "Male"},
    "native-country": {"United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
                       "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece",
                       "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy",
                       "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland",
                       "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti",
                       "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand",
                       "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong",
                       "Holand-Netherlands"},
    "income": {"<=50K", ">50K"},
}
for _k in ALLOWED:  # null policy: '?' is an allowed category everywhere except the label
    if _k != "income":
        ALLOWED[_k] = ALLOWED[_k] | {MISSING}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def md_table(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(lines)


def to_py(o):
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label_from(series):
    return series.astype(str).str.strip().str.startswith(">50K").astype(int)


# --------------------------------------------------------------------------
# Task 2: ingestion (raw data, unmodified)
# --------------------------------------------------------------------------
def make_synthetic(n=4000, seed=0):
    """Adult-like fake data, ONLY to test that the code runs. Not real results."""
    rng = np.random.default_rng(seed)
    edu = rng.choice(list(EDU_NUM), n)
    d = pd.DataFrame({
        "age": rng.integers(17, 91, n),
        "workclass": rng.choice(sorted(ALLOWED["workclass"] - {MISSING}), n),
        "fnlwgt": rng.integers(12285, 1490400, n),
        "education": edu,
        "education-num": [EDU_NUM[e] for e in edu],
        "marital-status": rng.choice(sorted(ALLOWED["marital-status"] - {MISSING}), n),
        "occupation": rng.choice(sorted(ALLOWED["occupation"] - {MISSING}), n),
        "relationship": rng.choice(sorted(ALLOWED["relationship"] - {MISSING}), n),
        "race": rng.choice(sorted(ALLOWED["race"] - {MISSING}), n),
        "sex": rng.choice(["Male", "Female"], n, p=[.67, .33]),
        "capital-gain": np.where(rng.random(n) < .08, rng.integers(100, 99999, n), 0),
        "capital-loss": np.where(rng.random(n) < .05, rng.integers(100, 4356, n), 0),
        "hours-per-week": rng.integers(1, 99, n),
        "native-country": rng.choice(["United-States", "Mexico", "India", "Germany"], n,
                                     p=[.9, .05, .03, .02]),
    })
    score = (d["education-num"] - 9) * .5 + (d["age"] - 40) * .03 + (d["hours-per-week"] - 40) * .04
    d[TARGET_RAW] = np.where(score + rng.normal(0, 2, n) > 1.5, ">50K", "<=50K")
    for c in ["workclass", "occupation", "native-country"]:
        d.loc[rng.random(n) < .05, c] = np.nan
    d = pd.concat([d, d.sample(25, random_state=1)], ignore_index=True)  # exact duplicates
    conf = d.sample(15, random_state=2).copy()  # feature duplicates with opposite label
    conf[TARGET_RAW] = np.where(conf[TARGET_RAW] == ">50K", "<=50K", ">50K")
    d = pd.concat([d, conf], ignore_index=True)
    for c in CATEGORICAL + [TARGET_RAW]:
        if c in d:
            d[c] = d[c].astype("category")
    return d


def ingest(cfg, synthetic):
    if synthetic:
        raw = make_synthetic(seed=cfg["seed"])
        log.warning("SYNTHETIC DATA: results are meaningless, this is a code test only")
    else:
        from sklearn.datasets import fetch_openml
        raw = fetch_openml(name=cfg["dataset"]["name"], version=cfg["dataset"]["version"],
                           as_frame=True, data_home=str(HERE / "data_cache")).frame
    rows, cols = raw.shape
    log.info("[ingest] rows=%d columns=%d", rows, cols)
    exp = (cfg["dataset"]["expected_rows"], cfg["dataset"]["expected_cols"])
    if not synthetic and (rows, cols) != exp:
        log.warning("[ingest] expected %s but got %s", exp, (rows, cols))
    raw_hash = hashlib.sha256(raw.to_csv(index=False, lineterminator="\n").encode()).hexdigest()
    log.info("[ingest] raw data sha256 (csv view) = %s", raw_hash)
    return raw, raw_hash


# --------------------------------------------------------------------------
# Task 3: validation (reports, never drops)
# --------------------------------------------------------------------------
def validate(raw):
    d = raw.rename(columns={TARGET_RAW: TARGET})
    rows = []

    def add(check, column, n_bad, detail="", status=None):
        rows.append({"check": check, "column": column,
                     "status": status or ("PASS" if n_bad == 0 else "VIOLATION"),
                     "violations": int(n_bad), "detail": detail})

    missing_cols = [c for c in SCHEMA if c not in d.columns]
    extra_cols = [c for c in d.columns if c not in SCHEMA]
    add("columns present", "(all)", len(missing_cols) + len(extra_cols),
        f"missing={missing_cols} extra={extra_cols}; {d.shape[1]} columns")
    d = d[[c for c in SCHEMA if c in d.columns]]

    for col, kind in SCHEMA.items():
        s = d[col]
        if kind == "int":
            if pd.api.types.is_integer_dtype(s):
                bad, note = 0, "integer dtype"
            elif pd.api.types.is_float_dtype(s):
                bad = int(s.isna().sum() + ((s.dropna() % 1) != 0).sum())
                note = "float dtype, " + ("all values integer-valued" if bad == 0 else "non-integer values")
            else:
                bad, note = len(s), "non-numeric dtype"
            add("type int", col, bad, f"observed dtype {s.dtype}; {note}")
        else:
            ok = isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s) \
                or pd.api.types.is_string_dtype(s)
            add("type category", col, 0 if ok else len(s), f"observed dtype {s.dtype}")

    for col, (lo, hi) in {**RANGES, **EXTRA_RANGES}.items():
        num = pd.to_numeric(d[col], errors="coerce")
        bad = ((num < lo) | (num > hi) | num.isna()).sum()
        tag = "range" if col in RANGES else "range (extra)"
        add(tag, col, bad, f"allowed [{lo}, {hi}], observed [{num.min():g}, {num.max():g}]")
    add("range (extra)", "fnlwgt", (pd.to_numeric(d["fnlwgt"], errors="coerce") <= 0).sum(),
        "fnlwgt must be > 0")

    for col in CATEGORICAL + [TARGET]:
        vals = d[col].dropna().astype(str).str.strip()
        unexpected = sorted(set(vals) - ALLOWED[col])
        add("allowed categories", col, vals.isin(unexpected).sum(),
            "unexpected: " + (", ".join(unexpected[:8]) or "none"))

    mism = (d["education"].astype(str).map(EDU_NUM) != pd.to_numeric(d["education-num"])).sum()
    add("education <-> education-num", "education", mism, "each education level has one number")

    miss = []
    for col in d.columns:
        n_nan = int(d[col].isna().sum())
        n_q = int((d[col].astype(str).str.strip() == MISSING).sum())
        miss.append({"column": col, "NaN": n_nan, "'?'": n_q, "missing_total": n_nan + n_q,
                     "pct": round(100 * (n_nan + n_q) / len(d), 2)})
    miss = pd.DataFrame(miss)
    for _, r in miss[miss.missing_total > 0].iterrows():
        add("missing values", r["column"], r["missing_total"],
            f"{r['pct']}% missing ('?' policy: allowed, imputed later)", status="ALLOWED")
    return pd.DataFrame(rows), miss, len(d)


def write_validation(checks, miss, n_rows, raw_shape, res):
    n_viol = int((checks.status == "VIOLATION").sum())
    txt = ["# Validation report", "",
           f"Rows checked: {n_rows} (rows dropped by validation: 0). Raw shape: {raw_shape}.",
           f"Checks with violations: **{n_viol}** of {len(checks)}.", "",
           "Policy: '?' (or NaN) means missing and is an allowed category. "
           "Violations are reported, rows are never dropped.", "",
           "## Checks", "", md_table(checks), "", "## Missing values per column", "",
           md_table(miss), ""]
    (res / "validation_report.md").write_text("\n".join(txt), encoding="utf-8")
    checks.to_csv(res / "validation_report.csv", index=False)
    log.info("[validate] %d checks, %d with violations (see validation_report.md)",
             len(checks), n_viol)
    for _, r in checks[checks.status == "VIOLATION"].iterrows():
        log.warning("[validate] VIOLATION %s | %s | %d rows | %s",
                    r["check"], r["column"], r["violations"], r["detail"])


# --------------------------------------------------------------------------
# Task 4: cleaning (everything learned from TRAIN only; the fitted class lives in
# lab04_cleaner.py so that results/preprocessor.joblib can be loaded from any script)
# --------------------------------------------------------------------------
def add_derived(df):
    out = df.copy()
    out["capital_net"] = out["capital-gain"] - out["capital-loss"]
    # age is >= 17 in Adult so age - 16 >= 1; the max() only guards invalid ages
    out["hours_per_age"] = out["hours-per-week"] / np.maximum(out["age"] - 16, 1)
    return out


# --------------------------------------------------------------------------
# Task 5: baseline
# --------------------------------------------------------------------------
def fit_eval(train, test, num_cols, cfg):
    pre = ColumnTransformer([("num", StandardScaler(), num_cols),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    model = Pipeline([("pre", pre),
                      ("clf", LogisticRegression(max_iter=cfg["baseline"]["max_iter"]))])
    cols = num_cols + CATEGORICAL
    model.fit(train[cols], train[LABEL])  # scaler/encoder are fitted on train inside the pipeline
    pred = model.predict(test[cols])
    return accuracy_score(test[LABEL], pred), f1_score(test[LABEL], pred)


# --------------------------------------------------------------------------
# Task 6: label-quality audit
# --------------------------------------------------------------------------
def row_issues(r):
    issues = []
    for col, (lo, hi) in {**RANGES, **EXTRA_RANGES}.items():
        v = pd.to_numeric(r[col], errors="coerce")
        if pd.isna(v) or v < lo or v > hi:
            issues.append(f"{col}={r[col]} outside [{lo},{hi}]")
    if not pd.to_numeric(r["fnlwgt"], errors="coerce") > 0:
        issues.append("fnlwgt <= 0")
    for col in CATEGORICAL + [TARGET]:
        v = r[col]
        if not pd.isna(v) and str(v).strip() not in ALLOWED[col]:
            issues.append(f"{col}='{v}' not in data dictionary")
    if EDU_NUM.get(str(r["education"])) != r["education-num"]:
        issues.append("education != education-num")
    if r["relationship"] == "Husband" and r["sex"] == "Female":
        issues.append("relationship=Husband but sex=Female")
    if r["relationship"] == "Wife" and r["sex"] == "Male":
        issues.append("relationship=Wife but sex=Male")
    return "; ".join(issues)


def label_audit(raw, cfg, res):
    d = raw.rename(columns={TARGET_RAW: TARGET}).copy()
    d["_label"] = d[TARGET].astype(str).str.strip()
    out = {"exact_duplicate_rows": int(d[FEATURES + ["_label"]].astype(str).duplicated().sum())}
    conflict_frames = []
    for name, cols in {"all_14_features": FEATURES,
                       "13_features_without_fnlwgt": [c for c in FEATURES if c != "fnlwgt"]}.items():
        key = pd.util.hash_pandas_object(d[cols].astype(str), index=False)
        g = d.groupby(key)["_label"].agg(["size", "nunique"])
        dup, conf = g[g["size"] > 1], g[g["nunique"] > 1]
        rows_conf = d[key.isin(conf.index)].copy()
        rows_conf.insert(0, "audit_variant", name)
        conflict_frames.append(rows_conf.drop(columns="_label"))
        out[name] = {"duplicate_groups": len(dup), "rows_in_duplicate_groups": int(dup["size"].sum()),
                     "conflicting_groups": len(conf), "rows_in_conflicting_groups": int(conf["size"].sum())}
    pd.concat(conflict_frames).to_csv(res / "duplicate_conflicts.csv", index=False)

    sample = d.drop(columns="_label").sample(n=cfg["label_audit"]["n_random_rows"],
                                             random_state=cfg["seed"])
    sample.insert(0, "row_id", sample.index)
    sample["automated_issues"] = [row_issues(r) for r in sample.drop(columns="row_id").to_dict("records")]
    sample["manual_check_ok (Y/N)"] = ""
    sample["notes"] = ""
    sample.to_csv(res / "manual_check_sample_50.csv", index=False)
    out["sample_rows"] = len(sample)
    out["sample_rows_with_automated_issues"] = int((sample["automated_issues"] != "").sum())

    a, b = out["all_14_features"], out["13_features_without_fnlwgt"]
    txt = ["# Label-quality audit", "",
           "## 1. Duplicate-conflict audit (full raw dataset)", "",
           f"- Exact duplicate rows (features + label): **{out['exact_duplicate_rows']}**",
           f"- Using all 14 features: {a['duplicate_groups']} duplicate groups "
           f"({a['rows_in_duplicate_groups']} rows); **{a['conflicting_groups']} groups with "
           f"conflicting income labels** ({a['rows_in_conflicting_groups']} rows).",
           f"- Without `fnlwgt` (a census sampling weight, 13 features): {b['duplicate_groups']} "
           f"duplicate groups; **{b['conflicting_groups']} conflicting groups** "
           f"({b['rows_in_conflicting_groups']} rows).", "",
           "Conflicting rows are listed in `duplicate_conflicts.csv`.", "",
           "## 2. 50 random rows vs the data dictionary", "",
           f"- Random sample of {out['sample_rows']} rows (seed {cfg['seed']}) saved to "
           "`manual_check_sample_50.csv`.",
           f"- Rows flagged by the automated impossible-value checks: "
           f"**{out['sample_rows_with_automated_issues']}** of {out['sample_rows']}.",
           "- Manual step: open the CSV, compare each row with the data dictionary, fill in the "
           "`manual_check_ok` column and write the final count in your report.", ""]
    (res / "label_audit.md").write_text("\n".join(txt), encoding="utf-8")
    log.info("[audit] exact dups=%d; conflicting groups (14 feats)=%d, (13 feats)=%d; "
             "sample rows with issues=%d/%d", out["exact_duplicate_rows"], a["conflicting_groups"],
             b["conflicting_groups"], out["sample_rows_with_automated_issues"], out["sample_rows"])
    return out


# --------------------------------------------------------------------------
# Task 7: hash, manifest, data card
# --------------------------------------------------------------------------
def write_data_card(stats, path):
    s = stats
    txt = f"""# Data card: UCI Adult (Census Income), processed for Lab 4

## Source
- **Dataset:** Adult / Census Income, loaded with `fetch_openml(name='adult', version=2, as_frame=True)` (OpenML copy of the UCI dataset).
- **Original source:** US Census Bureau, 1994 Current Population Survey. Extracted by Barry Becker; donated to UCI by Ronny Kohavi and Barry Becker (1996). Citation: R. Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid", KDD 1996.
- **Task:** predict whether a person earns more than 50K USD per year (income >50K) from 14 demographic and work features.
- **License:** UCI lists the Adult dataset under CC BY 4.0 (attribution required). Verify this on the dataset's UCI page before you submit.

## Size and composition
- Raw: {s['raw_rows']} rows x {s['raw_cols']} columns (14 features + income label).
- Features: 6 numeric (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) and 8 categorical.
- Label balance in raw data: {s['pos_rate']:.1f}% earn >50K (the majority class, <=50K, is about {100 - s['pos_rate']:.1f}%).
- Missing values (coded `?` or NaN): workclass {s['miss_workclass']}%, occupation {s['miss_occupation']}%, native-country {s['miss_native']}%.
- Processed (after the 70/30 stratified split): train {s['n_train']} rows (after dropping {s['n_dups']} exact duplicates from train only), test {s['n_test']} rows (untouched).
- Derived features: `capital_net` = capital-gain - capital-loss, `hours_per_age` = hours-per-week / (age - 16).

## Collection and labels
Labels come from reported annual income thresholded at 50K USD in 1994 dollars. They are not hand-labelled, so label quality was checked with a duplicate-conflict audit: {s['conflict_14']} groups of rows with identical features (all 14) have conflicting labels, and {s['conflict_13']} groups when the census weight `fnlwgt` is ignored. See `results/label_audit.md`.

## Known biases and limitations
- **Representation:** {s['pct_male']:.1f}% of rows are male, {s['pct_white']:.1f}% are White and {s['pct_us']:.1f}% have native-country United-States. Models can perform worse for small groups and are known to produce different error rates by sex and race.
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
- Run `python lab04_pipeline.py` (settings in `config.yaml`, seed {s['seed']}).
- SHA256 of processed data: train `{s['sha_train']}`, test `{s['sha_test']}`, combined `{s['sha_combined']}`.
- Libraries, config and fitted cleaning parameters are stored in `results/processed_config.yaml`.
"""
    path.write_text(txt, encoding="utf-8")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config.yaml"))
    ap.add_argument("--synthetic", action="store_true", help="fake data, code test only")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    res = HERE / cfg["output_dir"]
    res.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(res / "pipeline.log", mode="w", encoding="utf-8")])
    seed = cfg["seed"]
    np.random.seed(seed)

    # [2] ingest
    raw, raw_hash = ingest(cfg, args.synthetic)

    # [3] validate
    checks, miss, n_checked = validate(raw)
    write_validation(checks, miss, n_checked, raw.shape, res)

    # [4] split BEFORE cleaning
    df = raw.rename(columns={TARGET_RAW: TARGET})
    X, y = df[FEATURES], label_from(df[TARGET])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg["split"]["test_size"],
                                              stratify=y, random_state=seed)
    log.info("[split] train=%d test=%d | >50K share train=%.4f test=%.4f",
             len(X_tr), len(X_te), y_tr.mean(), y_te.mean())

    raw_train_dups = int(X_tr.assign(**{LABEL: y_tr}).astype(str).duplicated().sum())
    cleaner = TrainFittedCleaner(NUMERIC, CATEGORICAL, cfg["cleaning"]["clip_lower_quantile"],
                                 cfg["cleaning"]["clip_upper_quantile"],
                                 cfg["cleaning"]["missing_token"]).fit(X_tr)
    Xtr_c, rep_tr = cleaner.transform_with_report(X_tr)
    Xte_c, rep_te = cleaner.transform_with_report(X_te)
    log.info("[clean] fill values (from train): %s", cleaner.fill_values_)
    log.info("[clean] clip bounds (from train): %s", cleaner.clip_bounds_)
    log.info("[clean] train: imputed %s, clipped %s", rep_tr["imputed"], rep_tr["clipped"])
    log.info("[clean] test:  imputed %s, clipped %s (train statistics only)",
             rep_te["imputed"], rep_te["clipped"])

    train = Xtr_c.assign(**{LABEL: y_tr})
    test = Xte_c.assign(**{LABEL: y_te})
    dup_mask = train.duplicated(keep="first")
    n_dups = int(dup_mask.sum())
    if cfg["cleaning"]["drop_train_duplicates"]:
        train = train[~dup_mask]
    log.info("[clean] exact duplicates in raw train: %d; after imputation+clipping: %d; "
             "dropped from train: %d; test rows kept: %d", raw_train_dups, n_dups,
             n_dups if cfg["cleaning"]["drop_train_duplicates"] else 0, len(test))
    joblib.dump(cleaner, res / "preprocessor.joblib")
    log.info("[clean] saved results/preprocessor.joblib")

    # [5] derived features + baseline
    train, test = add_derived(train), add_derived(test)
    derived = ["capital_net", "hours_per_age"]
    rows = []
    for name, cols in [("without derived features", NUMERIC),
                       ("with capital_net + hours_per_age", NUMERIC + derived)]:
        acc, f1 = fit_eval(train, test, cols, cfg)
        rows.append({"features": name, "accuracy": round(acc, 4), "f1 (>50K)": round(f1, 4)})
        log.info("[baseline] %-34s accuracy=%.4f F1=%.4f", name, acc, f1)
    comp = pd.DataFrame(rows)
    comp["delta_accuracy"] = ["", round(rows[1]["accuracy"] - rows[0]["accuracy"], 4)]
    comp["delta_f1"] = ["", round(rows[1]["f1 (>50K)"] - rows[0]["f1 (>50K)"], 4)]
    comp.to_csv(res / "feature_comparison.csv", index=False)
    (res / "feature_comparison.md").write_text(md_table(comp) + "\n", encoding="utf-8")

    # [6] label audit
    audit = label_audit(raw, cfg, res)

    # [7] processed data, hashes, config yaml, data card
    out_cols = FEATURES + derived + [LABEL]
    for name, part in [("processed_train.csv", train), ("processed_test.csv", test)]:
        part[out_cols].to_csv(res / name, index=False, float_format="%.10g", lineterminator="\n")
    sha_tr, sha_te = sha256_file(res / "processed_train.csv"), sha256_file(res / "processed_test.csv")
    sha_all = hashlib.sha256((sha_tr + sha_te).encode()).hexdigest()
    log.info("[hash] processed_train.csv sha256 = %s", sha_tr)
    log.info("[hash] processed_test.csv  sha256 = %s", sha_te)
    log.info("[hash] combined            sha256 = %s", sha_all)

    manifest = {
        "config": cfg,
        "synthetic_data": bool(args.synthetic),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                        "joblib": joblib.__version__, "pyyaml": yaml.__version__},
        "raw_data_sha256_csv_view": raw_hash,
        "processed_sha256": {"processed_train.csv": sha_tr, "processed_test.csv": sha_te,
                             "combined": sha_all},
        "cleaning": {"fill_values_from_train": cleaner.fill_values_,
                     "clip_bounds_from_train": cleaner.clip_bounds_,
                     "train_rows_before_dedup": len(train) + (n_dups if cfg["cleaning"]["drop_train_duplicates"] else 0),
                     "train_rows_after_dedup": len(train), "test_rows": len(test),
                     "exact_duplicates_dropped_from_train": n_dups},
        "baseline_results": comp[["features", "accuracy", "f1 (>50K)"]].to_dict("records"),
        "label_audit": audit,
    }
    (res / "processed_config.yaml").write_text(yaml.safe_dump(to_py(manifest), sort_keys=False),
                                               encoding="utf-8")

    pct = lambda col, val: 100 * (df[col].astype(str).str.strip() == val).mean()
    mp = lambda col: float(miss.loc[miss.column == col, "pct"].iloc[0])
    write_data_card({
        "raw_rows": raw.shape[0], "raw_cols": raw.shape[1], "pos_rate": 100 * y.mean(),
        "miss_workclass": mp("workclass"), "miss_occupation": mp("occupation"),
        "miss_native": mp("native-country"), "n_train": len(train), "n_dups": n_dups,
        "n_test": len(test), "conflict_14": audit["all_14_features"]["conflicting_groups"],
        "conflict_13": audit["13_features_without_fnlwgt"]["conflicting_groups"],
        "pct_male": pct("sex", "Male"), "pct_white": pct("race", "White"),
        "pct_us": pct("native-country", "United-States"), "seed": seed,
        "sha_train": sha_tr, "sha_test": sha_te, "sha_combined": sha_all,
    }, HERE / "data_card.md")
    log.info("[done] outputs in %s ; data card in %s", res, HERE / "data_card.md")


if __name__ == "__main__":
    main()
