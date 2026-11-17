# Phishing Website Detection at a Fixed False-Positive Rate

DS 4021 – Machine Learning, Final Project

**Group members:** _add names_

## Research question

At a fixed 1% false-positive rate, how much phishing-detection recall do support vector machine,
random forest, gradient boosting, and neural network classifiers lose when restricted to features
computed from the URL alone, compared with the full set of URL, page-content, and external-service
features?

An L1-regularized logistic regression serves as the baseline.

## Data

Hannousse & Yahiouche, *Web page phishing detection*, Mendeley Data, v3 (2021),
doi: [10.17632/c2gw7fy2j4.3](https://doi.org/10.17632/c2gw7fy2j4.3), file `dataset_B_05_2020.csv`
(collected May 2020).

- 11,430 websites, exactly balanced (5,715 phishing, 5,715 legitimate)
- 87 features in three groups:
  - **URL (56):** `length_url` … `statistical_report`
  - **Page content (24):** `nb_hyperlinks` … `domain_with_copyright`
  - **External services (7):** `whois_registered_domain`, `domain_registration_length`, `domain_age`,
    `web_traffic`, `dns_record`, `google_index`, `page_rank`
- `url` is an identifier, not a feature; drop it before modeling. The outcome is `status`.

### Files in `Data/`

| File | Rows | Use |
|---|---|---|
| `phishing_train.csv` | 9,144 (80%) | All exploration, tuning, cross-validation and threshold selection |
| `phishing_reporting.csv` | 2,286 (20%) | **Untouched until the final evaluation.** Do not open, plot or summarize it before then. |

The split is stratified on `status` with `random_state=42`. To reproduce it:

```bash
python make_split.py
```

The script downloads the original file into `Data/raw/` (not committed, so the full dataset is never
loaded by accident), checks its SHA-256 against the published version, and rewrites the two CSVs
identically.

### Notes for later

- One URL appears twice (both labeled phishing, with different feature values); 174 rows share an
  identical feature vector with another row. All rows were kept so the reporting set is 20% of the
  original data.
- Six columns are constant across the whole dataset.
- `domain_age` and `domain_registration_length` use `-1` when the WHOIS lookup failed.
- Several features are heavy-tailed (e.g., `web_traffic`, `length_url`), so scaling matters for the SVM
  and neural network but not for the tree models.
