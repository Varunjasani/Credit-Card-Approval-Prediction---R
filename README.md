# Credit Card Approval Prediction

A machine learning project that predicts whether a credit card applicant is likely to be a **good** or **bad** customer, based on applicant demographics and their historical credit payment behavior. The project covers the full workflow: data cleaning, exploratory data analysis, statistical testing, feature engineering, handling class imbalance, and building and comparing several classification models.

## Problem Statement

Credit card issuers need to decide whether to approve an applicant. Using applicant information (income, education, family status, employment, housing, etc.) and their monthly credit record, the goal is to flag applicants who are likely to default (become a "bad" customer) so that approval decisions can be made more reliably.

A customer who is **30 or more days past due** on payments is treated as a **bad customer (target = 1)**; everyone else is treated as a **good customer (target = 0)**.

## Dataset

The project uses two source files:

- **`application_record.csv`** — applicant details such as gender, car/property ownership, income, education, family status, housing type, occupation, and the day-based fields used to derive age and years employed.
- **`credit_record.csv`** — the monthly payment status history for each applicant (`STATUS` column).

The two files are linked by the shared `ID` column.

> Note: This is a public dataset commonly used for credit-approval modeling. No private or personally identifiable information is included.

### Defining the target

The `STATUS` field is recoded into a binary target:

| STATUS | Meaning | Target |
|--------|---------|--------|
| C | Paid off that month | 0 (good) |
| X | No loan that month | 0 (good) |
| 0 | 1–29 days past due | 0 (good) |
| 1–5 | 30+ days past due (up to write-off) | 1 (bad) |

Each customer's records are then collapsed to a single row using the **worst** status they ever had (`aggregate(target ~ ID, FUN = max)`), so one applicant = one label.

## Repository Structure

```
.
├── Rscript.R                              # Exploratory data analysis + statistical testing
├── Credit_Card_Approval_Prediction.Rmd    # Data preprocessing + modeling notebook
├── plots/                                 # Saved EDA visualizations (PNG)
└── README.md
```

## Workflow

### 1. Exploratory Data Analysis (`Rscript.R`)

The EDA stage builds and merges the target, then explores the data through visualizations and statistical tests:

- Distribution of the target across **gender, education level, and marital status** (stacked bar charts)
- Income distribution across **education type, family status, and housing type** (box plots)
- **Age distribution** of applicants and income by **car ownership**
- Correlation analysis across numeric features (`corrplot`)
- **t-test** comparing annual income between defaulters and non-defaulters
- **Chi-square test** for association between categorical variables (e.g. family status and housing type)

Saved plots are available in the `plots/` folder.

### 2. Data Preprocessing & Feature Engineering (`Credit_Card_Approval_Prediction.Rmd`)

- Converting data types and recoding yes/no fields (gender, car, property) to 0/1
- Grouping high-cardinality categories (income type, education, family status, housing, occupation) into simpler buckets with `case_when`
- Deriving **AGE** and **YEARS_EMPLOYED** from the raw day-count fields, including an edge-case fix for pensioners with invalid employment values
- Standardizing annual income (z-score)

### 3. Handling Class Imbalance

The target is heavily imbalanced (far more good customers than bad). To address this, the majority class is reduced and the minority class is **up-sampled** so the training data is balanced before modeling.

### 4. Modeling

Several classification models are trained and compared:

- Logistic Regression (base and reduced-feature)
- Ridge Regression (L2 regularization, `glmnet`)
- Lasso Regression (L1 regularization, `glmnet`)
- Support Vector Machine (radial kernel)
- Random Forest

Models are evaluated using **accuracy, recall, confusion matrices, and ROC/AUC**. Recall is emphasized because correctly catching bad customers matters more than raw accuracy in an imbalanced setting. The project also includes a **decile mapping** step that ranks applicants by predicted risk.

## Key Findings

- Before balancing, models tended to predict only the majority (good) class — highlighting why class-imbalance handling is essential for this problem.
- After up-sampling, the models were able to identify bad customers, and recall became a meaningful metric to compare them.
- Statistical tests confirmed significant relationships between the target/income and several categorical features (e.g. income differs across education levels; family status and housing type are associated).

## Tools & Libraries

**Language:** R

**Key packages:** `dplyr`, `tidyr`, `ggplot2`, `caret`, `glmnet`, `randomForest`, `e1071`, `pROC`, `corrplot`, `janitor`, `scales`

## How to Run

1. Clone the repository and place `application_record.csv` and `credit_record.csv` in the working directory.
2. Install the required packages:
   ```r
   install.packages(c("dplyr", "tidyr", "ggplot2", "caret", "glmnet",
                      "randomForest", "e1071", "pROC", "corrplot",
                      "janitor", "scales"))
   ```
3. Run `Rscript.R` for the EDA and statistical analysis.
4. Open and run `Credit_Card_Approval.Rmd` for preprocessing and modeling.

## Notes

This project is for educational and demonstration purposes, showing an end-to-end data management and modeling workflow in R on a public dataset.
