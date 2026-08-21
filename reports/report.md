# Title

This project looks at lending data to explore what makes a loan default, how this can be predicted, and how models can work within the domain.

## Question

What this project is trying to explore is a classification problem: is this loan going to default or not? There are many features that can impact this question, and we'll get into them later. For a lender, models like the ones explored here, are quite interesting. If a model can comfortably predict if a loan will default or not they can cut cost.

## Data

The dataset used for this project is for LendingClub. It contains loans from 2007 to 2018Q4. There are ~2.2 million records. There were 1,345,350 records with resolved loan status, which were used to train the models. These records had a target definition of either "Fully Paid" or "Charged Off"/"Defaulted".

![Bar diagram](/diagrams/unresolved_fullypaid_default_bar.png)

The 915,318 unresolved loans had no final outcome, so they were dropped. There was a time-aware split to train the data on data pre 2016-10-01 and test data on post. The base rate of the resolved loans is 19.96%. It is used for setting a floor for a naive accuracy-based classifier. This means that an unlearned model that predicts "Fully Paid" for every loan will have an accuracy of ~80%. ROC-AUC (Receiver Operating Characteristic Area Under the Curve) will score 50% if ignoring all features. The project will use ROC-AUC and PR-AUC (Precision-Recall Area Under the Curve) instead of accuracy to evaluate the models.

## Method

### Leakage

When running a HistGradientBoostingClassifier model, it scored an ROC-AUC of ~99.91%, which is extremely high. This happened because of leakage. There were columns that populate only after the loan status is resolved and are not known to a lender when agreeing to a loan. These were columns like `total_pymnt`, `recoveries`, `last_pymnt_d`. After we dropped these columns the model scored ~72.56%, which is a drop of ~27 percentage points.

### Feature Selection

To select features for our model we had to drop some columns. Columns with free text were dropped because they were unstructured or too high cardinality to use. There were also vintage-limited columns that were dropped: LendingClub only started collecting these in December 2015, so ~76% of the records have them as null. `sec_app_*` and `*_joint` were also dropped since they are null for individual loans, and `application_type` already indicates whether a loan is joint or individual. `policy_code` was dropped since it is constant across every row, and `earliest_cr_line` was dropped since, as a date string with over 700 unique values, one-hot encoding it would blow up the feature space into hundreds of near-useless dummy columns.

### Train/Test Split

The data was split into a time-aware split at 2016-10-01. This was chosen rather than a random split since we did not want any leakage from future records to affect the training. This also mirrors how the model would actually be deployed.

### Preprocessing

Features are split into numeric or categorical features. The numeric transformer fills nulls with the median, then standardizes so every numeric feature is on the same scale. The categorical transformer fills nulls with the most frequent value, then uses `OneHotEncoder` to turn each category into 0/1 dummy columns, e.g. `home_ownership` becomes `home_ownership_RENT`, `home_ownership_OWN`, etc. Both are combined in a `ColumnTransformer`, which sends each column to the right transformer and drops anything not listed (`remainder="drop"`).

### Models Tried

First an implementation of logistic regression was tried from scratch. Gradient descent, sigmoid and BCE loss were all implemented from scratch (see [logistic_regression.py](/src/logistic_regression.py)).

Then `sklearn` was used to formally compare Logistic Regression, Random Forest and LightGBM. These three were chosen so linear and non-linear model outputs could be compared. Scored without tuning, using the mean across the 3 `TimeSeriesSplit` folds:

| Score              | Logistic Regression | Random Forest | LightGBM |
| ------------------ | ------------------- | ------------- | -------- |
| Mean ROC-AUC       | 0.7266              | 0.7186        | 0.7323   |
| Mean Avg Precision | 0.4072              | 0.3894        | 0.4159   |

### Validation and Tuning

To tune the models we used the last fold of the time series to see what the most optimal hyperparameters were. Hyperparameters are configuration variables set before training a machine learning model that control how the algorithm learns.

We only used the last fold for the search, not all 3 `TimeSeriesSplit` folds. Using all 3 would mean fitting every candidate 3 times, which is expensive since RF alone takes minutes per fit. Instead `RandomizedSearchCV` got the last fold's `(train_idx, val_idx)` pair as one fixed validation split: train on the earliest ~9/12 of the data, score on the most recent ~3/12. `val_idx` always comes after `train_idx` so this still respects time order, and it keeps the search cheap enough to actually run.

The search picks hyperparameters based on how well they score on that one slice. Reporting that same score as the final result would be misleading, since it's easy to land on a config that just happens to score well on the exact data it was judged on. So once the search finished, each winning pipeline was re-run through the same full 3-fold `cross_validate` used for the untuned baseline, producing these scores:

| Score              | Logistic Regression | Random Forest | LightGBM |
| ------------------ | ------------------- | ------------- | -------- |
| Mean ROC-AUC       | 0.7267              | 0.7271        | 0.7326   |
| Mean Avg Precision | 0.4072              | 0.4077        | 0.4172   |

These numbers come from the `cross_validate` run, not the "Validation ROC-AUC" printed during the search. Keeping those two separate is what makes the tuning honest instead of the search just grading itself. After tuning, RF overtakes LR, while LGBM barely moves despite 6x more search iterations, suggesting LGBM was already close to its ceiling. LR only moved by 0.001, which is attributable to noise. All three end up within 0.006 ROC-AUC of each other, with LightGBM > RF > LR.

The number of search iterations (`n_iter`) also differed per model: RF took much more time per fit, so it was given only 5 iterations, while LR and LGBM were given 15 and 30 respectively, both cheap enough to test whether they had already reached their ceiling.

### Evaluation metrics

Scores are reported as the mean ROC-AUC and mean Average Precision (PR-AUC) across the 3 `TimeSeriesSplit` folds, rather than a single train/test split score, so a model isn't judged on one lucky (or unlucky) fold.

### Cost-sensitive Thresholding

A 0.5 threshold assumes both mistakes cost the same, which they don't. Missing a default (FN) loses the full funded_amnt. Rejecting a good loan (FP) only loses the interest margin we would have earned (installment \* term - funded_amnt). We built a cost matrix from these two costs and swept thresholds to find the one that actually minimizes total dollar cost, instead of just guessing 0.5.
