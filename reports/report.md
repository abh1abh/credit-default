# Title

This project looks at lending data to explore what makes a loan default, how this can be predicted, and how models can work within the domain.

## Question

What this project is trying to explore is a classification problem: is this loan going to default or not? There are many features that can impact this question, and we'll get into them later. For a lender, models like the ones explored here, are quite interesting. If a model can comfortably predict if a loan will default or not they can cut cost.

## Data

The dataset used for this project is for LendingClub. It contains loans from 2007 to 2018Q4. There are ~2.2 million records. There were 1,345,350 records with resolved loan status, which were used to train the models. These records had a target definition of either "Fully Paid" or "Charged Off"/"Defaulted". The 915,318 unresolved loans had no final outcome, so they were dropped. There was a time-aware split to train the data on data pre 2016-10-01 and test data on post. The base rate of the resolved loans is 19.96%. It is used for setting a floor for a naive accuracy-based classifier. This means that an unlearned model that predicts "Fully Paid" for every loan will have an accuracy of ~80%. ROC-AUC (Receiver Operating Characteristic Area Under the Curve) will score 50% if ignoring all features. The project will use ROC-AUC and PR-AUC (Precision-Recall Area Under the Curve) instead of accuracy to evaluate the models.

## Method
