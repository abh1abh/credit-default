# Data

## The Dataset

1. Download the dataset zip from the Kaggle link above.
2. Extract accepted_2007_to_2018Q4.csv.gz into data/raw/.
3. Leave it gzipped — pandas reads gzip natively via pd.read_csv(path).
4. Ignore rejected_2007_to_2018Q4.csv.gz — rejected loans were never funded and have no repayment outcome, so they're useless for default prediction.

### Disclaimer

Data is never committed to git (see Repo Structure). It's a static, public, ~600MB+ file — treated as a build artifact, re-fetched via these instructions.
