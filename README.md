# MovieML

Can you tell whether a movie will make money before it comes out? MovieML predicts whether a film will be a **commercial success**, defined as revenue of at least 2× its budget, using **only information available before release**.

**Status:** complete. The full pipeline (data collection, feature engineering, modeling and evaluation) is in [`MovieML.ipynb`](MovieML.ipynb).

## Results

- **Best model:** a tuned Random Forest reached **0.728 ROC-AUC** on held-out future releases, beating logistic regression in 94.5% of bootstrap resamples.
- **Business impact:** in a simulated "greenlight" test, funding only the films the model favored raised the hit rate from **38.8% to 73.5%**.
- **Efficiency:** benchmarked training time and memory against the number of trees and settled on 100 trees for the final model.
- **What didn't help:** a "competing releases" feature (films opening within 14 days) was tested and dropped because it didn't improve the model.

## Data

- **Source:** [TMDB API](https://developer.themoviedb.org/) movie details and credits for films released 2010–2025.
- **Sampling:** random results pages instead of the most-voted films, so the dataset isn't biased toward blockbusters. (A first pull of the most-voted films came out 97% "successful", which is why I switched.)
- **Cleaning:** kept films with real budget and revenue figures and capped extreme ROI outliers, leaving **1,364 films**.

## Method

**No data leakage.** Every feature is something a studio would know before release. Revenue, ratings, vote counts and popularity are excluded on purpose.

**28 features**, including:
- Budget, runtime, release month and genre (one-hot)
- Franchise membership
- Director, cast and production-company track records: the average ROI of their *earlier* films only, computed in release order so a film never sees its own outcome or anything after it

**Temporal validation.** Films are sorted by release date and split 80/20, so the model is always tested on movies released after the ones it learned from. Hyperparameters were tuned with time-series cross-validation on the training set.

**Models compared:** a majority-class baseline, logistic regression, Random Forest (default and tuned), XGBoost, and a PyTorch neural network with dropout and early stopping.

## Tech

Python · pandas · scikit-learn · XGBoost · PyTorch · Google Colab · TMDB API

## Run it

1. Get a free API key from [TMDB](https://developer.themoviedb.org/).
2. Open `MovieML.ipynb` in Google Colab.
3. Add the key as a Colab secret named `TMDB_API_KEY`.
4. Run the cells in order. Data is saved to Google Drive so it survives runtime restarts.
