# Wine Classification Using Machine Learning

**Author:** Milad Zahmatkesh  
**Course:** Intensive Python for Data Science 
**University:** Southern Illinois University Edwardsville (SIUE)

## Overview

This project classifies wines into three classes based on their chemical properties using the sklearn wine dataset. Four different machine learning approaches are compared:

- KNN Classification
- Linear SVC
- Polynomial SVC
- KNN Regression (adapted for classification by rounding predictions)

Each method was tested across 50 different random train/test splits to account for the small dataset size (178 samples).

## Results

All four methods achieved strong results when using all 13 chemical features:

| Method | Avg Accuracy | Worst Case |
|--------|-------------|------------|
| KNN (k=27) | 97.0% | 89% |
| Linear SVC | 96.0% | 89% |
| Polynomial SVC (degree 3) | 96.0% | 89% |
| KNN Regression (k=13) | 97.0% | 92% |

KNN Regression with 13 neighbors was selected as the best model due to its high average accuracy and strong worst-case performance.

## Additional Analysis

- Correlation analysis and PCA were used to identify the most relevant features
- Feature reduction experiments using only the top correlated columns (total_phenols, flavanoids, od280/od315) showed significantly worse results
- Adding the alcohol feature improved reduced-feature performance but still fell short of using all 13 columns

## How to Run

1. Clone this repository
2. Open `wine_classification.ipynb` in Jupyter Notebook
3. Run all cells from top to bottom

The dataset is loaded directly from sklearn, so no additional downloads are needed.

## Tech Stack

- Python 3
- scikit-learn
- pandas
- NumPy
- Matplotlib
