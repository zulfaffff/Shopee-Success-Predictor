# Shopee Product Success Predictor

A machine learning system to classify e-commerce product performance using NLP and ensemble algorithms.

## Project Objective
To predict whether a product will achieve "Best-Seller" status (selling ≥1,000 units) based on title, price, and rating metrics. This helps sellers optimize their product listings to be more competitive and increase the chances of sales among thousands of new products that often fail to gain visibility.

## Technical Stack
* **Language:** Python 3.14.3
* **Libraries:** Scikit-Learn, Pandas, NLTK, Joblib
* **NLP:** TF-IDF Vectorization (500 features)
* **Algorithms:** Gradient Boosting Classifier, Random Forest

## Project Structure
* `src/preprocess.py`: Automated pipeline for NLP cleaning and feature engineering.
* `src/train.py`: Model training, K-Fold cross-validation, and serialization.
* `notebooks/EDA.ipynb`: Statistical analysis and model evaluation visualizations.
* `models/`: Serialized model binaries (.pkl).

## Implementation
1. **Preprocessing**: Cleans currency, standardizes sales units, and applies Indonesian stop-word removal.
2. **Feature Engineering**: Generates reputation scores, discount percentages, and text-length metrics.
3. **Training**: Executes ensemble models with 5-fold cross-validation.
4. **Evaluation**: Compares Gradient Boosting and Random Forest via Confusion Matrix and Feature Importance.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Train models: `python -m src.train`
3. View analysis: Open `notebooks/EDA.ipynb`

## Key Findings
* **Feature Importance**: Current Price, Name Length, and Discount Percentage are the strongest predictors of sales success.
* **NLP Insight**: Top keywords found in the title such as "Pro", "iPhone", "Max", and "Fingerprint", act as strong indicators that help the model distinguish successful listings.

## Contact
Zulfa Fauziyyah
Data Science & ML Engineering

Email: zulfafauziyyah454@gmail.com
LinkedIn: https://www.linkedin.com/in/zulfa-fauziyyah/
GitHub: 
Portfolio: [Link Website/Behance jika ada, kalau tidak ada hapus saja]