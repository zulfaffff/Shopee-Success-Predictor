import pandas as pd
import joblib
import os
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from src.preprocess import run_full_preprocessing

# 1. Load raw dataset
file_path = 'data/shopee_phone_case_dataset.csv'
df_raw = pd.read_csv(file_path, sep=';', encoding='cp1252')

# 2. Run automated cleaning and feature engineering
df = run_full_preprocessing(df_raw)

# 3. Define target variable based on sales threshold (1000 units)
threshold = 1000
y = df['units_sold_clean'].apply(lambda x: 1 if x >= threshold else 0).values

# 4. Text Vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=500)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text_final'])

# 5. Feature selection and matrix concatenation
selected_features = ['current_price_clean', 'discount_pct', 'name_len', 'desc_len', 'img_count']
numerical_data = df[selected_features].values
X_combined = hstack([numerical_data, tfidf_matrix])

# 6. Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Initialize Gradient Boosting and Random Forest models
model_gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
model_rf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=2, random_state=42)

# 8. Perform K-Fold Cross Validation for Gradient Boosting
kf = KFold(n_splits=5, shuffle=True, random_state=42)
print("--- RUNNING K-FOLD CROSS VALIDATION ---")
cv_gb = cross_val_score(model_gb, X_train, y_train, cv=kf)
print(f"GB Mean Accuracy: {cv_gb.mean():.4f}")

# 9. Train final models
print("Training models...")
model_gb.fit(X_train, y_train)
model_rf.fit(X_train, y_train)

# 10. Save trained models and vectorizer for deployment
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(model_gb, 'models/shopee_model_gb.pkl')
joblib.dump(model_rf, 'models/shopee_model_rf.pkl')
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')

print("SUCCESS: Models and Vectorizer are saved in /models folder!")