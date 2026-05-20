import pandas as pd
import nltk
import re
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_price(x):
    """Convert price string to integer."""
    if pd.isna(x) or str(x).strip() == "":
        return 0
    text = str(x).replace('Rp', '').replace('.', '').strip()
    if '-' in text:
        text = text.split('-')[0].strip()
    try:
        return int(text)
    except:
        return 0

def clean_units_sold(x):
    """Convert sales string (e.g., '1,2RB+') to integer."""
    if pd.isna(x) or str(x).strip() == "": return 0
    teks = str(x).upper().replace('+', '').replace('.', '').replace(' ', '').strip()
    if 'RB' in teks:
        teks = teks.replace('RB', '').replace(',', '.')
        try: return int(float(teks) * 1000)
        except: return 0
    try: return int(teks)
    except: return 0

def clean_rating_score(x):
    """Convert rating string to float."""
    if pd.isna(x) or str(x).strip() == "":
        return 0.0
    text = str(x).replace(',', '.')
    try:
        return float(text)
    except:
        return 0.0

def clean_rating_count(x):
    """Convert rating count string to integer."""
    if pd.isna(x) or str(x).strip() == "":
        return 0
    text = str(x).upper().replace('+', '').replace('.', '').replace(' ', '').strip()
    if 'RB' in text:
        text = text.replace('RB', '').replace(',', '.')
        try: return int(float(text) * 1000)
        except: return 0
    try:
        return int(float(text.replace(',', '.')))
    except:
        return 0

def remove_noise(text):
    """Remove URLs, special characters, and extra whitespaces."""
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_full_preprocessing(df):
    """Execute complete data cleaning and feature engineering pipeline."""
    # Clean numerical columns
    df['current_price_clean'] = df['current_price'].apply(clean_price)
    df['original_price_clean'] = df['original_price'].apply(clean_price)
    df.loc[df['original_price_clean'] == 0, 'original_price_clean'] = df['current_price_clean']
    df['units_sold_clean'] = df['units_sold'].apply(clean_units_sold)
    df['rating_score_clean'] = df['rating_score'].apply(clean_rating_score)
    df['rating_count_clean'] = df['rating_count'].apply(clean_rating_count)

    # Text imputation and cleaning
    df['desc'] = df['desc'].fillna("No description available")
    stop_words = set(stopwords.words('indonesian'))
    custom_noise = {
        'dan', 'yang', 'untuk', 'dengan', 'di', 'anda', 'kami', 'tidak',
        'jika', 'dari', 'akan', 'dalam', 'bisa', 'lebih', 'produk',
        'pengiriman', 'barang', 'ponsel', 'hp', 'model', 'for'
    }
    stop_words.update(custom_noise)

    def finalize_text(row):
        combined = str(row['product_title']) + " " + str(row['desc'])
        cleaned = remove_noise(combined)
        tokens = cleaned.split()
        filtered = [w for w in tokens if w not in stop_words and len(w) > 2]
        return " ".join(filtered)

    df['text_final'] = df.apply(finalize_text, axis=1)

    # Feature Engineering
    df['rep_score'] = df['rating_score_clean'] * df['rating_count_clean']
    df['discount_pct'] = ((df['original_price_clean'] - df['current_price_clean']) / df['original_price_clean'] * 100)
    df['discount_pct'] = df['discount_pct'].fillna(0)
    df['name_len'] = df['product_title'].apply(lambda x: len(str(x)))
    df['desc_len'] = df['desc'].apply(lambda x: len(str(x)))
    
    return df