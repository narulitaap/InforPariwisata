import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import googletrans
from googletrans import Translator
from google_trans_new import google_translator

st.title("Penerapan Metode Random Forest untuk Analisis Sentimen Ulasan Pengunjung Wisata Telaga Sarangan")
st.write("Narulita Arien Pramesti - 200411100065")
st.write("Niken Amalia - 200411100109")

dataset, preprocessing, implementation = st.tabs(["Dataset", "Preprocessing", "Implementation"])
with dataset:
    st.write("""# Dataset""")
    df = pd.read_csv("datasaranganmagetan.csv")
    df
    sumdata = len(df)
    st.success(f"#### Total Data : {sumdata}")
#===================================

with preprocessing:
    st.write("""# Preprocessing""")
    st.info("## Data setelah di preprocessing")
    data = pd.read_csv('hasillabel.csv', index_col=0)
    data

with implementation:
# Memuat dataset
    @st.cache(allow_output_mutation=True)
    def load_data():
        data = pd.read_csv('HasilBaru.csv')
        return data

    data = load_data()

# Pra-pemrosesan data
    nltk.download('stopwords')

    stopwords = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    def preprocess_text(text):
        # Case folding (mengubah ke huruf kecil)
        text = text.lower()
    
        # Text cleaning (menghapus karakter khusus)
        text = re.sub(r'\W+', ' ', text)
    
        # Tokenisasi (memisahkan teks menjadi token)
        tokens = word_tokenize(text)
    
        # Menghapus stopwords dan melakukan stemming
        clean_tokens = [stemmer.stem(token) for token in tokens if token not in stopwords]
    
        # Menggabungkan kembali token menjadi teks
        cleaned_text = ' '.join(clean_tokens)
    
        return cleaned_text

    data['processed_text'] = data['review_text'].apply(preprocess_text)

    # Menggunakan TF-IDF untuk mengubah teks menjadi fitur numerik
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(data['processed_text'])
    y = data['Sentiment']

    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Membuat dan melatih model Random Forest
    rf_classifier = RandomForestClassifier(n_estimators=100)
    rf_classifier.fit(X_train, y_train)

    # Memprediksi kategori untuk data uji
    y_pred = rf_classifier.predict(X_test)

    # Menghitung akurasi
    accuracy = accuracy_score(y_test, y_pred)

    # Menampilkan aplikasi Streamlit
    st.title('Analisis Sentimen Teks')
    st.write('Masukkan teks yang ingin dianalisis sentimennya:')
    text_input = st.text_area('Teks')

    if st.button('Analisis'):
    # Pra-pemrosesan teks
        preprocessed_text = preprocess_text(text_input)
    
        st.write('Hasil Pre-Processing:')
        st.write(preprocessed_text)
    
        # Mengubah teks menjadi fitur menggunakan TF-IDF
        text_features = vectorizer.transform([preprocessed_text])
    
        # Melakukan analisis sentimen
        sentiment = rf_classifier.predict(text_features)[0]
    
        st.write('Sentimen:', sentiment)

        st.write('Akurasi:',accuracy)