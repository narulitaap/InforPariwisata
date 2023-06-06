import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import nltk
nltk.download('punkt')
import re
from nltk.tokenize import RegexpTokenizer
nltk.download('stopwords')
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pickle
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("PEMROSESAN BAHASA ALAMI A")
st.write("### Dosen Pengampu : Dr. FIKA HASTARITA RACHMAN, ST., M.Eng")
st.write("#### Kelompok : 5")
st.write("##### Hambali Fitrianto - 200411100074")
st.write("##### Pramudya Dwi Febrianto - 200411100042")
st.write("##### Febrian Achmad Syahputra - 200411100106")

#Navbar
description, implementation = st.tabs(["Description", "Implementation"])

#data_set_description
with description:
    st.write("###### Judul : ANALISIS SENTIMEN PADA WISATA DIENG DENGAN ALGORITMA K-NEAREST NEIGHBOR (K-NN) ")
    st.write("""###### Penjelasan Prepocessing Data : """)
    st.write("""1. Case Folding :
    
    Case folding adalah proses dalam pemrosesan teks yang mengubah semua huruf dalam teks menjadi huruf kecil atau huruf besar. Tujuan dari case folding adalah untuk mengurangi variasi yang disebabkan oleh perbedaan huruf besar dan kecil dalam teks, sehingga mempermudah pemrosesan teks secara konsisten.
    
    Dalam case folding, biasanya semua huruf dalam teks dikonversi menjadi huruf kecil dengan menggunakan metode seperti lowercasing. Dengan demikian, perbedaan antara huruf besar dan huruf kecil tidak lagi diperhatikan dalam analisis teks, sehingga memungkinkan untuk mendapatkan hasil yang lebih konsisten dan mengurangi kompleksitas dalam pemrosesan teks.
    """)
    st.write("""2. Tokenize :

    Tokenisasi adalah proses pemisahan teks menjadi unit-unit yang lebih kecil yang disebut token. Token dapat berupa kata, frasa, atau simbol lainnya, tergantung pada tujuan dan aturan tokenisasi yang digunakan.

    Tujuan utama tokenisasi dalam pemrosesan bahasa alami (Natural Language Processing/NLP) adalah untuk memecah teks menjadi unit-unit yang lebih kecil agar dapat diolah lebih lanjut, misalnya dalam analisis teks, pembentukan model bahasa, atau klasifikasi teks.
    """)
    st.write("""3. Filtering (Stopword Removal) :

    Filtering atau Stopword Removal adalah proses penghapusan kata-kata yang dianggap tidak memiliki makna atau kontribusi yang signifikan dalam analisis teks. Kata-kata tersebut disebut sebagai stop words atau stopwords.

    Stopwords biasanya terdiri dari kata-kata umum seperti “a”, “an”, “the”, “is”, “in”, “on”, “and”, “or”, dll. Kata-kata ini sering muncul dalam teks namun memiliki sedikit kontribusi dalam pemahaman konten atau pengambilan informasi penting dari teks.

    Tujuan dari Filtering atau Stopword Removal adalah untuk membersihkan teks dari kata-kata yang tidak penting sehingga fokus dapat diarahkan pada kata-kata kunci yang lebih informatif dalam analisis teks. Dengan menghapus stopwords, kita dapat mengurangi dimensi data, meningkatkan efisiensi pemrosesan, dan memperbaiki kualitas hasil analisis.
    """)
    st.write("""4. Stemming :

    Stemming dalam pemrosesan bahasa alami (Natural Language Processing/NLP) adalah proses mengubah kata ke dalam bentuk dasarnya atau bentuk kata yang lebih sederhana, yang disebut sebagai “stem”. Stemming bertujuan untuk menghapus infleksi atau imbuhan pada kata sehingga kata-kata yang memiliki akar kata yang sama dapat diidentifikasi sebagai bentuk yang setara.
    """)
    
    st.write("""###### Penjelasan Ekstraksi Fitur : """)
    st.write("""TF-IDF :""")
    st.write("""Ditahap akhir dari text preprocessing adalah term-weighting .Term-weighting merupakan proses pemberian bobot term pada dokumen. Pembobotan ini digunakan nantinya oleh algoritma Machine Learning untuk klasifikasi dokumen. Ada beberapa metode yang dapat digunakan, salah satunya adalah TF-IDF (Term Frequency-Inverse Document Frequency).""")
    st.write("""TF (Term Frequency) :""")
    st.write("""TF (Term Frequency) adalah ukuran yang menggambarkan seberapa sering sebuah kata muncul dalam suatu dokumen. Menghitung TF melibatkan perbandingan jumlah kemunculan kata dengan jumlah kata keseluruhan dalam dokumen.""")
    st.write("""Perhitungan TF (Term Frequency) :
    
    TF(term) = (Jumlah kemunculan term dalam dokumen) / (Jumlah kata dalam dokumen)
    """)
    st.write("""DF (Document Frequency) :""")
    st.write("""DF (Document Frequency) adalah ukuran yang menggambarkan seberapa sering sebuah kata muncul dalam seluruh koleksi dokumen. DF menghitung jumlah dokumen yang mengandung kata tersebut.""")
    st.write("""Perhitungan DF (Document Frequency) :
    
    DF(term) = Jumlah dokumen yang mengandung term
    """)
    st.write("""IDF (Inverse Document Frequency) :""")
    st.write("""IDF (Inverse Document Frequency) adalah ukuran yang menggambarkan seberapa penting sebuah kata dalam seluruh koleksi dokumen. IDF dihitung dengan mengambil logaritma terbalik dari rasio total dokumen dengan jumlah dokumen yang mengandung kata tersebut. Tujuan IDF adalah memberikan bobot yang lebih besar pada kata-kata yang jarang muncul dalam seluruh koleksi dokumen.""")
    st.write("""Perhitungan IDF (Inverse Document Frequency) :
    
    IDF(term) = log((Total jumlah dokumen) / (DF(term)))
    """)
    st.write("""TF-IDF (Term Frequency-Inverse Document Frequency) :""")
    st.write("""TF-IDF (Term Frequency-Inverse Document Frequency) adalah metode yang menggabungkan informasi TF dan IDF. TF-IDF memberikan bobot yang lebih tinggi pada kata-kata yang sering muncul dalam dokumen tertentu (TF tinggi) dan jarang muncul dalam seluruh koleksi dokumen (IDF tinggi). Metode ini digunakan untuk mengevaluasi kepentingan relatif suatu kata dalam konteks dokumen.""")
    st.write("""Perhitungan TF-IDF (Term Frequency-Inverse Document Frequency) :
    
    TF-IDF(term, document) = TF(term, document) * IDF(term)
    """)
    st.write("""Dalam perhitungan TF-IDF, TF(term, document) adalah nilai TF untuk term dalam dokumen tertentu, dan IDF(term) adalah nilai IDF untuk term di seluruh koleksi dokumen.""")
    st.write("""Mengubah representasi teks ke dalam vektor :
    """)
    
    st.write("###### Aplikasi ini untuk : ")
    st.write("""ANALISIS SENTIMEN PADA WISATA DIENG DENGAN ALGORITMA K-NEAREST NEIGHBOR (K-NN)""")
    st.write("###### Source Code Aplikasi ada di Github anda bisa acces di link : https://github.com/BojayJaya/PBA-Kelompok-5")
    st.write("###### Untuk Wa saya anda bisa hubungi nomer ini : http://wa.me/6282138614807 ")

with implementation:
    st.write("""
    <center><h4 style = "text-align: justify;">APLIKASI ANALISIS SENTIMEN PADA WISATA DIENG DENGAN ALGORITMA K-NEAREST NEIGHBOR (K-NN)</h4></center>
    """,unsafe_allow_html=True)

    #Fractional Knapsack Problem
    #Getting input from user
    iu = st.text_area('Masukkan kata yang akan di analisa :')

    submit = st.button("submit")

    if submit:
        def prep_input_data(iu):
            ulasan_case_folding = iu.lower()

            #Cleansing
            clean_tag  = re.sub("@[A-Za-z0-9_]+","", ulasan_case_folding)
            clean_hashtag = re.sub("#[A-Za-z0-9_]+","", clean_tag)
            clean_https = re.sub(r'http\S+', '', clean_hashtag)
            clean_symbols = re.sub("[^a-zA-Z ]+"," ", clean_https)
            
            #Inisialisai fungsi tokenisasi dan stopword
            # stop_factory = StopWordRemoverFactory()
            tokenizer = RegexpTokenizer(r'cuaca\s+sejuk|telaga\s+sarangan|[\w\']+')
            tokens = tokenizer.tokenize(clean_symbols)

            #Stop Words
            stop_factory = StopWordRemoverFactory()
            data = stop_factory.get_stop_words()
            removed = []
            if tokens not in data:
                removed.append(tokens)

            #list to string
            gabung =' '.join([str(elem) for elem in removed])

            #Steaming
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            stem = stemmer.stem(gabung)
            return(ulasan_case_folding,clean_symbols,tokens,gabung,stem)

        #Dataset
        Data_ulasan = pd.read_csv("https://raw.githubusercontent.com/Niken-Amalia/Informatika_Pariwisata/main/datasaranganmagetan%20-%20Sheet1.csv")
        review_text = Data_ulasan['ulasan_hasil_preprocessing']
        sentimen = Data_ulasan['label']

        # TfidfVectorizer 
        tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')
        tfidf_wm = tfidfvectorizer.fit_transform(review_text)
        tfidf_tokens = tfidfvectorizer.get_feature_names_out()
        df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)
        with open('model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        
        #with open('tfidf.pkl', 'rb') as file:
         #   loaded_data_tfid = pickle.load(file)
        
        #tfidf_wm = loaded_data_tfid.fit_transform(ulasan_dataset)

        #Train test split
        x_train, x_test, y_train, y_test = train_test_split(tfidf_wm,y, test_size=0.3, random_state=5)
#         training, test, training_label, test_label  = train_test_split(tfidf_wm, sentimen,test_size=0.3, random_state=42)#Nilai X training dan Nilai X testing 70 30
#         training, test, training_label, test_label  = train_test_split(tfidf_wm, sentimen,test_size=0.4, random_state=42)#Nilai X training dan Nilai X testing 60 40
        # training_label, test_label = train_test_split(, test_size=0.2, random_state=42)#Nilai Y training dan Nilai Y testing    

        #model
        rf = loaded_model.fit(x_train, y_train)
        y_pred = rf.predict(x_test)

        #Evaluasi
        akurasi = accuracy_score(y_test, y_pred)
        akurasi_persen = akurasi * 100

        #Inputan 
        ulasan_case_folding,clean_symbols,tokens,gabung,stem = prep_input_data(iu)
        st.write('Case Folding')
        st.write(ulasan_case_folding)
        st.write('Cleaning Simbol')
        st.write(clean_symbols)
        st.write('Token')
        st.write(tokens)
        st.write('Stop Removal')
        st.write(gabung)
        st.write('Stemming')
        st.write(stem)

        
        #Prediksi
        y_pred = rf.predict(x_test)

        st.subheader('Akurasi')
        # st.info(akurasi)
        st.info(f"{akurasi_persen:.2f}%")

        st.subheader('Prediksi')
        if y_pred == "positif":
            st.success('Positif')
        elif y_pred == "netral":
            st.success('Netral')
        else:
            st.error('Negative')