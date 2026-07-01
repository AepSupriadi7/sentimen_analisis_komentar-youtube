# Analisis Komentar YouTube Berbasis Natural Language Processing (NLP)

## Deskripsi

Proyek ini merupakan aplikasi berbasis **Python Flask** yang dikembangkan untuk menganalisis komentar pada video YouTube menggunakan pendekatan **Natural Language Processing (NLP)**.

Aplikasi mampu melakukan pengambilan komentar secara otomatis melalui **YouTube Data API v3**, kemudian melakukan preprocessing teks, analisis sentimen, pemodelan topik, ekstraksi entitas tokoh, hingga visualisasi hasil analisis dalam bentuk dashboard interaktif.

Selain menghasilkan analisis sentimen, aplikasi ini juga membandingkan dua metode **Topic Modeling** (LDA dan BERTopic) serta dua metode **Named Entity Extraction** (Rule-Based dan Named Entity Recognition/NER).

---

# Fitur

- Crawling komentar YouTube menggunakan YouTube Data API v3
- Preprocessing teks bahasa Indonesia
- Analisis Sentimen menggunakan IndoBERT
- Topic Modeling menggunakan:
  - Latent Dirichlet Allocation (LDA)
  - BERTopic
- Ekstraksi Tokoh menggunakan:
  - Rule-Based
  - Named Entity Recognition (NER)
- Insight otomatis berdasarkan hasil analisis
- Network Analysis antar tokoh
- Dashboard interaktif berbasis Flask
- Export komentar asli dan hasil preprocessing ke CSV

---

# Metode yang Digunakan

| Tahapan               | Metode |
|----------             |---------|
| Crawling              | YouTube Data API v3 |
| Preprocessing         | Cleaning, Case Folding, Tokenizing, Stopword Removal, Slang Normalization, Stemming |
| Sentiment Analysis    | IndoBERT |
| Topic Modeling 1      | Latent Dirichlet Allocation (LDA) |
| Topic Modeling 2      | BERTopic (Transformer Embedding + UMAP + HDBSCAN + c-TF-IDF) |
| Entity Extraction 1   | Rule-Based |
| Entity Extraction 2   | Named Entity Recognition (XLM-RoBERTa) |
| Network Analysis      | NetworkX |
| Dashboard             | Flask + Bootstrap + Plotly |

---

# Alur Sistem

```
URL YouTube
      │
      ▼
YouTube Data API
      │
      ▼
Pengambilan Komentar
      │
      ▼
Penyimpanan Sementara (Pandas DataFrame)
      │
      ▼
Preprocessing
      │
      ▼
Analisis Sentimen (IndoBERT)
      │
      ├───────────────┐
      ▼               ▼
Topic Modeling     Entity Extraction
LDA                Rule-Based
BERTopic           NER
      │               │
      └───────┬───────┘
              ▼
      Insight Otomatis
              │
              ▼
      Network Analysis
              │
              ▼
 Dashboard Analisis NLP
```

---

# Struktur Folder

```
Project_NLP/
│
├── crawler/
├── preprocessing/
├── sentiment/
├── topic/
│   ├── lda_model.py
│   └── bertopic.py
│
├── ner/
├── network/
├── insight/
├── templates/
├── static/
├── app.py
├── requirements.txt
└── README.md
```

---

# Teknologi

- Python 3.10
- Flask
- Pandas
- NLTK
- Sastrawi
- Transformers
- Hugging Face
- BERTopic
- Gensim
- Plotly
- NetworkX
- Matplotlib

---

# Dataset

Dataset diperoleh secara langsung menggunakan **YouTube Data API v3** berdasarkan URL video yang dimasukkan oleh pengguna.

Data yang diambil meliputi:

- Nama pengguna
- Isi komentar
- Jumlah Like
- Tanggal komentar

---

# Hasil Analisis

Aplikasi menghasilkan beberapa informasi, antara lain:

- Distribusi Sentimen
- Topik Dominan menggunakan LDA
- Topik Dominan menggunakan BERTopic
- Perbandingan Topic Modeling
- Tokoh yang Dibicarakan (Rule-Based)
- Tokoh yang Dibicarakan (NER)
- Insight Otomatis
- Network Analysis Tokoh

---

# Perbandingan Metode

## Topic Modeling

| LDA                       | BERTopic |
|------                     |-----------|
| Berbasis probabilistik    | Berbasis Transformer Embedding |
| Cepat                     | Lebih akurat pada short text |
| Tidak memahami konteks    | Memahami hubungan semantik |
| Cocok sebagai baseline    | Cocok untuk komentar media sosial |

---

## Entity Extraction

| Rule-Based |              | NER |
|------------|              |-----|
| Menggunakan kamus nama    | Menggunakan model Transformer |
| Cepat                     | Lebih fleksibel |
| Tidak mengenali nama baru | Mampu mengenali nama baru |
| Mudah dikembangkan        | Akurasi lebih tinggi |

---

# Pengembang

**Aep Supriadi**

Program Magister Teknik Informatika

Fakultas Ilmu Komputer

Universitas Pamulang

Tahun Akademik 2025–2026

---

# Lisensi

Proyek ini dikembangkan untuk keperluan penelitian akademik dan pembelajaran pada Program Magister Teknik Informatika Universitas Pamulang.