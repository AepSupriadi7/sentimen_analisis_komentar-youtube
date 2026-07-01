import re
import string
import emoji
import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download NLTK (hanya sekali)
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

# Stopword bawaan bahasa Indonesia
stop_words = set(stopwords.words('indonesian'))

# Stopword tambahan khusus komentar YouTube Indonesia
custom_stopwords = {
    'nya', 'yg', 'ya', 'iya', 'gak', 'ga',
    'nggak', 'sih', 'nih', 'dong', 'lah',
    'kan', 'kok', 'aja', 'bang', 'bro',
    'pak', 'bu', 'om', 'deh', 'mah',
    'pun', 'gue', 'gua', 'aku', 'saya',
    'kami', 'kita', 'dia', 'si', 'yah',
    'kayak', 'kaya', 'video', 'youtube',
    'komen', 'komentar'
}

stop_words.update(custom_stopwords)

# Kamus slang Indonesia
slang_dict = {
    "yg": "yang",
    "bgt": "banget",
    "jg": "juga",
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "tdk": "tidak",
    "gpp": "tidak apa apa",
    "klo": "kalau",
    "krn": "karena",
    "dgn": "dengan",
    "utk": "untuk",
    "aja": "saja",
    "dr": "dari",
    "udh": "sudah",
    "sdh": "sudah",
    "blm": "belum",
    "tp": "tetapi",
    "sy": "saya",
    "gw": "saya",
    "gua": "saya",
    "lu": "kamu",
    "luw": "kamu",
    "bs": "bisa",
    "kek": "seperti",
    "sm": "sama",
    "trs": "terus",
    "jd": "jadi",
    "pd": "pada",
    "org": "orang",
    "smua": "semua",
    "y": "iya",
    "lg": "lagi",
    "jgan": "jangan"
}

factory = StemmerFactory()
stemmer = factory.create_stemmer()


def normalize_slang(text, dictionary):

    words = text.split()

    normalized_words = [
        dictionary[word]
        if word in dictionary
        else word
        for word in words
    ]

    return " ".join(normalized_words)


def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Hapus URL
    text = re.sub(
        r'http\S+|www\S+|https\S+',
        '',
        text
    )

    # Hapus mention
    text = re.sub(
        r'@\w+',
        '',
        text
    )

    # Hapus hashtag
    text = re.sub(
        r'#\w+',
        '',
        text
    )

    # Hapus emoji
    text = emoji.replace_emoji(
        text,
        replace=''
    )

    # Hapus ekspresi tertawa
    text = re.sub(
        r'\b(wk|wkwk|haha|hehe|xixi)+\w*\b',
        '',
        text
    )

    # Hapus angka
    text = re.sub(
        r'\d+',
        '',
        text
    )

    # Hapus huruf berulang
    # baguuuusss -> bagus
    text = re.sub(
        r'(.)\1{2,}',
        r'\1',
        text
    )

    # Hapus tanda baca
    text = text.translate(
        str.maketrans(
            '',
            '',
            string.punctuation
        )
    )

    # Hapus spasi berlebih
    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    # Normalisasi slang
    text = normalize_slang(
        text,
        slang_dict
    )

    # Tokenisasi
    tokens = word_tokenize(text)

    # Stopword removal
    tokens = [
        word for word in tokens
        if word not in stop_words
        and len(word) > 2
    ]

    return " ".join(tokens)


# Untuk sentiment analysis (PAKAI STEMMING)
def preprocess_text(text):

    text = clean_text(text)

    text = stemmer.stem(text)

    return text


# Untuk topic modeling (TANPA STEMMING)
def preprocess_topic(text):

    text = clean_text(text)

    return text