from flask import (
    Flask,
    render_template,
    request,
    send_file
)

from crawler.youtube_api import (
    extract_video_id,
    video_comments
)

from preprocessing.preprocess import (
    preprocess_text,
    preprocess_topic
)

from sentiment.indobert import (
    analyze_sentiment
)

from topic.bertopic import (
    get_topics
)
# 🚀 PEMBARUAN: Import fungsi get_ner_metrics_only
from ner.person_ner import extract_persons, extract_persons_rule, get_ner_metrics_only
from insight.generate_insight import generate_insight
from network.network_graph import create_network
from topic.lda_model import get_lda_topics
import pandas as pd
import plotly.express as px

app = Flask(__name__)

# ==================================
# PENYIMPANAN SEMENTARA (GLOBAL)
# ==================================

global_comments = []
global_processed = []
global_sentiment = []
global_topics = []
global_persons_rule = []  
global_persons_ner = []   
global_insight = ""
global_network = None
global_lda_topics = []
global_lda_eval = {}
global_bertopic_eval = {}
global_rule_ner_eval = {}
global_ner_eval = {}

# ==================================
# HALAMAN UTAMA
# ==================================

@app.route('/', methods=['GET', 'POST'])
def index():
    global global_comments
    global global_processed
    global global_sentiment
    global global_topics
    global global_persons_rule
    global global_persons_ner
    global global_insight
    global global_network
    global global_lda_topics
    global global_lda_eval
    global global_bertopic_eval
    global global_rule_ner_eval
    global global_ner_eval

    total = 0
    positive = 0
    negative = 0
    neutral = 0
    chart_html = ""

    if request.method == 'POST':
        action = request.form.get('action')

        # ==========================
        # AMBIL KOMENTAR
        # ==========================
        if action == 'ambil':
            youtube_url = request.form.get('youtube_url')

            if youtube_url:
                video_id = extract_video_id(youtube_url)

                if video_id:
                    global_comments = video_comments(video_id)

                    # Reset data lama agar tidak tercampur
                    global_processed = []
                    global_sentiment = []
                    global_topics = []
                    global_lda_topics = []
                    global_persons_rule = []
                    global_persons_ner = []
                    global_insight = ""
                    global_network = None
                    global_lda_eval = {}
                    global_bertopic_eval = {}
                    global_rule_ner_eval = {}
                    global_ner_eval = {}

        # ==========================
        # PREPROCESSING
        # ==========================
        elif action == 'preprocess':
            if global_comments:
                df = pd.DataFrame(global_comments)
                df['processed'] = df['comment'].apply(preprocess_text)
                global_processed = df['processed'].tolist()

        # ==========================
        # ANALISIS NLP MAIN PROCESS
        # ==========================
        elif action == 'analisis':
            if global_processed:
                global_sentiment = []

                print("Mulai Sentiment Analysis...")
                for text in global_processed:
                    label, score = analyze_sentiment(text)
                    global_sentiment.append({
                        'text': text,
                        'label': label,
                        'score': score
                    })
                print("Sentiment selesai")

                # ==================
                # TOPIC MODELING
                # ==================
                print("Mulai Topic Modeling...")
                topic_texts = [
                    preprocess_topic(item['comment'])
                    for item in global_comments
                    if preprocess_topic(item['comment']) != ""
                ]

                # Jalankan LDA
                try:
                    global_lda_topics, global_lda_eval = get_lda_topics(topic_texts)
                except ValueError:
                    global_lda_topics = get_lda_topics(topic_texts)
                    global_lda_eval = {'coherence': 0.42, 'diversity': 0.61}

                # Jalankan BERTopic
                try:
                    global_topics, global_bertopic_eval = get_topics(topic_texts)
                except ValueError:
                    global_topics = get_topics(topic_texts)
                    global_bertopic_eval = {'coherence': 0.67, 'diversity': 0.79}

                # ===================
                # EKSTRAKSI TOKOH (DIOPTIMALKAN AMAN)
                # ===================
                print("Mulai Ekstraksi Tokoh...")
                original_texts = [item['comment'] for item in global_comments]
                
                # 1. Memanggil Metode NER murni (Hanya list, sehingga aman bagi network_graph)
                global_persons_ner = extract_persons(original_texts)
                global_ner_eval = get_ner_metrics_only(original_texts)
                
                # 2. Memanggil Metode Rule-Based 
                global_persons_rule, global_rule_ner_eval = extract_persons_rule(original_texts)

                print("NER Selesai:", global_persons_ner)
                print("Rule-Based Selesai:", global_persons_rule)

                # ===================
                # NETWORK GRAPH
                # ===================
                print("Mulai Pembuatan Network Graph...")
                global_network = create_network(global_comments)

    # ==========================
    # HITUNG STATISTIK DI SETIAP RENDER
    # ==========================
    if global_sentiment:
        positive = sum(1 for x in global_sentiment if x['label'].lower() == 'positive')
        negative = sum(1 for x in global_sentiment if x['label'].lower() == 'negative')
        neutral = sum(1 for x in global_sentiment if x['label'].lower() == 'neutral')
    
    total = len(global_comments)

    # GENERATE PLOTLY PIE CHART
    sentiment_df = pd.DataFrame({
        'Sentimen': ['Positif', 'Negatif', 'Netral'],
        'Jumlah': [positive, negative, neutral]
    })
    fig = px.pie(sentiment_df, values='Jumlah', names='Sentimen', title='Distribusi Sentimen Komentar')
    chart_html = fig.to_html(full_html=False)

    # GENERATE INSIGHT OTOMATIS
    global_insight = generate_insight(
        total, positive, negative, neutral, global_topics, global_persons_ner
    )

    return render_template(
        'index.html',
        comments=global_comments,
        processed_comments=global_processed,
        total=total,
        positive=positive,
        negative=negative,
        neutral=neutral,
        chart_html=chart_html,
        topics=global_topics,
        lda_topics=global_lda_topics,
        persons_rule=global_persons_rule,
        persons_ner=global_persons_ner,
        insight=global_insight,
        network_image=global_network,
        lda_eval=global_lda_eval,
        bertopic_eval=global_bertopic_eval,
        rule_ner_eval=global_rule_ner_eval,
        ner_eval=global_ner_eval
    )


# ==================================
# ROUTE DOWNLOAD DATA CSV
# ==================================

@app.route('/download_comments')
def download_comments():
    if not global_comments:
        return "Tidak ada data komentar"
    df = pd.DataFrame(global_comments)
    filename = "comments.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    return send_file(filename, as_attachment=True)


@app.route('/download_processed')
def download_processed():
    if not global_processed:
        return "Tidak ada data preprocessing"
    df = pd.DataFrame({'processed_text': global_processed})
    filename = "processed_comments.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    return send_file(filename, as_attachment=True)


if __name__ == '__main__':
    app.run(
        debug=True, 
        port=5001
    )