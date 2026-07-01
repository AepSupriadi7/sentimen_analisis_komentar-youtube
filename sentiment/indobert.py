from transformers import pipeline

# Load model sekali saja
sentiment_model = pipeline(
    "sentiment-analysis",
    model="w11wo/indonesian-roberta-base-sentiment-classifier",
    tokenizer="w11wo/indonesian-roberta-base-sentiment-classifier"
)


def analyze_sentiment(text):

    try:

        result = sentiment_model(text[:512])[0]

        label = result['label']
        score = round(result['score'], 4)

        return label, score

    except Exception as e:

        return "neutral", 0.0