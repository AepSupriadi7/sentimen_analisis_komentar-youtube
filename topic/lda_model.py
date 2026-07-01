from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def get_lda_topics(texts, n_topics=5):

    if len(texts) < 10:
        return []

    try:
        vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            stop_words=None
        )
        texts = [
            x for x in texts
            if len(x.split()) >= 3
        ]
        dtm = vectorizer.fit_transform(texts)

        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42
        )

        lda.fit(dtm)

        words = vectorizer.get_feature_names_out()

        topics = []

        for topic_idx, topic in enumerate(lda.components_):

            top_words = [
                words[i]
                for i in topic.argsort()[:-4:-1]
            ]

            topic_name = ", ".join(top_words)

            topic_count = 0

            topic_distribution = lda.transform(dtm)

            for row in topic_distribution:
                if row.argmax() == topic_idx:
                    topic_count += 1

            topics.append({
                "topic": topic_name,
                "count": topic_count
            })

        topics = sorted(
            topics,
            key=lambda x: x["count"],
            reverse=True
        )

        return topics

    except:
        return []