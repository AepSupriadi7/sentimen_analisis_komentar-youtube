def generate_insight(
        total,
        positive,
        negative,
        neutral,
        topics,
        persons):

    sentiment = "netral"
    persen = 0

    if total > 0:

        p_pos = positive / total * 100
        p_neg = negative / total * 100
        p_neu = neutral / total * 100

        if p_pos >= p_neg and p_pos >= p_neu:
            sentiment = "positif"
            persen = p_pos

        elif p_neg >= p_pos and p_neg >= p_neu:
            sentiment = "negatif"
            persen = p_neg

        else:
            sentiment = "netral"
            persen = p_neu

    topik = "-"

    if topics:
        topik = topics[0]['topic']

    tokoh = "-"

    if persons:
        tokoh = persons[0]['name']

    insight = f"""
Berdasarkan analisis terhadap {total} komentar YouTube,
mayoritas komentar memiliki sentimen {sentiment}
sebesar {persen:.1f}%.
Topik yang paling sering dibahas adalah {topik}.
Tokoh yang paling banyak disebut adalah {tokoh}.
"""

    return insight.strip()