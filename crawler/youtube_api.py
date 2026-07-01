import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs

# Load file .env
load_dotenv()

# Ambil API Key dari .env
API_KEY = os.getenv("YOUTUBE_API_KEY")


def extract_video_id(url):
    """
    Mengambil video ID dari URL YouTube
    """
    parsed_url = urlparse(url)

    if parsed_url.hostname in ['www.youtube.com', 'youtube.com']:
        return parse_qs(parsed_url.query).get('v', [None])[0]

    elif parsed_url.hostname == 'youtu.be':
        return parsed_url.path[1:]

    return None


def video_comments(video_id):
    """
    Mengambil SEMUA komentar YouTube berdasarkan video ID tanpa batasan
    """
    youtube = build(
        'youtube',
        'v3',
        developerKey=API_KEY
    )

    comments = []
    next_page_token = None

    print(f"Mulai menarik semua komentar untuk Video ID: {video_id}...")

    while True:
        try:
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,  # Tetap 100 karena ini batas maksimal per halaman dari Google
                pageToken=next_page_token,
                textFormat='plainText'
            )

            response = request.execute()

            for item in response.get('items', []):
                snippet = item['snippet']['topLevelComment']['snippet']

                comments.append({
                    'publishedAt': snippet['publishedAt'],
                    'authorDisplayName': snippet['authorDisplayName'],
                    'comment': snippet['textDisplay'],
                    'likeCount': snippet['likeCount']
                })

            print(f"Berhasil menarik {len(comments)} komentar...")
            next_page_token = response.get('nextPageToken')

            # 🚀 KUNCI PERUBAHAN: Loop hanya berhenti jika nextPageToken tidak ada lagi (komentar habis)
            if not next_page_token:
                break

        except Exception as e:
            print(f"Proses terhenti (Kuota API habis atau error): {e}")
            break

    print(f"Selesai! Total komentar yang berhasil ditarik: {len(comments)}")
    return comments