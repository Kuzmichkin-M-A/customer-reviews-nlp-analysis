"""Small, explainable EN/RU baseline. Ratings are never sentiment features."""
import re
from functools import lru_cache
from collections import Counter
import numpy as np
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import MiniBatchKMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

RU_STOP = set('и в во на с со из у а но или это как к по за для что то мы вы они он она оно был была были быть очень уже еще всё все тоже при о об от до бы же ли я мне меня нас вам его ее их тут там этот такой свой наш ваш когда если чтобы чем чем-то'.split())
STOP = (set(ENGLISH_STOP_WORDS) | RU_STOP) - {'not', 'no', 'never', 'не', 'нет', 'ни'}
STEM = {'ru': SnowballStemmer('russian'), 'en': SnowballStemmer('english')}
VADER = SentimentIntensityAnalyzer()
POS_RU = set('хорош отличный прекрасный вкусный вкусно свежий быстрый быстро вежливый чистый чисто уютный удобно удобный понравиться понравился понравилось рекомендовать рекомендую спасибо доволен довольный лучший супер замечательный качественный качественно недорогой приветливый великолепный комфортный'.split())
NEG_RU = set('плохой плохо ужасный ужасно отвратительный грубый грубость грязный грязно медленный медленно долго дорогой дорого холодный невкусный испорченный хамство хамить сломанный сломался опоздать опоздал опоздала задержка обман разочарован разочарование душно очередь шумный шумно проблема неработающий'.split())
POS_STEMS = {STEM['ru'].stem(w) for w in POS_RU}
NEG_STEMS = {STEM['ru'].stem(w) for w in NEG_RU}


def detect_language(text):
    ru = len(re.findall('[а-яё]', text.lower()))
    en = len(re.findall('[a-z]', text.lower()))
    return 'ru' if ru > en else 'en'


def tokens(text):
    return re.findall(r"[a-zа-яё]+(?:'[a-z]+)?", str(text).lower())


@lru_cache(maxsize=50000)
def clean_text(text):
    lang = detect_language(str(text))
    return ' '.join(STEM[lang].stem(w) for w in tokens(text) if w not in STOP and len(w) > 1)


def split_clauses(text):
    return [s.strip() for s in re.split(r'[.!?;\n]+|,\s*|\b(?:but|however|although|но|зато|однако)\b', str(text), flags=re.I) if s.strip()]


def sentiment(text, language=None):
    lang = language or detect_language(text)
    words = tokens(text)
    if lang == 'en':
        result = VADER.polarity_scores(text)
        score = result['compound']
        positive, negative = result['pos'] > 0, result['neg'] > 0
        hits = sum(w in VADER.lexicon for w in words)
    else:
        values = []
        for i, word in enumerate(words):
            stem = STEM['ru'].stem(word)
            val = 1 if stem in POS_STEMS else -1 if stem in NEG_STEMS else 0
            if val and any(w in {'не', 'нет', 'ни'} for w in words[max(0, i-3):i]):
                val *= -1
            if val:
                values.append(val)
        if re.search(r'не\s+работает|не\s+стоит\s+(?:своих\s+)?денег', text, re.I):
            values.append(-1)
        positive, negative = 1 in values, -1 in values
        hits = len(values)
        score = sum(values) / max(1, len(values))
    label = 'Positive' if score >= 0.05 else 'Negative' if score <= -0.05 else 'Neutral'
    mixed = positive and negative
    if lang == 'en' and mixed:
        # A weak negative word inside an otherwise positive narrative is not enough.
        clause_scores = [VADER.polarity_scores(c) for c in split_clauses(text)]
        mixed = (any(c['compound'] >= .2 for c in clause_scores) and
                 any(c['compound'] <= -.2 for c in clause_scores)) or any(
                     c['pos'] >= .2 and c['neg'] >= .2 for c in clause_scores)
    if mixed:
        label = 'Mixed'
    sarcasm = bool(re.search(r'ага[, ]+конечно|ну да[,!]|спасибо за (?:испорч|ужас)|yeah[, ]+right|thanks for nothing|/s\b', text, re.I))
    evidence = 'none' if hits == 0 else 'limited' if hits == 1 or mixed or sarcasm else 'multiple'
    return {'Sentiment_Class': label, 'Text_Score': round(float(score), 4),
            'Mixed': mixed, 'Check_Sarcasm': sarcasm, 'Evidence': evidence, 'Language': lang}


def enrich(frame):
    df = frame.copy()
    values = [sentiment(t) for t in df['Review_Text']]
    for col in ('Sentiment_Class', 'Text_Score', 'Mixed', 'Check_Sarcasm', 'Evidence', 'Language'):
        df[col] = [r[col] for r in values]
    df['Clean_Text'] = df['Review_Text'].map(clean_text)
    df['Is_Contradictory'] = ((df['Rating_Class'] == 'Positive') & (df['Sentiment_Class'] == 'Negative')) | ((df['Rating_Class'] == 'Negative') & (df['Sentiment_Class'] == 'Positive'))
    return df


def get_top_ngrams(corpus, n=2, limit=10):
    texts = list(corpus)
    if not texts:
        return []
    try:
        vec = CountVectorizer(ngram_range=(n, n), token_pattern=r'(?u)\b\w\w+\b')
        matrix = vec.fit_transform(texts)
    except ValueError:
        return []
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    order = counts.argsort()[::-1][:limit]
    return [(vec.get_feature_names_out()[i], int(counts[i])) for i in order]


@lru_cache(maxsize=2)
def embedding_model(path):
    from pathlib import Path
    if not Path(path).is_dir():
        raise ValueError('Select an existing local model folder / Укажите папку локальной модели.')
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(path, local_files_only=True, trust_remote_code=False)


def analyze_clusters(frame, n_clusters=6, method='tfidf', model_path=''):
    df = frame.copy()
    df['Cluster'] = -1
    valid = df['Clean_Text'].str.strip().ne('')
    if not valid.any():
        return df, {-1: 'No vocabulary / Нет значимых слов'}
    vec = TfidfVectorizer(max_features=4000, ngram_range=(1, 2))
    lexical = vec.fit_transform(df.loc[valid, 'Clean_Text'])
    matrix = lexical
    if method == 'embeddings':
        matrix = embedding_model(model_path).encode(df.loc[valid, 'Review_Text'].tolist(),
                    batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    count = min(int(n_clusters), int(valid.sum()), df.loc[valid, 'Clean_Text'].nunique())
    if count == 1:
        labels = np.zeros(int(valid.sum()), dtype=int)
    else:
        model = MiniBatchKMeans(n_clusters=count, random_state=42, batch_size=1024, n_init=3)
        labels = model.fit_predict(matrix)
    df.loc[valid, 'Cluster'] = labels
    names = {}
    terms = vec.get_feature_names_out()
    for label in sorted(set(labels)):
        weights = np.asarray(lexical[labels == label].mean(axis=0)).ravel()
        top = [terms[i] for i in weights.argsort()[::-1] if weights[i] > 0][:3]
        # Readable aspect heading, with keywords as supporting hints.
        from aspects import PATTERNS, title
        original_texts = df.loc[valid].loc[labels == label, 'Review_Text']
        aspect_counts = {k: sum(bool(pattern.search(t)) for t in original_texts)
                         for k, pattern in PATTERNS.items() if k != 'other'}
        best = max(aspect_counts, key=aspect_counts.get)
        heading = title(best, detect_language(' '.join(original_texts.head(20)))) if aspect_counts[best] else 'Other topic / Другая тема'
        names[int(label)] = heading + ': ' + ' / '.join(top)
    if (~valid).any():
        names[-1] = 'No vocabulary / Нет значимых слов'
    return df, names


def generate_wordcloud(text_series):
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    counts = Counter(' '.join(text_series).split())
    if not counts:
        return None
    cloud = WordCloud(width=1000, height=400, background_color='white', random_state=42,
                      collocations=False).generate_from_frequencies(counts)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(cloud)
    ax.axis('off')
    return fig
