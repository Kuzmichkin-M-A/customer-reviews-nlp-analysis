import re
import pandas as pd
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Загрузка ресурсов NLTK
# nltk.download('stopwords')
# nltk.download('wordnet')

stop_words = set(stopwords.words('english')) - {'not', 'no'}
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    '''Полный цикл очистки текста'''
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    # токенизация
    tokens = nltk.word_tokenize(text)
    # удаление стоп-слов коротких слов и лемматизация
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(cleaned)

@st.cache_data
def get_top_ngrams(corpus, n=2, limit=10):
    '''Выделение популярных словосочетаний'''
    vec = TfidfVectorizer(ngram_range=(n, n), token_pattern=r'(?u)\b[a-zA-Z]+\b').fit(corpus)

    X = vec.transform(corpus)
    sums = X.sum(axis=0)

    words_freq = [(word, sums[0, idx])for word, idx in vec.vocabulary_.items()]

    return sorted(words_freq, key=lambda x: x[1], reverse=True)[:limit]

@st.cache_data
def analyze_clusters(filtered_df, n_clusters=5):
    '''Разбивает отзывы на группы и определяет тему для каждой'''
    df = filtered_df.copy()

    if df.empty:
        return df, {}

    n_samples = len(df)

    if n_samples < 2:
        df['Cluster'] = 0
        cluster_themes = {0: 'Недостаточно отзывов для анализа'}
        return df, cluster_themes

    if n_samples < n_clusters:
        # Автоматически снижаем количество кластеров до количества отзывов
        n_clusters = n_samples
    
    # векторизация
    vectorizer = TfidfVectorizer(
        max_features=1000, 
        ngram_range=(1, 2), 
        stop_words='english'
    )
    X = vectorizer.fit_transform(df['Clean_Text'])

    # кластеризация 
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
    df['Cluster'] = kmeans.fit_predict(X)

    # Извлечение имени для каждого кластера
    cluster_themes = {}
    feature_names = vectorizer.get_feature_names_out()
    
    for i in range(n_clusters):
        # получаем средние значения TF IDF для всех значений в кластере
        cluster_center = kmeans.cluster_centers_[i]
        # берем 3 слова с самым высоким весом
        top_indices = cluster_center.argsort()[-3:][::-1]
        theme_name = ' & '.join([feature_names[idx] for idx in top_indices])
        cluster_themes[i] = theme_name

    return df, cluster_themes

@st.cache_data
def get_cluster_stats(df, cluster_themes):
    '''Считает статистику по каждому найденному кластеру (аспекту).'''
    stats = []
    for i, theme in cluster_themes.items():
        cluster_data = df[df['Cluster'] == i]
        avg_rating = cluster_data['Rating'].mean()
        sentiment_dist = cluster_data['Sentiment_Class'].value_counts(normalize=True).to_dict()
        
        stats.append({
            'cluster_id': i,
            'Theme': theme.capitalize(),
            'Count': len(cluster_data),
            'Avg_Rating': round(avg_rating, 2),
            'Pos_Share': round(sentiment_dist.get('Positive', 0) * 100, 1),
            'Neg_Share': round(sentiment_dist.get('Negative', 0) * 100, 1)
        })
    return pd.DataFrame(stats)


@st.cache_data
def generate_wordcloud(text_series):
    # Соединяем все тексты в одну гигантскую строку
    full_text = ' '.join(text_series.astype(str))
    if not full_text.strip():
        return None
    
    # Генерируем облако
    wc = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(full_text)
    
    # Создаем matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    return fig