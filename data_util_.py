import streamlit as st
import pandas as pd
import numpy as np
from nlp_ import clean_text
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from multiprocessing import Pool, cpu_count

analyzer = SentimentIntensityAnalyzer()

def _analyze_single_text(text):
    score = analyzer.polarity_scores(str(text))['compound']
    if score >= 0.05: return score, 'Positive'
    elif score <= -0.05: return score, 'Negative'
    else: return score, 'Neutral'

@st.cache_data(show_spinner='Загрузка и предобработка данных...')
def load_data(file):
    '''Загрузка CSV и предобработка
    Загружается датасет, проверяется структура (наличие необходимых столбцов, обработка пропусков)
    Анализируется тональность
    '''
    try:
        df = pd.read_csv(file, encoding='cp1252')
        required_cols = ['Review_ID', 'Rating', 'Year_Month', 'Reviewer_Location', 'Review_Text', 'Branch']
        
        # Проверка наличия колонок
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return None, f'В файле отсутствуют колонки: {', '.join(missing_cols)}'
        
        # Обработка пропусков
        df = df.dropna(subset=['Review_Text', 'Rating'])
        
        # Приведение типов
        df['Rating'] = df['Rating'].astype(int)
        df['Year_Month'] = pd.to_datetime(df['Year_Month'], format='%Y-%m', errors='coerce')
        df = df.dropna(subset=['Year_Month'])
        df['Year_Month'] = df['Year_Month'].dt.to_period('M').astype(str)
        
        # Оценка классов отзывов
        df['Rating_Class'] = df['Rating'].apply(lambda x: 'Positive' if x >= 4 else ('Negative' if x <= 2 else 'Neutral'))

        df['Clean_Text'] = df['Review_Text'].apply(clean_text)

        # Анализ тональности
        with Pool(cpu_count()) as pool:
            vader_results = pool.map(_analyze_single_text, df['Review_Text'].tolist())
        
        # Распаковываем результаты в DataFrame
        df['Vader_Score'] = [r[0] for r in vader_results]
        df['Sentiment_Class'] = [r[1] for r in vader_results]
        
        df['Is_Contradictory'] = ((df['Rating_Class'] == 'Positive') & (df['Sentiment_Class'] == 'Negative')) | \
                                 ((df['Rating_Class'] == 'Negative') & (df['Sentiment_Class'] == 'Positive'))
        
        return df, None
    except Exception as e:
        return None, str(e)

def apply_filters(df, branch, rating, location):
    '''Фильтрация датасета по выбору
    создаем копию датасета
    если стоит не алл применяем определенный фильтр
    '''
    filtered_df = df.copy()
    if branch != 'All':
        filtered_df = filtered_df[filtered_df['Branch'] == branch]
    if rating != 'All':
        filtered_df = filtered_df[filtered_df['Rating'] == int(rating)]
    if location != 'All':
        filtered_df = filtered_df[filtered_df['Reviewer_Location'] == location]
    return filtered_df