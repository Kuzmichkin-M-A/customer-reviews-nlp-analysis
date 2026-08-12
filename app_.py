import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from data_util_ import load_data, apply_filters
from nlp_ import clean_text, get_top_ngrams, analyze_clusters, get_cluster_stats, generate_wordcloud

st.set_page_config(page_title='Анализ отзывов', layout='wide')

st.title('Анализ отзывов')
st.markdown('---')

# Загрузка и фильтры
st.sidebar.header('Настройки данных')
uploaded_file = st.sidebar.file_uploader('Загрузите CSV-датасет', type=['csv'])

welcome_placeholder = st.empty()

if not uploaded_file:

    welcome_placeholder.info(
        'Пожалуйста, загрузите CSV-файл для начала анализа. '
        'Ожидаемые колонки: Review_ID, Rating, Year_Month, Reviewer_Location, Review_Text, Branch.')
    
else:
    welcome_placeholder.empty()

    df, error = load_data(uploaded_file)
    
    if error:
        st.error(error)
    else:
        # Инициализация фильтров
        branches = ['All'] + sorted(df['Branch'].unique().tolist())
        locations = ['All'] + sorted(df['Reviewer_Location'].unique().tolist())
        
        sel_branch = st.sidebar.selectbox('Филиал (Branch)', branches)
        sel_rating = st.sidebar.selectbox('Рейтинг', ['All', '5', '4', '3', '2', '1'])
        sel_loc = st.sidebar.selectbox('Локация автора', locations)
        
        filtered_df = apply_filters(df, sel_branch, sel_rating, sel_loc)

        # три главные вкладки
        tab1, tab2, tab3 = st.tabs(['Общая статистика', 'Анализ текста', 'Итоговое резюме'])
        
        with tab1:
            st.subheader('Обзор выборки')
            col1, col2, col3 = st.columns(3)
            col1.metric('Всего отзывов', len(filtered_df))
            col2.metric('Средний рейтинг', round(filtered_df['Rating'].mean(), 2))
            col3.metric('Локаций', filtered_df['Reviewer_Location'].nunique())
            
            # График распределения рейтингов
            fig_rating = px.histogram(filtered_df, x='Rating', color='Sentiment_Class', 
                                     title='Распределение оценок', barmode='group',
                                     color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#f1c40f', 'Negative':'#e74c3c'},
                                      labels={
                                          'Rating': 'Рейтинг',
                                          'Sentiment_Class': 'Тональность'})
            st.plotly_chart(fig_rating, use_container_width=True)
            
            rating_dist = filtered_df.groupby(['Year_Month','Branch']).size().reset_index(name = 'Counts')
            fig_ratign_by_branch = px.line(rating_dist, x = 'Year_Month', y = 'Counts', color = 'Branch',
                                           title = 'Динамика оценок во времени',
                                           color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#f1c40f', 'Negative':'#e74c3c'},
                                           labels={
                                               'Branch': 'Филиал',
                                               'Year_Month': 'Месяц и год',
                                               'Counts': 'Количество отзывов'})
            st.plotly_chart(fig_ratign_by_branch, use_container_width=True)
            
            # Динамика во времени
            time_dist = filtered_df.groupby('Year_Month').size().reset_index(name='Counts')
            fig_time = px.line(time_dist, x='Year_Month', y='Counts', title='Общая активность отзывов во времени',
                            labels={'Year_Month': 'Месяц и год', 'Counts': 'Количество отзывов'})
            st.plotly_chart(fig_time, use_container_width=True)

            #Локации  
            loc_counts = (filtered_df['Reviewer_Location'].value_counts().head(5).reset_index())
            loc_counts.columns = ['Reviewer_Location', 'Count']
            
            fig_loc = px.bar(loc_counts, x='Reviewer_Location', y='Count', title='Количество отзывов по локациям', text='Count',
                            labels={'Reviewer_Location' : 'Лоакция комментатора', 'Count': 'Количество отзывов'})
            
            fig_loc.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_loc, use_container_width=True)

            # Распределение отзывов по филиалам 
            branch_counts = (filtered_df['Branch'].value_counts().reset_index())
            branch_counts.columns = ['Branch', 'Count']
            
            fig_branch = px.pie(branch_counts, names='Branch', values='Count', title='Доля отзывов по филиалам', hole=0.6)

            fig_branch.update_layout(height=500, margin=dict(t=40, b=20, l=20, r=20))
            
            st.plotly_chart(fig_branch, use_container_width=True)
        
        with tab2:
            st.subheader('NLP Аналитика')

            # для ускорения отсекаем часть строк
            if len(filtered_df) > 10000:
                df_visual = filtered_df.sample(n=10000, random_state=42)
            else:
                df_visual = filtered_df
            
            col_word1, col_word2 = st.columns(2)
            
            with col_word1:
                st.write('**Облако ключевых слов**')
                with st.spinner('Генерация облака слов...'):
                    fig_wc = generate_wordcloud(df_visual['Clean_Text'])
                    
                    if fig_wc is not None:
                        st.pyplot(fig_wc)
                    else:
                        st.info('Нет доступного текста для построения облака слов.')
                
            with col_word2:
                st.write('**Топ-10 биграмм (словосочетаний)**')
                bigrams = get_top_ngrams(df_visual['Clean_Text'], n=2)
                bigr_df = pd.DataFrame(bigrams, columns=['Phrase', 'Score'])
                    
                if not bigr_df.empty:
                    fig_bg = px.bar(bigr_df, y='Phrase', x='Score', orientation='h', color='Score')

                    fig_bg.update_layout(yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_bg, use_container_width=True)
                else:
                    st.info('Недостаточно данных для расчета биграмм.')

            st.markdown('---')
            st.subheader('Анализ скрытых противоречий (Рейтинг vs Текст)')
        
            contradictions_df = filtered_df[filtered_df['Is_Contradictory'] == True]
        
            st.metric('Противоречивых отзывов', len(contradictions_df), help='Отзывы, где оценка пользователя расходится с тональностью написанного комментария.')
                    
            if not contradictions_df.empty:
                max_rows = min(len(contradictions_df), 100)
                num_show = st.number_input('Сколько отзывов показать?', min_value=1, max_value=max_rows, value=min(5, max_rows), step=1)
                st.write(f'**Отображено {num_show} из {len(contradictions_df)} противоречивых отзывов:**')
                # Фиксированная высота, чтобы таблица скроллилась, если строк много) 
                st.dataframe(contradictions_df[['Rating', 'Sentiment_Class', 'Review_Text']].head(num_show), width='stretch') 
            else:
                st.success('Противоречий не обнаружено! Все оценки соответствуют текстам.')

        with tab3:
            st.subheader('Автоматическое резюме на основе ML-кластеризации')
            
            # Запускаем кластеризацию 
            n_cl = st.slider('Количество тем для поиска', 3, 10, 6)
            
            with st.spinner('Алгоритм ищет скрытые темы в отзывах...'):
                df_clustered, themes = analyze_clusters(filtered_df, n_clusters=n_cl)
                stats_df = get_cluster_stats(df_clustered, themes)
            
            # Выводим summary 
            col_sum1, col_sum2 = st.columns(2)
            
            with col_sum1:
                st.success('Основные плюсы (Темы с высоким рейтингом)')
                # Сортируем по доле позитива и берем топ-3
                top_pos = stats_df.sort_values(by='Pos_Share', ascending=False).head(3)
                for _, row in top_pos.iterrows():
                    st.markdown(f'**{row['Theme']}**')
                    st.caption(f'Доля позитива: {row['Pos_Share']}% | Ср. оценка: {row['Avg_Rating']}')
        
            with col_sum2:
                st.error('Основные минусы (Темы с жалобами)')
                # Сортируем по доле негатива
                top_neg = stats_df.sort_values(by='Neg_Share', ascending=False).head(3)
                for _, row in top_neg.iterrows():
                    st.markdown(f'**{row['Theme']}**')
                    st.caption(f'Доля негатива: {row['Neg_Share']}% | Ср. оценка: {row['Avg_Rating']}')
        
            st.write('---')
            st.write('**Детальная таблица аспектов:**')
            st.dataframe(stats_df[['Theme', 'Count', 'Avg_Rating', 'Pos_Share', 'Neg_Share']], use_container_width=True)
