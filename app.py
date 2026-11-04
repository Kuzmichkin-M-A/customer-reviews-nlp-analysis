"""ReviewLens Streamlit app. Run with: streamlit run app.py"""
from pathlib import Path
from io import BytesIO
from datetime import datetime, timezone
import hashlib
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from reviewlens.data_io import read_table, suggest_columns, prepare_data, csv_bytes, ALIASES
from reviewlens.nlp import enrich, analyze_clusters, get_top_ngrams, generate_wordcloud
from reviewlens.aspects import extract_aspects, apply_aspect_edits, aspect_stats, title
from reviewlens.reporting import top_findings, make_report, compare_periods
from reviewlens.style import APP_NAME, CSS, PRIMARY, SENTIMENT_COLORS, style_fig, translate_legend

ROOT = Path(__file__).parent
st.set_page_config(page_title=APP_NAME, page_icon='🔍', layout='wide')
st.markdown(CSS, unsafe_allow_html=True)
st.sidebar.markdown(f'## 🔍 {APP_NAME}')
lang = st.sidebar.selectbox('Язык интерфейса / Interface language', ['ru', 'en'])


def t(ru, en):
    return ru if lang == 'ru' else en


SENTIMENT = {'Positive': t('Положительная', 'Positive'), 'Negative': t('Отрицательная', 'Negative'),
             'Neutral': t('Нейтральная / нет сигнала', 'Neutral / no signal'), 'Mixed': t('Смешанная', 'Mixed')}
COLORS = SENTIMENT_COLORS


def show_chart(fig, height=400):
    '''Показывает график Plotly в едином стиле с переведённой легендой тональности'''
    fig = translate_legend(style_fig(fig, height=height), SENTIMENT)
    fig.update_traces(textposition='outside', cliponaxis=False, selector=dict(type='bar'))
    st.plotly_chart(fig, width='stretch', theme=None)


def describe_meta(meta):
    '''Человекочитаемое описание того, как прочитан файл'''
    if meta.get('encoding') == 'XLSX':
        return t(f"Excel · лист {meta.get('sheet', '')}", f"Excel · sheet {meta.get('sheet', '')}")
    separator = str(meta.get('separator', '')).strip("'\"")
    separator = {'\\t': 'TAB', '\t': 'TAB'}.get(separator, separator)
    return t(f"кодировка {meta.get('encoding')} · разделитель «{separator}»",
             f"encoding {meta.get('encoding')} · delimiter “{separator}”")
FIELD_LABELS = {'Review_Text': t('Текст отзыва *', 'Review text *'), 'Rating': t('Оценка 1–5', 'Rating 1–5'),
                'Date': t('Дата (ISO: ГГГГ-ММ-ДД)', 'Date (ISO: YYYY-MM-DD)'),
                'Branch': t('Филиал / компания / товар', 'Branch / company / product'),
                'Reviewer_Location': t('Локация автора', 'Reviewer location'),
                'Original_ID': t('Исходный ID', 'Original ID'), 'Source_URL': t('Ссылка на отзыв', 'Review URL')}


@st.cache_data(show_spinner=False, max_entries=4)
def parsed(payload, name, encoding, sep, sheet):
    return read_table(payload, name, encoding, sep, sheet)


@st.cache_data(show_spinner=False, max_entries=4)
def processed(raw, mapping):
    df, validation = prepare_data(raw, mapping)
    return enrich(df), validation


@st.cache_data(show_spinner=False, max_entries=8)
def mentions_for(df, template, interface_language):
    return extract_aspects(df, template, interface_language)


@st.cache_data(show_spinner=False, max_entries=4)
def clusters_for(df, count, method, model_path):
    return analyze_clusters(df, count, method, model_path)


def evidence(group, reviews, limit=5):
    joined = group.merge(reviews, on='Review_ID', suffixes=('_Aspect', ''), validate='many_to_one')
    for row in joined.head(limit).itertuples():
        st.caption(SENTIMENT.get(getattr(row, 'Sentiment_Class_Aspect', row.Sentiment_Class), ''))
        st.text(getattr(row, 'Snippet', row.Review_Text))
        with st.popover(f"{row.Review_ID} · {t('исходная строка', 'source row')} {row.Source_Row}"):
            st.text(row.Review_Text)
            st.caption(f"{t('Оценка', 'Rating')}: {row.Rating} · {row.Branch}")
            if pd.notna(row.Source_URL) and str(row.Source_URL).startswith(('https://', 'http://')):
                st.link_button(t('Открыть ссылку из файла', 'Open source URL'), str(row.Source_URL))


if 'reviews' not in st.session_state:
    st.markdown(
        '<div class="hero-title">' + t('Анализ отзывов <span>с доказательствами</span>',
                                       'Review analysis <span>with evidence</span>') + '</div>'
        '<div class="hero-sub">' + t('Загрузите отзывы клиентов — и увидите главные проблемы и сильные стороны. '
                                     'Каждый вывод можно проверить по исходным отзывам.',
                                     'Upload customer reviews to see the main problems and strengths. '
                                     'Every finding can be checked against the source reviews.') + '</div>',
        unsafe_allow_html=True)
else:
    st.markdown(f'<div class="report-title">{APP_NAME} · <span>{t("Отчёт", "report")}</span></div>',
                unsafe_allow_html=True)
    st.caption(t('Анализ отзывов с проверяемыми выводами · учебный прототип', 'Review analysis with traceable findings · educational prototype'))
st.sidebar.caption(t('Сервис анализа отзывов', 'Review analysis service'))

with st.sidebar:
    st.header(t('Загрузка', 'Import'), anchor=False)
    uploaded = st.file_uploader(t('Файл с отзывами', 'Reviews file'), type=['csv', 'xlsx'])
    demo = st.selectbox(t('Или пример', 'Or demo'), ['—', 'RU demo', 'EN demo'])
    encoding = st.selectbox(t('Кодировка CSV', 'CSV encoding'), ['auto', 'utf-8-sig', 'cp1251', 'cp1252', 'utf-16'])
    sep_name = st.selectbox(t('Разделитель CSV', 'CSV delimiter'), ['auto', ',', ';', 'TAB', '|'])
    sep = '\t' if sep_name == 'TAB' else sep_name

payload = uploaded.getvalue() if uploaded else None
filename = uploaded.name if uploaded else None
if payload is None and demo != '—':
    filename = 'demo_ru.csv' if demo == 'RU demo' else 'demo_en.csv'
    payload = (ROOT/'data'/filename).read_bytes()

if payload is not None:
    digest = hashlib.sha256(payload).hexdigest()[:16]
    sheet = 0
    try:
        if filename.lower().endswith('.xlsx'):
            sheets = pd.ExcelFile(BytesIO(payload)).sheet_names
            sheet = st.selectbox(t('Лист Excel', 'Excel sheet'), sheets)
        raw, meta = parsed(payload, filename, encoding, sep, sheet)
        input_identity = f'{digest}-{sheet}'
        active_input = st.session_state.get('input_identity') == input_identity
        import_container = st.sidebar if active_input else st
        with import_container.expander(t('Изменить сопоставление колонок', 'Change column mapping') if active_input else t('Предпросмотр и сопоставление колонок', 'Preview and column mapping'),
                         expanded=st.session_state.get('input_identity') != input_identity):
            st.dataframe(raw.head(5), hide_index=True)
            st.caption(f"{len(raw)} {t('строк', 'rows')} · {describe_meta(meta)}")
            suggestions = suggest_columns(raw.columns)
            with st.form(f'mapping-{digest}-{sheet}'):
                mapping = {}
                fields = st.columns(2)
                for i, field in enumerate(ALIASES):
                    options = [None] + raw.columns.tolist()
                    mapping[field] = fields[i % 2].selectbox(FIELD_LABELS[field], options,
                        index=options.index(suggestions[field]),
                        format_func=lambda v: t('Не используется', 'Not used') if v is None else v,
                        key=f'map-{digest}-{sheet}-{field}')
                analyze = st.form_submit_button(t('Анализировать', 'Analyze'), type='primary')
            if analyze:
                with st.spinner(t('Обработка текстов…', 'Processing texts…')):
                    reviews, validation = processed(raw, mapping)
                if reviews.empty:
                    st.error(t('После проверки не осталось текстовых отзывов.', 'No text reviews remain after validation.'))
                else:
                    run_id = hashlib.sha256((digest + str(sheet) + json.dumps(mapping, sort_keys=True)).encode()).hexdigest()[:16]
                    st.session_state.update(reviews=reviews, validation=validation, source=filename, run_id=run_id,
                                            input_identity=input_identity)
                    history = st.session_state.setdefault('history', [])
                    history.append({'Time_UTC': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                                    'File': filename, 'Run': run_id, 'Reviews': len(reviews),
                                    'Negative_share': round(reviews['Sentiment_Class'].eq('Negative').mean()*100, 1)})
                    st.session_state['history'] = history[-20:]
                    st.rerun()
    except (ValueError, UnicodeError, OSError, ImportError) as exc:
        st.error(f"{t('Ошибка загрузки', 'Import error')}: {exc}")

if 'reviews' not in st.session_state:
    if payload is None:
        # Главная страница до загрузки данных
        st.write('')
        features = [
            ('🎯', t('Главные выводы', 'Main findings'), t('До 3 проблем и 3 сильных сторон по аспектам', 'Up to 3 problems and 3 strengths by aspect')),
            ('🔎', t('Доказательства', 'Evidence'), t('Цитаты и исходные отзывы под каждым выводом', 'Quotes and source reviews under every finding')),
            ('🆚', t('Сравнение', 'Comparison'), t('Периоды, филиалы, компании или товары', 'Periods, branches, companies or products')),
            ('🌐', t('RU и EN', 'RU and EN'), t('Отзывы и интерфейс на русском и английском', 'Reviews and interface in Russian and English')),
        ]
        for col, (icon, name, text) in zip(st.columns(4), features):
            with col:
                with st.container(border=True, height=190):
                    st.markdown(f'<div class="feature-icon">{icon}</div>', unsafe_allow_html=True)
                    st.markdown(f'**{name}**')
                    st.caption(text)
        st.write('')
        st.subheader(t('Как это работает', 'How it works'), anchor=False)
        steps = [
            (t('Загрузите файл', 'Upload a file'), t('CSV или XLSX — или откройте пример слева', 'CSV or XLSX — or open a demo on the left')),
            (t('Проверьте колонки', 'Check the columns'), t('Обязательна только колонка с текстом', 'Only the text column is required')),
            (t('Нажмите «Анализировать»', 'Click Analyze'), t('Откройте цитаты под выводами', 'Open the quotes under each finding')),
        ]
        for i, (col, (name, text)) in enumerate(zip(st.columns(3), steps), start=1):
            with col:
                st.markdown(f'<span class="step-num">{i}</span><b>{name}</b>', unsafe_allow_html=True)
                st.caption(text)
        st.markdown('<div class="cta">' + t('⬅️ Чтобы начать, загрузите файл или выберите пример в панели слева',
                                            '⬅️ To start, upload a file or choose a demo in the left panel') + '</div>',
                    unsafe_allow_html=True)
        with st.expander(t('Какой формат файла нужен?', 'What file format is needed?')):
            st.markdown(t('Одна строка — один отзыв. Обязательна только колонка с текстом; оценка, дата, филиал и локация — по желанию. '
                          'Рейтинг не определяет тональность текста. Даты: ГГГГ-ММ или ГГГГ-ММ-ДД.',
                          'One row per review. Only the text column is required; rating, date, branch and location are optional. '
                          'Rating never determines text sentiment. Dates: YYYY-MM or YYYY-MM-DD.'))
            st.download_button(t('Скачать шаблон CSV', 'Download CSV template'),
                               (ROOT/'data'/'template.csv').read_bytes(), 'reviews_template.csv', 'text/csv')
    else:
        st.info(t('Проверьте сопоставление колонок и нажмите «Анализировать».', 'Check the column mapping and click Analyze.'))
    st.stop()

reviews = st.session_state.reviews
run_id = st.session_state.run_id
source = st.session_state.source
st.caption(f"{t('Активный анализ', 'Active analysis')}: {source} · {run_id}")
with st.sidebar.expander(t('Проверка данных', 'Data validation')):
    st.json(st.session_state.validation)
    st.write(t('Удаляются только строки без буквенного текста. Невалидные оценки и даты остаются пропусками. Повторяющиеся тексты сохраняются: это могут быть разные отзывы. Исходная строка сохраняется.',
               'Only rows without alphabetic text are removed. Invalid ratings and dates become missing values. Repeated texts are kept: they may be separate reviews. Source row numbers are preserved.'))

with st.sidebar:
    st.header(t('Фильтры', 'Filters'), anchor=False)
    df = reviews.copy()
    for col, label in [('Branch', FIELD_LABELS['Branch']), ('Reviewer_Location', FIELD_LABELS['Reviewer_Location']),
                       ('Sentiment_Class', t('Тональность текста', 'Text sentiment'))]:
        choices = st.multiselect(label, sorted(reviews[col].unique()), key=f'filter-{col}-{run_id}',
                                format_func=lambda v: SENTIMENT.get(v, v), placeholder=t('Все', 'All'))
        if choices:
            df = df[df[col].isin(choices)]
    rating_filter = st.multiselect(t('Рейтинг', 'Rating'), [1, 2, 3, 4, 5], key=f'rating-{run_id}',
                                   placeholder=t('Все', 'All'))
    if rating_filter:
        df = df[df['Rating'].isin(rating_filter)]
    dated = reviews['Date'].dropna()
    if not dated.empty and st.checkbox(t('Ограничить даты', 'Filter by date'), key=f'dates-{run_id}'):
        dates = st.date_input(t('Диапазон дат', 'Date range'), (dated.min().date(), dated.max().date()), key=f'range-{run_id}')
        if len(dates) == 2:
            df = df[df['Date'].ge(pd.Timestamp(dates[0])) & df['Date'].lt(pd.Timestamp(dates[1]) + pd.Timedelta(days=1))]
    template = st.selectbox(t('Шаблон аспектов', 'Aspect template'), ['general', 'food', 'ecommerce', 'services'],
        format_func=lambda x: {'general': t('Общий / развлечения', 'General / entertainment'), 'food': t('Общепит', 'Food service'),
                              'ecommerce': 'E-commerce', 'services': t('Услуги', 'Services')}[x])

if df.empty:
    st.warning(t('По этим фильтрам отзывов нет. Измените фильтры.', 'No reviews match. Adjust the filters.'))
    st.stop()
mentions = mentions_for(df, template, lang)
edit_key = f'editor-{run_id}-{template}-{lang}'
with st.expander(t('Редактор аспектов: переименовать, объединить, скрыть', 'Aspect editor: rename, merge, hide')):
    st.write(t('Одинаковое название объединяет аспекты. Каждый отзыв в объединённом аспекте учитывается один раз. Скрытые аспекты исключаются из сводки.',
               'Use the same name to merge aspects. A review is counted once in a merged aspect. Hidden aspects are excluded from the summary.'))
    # Include all template aspects, so edits survive changes to filters.
    from reviewlens.aspects import TEMPLATES
    defaults = pd.DataFrame([{'Aspect_Key': k, 'Name': title(k, lang), 'Hidden': False} for k in TEMPLATES[template]])
    if f'restored-{edit_key}' in st.session_state:
        st.session_state[f'base-{edit_key}'] = st.session_state.pop(f'restored-{edit_key}')
        st.session_state.pop(edit_key, None)
    defaults = st.session_state.get(f'base-{edit_key}', defaults)
    edits = st.data_editor(defaults, hide_index=True, disabled=['Aspect_Key'], key=edit_key,
                           column_config={'Name': st.column_config.TextColumn(t('Название', 'Name'), required=True),
                                          'Hidden': st.column_config.CheckboxColumn(t('Скрыть', 'Hide'))})
mentions = apply_aspect_edits(mentions, edits)
stats = aspect_stats(mentions, df)
named = stats[~stats['Aspect'].isin(mentions.loc[mentions['Aspect_Key'].eq('other'), 'Aspect'])]
report = make_report(named, df, source, lang)

metrics = st.columns(4)
metrics[0].metric(t('Отзывов', 'Reviews'), len(df))
metrics[1].metric(t('Отрицательных по тексту', 'Negative by text'), f"{df['Sentiment_Class'].eq('Negative').mean():.1%}")
metrics[2].metric(t('Смешанных', 'Mixed'), int(df['Mixed'].sum()))
metrics[3].metric(t('Средняя оценка', 'Average rating'), f"{df['Rating'].mean():.2f}" if df['Rating'].notna().any() else '—')
tabs = st.tabs(['🎯 ' + t('Главные выводы', 'Main findings'), '📊 ' + t('Статистика', 'Statistics'),
                '💬 ' + t('Темы и слова', 'Topics and words'), '🆚 ' + t('Периоды и сравнение', 'Periods and comparison'),
                '📄 ' + t('Исходные отзывы', 'Source reviews'), '⬇️ ' + t('История и экспорт', 'History and export')])

with tabs[0]:
    st.caption(t('По числу полярных упоминаний: до 3 проблем и 3 сильных сторон. Это словарные подсказки для проверки. «Без определённого аспекта» в лидеры не входит.',
                 'By polar mention count: up to 3 problems and 3 strengths. These are lexicon-based hints to review. Unspecified aspects are excluded from leaders.'))
    for column, polarity, heading in zip(st.columns(2), ['Negative', 'Positive'], [t('Главные проблемы', 'Main problems'), t('Сильные стороны', 'Strengths')]):
        with column:
            st.subheader(heading, anchor=False)
            findings = top_findings(named, polarity)
            if findings.empty:
                st.info(t('Подходящих упоминаний не найдено.', 'No matching mentions found.'))
            for item in findings.itertuples():
                count = getattr(item, polarity)
                with st.expander(f'{item.Aspect}: {count} / {item.Count}', expanded=True):
                    st.caption(t('Числитель — отзывы с этой тональностью по аспекту, знаменатель — все отзывы об аспекте.',
                                 'Numerator: reviews with this aspect sentiment. Denominator: all reviews mentioning the aspect.'))
                    selected = mentions[(mentions['Aspect'] == item.Aspect) & (mentions['Sentiment_Class'] == polarity)]
                    evidence(selected, df)
    st.subheader(t('Все аспекты', 'All aspects'), anchor=False)
    st.dataframe(stats, hide_index=True)
    st.caption(t('Один отзыв может упоминать несколько аспектов. Доли Positive и Negative не включают Mixed. Примеры приведены в порядке строк источника.',
                 'One review can mention multiple aspects. Positive and Negative shares exclude Mixed. Examples follow source row order.'))
    if not stats.empty:
        aspect = st.selectbox(t('Проверить аспект', 'Inspect an aspect'), stats['Aspect'])
        evidence(mentions[mentions['Aspect'] == aspect], df)
        with st.form(f'feedback-{run_id}'):
            reason = st.text_input(t('Почему вывод неверен?', 'Why is the finding incorrect?'))
            if st.form_submit_button(t('Вывод неверен — сохранить замечание', 'Flag incorrect finding')):
                st.session_state.setdefault('feedback', []).append({'Run': run_id, 'Aspect': aspect, 'Comment': reason})
                st.success(t('Замечание сохранено в сеансе. Скачайте его во вкладке экспорта.', 'Feedback saved in this session. Download it in the export tab.'))

with tabs[1]:
    labels = {'Sentiment': t('Тональность', 'Sentiment'), 'Sentiment_Class': t('Тональность текста', 'Text sentiment'),
              'Count': t('Количество отзывов', 'Reviews'), 'Rating': t('Оценка', 'Rating'), 'Month': t('Месяц', 'Month')}
    counts = df['Sentiment_Class'].value_counts().rename_axis('Sentiment').reset_index(name='Count')
    counts['Label'] = counts['Sentiment'].map(SENTIMENT)
    fig = px.bar(counts, x='Label', y='Count', color='Sentiment', color_discrete_map=COLORS, text='Count',
                 title=t('Тональность текста', 'Text sentiment'), labels={**labels, 'Label': ''})
    fig.update_layout(showlegend=False)
    show_chart(fig, height=360)
    if df['Rating'].notna().any():
        fig = px.histogram(df.dropna(subset=['Rating']), x='Rating', color='Sentiment_Class',
                           color_discrete_map=COLORS, barmode='group', nbins=5,
                           title=t('Оценки и тональность текста', 'Ratings and text sentiment'), labels=labels)
        fig.update_yaxes(title_text=labels['Count'])
        fig.update_xaxes(dtick=1)
        show_chart(fig)
    if df['Date'].notna().any():
        timeline = df.dropna(subset=['Date']).assign(Month=lambda x: x['Date'].dt.to_period('M').astype(str)).groupby(['Month', 'Sentiment_Class']).size().reset_index(name='Count')
        show_chart(px.line(timeline, x='Month', y='Count', color='Sentiment_Class', color_discrete_map=COLORS, markers=True,
                           title=t('Количество отзывов по месяцам', 'Review volume by month'), labels=labels))
    for column, col in zip(st.columns(2), ('Branch', 'Reviewer_Location')):
        dist = df[col].value_counts().head(10).rename_axis(col).reset_index(name='Count')
        fig = px.bar(dist, x='Count', y=col, orientation='h', text='Count', color_discrete_sequence=[PRIMARY],
                     title=FIELD_LABELS[col], labels={col: '', 'Count': labels['Count']})
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        with column:
            show_chart(fig)
    flagged = df[df['Is_Contradictory'] | df['Check_Sarcasm'] | df['Mixed']]
    st.subheader(t('Отзывы для ручной проверки', 'Reviews for manual review'), anchor=False)
    st.caption(t('Несовпадение рейтинга и текста, смешанная тональность или маркер возможной иронии. Это не доказательство сарказма.',
                 'Rating/text disagreement, mixed sentiment or a possible irony marker. These flags do not prove sarcasm.'))
    st.dataframe(flagged[['Review_ID', 'Review_Text', 'Rating', 'Sentiment_Class', 'Is_Contradictory', 'Mixed', 'Check_Sarcasm']], hide_index=True)

with tabs[2]:
    st.caption(t('Автоматические темы — отдельная исследовательская группировка. Названия состоят из ключевых слов. Это не аспектная тональность.',
                 'Automatic topics are a separate exploratory grouping. Labels use keywords. Topic clustering is separate from aspect sentiment.'))
    method = st.selectbox(t('Метод группировки', 'Clustering method'), ['tfidf', 'embeddings'])
    model_path = st.text_input(t('Папка локальной мультиязычной Sentence Transformers модели', 'Local multilingual Sentence Transformers model folder')) if method == 'embeddings' else ''
    count = st.slider(t('Максимум тем', 'Maximum topics'), 1, 10, 6)
    if st.button(t('Найти темы', 'Find topics')):
        try:
            with st.spinner(t('Поиск тем…', 'Finding topics…')):
                clustered, themes = clusters_for(df, count, method, model_path)
            st.session_state['topics'] = (run_id, hashlib.sha256(csv_bytes(df[['Review_ID']])).hexdigest(), clustered, themes, method)
        except Exception as exc:
            st.error(f"{t('Не удалось выполнить группировку. TF-IDF доступен без модели.', 'Clustering failed. TF-IDF works without a model.')} {exc}")
    saved = st.session_state.get('topics')
    current_filter = hashlib.sha256(csv_bytes(df[['Review_ID']])).hexdigest()
    if saved and saved[0] == run_id and saved[1] == current_filter:
        _, _, clustered, themes, used_method = saved
        st.caption(f'{t("Рассчитанный метод", "Computed method")}: {used_method}')
        theme_id = st.selectbox(t('Тема', 'Topic'), list(themes), format_func=lambda i: f'{i}: {themes[i]}')
        group = clustered[clustered['Cluster'] == theme_id]
        st.write(f'{len(group)} {t("отзывов", "reviews")}')
        evidence(group[['Review_ID']], df)
        st.download_button(t('Темы CSV', 'Topics CSV'), csv_bytes(clustered), 'topics.csv', 'text/csv')
    grams = get_top_ngrams(df['Clean_Text'])
    if grams:
        fig = px.bar(pd.DataFrame(grams, columns=['Phrase', 'Count']), x='Count', y='Phrase', orientation='h',
                     color_discrete_sequence=[PRIMARY],
                     title=t('Частые биграммы (число вхождений)', 'Frequent bigrams (occurrences)'),
                     labels={'Phrase': '', 'Count': t('Число вхождений', 'Occurrences')})
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        show_chart(fig)
    if st.checkbox(t('Показать облако слов', 'Show word cloud')):
        fig = generate_wordcloud(df['Clean_Text'])
        if fig:
            st.pyplot(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)
        else:
            st.info(t('Нет значимых слов.', 'No significant words.'))

with tabs[3]:
    st.subheader(t('Сравнение периодов и дайджест', 'Period comparison and digest'), anchor=False)
    st.caption(t('Даты берутся из текущей отфильтрованной выборки. Дайджест формируется по кнопке и скачивается. Фоновых рассылок нет.',
                 'Dates use the current filtered selection. Generate and download a digest manually.'))
    dated = df['Date'].dropna()
    if not dated.empty:
        end = dated.max().normalize()
        a = st.date_input(t('Предыдущий период', 'Previous period'), ((end-pd.Timedelta(days=13)).date(), (end-pd.Timedelta(days=7)).date()))
        b = st.date_input(t('Текущий период', 'Current period'), ((end-pd.Timedelta(days=6)).date(), end.date()))
        threshold = st.slider(t('Порог роста негатива, п.п.', 'Negative share increase threshold, pp'), 1, 100, 15)
        minimum = st.number_input(t('Минимум отзывов об аспекте в каждом периоде', 'Minimum aspect reviews in each period'), 1, 1000, 5)
        if len(a) == 2 and len(b) == 2:
            try:
                comparison = compare_periods(df, mentions, a, b, threshold, minimum)
                st.dataframe(comparison, hide_index=True)
                st.caption(t('Change_pp — разница долей негатива в процентных пунктах. При отсутствии отзывов доля неизвестна. Alert срабатывает только при достаточной выборке в обоих периодах.',
                             'Change_pp is the negative share difference in percentage points. Shares are unknown when no reviews exist. Alert requires sufficient reviews in both periods.'))
                if comparison['Alert'].any():
                    st.warning(t('Рост негатива: ', 'Increasing negative share: ') + ', '.join(comparison.loc[comparison['Alert'], 'Aspect']))
                current = df[df['Date'].ge(pd.Timestamp(b[0])) & df['Date'].lt(pd.Timestamp(b[1])+pd.Timedelta(days=1))]
                current_mentions = mentions[mentions['Review_ID'].isin(current['Review_ID'])]
                digest_report = make_report(aspect_stats(current_mentions[current_mentions['Aspect_Key'] != 'other'], current), current, f'{source} {b[0]} — {b[1]}', lang)
                st.download_button(t('Скачать дайджест текущего периода', 'Download current period digest'), digest_report.encode('utf-8'), 'digest.txt', 'text/plain')
                st.download_button(t('Сравнение CSV', 'Comparison CSV'), csv_bytes(comparison), 'periods.csv', 'text/csv')
            except ValueError as exc:
                st.warning(str(exc))
    else:
        st.info(t('Для сравнения периодов нужна колонка с датой.', 'A date column is needed for period comparison.'))
    if df['Branch'].nunique() > 1:
        st.subheader(t('Сравнение филиалов / компаний / товаров', 'Compare branches / companies / products'), anchor=False)
        groups = st.multiselect(t('До 4 групп из файла', 'Up to 4 groups from the file'), sorted(df['Branch'].unique()), max_selections=4)
        pieces = []
        for branch in groups:
            subset = df[df['Branch'] == branch]
            item = aspect_stats(mentions[mentions['Review_ID'].isin(subset['Review_ID'])], subset)
            item['Branch'] = branch
            pieces.append(item)
        if pieces:
            st.dataframe(pd.concat(pieces, ignore_index=True), hide_index=True)

with tabs[4]:
    st.dataframe(df.drop(columns=['Clean_Text']), hide_index=True)
    ids = df['Review_ID'].tolist()
    selected = st.selectbox(t('Открыть исходный отзыв по ID', 'Open a source review by ID'), ids)
    shown_ids = [selected]
    for rid in shown_ids:
        row = df[df['Review_ID'] == rid].iloc[0]
        with st.expander(f"{rid} · {t('строка', 'row')} {row.Source_Row}", expanded=rid == selected):
            st.text(row.Review_Text)
            st.caption(f"{SENTIMENT[row.Sentiment_Class]} · {t('Оценка', 'Rating')}: {row.Rating} · {row.Branch}")
            if pd.notna(row.Source_URL) and str(row.Source_URL).startswith(('https://', 'http://')):
                st.link_button(t('Открыть ссылку из файла', 'Open source URL'), str(row.Source_URL))

with tabs[5]:
    st.subheader(t('Скачать результаты', 'Download results'), anchor=False)
    st.download_button(t('Краткий отчёт TXT', 'Summary TXT'), report.encode('utf-8'), 'report.txt', 'text/plain')
    st.download_button(t('Аспекты CSV', 'Aspect stats CSV'), csv_bytes(stats), 'aspects.csv', 'text/csv')
    st.download_button(t('Упоминания с цитатами CSV', 'Mentions with quotes CSV'), csv_bytes(mentions), 'mentions.csv', 'text/csv')
    st.download_button(t('Разобранные отзывы CSV', 'Analyzed reviews CSV'), csv_bytes(df), 'reviews_analyzed.csv', 'text/csv')
    st.download_button(t('Настройки аспектов JSON', 'Aspect settings JSON'), edits.to_json(orient='records', force_ascii=False, indent=2).encode('utf-8'), 'aspect_settings.json')
    settings_file = st.file_uploader(t('Восстановить настройки аспектов JSON', 'Restore aspect settings JSON'), type=['json'])
    if settings_file and st.button(t('Применить настройки', 'Apply settings')):
        try:
            restored = pd.DataFrame(json.loads(settings_file.getvalue()))
            if set(restored.columns) != {'Aspect_Key', 'Name', 'Hidden'} or not restored['Aspect_Key'].is_unique:
                raise ValueError('Invalid settings schema')
            if not restored['Aspect_Key'].isin(TEMPLATES[template]).all() or not restored['Hidden'].map(lambda x: isinstance(x, bool)).all():
                raise ValueError('Settings do not match the selected template')
            if not restored['Name'].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
                raise ValueError('Names must be non-empty strings')
            st.session_state[f'restored-{edit_key}'] = restored
            st.rerun()
        except (ValueError, TypeError, KeyError) as exc:
            st.error(str(exc))
    history = pd.DataFrame(st.session_state.get('history', []))
    st.subheader(t('История загрузок этого сеанса', 'Upload history for this session'), anchor=False)
    st.caption(t('Хранятся последние 20 запусков. История и обратная связь исчезают при закрытии сеанса. Скачайте нужные результаты. Negative_share считается по тексту и не включает Mixed.',
                 'Last 20 runs. History and feedback disappear when the session ends. Download results to keep them. Negative_share uses text and excludes Mixed.'))
    st.dataframe(history, hide_index=True)
    st.download_button(t('История CSV', 'History CSV'), csv_bytes(history), 'history.csv', 'text/csv')
    feedback = pd.DataFrame(st.session_state.get('feedback', []), columns=['Run', 'Aspect', 'Comment'])
    st.download_button(t('Замечания CSV', 'Feedback CSV'), csv_bytes(feedback), 'feedback.csv', 'text/csv')
    st.caption(t('Обратная связь предназначена для ручного разбора и не обучает алгоритм автоматически.', 'Feedback is for manual review and does not automatically train the algorithm.'))

with st.expander(t('Метод и ограничения', 'Method and limitations')):
    st.write(t('EN: VADER. RU: небольшой словарь со стеммингом и учётом отрицаний. Аспекты: ключевые слова и фрагменты предложений. Neutral означает нейтральный текст или отсутствие словарного сигнала. Evidence — наличие лексических сигналов, а не вероятность правильности. Сложные отрицания, сленг, ирония и несколько аспектов в одном фрагменте могут давать ошибки.',
               'EN: VADER. RU: a small stemmed lexicon with negation rules. Aspects: keywords and clauses. Neutral means neutral text or no lexicon signal. Evidence describes lexical signals, not a probability of correctness. Complex negation, slang, irony and multiple aspects in one clause can cause errors.'))
    st.write(t('Sentence Transformers используется только для группировки и только из локальной папки модели. Работа без API и автоматического скачивания моделей. Для оценки качества есть отдельная учебная размеченная выборка.',
               'Sentence Transformers is used only for clustering, from a local model folder. No API or automatic model downloads. A separate educational labeled fixture is provided for quality checks.'))
