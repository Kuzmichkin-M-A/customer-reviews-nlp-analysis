"""Aspect templates + clause-level sentiment, deliberately transparent heuristics."""
import re
import pandas as pd
from reviewlens.nlp import sentiment, split_clauses

# Stable keys keep saved edits independent of the interface language.
ASPECTS = {
    'service': ('Обслуживание', 'Service', r'персонал\w*|официант\w*|сервис\w*|обслуж\w*|сотрудник\w*|staff|service|waiter\w*|employee\w*|rude|груб\w*|хам\w*'),
    'food': ('Еда и напитки', 'Food and drinks', r'ед[ауы]|кухн\w*|блюд\w*|вкус\w*|кофе|напит\w*|food|meal\w*|dish\w*|coffee|tasty|delicious|breakfast|restaurant\w*'),
    'price': ('Цена и ценность', 'Price and value', r'цен[аыуе]\w*|дорог\w*|дешев\w*|денег|стоим\w*|price\w*|expensive|cheap|money|cost\w*|value'),
    'waiting': ('Очереди и ожидание', 'Queues and waiting', r'очеред\w*|ожидан\w*|жда\w*|долго|queue\w*|wait\w*|line\w*'),
    'cleanliness': ('Чистота', 'Cleanliness', r'чист\w*|гряз\w*|уборк\w*|clean\w*|dirty|dirt\w*'),
    'atmosphere': ('Атмосфера и комфорт', 'Atmosphere and comfort', r'атмосфер\w*|уют\w*|душно|кондиционер\w*|шум\w*|комфорт\w*|atmosphere|cozy|noisy|noise|comfortable|crowd\w*|hot'),
    'delivery': ('Доставка', 'Delivery', r'достав\w*|курьер\w*|опозд\w*|delivery|deliver\w*|courier\w*|shipping'),
    'quality': ('Качество товара', 'Product quality', r'качеств\w*|товар\w*|брак\w*|слома\w*|quality|product\w*|broken|defect\w*'),
    'returns': ('Возврат и поддержка', 'Returns and support', r'возврат\w*|поддерж\w*|refund\w*|return\w*|support'),
    'attractions': ('Аттракционы и развлечения', 'Attractions and entertainment', r'аттракцион\w*|развлеч\w*|ride\w*|attraction\w*|parade\w*|show\w*|firework\w*'),
    'other': ('Без определённого аспекта', 'Unspecified aspect', r'(?!)'),
}
TEMPLATES = {
    'general': list(ASPECTS),
    'food': ['service', 'food', 'price', 'waiting', 'cleanliness', 'atmosphere', 'delivery', 'other'],
    'ecommerce': ['quality', 'price', 'delivery', 'returns', 'service', 'other'],
    'services': ['service', 'price', 'waiting', 'cleanliness', 'atmosphere', 'other'],
}
PATTERNS = {key: re.compile(r'\b(?:' + value[2] + r')\b', re.I) for key, value in ASPECTS.items()}
MENTION_COLS = ['Review_ID', 'Aspect_Key', 'Aspect', 'Sentiment_Class', 'Snippet', 'Evidence']


def title(key, lang='ru'):
    return ASPECTS[key][0 if lang == 'ru' else 1]


def combine_labels(labels):
    labels = set(labels)
    if 'Mixed' in labels or {'Positive', 'Negative'} <= labels:
        return 'Mixed'
    return 'Positive' if 'Positive' in labels else 'Negative' if 'Negative' in labels else 'Neutral'


def extract_aspects(df, template='general', lang='ru'):
    records = []
    for row in df.itertuples():
        found = {}
        clauses = split_clauses(row.Review_Text)
        for clause in clauses:
            # Conjunctions separate different targets only when both halves have an aspect.
            parts = re.split(r'\b(?:and|и)\b', clause, flags=re.I)
            parts_with_aspects = sum(any(PATTERNS[k].search(p) for k in TEMPLATES[template]) for p in parts)
            units = parts if parts_with_aspects >= 2 else [clause]
            for unit in units:
                for key in TEMPLATES[template]:
                    if PATTERNS[key].search(unit):
                        found.setdefault(key, []).append(unit.strip())
        if not found:
            found = {'other': [row.Review_Text]}
        for key, snippets in found.items():
            results = [sentiment(s, row.Language) for s in snippets]
            records.append({'Review_ID': row.Review_ID, 'Aspect_Key': key, 'Aspect': title(key, lang),
                            'Sentiment_Class': combine_labels(r['Sentiment_Class'] for r in results),
                            'Snippet': ' … '.join(dict.fromkeys(snippets)),
                            'Evidence': 'none' if all(r['Evidence'] == 'none' for r in results) else 'limited'})
    return pd.DataFrame(records, columns=MENTION_COLS)


def apply_aspect_edits(mentions, edits):
    """Same destination merges themes; each review counted once after a merge."""
    if mentions.empty:
        return mentions.copy()
    edit_map = edits.set_index('Aspect_Key').to_dict('index')
    rows = []
    for row in mentions.to_dict('records'):
        edit = edit_map.get(row['Aspect_Key'], {})
        if edit.get('Hidden', False):
            continue
        name = str(edit.get('Name', row['Aspect'])).strip()
        row['Aspect'] = name or row['Aspect']
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=MENTION_COLS)
    out = []
    for (rid, name), group in pd.DataFrame(rows).groupby(['Review_ID', 'Aspect'], sort=False):
        out.append({'Review_ID': rid, 'Aspect': name,
                    'Aspect_Key': '|'.join(sorted(group['Aspect_Key'].unique())),
                    'Sentiment_Class': combine_labels(group['Sentiment_Class']),
                    'Snippet': ' … '.join(dict.fromkeys(group['Snippet'])),
                    'Evidence': 'none' if group['Evidence'].eq('none').all() else 'limited'})
    return pd.DataFrame(out, columns=MENTION_COLS)


def aspect_stats(mentions, reviews):
    columns = ['Aspect', 'Count', 'Positive', 'Negative', 'Mixed', 'Neutral', 'Pos_Share', 'Neg_Share', 'Avg_Rating']
    if mentions.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for aspect, group in mentions.groupby('Aspect', sort=False):
        counts = group['Sentiment_Class'].value_counts()
        ids = set(group['Review_ID'])
        ratings = reviews.loc[reviews['Review_ID'].isin(ids), 'Rating']
        row = {'Aspect': aspect, 'Count': len(ids), 'Avg_Rating': ratings.mean()}
        row.update({s: int(counts.get(s, 0)) for s in ('Positive', 'Negative', 'Mixed', 'Neutral')})
        row.update(Pos_Share=round(row['Positive']/len(ids)*100, 1), Neg_Share=round(row['Negative']/len(ids)*100, 1))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)
