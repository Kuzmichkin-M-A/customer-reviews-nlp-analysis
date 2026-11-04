"""Reproducible summaries; counters always derive from actual mention rows."""
import pandas as pd
from reviewlens.aspects import aspect_stats


def top_findings(stats, polarity):
    if stats.empty:
        return stats.copy()
    # Do not invent strengths/problems when there are no polar mentions.
    return stats.loc[stats[polarity] > 0].sort_values(
        [polarity, 'Count', 'Aspect'], ascending=[False, False, True]).head(3)


def make_report(stats, reviews, source, lang='ru'):
    ru = lang == 'ru'
    lines = [f'ReviewLens — {source}',
             f"{'Отзывов' if ru else 'Reviews'}: {len(reviews)}",
             'Метод: словарная тональность EN/RU, аспекты по ключевым словам.' if ru else 'Method: EN/RU lexicon sentiment, keyword-based aspects.',
             'Доли считаются от отзывов, упоминающих аспект; смешанные отзывы выделены отдельно.' if ru else 'Shares use reviews mentioning an aspect; mixed sentiment is separate.']
    for polarity, label in [('Negative', 'Проблемы' if ru else 'Problems'), ('Positive', 'Сильные стороны' if ru else 'Strengths')]:
        lines += ['', label]
        findings = top_findings(stats, polarity)
        if findings.empty:
            lines += ['Нет найденных упоминаний.' if ru else 'No matching mentions.']
        for row in findings.itertuples():
            count = getattr(row, polarity)
            lines.append(f'- {row.Aspect}: {count} / {row.Count}')
    return '\n'.join(lines)


def compare_periods(reviews, mentions, previous, current, threshold=15, min_reviews=5):
    """Disjoint inclusive ranges. Dates absent from source stay outside comparison."""
    a, b = (pd.Timestamp(x) for x in previous)
    c, d = (pd.Timestamp(x) for x in current)
    if a > b or c > d or b >= c:
        raise ValueError('Periods must be ordered and disjoint / Периоды должны идти по порядку и не пересекаться.')
    frames = []
    for start, end in ((a, b), (c, d)):
        subset = reviews[reviews['Date'].ge(start) & reviews['Date'].lt(end + pd.Timedelta(days=1))]
        frame = aspect_stats(mentions[mentions['Review_ID'].isin(subset['Review_ID'])], subset)
        frames.append(frame.set_index('Aspect')[['Count', 'Neg_Share']])
    joined = frames[0].add_suffix('_Before').join(frames[1].add_suffix('_After'), how='outer')
    for col in ('Count_Before', 'Count_After'):
        joined[col] = joined[col].fillna(0).astype(int)
    joined['Change_pp'] = joined['Neg_Share_After'] - joined['Neg_Share_Before']
    joined['Alert'] = ((joined['Count_Before'] >= min_reviews) & (joined['Count_After'] >= min_reviews) & (joined['Change_pp'] >= threshold))
    return joined.reset_index()
