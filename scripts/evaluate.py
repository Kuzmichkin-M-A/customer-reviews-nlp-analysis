"""Diagnostic educational fixtures, not an independent benchmark."""
from pathlib import Path
import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from reviewlens.data_io import prepare_data, suggest_columns
from reviewlens.nlp import sentiment, enrich
from reviewlens.aspects import extract_aspects

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['Positive', 'Negative', 'Neutral', 'Mixed']


def run():
    sample = pd.read_csv(ROOT/'data'/'evaluation'/'quality_sample.csv')
    analyzer = SentimentIntensityAnalyzer()
    def old(text):
        score = analyzer.polarity_scores(text)['compound']
        return 'Positive' if score >= .05 else 'Negative' if score <= -.05 else 'Neutral'
    sample['Original_VADER'] = sample['Review_Text'].map(old)
    sample['V2'] = sample['Review_Text'].map(lambda t: sentiment(t)['Sentiment_Class'])
    metrics = []
    for language, group in sample.groupby('Language'):
        for method in ('Original_VADER', 'V2'):
            metrics.append({'Language': language, 'Method': method, 'N': len(group),
                'Accuracy': round(accuracy_score(group['Expected'], group[method]), 4),
                'Macro_F1_4_classes': round(f1_score(group['Expected'], group[method], labels=LABELS,
                                                       average='macro', zero_division=0), 4)})
    aspects = pd.read_csv(ROOT/'data'/'evaluation'/'aspect_quality_sample.csv')
    frame = enrich(prepare_data(aspects, suggest_columns(aspects.columns))[0])
    mentions = extract_aspects(frame)
    aspect_metrics = []
    aspect_predictions = []
    for language in ('en', 'ru'):
        tp = fp = fn = exact = total = 0
        for i, row in aspects[aspects['Language'] == language].iterrows():
            expected = set(row.Expected.split('|'))
            selected = mentions[mentions['Review_ID'] == frame.iloc[i]['Review_ID']]
            predicted = {f'{r.Aspect_Key}:{r.Sentiment_Class}' for r in selected.itertuples()}
            tp += len(expected & predicted)
            fp += len(predicted - expected)
            fn += len(expected - predicted)
            exact += predicted == expected
            total += 1
            aspect_predictions.append({'Language': language, 'Review_Text': row.Review_Text,
                                       'Expected': row.Expected, 'Predicted': '|'.join(sorted(predicted))})
        aspect_metrics.append({'Language': language, 'N': total, 'Exact_review_match': round(exact/total, 4),
                               'Aspect_sentiment_micro_F1': round(2*tp/max(1, 2*tp+fp+fn), 4)})
    out = ROOT/'validation'
    out.mkdir(exist_ok=True)
    pd.DataFrame(metrics).to_csv(out/'quality_metrics.csv', index=False, encoding='utf-8-sig')
    sample.to_csv(out/'quality_predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(aspect_metrics).to_csv(out/'aspect_quality_metrics.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(aspect_predictions).to_csv(out/'aspect_quality_predictions.csv', index=False, encoding='utf-8-sig')
    print(json.dumps({'sentiment': metrics, 'aspects': aspect_metrics}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
