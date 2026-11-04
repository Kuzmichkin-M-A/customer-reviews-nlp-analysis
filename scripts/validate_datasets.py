"""Integration check on both user supplied datasets; no sample truncation."""
from pathlib import Path
from time import perf_counter
import json
from reviewlens.data_io import read_table, prepare_data, suggest_columns
from reviewlens.nlp import enrich, analyze_clusters
from reviewlens.aspects import extract_aspects, aspect_stats

ROOT = Path(__file__).resolve().parents[1]


def run():
    results = []
    for name in ('test_reviews_ru_5000.csv', 'DisneylandReviews.csv'):
        source = ROOT/'data'/'datasets'/name
        if not source.exists():
            print(f'Skipped {name}: put it into data/datasets/ to include it.', flush=True)
            continue
        start = perf_counter()
        raw, meta = read_table(source.read_bytes(), name)
        frame, checks = prepare_data(raw, suggest_columns(raw.columns))
        frame = enrich(frame)
        mentions = extract_aspects(frame)
        stats = aspect_stats(mentions, frame)
        clustered, themes = analyze_clusters(frame)
        assert len(frame) == len(clustered)
        assert frame['Review_ID'].is_unique
        assert set(mentions['Review_ID']) <= set(frame['Review_ID'])
        assert not mentions.duplicated(['Review_ID', 'Aspect_Key']).any()
        for row in stats.itertuples():
            assert row.Count == row.Positive + row.Negative + row.Mixed + row.Neutral
            assert row.Count <= len(frame)
        results.append({'File': name, 'Read': meta, 'Validation': checks, 'Mentions': len(mentions),
                        'Topics': len(themes), 'Sentiment': frame['Sentiment_Class'].value_counts().to_dict(),
                        'Seconds': round(perf_counter()-start, 2)})
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    out = ROOT/'validation'
    out.mkdir(exist_ok=True)
    (out/'datasets.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    run()
