"""Optional smoke check; requires a prepared local model, runs offline."""
import os
from pathlib import Path
import json
import argparse
from time import perf_counter

os.environ['HF_HUB_OFFLINE'] = '1'
from reviewlens.data_io import read_table, prepare_data, suggest_columns
from reviewlens.nlp import enrich, analyze_clusters

ROOT = Path(__file__).resolve().parents[1]


def run(model_path):
    start = perf_counter()
    frames = []
    for filename in ('demo_ru.csv', 'demo_en.csv'):
        raw, _ = read_table((ROOT/'data'/filename).read_bytes(), filename)
        frame, _ = prepare_data(raw, suggest_columns(raw.columns))
        frame['Review_ID'] = filename + ':' + frame['Review_ID']
        frames.append(frame)
    import pandas as pd
    frame = enrich(pd.concat(frames, ignore_index=True))
    grouped, themes = analyze_clusters(frame, 6, 'embeddings', model_path)
    assert len(grouped) == 60
    assert grouped['Cluster'].ge(0).all()
    assert len(themes) > 1
    result = {'Model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
              'Mode': 'local_files_only, HF_HUB_OFFLINE=1', 'Reviews': len(frame),
              'Languages': sorted(frame['Language'].unique()), 'Topics': themes,
              'Seconds': round(perf_counter()-start, 2),
              'Note': 'Execution check only; no claim of improved clustering quality.'}
    (ROOT/'validation'/'embeddings.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=str(ROOT/'models'/'paraphrase-multilingual-MiniLM-L12-v2'))
    run(parser.parse_args().model)
