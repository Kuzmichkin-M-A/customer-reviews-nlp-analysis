import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_util_ import read_table, suggest_columns, prepare_data, csv_bytes
from nlp_ import enrich, sentiment, clean_text, analyze_clusters, get_top_ngrams
from aspects import extract_aspects, apply_aspect_edits, aspect_stats
from reporting import compare_periods, top_findings, make_report


def reviews(texts, **extra):
    raw = pd.DataFrame({'Review_Text': texts, **extra})
    return enrich(prepare_data(raw, suggest_columns(raw.columns))[0])


class ImportTests(unittest.TestCase):
    def test_encoding_separator_and_mapping(self):
        for encoding in ('utf-8-sig', 'cp1251'):
            raw, meta = read_table('текст;оценка\n"Отличный сервис; спасибо";5\n'.encode(encoding), 'a.csv')
            data, checks = prepare_data(raw, suggest_columns(raw.columns))
            self.assertEqual(data.iloc[0]['Review_Text'], 'Отличный сервис; спасибо')
            self.assertEqual(data.iloc[0]['Rating'], 5)
            self.assertEqual(meta['separator'], "';'")
            self.assertEqual(checks['kept'], 1)

    def test_optional_fields_and_stable_source_rows(self):
        raw = pd.DataFrame({'text': ['good service', '', None, '123!!', 'great food'],
                            'rating': ['6', '5', '4', '3', 'bad'],
                            'date': ['missing', '2026-01', '', '', '2026-02-01']})
        frame, stats = prepare_data(raw, suggest_columns(raw.columns))
        self.assertEqual(frame['Source_Row'].tolist(), [2, 6])
        self.assertEqual(frame['Review_ID'].tolist(), ['R000002', 'R000006'])
        self.assertEqual(stats['removed_text'], 3)
        self.assertEqual(stats['invalid_ratings'], 2)
        self.assertTrue(frame['Rating'].isna().all())
        self.assertEqual(frame['Date'].notna().sum(), 1)

    def test_text_only_and_empty_errors(self):
        frame = reviews(['good', 'good'])
        self.assertTrue(frame['Rating'].isna().all())
        self.assertEqual(frame['Review_ID'].nunique(), 2)
        with self.assertRaises(ValueError):
            read_table(b'text\n', 'empty.csv')
        with self.assertRaises(ValueError):
            prepare_data(pd.DataFrame({'x': ['abc']}), {})

    def test_safe_csv(self):
        value = csv_bytes(pd.DataFrame({'text': ['=1+1', '@A', 'normal', '  +SUM(A1)']})).decode('utf-8-sig')
        self.assertIn("'=1+1", value)
        self.assertIn("'  +SUM(A1)", value)


class NLPTests(unittest.TestCase):
    def test_rating_independence_and_russian(self):
        frame = reviews(['Ужасное обслуживание', 'Ужасное обслуживание'], Rating=['1', '5'])
        self.assertEqual(frame['Sentiment_Class'].tolist(), ['Negative', 'Negative'])
        self.assertEqual(frame['Is_Contradictory'].tolist(), [False, True])
        self.assertTrue(any('а' <= c <= 'я' for c in clean_text('Отличный персонал')))
        self.assertEqual(sentiment('Не рекомендую')['Sentiment_Class'], 'Negative')
        self.assertEqual(sentiment('Не стоит своих денег')['Sentiment_Class'], 'Negative')
        self.assertEqual(sentiment('The service is not good')['Sentiment_Class'], 'Negative')

    def test_aspect_locality_and_mixed(self):
        data = reviews(['Еда вкусная, но персонал грубый.', 'The food is delicious but the staff are rude.'])
        mentions = extract_aspects(data)
        for rid in data['Review_ID']:
            group = mentions[mentions['Review_ID'] == rid].set_index('Aspect_Key')
            self.assertEqual(group.loc['food', 'Sentiment_Class'], 'Positive')
            self.assertEqual(group.loc['service', 'Sentiment_Class'], 'Negative')
        self.assertTrue(data['Mixed'].all())

    def test_editor_merge_hide_and_recount(self):
        data = reviews(['Еда вкусная, но персонал грубый.', 'Еда отличная'])
        mentions = extract_aspects(data)
        edits = pd.DataFrame([{'Aspect_Key': 'food', 'Name': 'Опыт', 'Hidden': False},
                              {'Aspect_Key': 'service', 'Name': 'Опыт', 'Hidden': False}])
        merged = apply_aspect_edits(mentions, edits)
        self.assertEqual(len(merged), 2)
        stats = aspect_stats(merged, data).iloc[0]
        self.assertEqual(stats['Count'], 2)
        self.assertEqual(stats['Mixed'], 1)
        self.assertEqual(stats['Positive'], 1)
        edits['Hidden'] = True
        self.assertTrue(apply_aspect_edits(mentions, edits).empty)

    def test_cluster_edges_and_determinism(self):
        self.assertEqual(get_top_ngrams(['']), [])
        single = reviews(['the and a'])
        grouped, names = analyze_clusters(single)
        self.assertEqual(grouped.iloc[0]['Cluster'], -1)
        same = reviews(['good food'] * 4)
        grouped, names = analyze_clusters(same, 10)
        self.assertEqual(grouped['Cluster'].nunique(), 1)
        data = reviews(['great food', 'bad service', 'fast delivery', 'great food'])
        a, _ = analyze_clusters(data, 2)
        b, _ = analyze_clusters(data, 2)
        self.assertEqual(a['Cluster'].tolist(), b['Cluster'].tolist())
        self.assertEqual(get_top_ngrams(['food good food good'], limit=1), [('food good', 2)])
        with self.assertRaises((ValueError, ImportError)):
            analyze_clusters(data, method='embeddings', model_path='does-not-exist')


class ReportingTests(unittest.TestCase):
    def test_periods_alerts_and_missing_denominator(self):
        data = reviews(['Отличный персонал'] * 5 + ['Персонал грубый'] * 5,
                       Date=['2026-01-01']*5 + ['2026-01-08']*5)
        mentions = extract_aspects(data)
        result = compare_periods(data, mentions, ('2026-01-01', '2026-01-07'), ('2026-01-08', '2026-01-14'))
        self.assertTrue(result.iloc[0]['Alert'])
        self.assertEqual(result.iloc[0]['Change_pp'], 100)
        sparse = compare_periods(data, mentions, ('2025-12-01', '2025-12-07'), ('2026-01-08', '2026-01-14'))
        self.assertFalse(sparse.iloc[0]['Alert'])
        self.assertTrue(pd.isna(sparse.iloc[0]['Neg_Share_Before']))
        with self.assertRaises(ValueError):
            compare_periods(data, mentions, ('2026-01-01', '2026-01-08'), ('2026-01-08', '2026-01-14'))

    def test_no_fabricated_findings(self):
        data = reviews(['Еда отличная'])
        stats = aspect_stats(extract_aspects(data), data)
        self.assertTrue(top_findings(stats, 'Negative').empty)
        self.assertIn('Нет найденных упоминаний', make_report(stats, data, 'test'))


if __name__ == '__main__':
    unittest.main()
