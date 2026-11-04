import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest
import pandas as pd
from reviewlens.data_io import prepare_data, suggest_columns
from reviewlens.nlp import enrich
from reviewlens.aspects import TEMPLATES, title

ROOT = Path(__file__).resolve().parents[1]


class AppTests(unittest.TestCase):
    def run_demo(self, demo, language='ru'):
        app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=90).run()
        self.assertFalse(app.exception)
        if language == 'en':
            app.sidebar.selectbox[0].set_value('en').run()
        app.sidebar.selectbox[1].set_value(demo).run()
        label = 'Анализировать' if language == 'ru' else 'Analyze'
        next(b for b in app.button if b.label == label).click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, '30')
        next(b for b in app.button if b.label in ('Найти темы', 'Find topics')).click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state['topics'][3])
        return app

    def test_russian_demo_and_filters(self):
        app = self.run_demo('RU demo')
        # Empty filtered subsets must show a message instead of crashing.
        app.sidebar.multiselect[2].set_value(['Negative']).run()
        app.sidebar.multiselect[3].set_value([5]).run()
        self.assertFalse(app.exception)

    def test_english_demo(self):
        self.run_demo('EN demo', 'en')

    def test_text_only_all_aspects_hidden(self):
        raw = pd.DataFrame({'text': ['The service was excellent.']})
        frame = enrich(prepare_data(raw, suggest_columns(raw.columns))[0])
        app = AppTest.from_file(str(ROOT/'app.py'), default_timeout=90)
        for key, value in {'reviews': frame, 'validation': {}, 'source': 'text-only.csv', 'run_id': 'unit'}.items():
            app.session_state[key] = value
        app.session_state['restored-editor-unit-general-ru'] = pd.DataFrame([
            {'Aspect_Key': key, 'Name': title(key), 'Hidden': True} for key in TEMPLATES['general']])
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, '1')
        self.assertEqual(app.metric[3].value, '—')
        self.assertTrue(app.session_state['base-editor-unit-general-ru']['Hidden'].all())

    def test_source_quotes_and_feedback(self):
        app = self.run_demo('RU demo')
        self.assertTrue(app.get('popover'))
        next(x for x in app.text_input if x.label == 'Почему вывод неверен?').set_value('Проверить аспект')
        next(x for x in app.button if x.label == 'Вывод неверен — сохранить замечание').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['feedback'][-1]['Comment'], 'Проверить аспект')


if __name__ == '__main__':
    unittest.main()
