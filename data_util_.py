"""File import and validation. No Streamlit state or network access."""
from io import BytesIO, StringIO
import csv
import re
import pandas as pd

ALIASES = {
    'Review_Text': ['review_text', 'text', 'review', 'отзыв', 'текст', 'текст отзыва', 'content'],
    'Rating': ['rating', 'stars', 'score', 'оценка', 'рейтинг'],
    'Date': ['year_month', 'date', 'дата', 'дата отзыва'],
    'Branch': ['branch', 'company', 'product', 'филиал', 'компания', 'товар'],
    'Reviewer_Location': ['reviewer_location', 'location', 'город', 'локация'],
    'Original_ID': ['review_id', 'id', 'идентификатор'],
    'Source_URL': ['url', 'link', 'ссылка'],
}


def suggest_columns(columns):
    lookup = {str(c).strip().casefold(): c for c in columns}
    return {field: next((lookup[a] for a in names if a in lookup), None)
            for field, names in ALIASES.items()}


def read_table(payload, filename, encoding='auto', separator='auto', sheet=0):
    if filename.lower().endswith('.xlsx'):
        frame = pd.read_excel(BytesIO(payload), sheet_name=sheet, dtype=str)
        meta = {'encoding': 'XLSX', 'separator': '', 'sheet': str(sheet)}
    elif filename.lower().endswith('.csv'):
        if encoding == 'auto':
            text = None
            for enc in ('utf-8-sig', 'cp1251', 'cp1252'):
                try:
                    candidate = payload.decode(enc)
                    # cp1251 can decode most Western bytes: reject mojibake-like candidates.
                    if enc == 'cp1251':
                        sample = candidate[:20000]
                        letters = len(re.findall('[A-Za-zА-Яа-яЁё]', sample))
                        if len(re.findall('[А-Яа-яЁё]', sample)) / max(1, letters) < 0.2:
                            continue
                    text, encoding = candidate, enc
                    break
                except UnicodeDecodeError:
                    pass
            if text is None:
                raise ValueError('Cannot decode CSV. Select an encoding manually / Выберите кодировку.')
        else:
            text = payload.decode(encoding)
        if separator == 'auto':
            try:
                separator = csv.Sniffer().sniff(text[:65536], delimiters=',;\t|').delimiter
            except csv.Error:
                separator = ','
        frame = pd.read_csv(StringIO(text), sep=separator, dtype=str, keep_default_na=False)
        meta = {'encoding': encoding, 'separator': repr(separator)}
    else:
        raise ValueError('Supported formats: CSV, XLSX / Поддерживаются CSV и XLSX.')
    frame.columns = [str(c).strip() for c in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError('Duplicate column names / Повторяющиеся названия колонок.')
    if frame.empty:
        raise ValueError('The file has no rows / В файле нет строк.')
    return frame, meta


def prepare_data(raw, mapping):
    text_col = mapping.get('Review_Text')
    if not text_col or text_col not in raw:
        raise ValueError('Select a text column / Выберите колонку с текстом отзыва.')
    frame = pd.DataFrame(index=raw.index)
    # Stable source row number (header = row 1), also works for repeated external IDs.
    frame['Source_Row'] = range(2, len(raw) + 2)
    frame['Review_ID'] = frame['Source_Row'].map(lambda n: f'R{n:06d}')
    for field in ALIASES:
        col = mapping.get(field)
        frame[field] = raw[col] if col in raw.columns else pd.NA
    valid = frame['Review_Text'].map(lambda t: isinstance(t, str) and bool(re.search('[a-zA-Zа-яА-ЯёЁ]', t)))
    dropped = int((~valid).sum())
    frame = frame.loc[valid].copy()
    frame['Review_Text'] = frame['Review_Text'].str.strip()
    numeric = pd.to_numeric(frame['Rating'].astype('string').str.replace(',', '.', regex=False), errors='coerce')
    valid_rating = (numeric.between(1, 5) & (numeric % 1 == 0)).fillna(False)
    invalid_ratings = int((frame['Rating'].notna() & frame['Rating'].ne('') & ~valid_rating).sum())
    frame['Rating'] = numeric.where(valid_rating)
    dates = frame['Date'].astype('string').str.strip()
    dates = dates.replace({'': pd.NA, 'missing': pd.NA})
    frame['Date'] = pd.to_datetime(dates, format='mixed', errors='coerce', dayfirst=False, utc=True).dt.tz_convert(None)
    invalid_dates = int((dates.notna() & frame['Date'].isna()).sum())
    for field in ('Branch', 'Reviewer_Location'):
        frame[field] = frame[field].fillna('').astype(str).str.strip().replace('', '—')
    frame['Rating_Class'] = frame['Rating'].map(
        lambda x: 'Unknown' if pd.isna(x) else 'Positive' if x >= 4 else 'Negative' if x <= 2 else 'Neutral')
    duplicates = int(frame['Review_Text'].duplicated().sum())
    return frame.reset_index(drop=True), {
        'input': len(raw), 'kept': len(frame), 'removed_text': dropped,
        'invalid_ratings': invalid_ratings, 'invalid_dates': invalid_dates,
        'repeated_texts_kept': duplicates,
    }


def csv_bytes(frame):
    """Excel-friendly CSV; escape formula-like strings in user supplied cells."""
    out = frame.copy()
    for col in out.select_dtypes(include=['object', 'string']).columns:
        out[col] = out[col].map(lambda v: "'" + v if isinstance(v, str) and v.lstrip().startswith(('=', '+', '-', '@')) else v)
    return out.to_csv(index=False).encode('utf-8-sig')
