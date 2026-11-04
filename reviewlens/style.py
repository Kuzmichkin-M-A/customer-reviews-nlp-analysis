"""Единое оформление интерфейса: тёмная тема с оранжевым акцентом.

Тема самого Streamlit задаётся в .streamlit/config.toml, здесь — цвета графиков,
CSS-дополнения и общий стиль Plotly. Логики анализа здесь нет.
"""

APP_NAME = 'ReviewLens'  # рабочее название, меняется в одном месте

BG = '#0F1117'
CARD = '#1A1D26'
BORDER = '#2A2F3A'
TEXT = '#E5E7EB'
MUTED = '#9CA3AF'
PRIMARY = '#FF7A1A'

# Цвета тональности, контрастные на тёмном фоне. Mixed — фиолетовый,
# чтобы не путать его с оранжевым акцентом интерфейса.
SENTIMENT_COLORS = {'Positive': '#22C55E', 'Negative': '#EF4444',
                    'Neutral': '#94A3B8', 'Mixed': '#A78BFA'}
PALETTE = ['#FF7A1A', '#38BDF8', '#A78BFA', '#22C55E', '#F472B6', '#FACC15']

CSS = f'''
<style>
[data-testid="stMetric"] {{
    background: {CARD};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 16px;
}}
.hero-title {{font-size: 3.2rem !important; font-weight: 800; line-height: 1.1; margin: 0.5rem 0 0.6rem 0;}}
.hero-title span {{color: {PRIMARY};}}
.hero-sub {{font-size: 1.2rem !important; color: {MUTED}; max-width: 760px; line-height: 1.5;}}
.report-title {{font-size: 2rem !important; font-weight: 700; margin: 0.2rem 0 0.2rem 0;}}
.report-title span {{color: {PRIMARY};}}
.feature-icon {{font-size: 1.8rem;}}
.step-num {{display: inline-block; width: 28px; height: 28px; border-radius: 50%;
           background: {PRIMARY}; color: {BG}; font-weight: 700;
           text-align: center; line-height: 28px; margin-right: 8px;}}
.cta {{margin-top: 1.5rem; padding: 14px 18px; border-left: 4px solid {PRIMARY};
      background: {CARD}; border-radius: 8px;}}
</style>
'''


def style_fig(fig, height=400):
    '''Применяет общий стиль к графику Plotly'''
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=height,
        font=dict(family='Arial', size=13, color=TEXT),
        title=dict(font=dict(size=16)),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation='h', y=-0.2),
        colorway=PALETTE,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=BORDER)
    return fig


def translate_legend(fig, names):
    '''Переводит названия серий в легенде (например, Positive → Положительная)'''
    fig.for_each_trace(lambda t: t.update(name=names.get(t.name, t.name)))
    return fig
